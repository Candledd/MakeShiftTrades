"""Machine Learning Confidence Layer for ICT / SMC Signals
===========================================================

Architecture
------------
  Feature engineering   18 per-bar features derived from OHLCV +
                        SMC indicators (FVG, MS, OB, Engulfing, Liquidity).
  Training universe     SPY, QQQ, IWM, GLD, TLT, ES=F, NQ=F  (daily, 2 yr).
  Model                 GradientBoostingClassifier (3-class: BUY/HOLD/SELL).
  Labels                Forward 3-bar return sign, thresholded at ±0.5 × ATR.
  Output                dict with signal, confidence (0-100), probabilities.

The model is trained lazily on the first call (background thread started by
server.py) and cached in memory for the lifetime of the server process.
Training takes ~10-20 s; inference is <1 s.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np
import pandas as pd

from charts.data import fetch_ohlcv
from charts.indicators.engulfing import detect_engulfing
from charts.indicators.fvg import detect_fvg
from charts.indicators.liquidity import detect_liquidity_levels
from charts.indicators.price_action import (
    detect_market_structure,
    detect_order_blocks,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Training configuration
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_SYMBOLS  = ["SPY", "QQQ", "IWM", "GLD", "TLT", "ES=F", "NQ=F"]
TRAIN_PERIOD   = "2y"
TRAIN_INTERVAL = "1d"
FWD_BARS       = 3       # forward bars used for label
ATR_MULT       = 0.5     # label threshold multiplier (×ATR)

# int class  → signal string
LABEL_MAP = {0: "BUY", 1: "HOLD", 2: "SELL"}


# ─────────────────────────────────────────────────────────────────────────────
# Technical helpers (all vectorized, no future-leak)
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Per-bar SMC feature aggregators
# ─────────────────────────────────────────────────────────────────────────────

def _make_trend_series(
    df: pd.DataFrame, ms_events: list[dict]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bar: trend (+1 bull / -1 bear / 0 none), 20-bar BOS & CHoCH counts."""
    n        = len(df)
    trend    = np.zeros(n, dtype=np.float32)
    bos_pt   = np.zeros(n, dtype=np.float32)
    choch_pt = np.zeros(n, dtype=np.float32)

    for ev in ms_events:
        ts = ev["date"]
        try:
            idx = df.index.get_loc(ts)
        except KeyError:
            idx = int(df.index.searchsorted(ts))
        if idx >= n:
            continue
        t = ev.get("type", "")
        trend[idx:] = 1.0 if "bull" in t else -1.0
        label = ev.get("label", "")
        if label == "BOS":
            bos_pt[idx] += 1.0
        elif label == "CHoCH":
            choch_pt[idx] += 1.0

    bos20   = pd.Series(bos_pt,   index=df.index).rolling(20, min_periods=1).sum().to_numpy()
    choch20 = pd.Series(choch_pt, index=df.index).rolling(20, min_periods=1).sum().to_numpy()
    return trend, bos20, choch20


def _make_fvg_features(
    df: pd.DataFrame, fvg_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-bar: bull/bear FVG active-at-price flags + active counts."""
    n      = len(df)
    closes = df["Close"].to_numpy()
    bull_active = np.zeros(n, dtype=np.float32)
    bear_active = np.zeros(n, dtype=np.float32)
    bull_cnt    = np.zeros(n, dtype=np.float32)
    bear_cnt    = np.zeros(n, dtype=np.float32)

    if fvg_df.empty:
        return bull_active, bear_active, bull_cnt, bear_cnt

    for _, row in fvg_df.iterrows():
        start = int(row["origin_idx"])
        try:
            end_bar = df.index.get_loc(row["end_date"])
        except KeyError:
            end_bar = int(df.index.searchsorted(row["end_date"]))

        top    = float(row["top"])
        bottom = float(row["bottom"])
        is_bull = row["type"] == "bullish"

        for i in range(start, min(int(end_bar) + 1, n)):
            if is_bull:
                bull_cnt[i] += 1.0
                if bottom <= closes[i] <= top:
                    bull_active[i] = 1.0
            else:
                bear_cnt[i] += 1.0
                if bottom <= closes[i] <= top:
                    bear_active[i] = 1.0

    return bull_active, bear_active, bull_cnt, bear_cnt


def _make_ob_features(
    df: pd.DataFrame, obs: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar: 1 if price is within 1.5×OB-height of a bull/bear Order Block."""
    n      = len(df)
    closes = df["Close"].to_numpy()
    bull_ob = np.zeros(n, dtype=np.float32)
    bear_ob = np.zeros(n, dtype=np.float32)

    for ob in obs:
        try:
            ob_idx = df.index.get_loc(ob["date"])
        except KeyError:
            ob_idx = int(df.index.searchsorted(ob["date"]))

        top     = float(ob["top"])
        bottom  = float(ob["bottom"])
        tol     = (top - bottom) * 1.5
        is_bull = ob["type"] == "bullish"

        for i in range(int(ob_idx), n):
            c = closes[i]
            if is_bull and (bottom - tol) <= c <= (top + tol):
                bull_ob[i] = 1.0
            elif not is_bull and (bottom - tol) <= c <= (top + tol):
                bear_ob[i] = 1.0

    return bull_ob, bear_ob


def _make_engulf_features(
    df: pd.DataFrame, engulf_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar: 1 if a bull/bear engulfing occurred within the last 3 bars."""
    n    = len(df)
    bull = np.zeros(n, dtype=np.float32)
    bear = np.zeros(n, dtype=np.float32)

    if engulf_df.empty:
        return bull, bear

    for _, row in engulf_df.iterrows():
        try:
            idx = df.index.get_loc(row["date"])
        except KeyError:
            idx = int(df.index.searchsorted(row["date"]))
        # Mark the engulfing bar and the 2 bars that follow
        for j in range(int(idx), min(int(idx) + 3, n)):
            if row["type"] == "bullish":
                bull[j] = 1.0
            else:
                bear[j] = 1.0

    return bull, bear


def _make_liq_features(
    df: pd.DataFrame, levels: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar: distance to nearest high/low liquidity level, in ATR units."""
    n      = len(df)
    closes = df["Close"].to_numpy()
    atr_arr = _atr(df["High"], df["Low"], df["Close"]).to_numpy()
    liq_high = np.full(n, 10.0, dtype=np.float32)
    liq_low  = np.full(n, 10.0, dtype=np.float32)

    if not levels:
        return liq_high, liq_low

    high_prices = np.array([l["price"] for l in levels if l["dir"] == "high"])
    low_prices  = np.array([l["price"] for l in levels if l["dir"] == "low"])

    for i in range(n):
        c   = closes[i]
        atr = float(atr_arr[i]) if (not np.isnan(atr_arr[i]) and atr_arr[i] > 0) else 1.0

        if len(high_prices):
            above = high_prices[high_prices > c]
            liq_high[i] = float((above.min() - c) / atr) if len(above) else 10.0

        if len(low_prices):
            below = low_prices[low_prices < c]
            liq_low[i] = float((c - below.max()) / atr) if len(below) else 10.0

    return np.clip(liq_high, 0, 20), np.clip(liq_low, 0, 20)


# ─────────────────────────────────────────────────────────────────────────────
# Full feature matrix
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "rsi14", "atr_norm", "vol_ratio", "ret5", "ret20",
    "body_ratio", "candle_dir",
    "trend", "trend_age_norm", "bos_cnt20", "choch_cnt20",
    "bull_fvg_active", "bear_fvg_active", "bull_fvg_cnt", "bear_fvg_cnt",
    "bull_ob", "bear_ob",
    "bull_engulf", "bear_engulf",
    "liq_high_atr", "liq_low_atr",
]


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the full ML feature matrix from an OHLCV DataFrame.

    All features are strictly backward-looking (no future-leak).
    Returns a DataFrame indexed like ``df`` with columns = FEATURE_NAMES.
    Warmup rows contain NaN and should be dropped by the caller.
    """
    n      = len(df)
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # ── Technical (vectorized) ───────────────────────────────────────────
    rsi14   = _rsi(close).to_numpy()
    atr14   = _atr(high, low, close).to_numpy()
    atr_norm = np.where(close.to_numpy() > 0, atr14 / close.to_numpy(), np.nan)

    vol_sma20 = volume.rolling(20, min_periods=1).mean()
    vol_ratio = (volume / vol_sma20.replace(0, np.nan)).to_numpy()

    ret5  = close.pct_change(5).to_numpy()
    ret20 = close.pct_change(20).to_numpy()

    o = df["Open"].to_numpy()
    h = high.to_numpy()
    l = low.to_numpy()
    c = close.to_numpy()
    body_ratio = np.where((h - l) > 0, np.abs(c - o) / (h - l), 0.5)
    candle_dir = (c > o).astype(np.float32)

    # ── SMC indicators (each computed once) ─────────────────────────────
    ms_events = detect_market_structure(df, term="intermediate")
    fvg_df    = detect_fvg(df)
    obs       = detect_order_blocks(df, term="intermediate")
    engulf_df = detect_engulfing(df)
    levels    = detect_liquidity_levels(df)

    trend_arr, bos_arr, choch_arr = _make_trend_series(df, ms_events)

    # trend_age_norm: bars since last trend flip, normalized to [0, 1] over 50 bars
    trend_age = np.zeros(n, dtype=np.float32)
    last_flip = 0
    for i in range(1, n):
        if trend_arr[i] != trend_arr[i - 1] and trend_arr[i - 1] != 0.0:
            last_flip = i
        trend_age[i] = float(i - last_flip)
    trend_age_norm = np.minimum(trend_age / 50.0, 1.0)

    bull_fvg_active, bear_fvg_active, bull_fvg_cnt, bear_fvg_cnt = \
        _make_fvg_features(df, fvg_df)
    bull_ob, bear_ob     = _make_ob_features(df, obs)
    bull_engulf, bear_engulf = _make_engulf_features(df, engulf_df)
    liq_high_atr, liq_low_atr = _make_liq_features(df, levels)

    return pd.DataFrame({
        "rsi14":           rsi14,
        "atr_norm":        atr_norm,
        "vol_ratio":       vol_ratio,
        "ret5":            ret5,
        "ret20":           ret20,
        "body_ratio":      body_ratio,
        "candle_dir":      candle_dir,
        "trend":           trend_arr,
        "trend_age_norm":  trend_age_norm,
        "bos_cnt20":       bos_arr,
        "choch_cnt20":     choch_arr,
        "bull_fvg_active": bull_fvg_active,
        "bear_fvg_active": bear_fvg_active,
        "bull_fvg_cnt":    bull_fvg_cnt,
        "bear_fvg_cnt":    bear_fvg_cnt,
        "bull_ob":         bull_ob,
        "bear_ob":         bear_ob,
        "bull_engulf":     bull_engulf,
        "bear_engulf":     bear_engulf,
        "liq_high_atr":    liq_high_atr,
        "liq_low_atr":     liq_low_atr,
    }, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# Label generation
# ─────────────────────────────────────────────────────────────────────────────

def make_labels(
    df: pd.DataFrame,
    fwd: int = FWD_BARS,
    mult: float = ATR_MULT,
) -> pd.Series:
    """3-class label: 0=BUY, 1=HOLD, 2=SELL.

    A bar is BUY  if close[t+fwd] > close[t] + mult×ATR.
    A bar is SELL if close[t+fwd] < close[t] - mult×ATR.
    Otherwise HOLD.
    """
    atr     = _atr(df["High"], df["Low"], df["Close"])
    fwd_ret = df["Close"].shift(-fwd) - df["Close"]
    thresh  = atr * mult
    labels  = pd.Series(1, index=df.index, dtype=int)   # default: HOLD
    labels[fwd_ret >  thresh] = 0   # BUY
    labels[fwd_ret < -thresh] = 2   # SELL
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Model class
# ─────────────────────────────────────────────────────────────────────────────

class SMCMLModel:
    """Gradient-Boosted classifier trained on SMC + technical features.

    Training universe: ``TRAIN_SYMBOLS`` (daily bars, 2 years).
    The model learns which combinations of SMC signals historically led to
    profitable 3-bar forward moves, outputting a BUY / HOLD / SELL prediction
    with a calibrated probability.

    Usage
    -----
    model = SMCMLModel()
    model.fit()                       # trains; ~10-20 s
    result = model.predict(df)        # instant
    # result = {"signal": "BUY", "confidence": 72.4, "probabilities": {...}}
    """

    def __init__(self) -> None:
        self._clf      = None
        self._scaler   = None
        self._trained  = False
        self._lock     = threading.Lock()
        self._train_ts = 0.0

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def fit(self, symbols: list[str] = TRAIN_SYMBOLS) -> None:
        """Train (or retrain) the classifier on historical multi-symbol data."""
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.error(
                "scikit-learn is not installed. "
                "Run: pip install 'scikit-learn>=1.3.0'"
            )
            return

        logger.info("SMCMLModel: starting training on %s …", symbols)
        t0 = time.time()

        all_X: list[pd.DataFrame] = []
        all_y: list[pd.Series]    = []

        for sym in symbols:
            try:
                df = fetch_ohlcv(sym, period=TRAIN_PERIOD, interval=TRAIN_INTERVAL)
                if len(df) < 60:
                    logger.warning("Skipping %s — only %d bars.", sym, len(df))
                    continue

                feats  = extract_features(df)
                labels = make_labels(df)

                combined = feats.copy()
                combined["_label"] = labels
                # Drop warmup NaNs and the last FWD_BARS rows (no valid forward label)
                combined = combined.dropna().iloc[:-FWD_BARS]

                if len(combined) < 20:
                    logger.warning("Skipping %s — not enough clean rows.", sym)
                    continue

                all_X.append(combined[FEATURE_NAMES])
                all_y.append(combined["_label"])
                logger.info("  %s: %d rows", sym, len(combined))

            except Exception as exc:
                logger.warning("Training data failed for %s: %s", sym, exc)

        if not all_X:
            logger.error("No training data collected. Model NOT trained.")
            return

        X = pd.concat(all_X, ignore_index=True).clip(-10, 10)
        y = pd.concat(all_y, ignore_index=True)

        scaler = StandardScaler()
        X_s    = scaler.fit_transform(X)

        clf = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            random_state=42,
        )
        clf.fit(X_s, y)

        with self._lock:
            self._clf      = clf
            self._scaler   = scaler
            self._trained  = True
            self._train_ts = time.time()

        dist = {LABEL_MAP[int(k)]: int(v) for k, v in y.value_counts().sort_index().items()}
        logger.info(
            "SMCMLModel: trained. %d samples | %.1f s | label dist: %s",
            len(X), time.time() - t0, dist,
        )

    def ensure_trained(self) -> None:
        """Trigger training if the model is untrained or older than 24 hours."""
        hours_stale = (time.time() - self._train_ts) / 3600
        if not self._trained or hours_stale > 24:
            self.fit()

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def predict(self, df: pd.DataFrame) -> dict:
        """Score the last bar of *df*.

        Returns
        -------
        dict with keys:
          signal        : 'BUY' | 'SELL' | 'HOLD'
          confidence    : float 0-100 (probability of the predicted class)
          probabilities : {'BUY': float, 'HOLD': float, 'SELL': float}
          trained       : bool
        """
        if not self._trained:
            return {
                "signal":        "HOLD",
                "confidence":    0.0,
                "probabilities": {"BUY": 0.0, "HOLD": 100.0, "SELL": 0.0},
                "trained":       False,
            }

        try:
            feats     = extract_features(df)
            last_row  = feats[FEATURE_NAMES].iloc[[-1]].fillna(0.0).clip(-10, 10)
            X_s       = self._scaler.transform(last_row)
            proba     = self._clf.predict_proba(X_s)[0]

            prob_map: dict[str, float] = {"BUY": 0.0, "HOLD": 0.0, "SELL": 0.0}
            for i, cls_idx in enumerate(self._clf.classes_):
                prob_map[LABEL_MAP[int(cls_idx)]] = float(proba[i]) * 100

            pred_cls   = int(self._clf.classes_[int(np.argmax(proba))])
            signal     = LABEL_MAP[pred_cls]
            confidence = float(np.max(proba)) * 100

            return {
                "signal":        signal,
                "confidence":    round(confidence, 1),
                "probabilities": {k: round(v, 1) for k, v in prob_map.items()},
                "trained":       True,
            }

        except Exception as exc:
            logger.exception("ML predict failed: %s", exc)
            return {
                "signal":        "HOLD",
                "confidence":    0.0,
                "probabilities": {"BUY": 0.0, "HOLD": 100.0, "SELL": 0.0},
                "trained":       True,
            }


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_model: Optional[SMCMLModel] = None
_model_lock = threading.Lock()


def get_model() -> SMCMLModel:
    """Return the process-wide model singleton (creates it if needed)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SMCMLModel()
    return _model
