"""Machine Learning Confidence Layer for ICT / SMC Signals
===========================================================

Architecture
------------
    Feature engineering   26 per-bar features derived from OHLCV +
                        SMC indicators (FVG, MS, OB, Engulfing, Liquidity).
  Training universe     SPY, QQQ, IWM, GLD, TLT, ES=F, NQ=F  (daily, 2 yr).
  Model                 GradientBoostingClassifier (3-class: BUY/HOLD/SELL).
  Labels                Forward 3-bar return sign, thresholded at ±0.5 × ATR.
  Output                dict with signal, confidence (0-100), probabilities.

Modeling notes
--------------
This implementation applies three practices supported by scikit-learn guidance
and common quantitative ML literature:
    1) Time-ordered split (train on past, validate on future) to reduce leakage.
    2) Class-imbalance-aware weighting via compute_class_weight("balanced").
    3) Probability calibration on holdout data for better confidence estimates.

The model is trained lazily on the first call (background thread started by
server.py) and cached in memory for the lifetime of the server process.
Training takes ~10-20 s; inference is <1 s.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

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
TRAIN_SPLIT    = 0.80    # chronological train/validation split per symbol
MIN_ROWS       = 120     # minimum clean rows per symbol to keep
MIN_CAL_ROWS   = 150     # minimum total validation rows for calibration
FEEDBACK_FILE  = Path(__file__).resolve().parent.parent / "trade_feedback.jsonl"
FEEDBACK_MAX_ROWS = 3000
FEEDBACK_RETRAIN_EVERY = 5
FEEDBACK_WEIGHT_BOOST = 1.8

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
    "ret1", "volatility20", "sma20_dist", "range_atr", "atr_slope5",
    "volatility_regime", "momentum_consistency", "volume_shock",
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
    ret1  = close.pct_change(1).to_numpy()

    volatility20 = close.pct_change().rolling(20, min_periods=5).std().to_numpy()
    sma20 = close.rolling(20, min_periods=5).mean().to_numpy()
    sma20_dist = np.where(sma20 != 0, (close.to_numpy() - sma20) / sma20, np.nan)

    o = df["Open"].to_numpy()
    h = high.to_numpy()
    l = low.to_numpy()
    c = close.to_numpy()
    body_ratio = np.where((h - l) > 0, np.abs(c - o) / (h - l), 0.5)
    candle_dir = (c > o).astype(np.float32)
    range_atr = np.where(atr14 > 0, (h - l) / atr14, np.nan)

    atr_s = pd.Series(atr14, index=df.index)
    atr_slope5 = atr_s.pct_change(5).to_numpy()

    vol_regime_slow = pd.Series(volatility20, index=df.index).rolling(100, min_periods=20).mean()
    volatility_regime = np.where(vol_regime_slow.to_numpy() > 0,
                                 volatility20 / vol_regime_slow.to_numpy(),
                                 np.nan)

    up_days = close.diff().gt(0).rolling(10, min_periods=5).mean().to_numpy()
    momentum_consistency = np.where(np.isnan(up_days), np.nan, np.abs(up_days - 0.5) * 2.0)

    vol_med20 = volume.rolling(20, min_periods=5).median().replace(0, np.nan)
    volume_shock = (volume / vol_med20).to_numpy()

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
        "ret1":            ret1,
        "volatility20":    volatility20,
        "sma20_dist":      sma20_dist,
        "range_atr":       range_atr,
        "atr_slope5":      atr_slope5,
        "volatility_regime": volatility_regime,
        "momentum_consistency": momentum_consistency,
        "volume_shock":    volume_shock,
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
        self._calibrator = None
        self._scaler   = None
        self._trained  = False
        self._training = False
        self._lock     = threading.Lock()
        self._train_ts = 0.0
        self._last_train_attempt_ts = 0.0
        self._feedback_since_retrain = 0

    def _needs_training(self, max_age_hours: float = 24.0) -> bool:
        """Return True if model is missing or stale."""
        if not self._trained:
            return True
        if self._train_ts <= 0:
            return True
        return ((time.time() - self._train_ts) / 3600.0) > max_age_hours

    def _is_training(self) -> bool:
        with self._lock:
            return self._training

    def start_training_async(self, force: bool = False) -> bool:
        """Start background training if needed. Returns True when a thread is started."""
        if (not force) and (not self._needs_training()):
            return False

        t = threading.Thread(target=self.fit, kwargs={"force": force}, daemon=True)
        t.start()
        return True

    def _feedback_label_from_record(self, record: dict[str, Any]) -> int:
        """Map trade outcome feedback into the 3-class target space."""
        reason = str(record.get("result_reason", "")).lower()
        side = str(record.get("side", "")).upper()

        if reason == "take_profit_hit":
            return 0 if side == "BUY" else 2
        if reason == "stop_loss_hit":
            return 2 if side == "BUY" else 0
        if reason.startswith("order_") or reason.startswith("terminal_"):
            return 1

        # Fallback for unknown outcomes.
        return 1

    def _load_feedback_dataset(self) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """Load persisted feedback samples as (X, y, weight)."""
        if not FEEDBACK_FILE.exists():
            return pd.DataFrame(columns=FEATURE_NAMES), pd.Series(dtype=int), np.array([], dtype=float)

        rows: list[dict[str, Any]] = []
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
            for ln in lines[-FEEDBACK_MAX_ROWS:]:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                feats = rec.get("features") if isinstance(rec.get("features"), dict) else None
                if not feats:
                    continue
                row = {name: float(feats.get(name, 0.0) or 0.0) for name in FEATURE_NAMES}
                row["_label"] = int(rec.get("label", 1))
                row["_weight"] = float(rec.get("weight", 1.0) or 1.0)
                rows.append(row)
        except OSError as exc:
            logger.warning("Could not read feedback dataset: %s", exc)
            return pd.DataFrame(columns=FEATURE_NAMES), pd.Series(dtype=int), np.array([], dtype=float)

        if not rows:
            return pd.DataFrame(columns=FEATURE_NAMES), pd.Series(dtype=int), np.array([], dtype=float)

        df = pd.DataFrame(rows)
        X = df[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10, 10)
        y = df["_label"].astype(int)
        w = df["_weight"].to_numpy(dtype=float)
        return X, y, w

    def add_trade_feedback(self, record: dict[str, Any]) -> bool:
        """Persist a realized paper-trade outcome as a supervised feedback sample."""
        feature_row = record.get("feature_row")
        if not isinstance(feature_row, dict):
            return False

        label = self._feedback_label_from_record(record)
        reason = str(record.get("result_reason", "unknown"))
        weight = FEEDBACK_WEIGHT_BOOST if reason in {"take_profit_hit", "stop_loss_hit"} else 1.0

        payload = {
            "ts": time.time(),
            "ticker": record.get("ticker"),
            "side": record.get("side"),
            "reason": reason,
            "worked": bool(record.get("worked", False)),
            "label": int(label),
            "weight": float(weight),
            "features": {name: float(feature_row.get(name, 0.0) or 0.0) for name in FEATURE_NAMES},
        }

        try:
            with open(FEEDBACK_FILE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, separators=(",", ":")) + "\n")
        except OSError as exc:
            logger.warning("Could not append trade feedback: %s", exc)
            return False

        self._feedback_since_retrain += 1
        if self._feedback_since_retrain >= FEEDBACK_RETRAIN_EVERY:
            # Force retrain so new feedback is incorporated promptly.
            started = self.start_training_async(force=True)
            if started:
                self._feedback_since_retrain = 0

        return True

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def fit(self, symbols: list[str] = TRAIN_SYMBOLS, force: bool = False) -> None:
        """Train (or retrain) the classifier on historical multi-symbol data."""
        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
            from sklearn.preprocessing import StandardScaler
            from sklearn.utils.class_weight import compute_class_weight
        except ImportError:
            logger.error(
                "scikit-learn is not installed. "
                "Run: pip install 'scikit-learn>=1.3.0'"
            )
            return

        with self._lock:
            now = time.time()
            if self._training:
                logger.info("SMCMLModel: training already in progress; skipping new fit call.")
                return

            if (not force) and (now - self._last_train_attempt_ts < 120):
                logger.info("SMCMLModel: fit call skipped due to cooldown.")
                return

            if (not force) and (not self._needs_training(max_age_hours=24.0)):
                logger.info("SMCMLModel: model is fresh; skipping fit.")
                return

            self._training = True
            self._last_train_attempt_ts = now

        logger.info("SMCMLModel: starting training on %s …", symbols)
        t0 = time.time()

        try:
            train_X_parts: list[pd.DataFrame] = []
            train_y_parts: list[pd.Series]    = []
            valid_X_parts: list[pd.DataFrame] = []
            valid_y_parts: list[pd.Series]    = []

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
                    combined = combined.dropna().iloc[:-FWD_BARS]

                    if len(combined) < MIN_ROWS:
                        logger.warning("Skipping %s — not enough clean rows.", sym)
                        continue

                    split_idx = int(len(combined) * TRAIN_SPLIT)
                    split_idx = max(30, min(split_idx, len(combined) - 20))

                    train_df = combined.iloc[:split_idx]
                    valid_df = combined.iloc[split_idx:]

                    train_X_parts.append(train_df[FEATURE_NAMES])
                    train_y_parts.append(train_df["_label"])
                    if len(valid_df) >= 5:
                        valid_X_parts.append(valid_df[FEATURE_NAMES])
                        valid_y_parts.append(valid_df["_label"])

                    logger.info(
                        "  %s: train=%d valid=%d",
                        sym,
                        len(train_df),
                        len(valid_df),
                    )

                except Exception as exc:
                    logger.warning("Training data failed for %s: %s", sym, exc)

            if not train_X_parts:
                logger.error("No training data collected. Model NOT trained.")
                return

            X_train = pd.concat(train_X_parts, ignore_index=True).clip(-10, 10)
            y_train = pd.concat(train_y_parts, ignore_index=True)
            feedback_X, feedback_y, feedback_w = self._load_feedback_dataset()
            n_feedback = len(feedback_X)
            if n_feedback:
                X_train = pd.concat([X_train, feedback_X], ignore_index=True)
                y_train = pd.concat([y_train, feedback_y], ignore_index=True)

            X_valid = None
            y_valid = None
            if valid_X_parts:
                X_valid = pd.concat(valid_X_parts, ignore_index=True).clip(-10, 10)
                y_valid = pd.concat(valid_y_parts, ignore_index=True)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)

            classes = np.unique(y_train)
            class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
            weight_map = {int(c): float(w) for c, w in zip(classes, class_weights)}
            sample_weight = y_train.map(weight_map).to_numpy(dtype=float).copy()
            if n_feedback:
                sample_weight[-n_feedback:] *= np.clip(feedback_w, 0.5, 3.0)

            clf = GradientBoostingClassifier(
                n_estimators=220,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.7,
                random_state=42,
            )
            clf.fit(X_train_s, y_train, sample_weight=sample_weight)

            calibrator = None
            metrics: dict[str, float] = {}
            if X_valid is not None and y_valid is not None and len(X_valid) >= 10:
                X_valid_s = scaler.transform(X_valid)
                raw_pred = clf.predict(X_valid_s)
                metrics["balanced_acc"] = float(balanced_accuracy_score(y_valid, raw_pred))
                metrics["macro_f1"] = float(f1_score(y_valid, raw_pred, average="macro", zero_division=0))

                if len(X_valid) >= MIN_CAL_ROWS and y_valid.nunique() >= 2:
                    try:
                        calibrator = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
                        calibrator.fit(X_valid_s, y_valid)
                        cal_proba = calibrator.predict_proba(X_valid_s)
                        metrics["cal_log_loss"] = float(log_loss(y_valid, cal_proba, labels=calibrator.classes_))
                    except Exception as exc:
                        # Newer scikit-learn versions removed/changed prefit behavior.
                        # Fallback to FrozenEstimator-based calibration when available.
                        try:
                            from sklearn.frozen import FrozenEstimator

                            calibrator = CalibratedClassifierCV(
                                FrozenEstimator(clf),
                                method="sigmoid",
                            )
                            calibrator.fit(X_valid_s, y_valid)
                            cal_proba = calibrator.predict_proba(X_valid_s)
                            metrics["cal_log_loss"] = float(
                                log_loss(y_valid, cal_proba, labels=calibrator.classes_)
                            )
                            logger.info("Calibration used FrozenEstimator fallback path.")
                        except Exception as fallback_exc:
                            logger.warning(
                                "Calibration skipped due to errors (prefit=%s, fallback=%s)",
                                exc,
                                fallback_exc,
                            )

            with self._lock:
                self._clf      = clf
                self._calibrator = calibrator
                self._scaler   = scaler
                self._trained  = True
                self._train_ts = time.time()

            dist = {LABEL_MAP[int(k)]: int(v) for k, v in y_train.value_counts().sort_index().items()}
            logger.info(
                "SMCMLModel: trained. train=%d (feedback=%d) valid=%d | %.1f s | label dist: %s | metrics=%s | calibrated=%s",
                len(X_train),
                n_feedback,
                0 if X_valid is None else len(X_valid),
                time.time() - t0,
                dist,
                {k: round(v, 4) for k, v in metrics.items()},
                calibrator is not None,
            )
        finally:
            with self._lock:
                self._training = False

    def ensure_trained(self, async_mode: bool = True) -> None:
        """Ensure training exists; optionally trigger non-blocking background training."""
        if not self._needs_training(max_age_hours=24.0):
            return
        if async_mode:
            self.start_training_async()
        else:
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
          training      : bool
        """
        # Lazy behavior: first prediction can kick off background training.
        self.ensure_trained(async_mode=True)

        if not self._trained:
            return {
                "signal":        "HOLD",
                "confidence":    0.0,
                "probabilities": {"BUY": 0.0, "HOLD": 0.0, "SELL": 0.0},
                "trained":       False,
                "training":      self._is_training(),
            }

        try:
            from sklearn.preprocessing import StandardScaler as _StandardScaler

            feats       = extract_features(df)
            feat_matrix = feats[FEATURE_NAMES].fillna(0.0).clip(-10, 10)

            # Use a local scaler fitted on the current df so that feature
            # magnitudes are normalised relative to the same timeframe.
            # The training-time scaler was fitted on daily bars; applying it
            # directly to intraday data (e.g. 5 m) would compress all
            # return-based features near zero and cause the model to default
            # to HOLD with near-100 % confidence on every intraday bar.
            local_scaler = _StandardScaler()
            feat_matrix_s = local_scaler.fit_transform(feat_matrix)
            last_row_s    = feat_matrix_s[[-1]]
            X_s           = last_row_s
            model_for_proba = self._calibrator if self._calibrator is not None else self._clf
            proba     = model_for_proba.predict_proba(X_s)[0]

            prob_map: dict[str, float] = {"BUY": 0.0, "HOLD": 0.0, "SELL": 0.0}
            for i, cls_idx in enumerate(model_for_proba.classes_):
                prob_map[LABEL_MAP[int(cls_idx)]] = float(proba[i]) * 100

            pred_cls   = int(model_for_proba.classes_[int(np.argmax(proba))])
            signal     = LABEL_MAP[pred_cls]
            confidence = float(np.max(proba)) * 100

            return {
                "signal":        signal,
                "confidence":    round(confidence, 1),
                "probabilities": {k: round(v, 1) for k, v in prob_map.items()},
                "trained":       True,
                "training":      self._is_training(),
            }

        except Exception as exc:
            logger.exception("ML predict failed: %s", exc)
            return {
                "signal":        "HOLD",
                "confidence":    0.0,
                "probabilities": {"BUY": 0.0, "HOLD": 0.0, "SELL": 0.0},
                "trained":       True,
                "training":      self._is_training(),
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
