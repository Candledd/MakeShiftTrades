"""HMM-based Market Regime Classifier for MakeShiftTrades.

Fits a Hidden Markov Model (or fallback Gaussian Mixture) on SPY log returns
and rolling volatility to detect latent market regimes.  States are mapped to
human-readable labels usable by the engine's routing and risk pipeline.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Backend selection ──────────────────────────────────────────────────────────
HMM_AVAILABLE = False
try:
    import hmmlearn  # noqa: F401 — check availability only

    HMM_AVAILABLE = True
except ImportError:
    logger.warning(
        "hmmlearn not available; falling back to GaussianMixture "
        "for regime classification"
    )


class RegimeClassifier:
    """HMM-driven market regime classifier.

    Fits a Hidden Markov Model (or fallback Gaussian Mixture) on log returns
    and rolling volatility of a market proxy (e.g. SPY) to detect latent
    market regimes.

    After fitting, states are reordered by mean log return (descending) so that:

        state 0  →  "Bullish Calm"    (highest mean return, lower vol)
        state 1  →  "Sideways"        (middle — optional 3rd state)
        state 2  →  "Bearish Volatile" (lowest mean return, higher vol)

    For a 2-state model the mapping is simply 0 → Bullish Calm, 1 → Bearish Volatile.
    """

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state

        self.model = None
        self._is_fitted = False
        self._state_remap: dict[int, int] = {}

        # Human-readable regime labels — reassigned during fit() based on state count.
        self.state_labels: list[str] = [
            "Bullish Calm",
            "Sideways",
            "Bearish Volatile",
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # Feature engineering
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _prepare_features(df: pd.DataFrame) -> np.ndarray:
        """Extract log returns and 21-day rolling volatility from price data.

        Returns a 2-D array of shape ``(N, 2)`` with columns
        ``[log_return, sqrt(rolling_mean_sq_return)]``.
        """
        close = df["Close"] if "Close" in df.columns else df["close"]
        log_ret = np.log(close / close.shift(1)).dropna().values.reshape(-1, 1)

        # 21-trading-day squared-return rolling average (~1 calendar month)
        sq_ret = log_ret.reshape(-1) ** 2
        vol_21 = (
            pd.Series(sq_ret).rolling(window=21).mean().replace(0, 1e-12).values
        )
        vol_feature = np.sqrt(vol_21).reshape(-1, 1)

        # Align both features by the shorter length
        min_len = min(len(log_ret), len(vol_feature))
        features = np.column_stack(
            [log_ret[-min_len:], vol_feature[-min_len:]]
        )

        # Strip remaining NaN / inf rows
        features = features[~np.isnan(features).any(axis=1)]
        features = features[~np.isinf(features).any(axis=1)]
        return features

    # ──────────────────────────────────────────────────────────────────────────
    # Fitting
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> None:
        """Fit the HMM (or fallback GMM) on market proxy price data.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with at least a ``'Close'`` column (case-insensitive
            fallback to ``'close'``).
        """
        features = self._prepare_features(df)
        if len(features) < self.n_states * 10:
            logger.warning(
                "Not enough data points (%d) to fit %d-state HMM",
                len(features),
                self.n_states,
            )
            self._is_fitted = False
            return

        if HMM_AVAILABLE:
            from hmmlearn.hmm import GaussianHMM as _HMMBackend

            self.model = _HMMBackend(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=self.n_iter,
                random_state=self.random_state,
                init_params="stmc",
                params="stmc",
                tol=1e-4,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(features)
        else:
            from sklearn.mixture import GaussianMixture as _GMMBackend

            self.model = _GMMBackend(
                n_components=self.n_states,
                covariance_type="full",
                random_state=self.random_state,
                max_iter=self.n_iter,
                n_init=5,
            )
            self.model.fit(features)

        self._is_fitted = True

        # Reorder states so that state 0 = highest mean log return
        means = self.model.means_[:, 0]  # type: ignore[union-attr]

        order = np.argsort(means)[::-1]  # descending
        self._state_remap = {old: new for new, old in enumerate(order)}

        if self.n_states == 3:
            self.state_labels = ["Bullish Calm", "Sideways", "Bearish Volatile"]
        elif self.n_states == 2:
            self.state_labels = ["Bullish Calm", "Bearish Volatile"]
        # If custom n_states, labels stay at their positional default

        logger.info(
            "RegimeClassifier fitted (%d states). State means (log-ret): %s",
            self.n_states,
            [f"{m:.6f}" for m in means[order]],
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> tuple[str, int]:
        """Predict the **current** market regime.

        Uses the last observation in the feature matrix to determine the
        latent state.

        Parameters
        ----------
        df : pd.DataFrame
            Price data with at least a ``'Close'`` column.

        Returns
        -------
        (regime_label, state_id) : tuple[str, int]
            Human-readable regime string and the **remapped** state index
            (0 = "Bullish Calm", …, N-1).  Returns ``("unknown", -1)`` if
            the model has not been fitted or if prediction fails.
        """
        if not self._is_fitted or self.model is None:
            return ("unknown", -1)

        try:
            features = self._prepare_features(df)
            if len(features) < 5:
                return ("unknown", -1)

            raw_state: int = self.model.predict(features)[-1]  # type: ignore[union-attr]

            remapped = self._state_remap.get(raw_state, raw_state)
            if remapped is None:
                return ("unknown", -1)
            label = (
                self.state_labels[remapped]
                if remapped < len(self.state_labels)
                else "unknown"
            )
            return (label, remapped)
        except Exception as exc:
            logger.warning("RegimeClassifier.predict failed: %s", exc)
            return ("unknown", -1)
