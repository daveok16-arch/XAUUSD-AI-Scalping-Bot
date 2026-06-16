"""
Advanced Feature Engineering Module for XAUUSD Scalping

Generates comprehensive technical indicators and features for ML models.
Optimized for Kaggle environment.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import ta


logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Generate technical indicator features for XAUUSD scalping ML model."""

    # Feature groups
    PRICE_FEATURES = [
        "returns", "log_returns", "high_low_range", "open_close_range",
        "body_size", "upper_shadow", "lower_shadow", "direction"
    ]

    MOMENTUM_FEATURES = [
        "rsi_14", "rsi_7", "macd", "macd_signal", "macd_hist",
        "stoch_k", "stoch_d", "williams_r", "cci_20", "momentum_10"
    ]

    TREND_FEATURES = [
        "sma_10", "sma_20", "sma_50", "ema_10", "ema_20", "ema_50",
        "adx", "adx_pos", "adx_neg", "trend_direction"
    ]

    VOLATILITY_FEATURES = [
        "atr_14", "atr_7", "bollinger_upper", "bollinger_middle",
        "bollinger_lower", "bollinger_width", "bollinger_pct",
        "keltner_upper", "keltner_lower", "keltner_width"
    ]

    VOLUME_FEATURES = [
        "volume_sma_20", "volume_ratio", "obv", "cmf_20"
    ]

    def __init__(self, lookback_window: int = 100):
        """
        Initialize feature engineer.

        Args:
            lookback_window: Minimum bars needed for feature calculation
        """
        self.lookback_window = lookback_window
        self.feature_names: List[str] = []

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create complete feature set from OHLCV data.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]

        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering...")
        df = df.copy()

        # Ensure standard column names
        df.columns = [c.lower() for c in df.columns]

        # Price-based features
        df = self._add_price_features(df)

        # Momentum indicators
        df = self._add_momentum_features(df)

        # Trend indicators
        df = self._add_trend_features(df)

        # Volatility indicators
        df = self._add_volatility_features(df)

        # Volume features
        df = self._add_volume_features(df)

        # Lag features
        df = self._add_lag_features(df)

        # Time-based features
        df = self._add_time_features(df)

        # Drop NaN values from indicator calculations
        initial_rows = len(df)
        df = df.dropna()
        dropped = initial_rows - len(df)
        logger.info(f"Dropped {dropped} rows due to NaN values from indicator calculations")

        # Store feature names (exclude target columns)
        exclude_cols = ["open", "high", "low", "close", "volume", "target"]
        self.feature_names = [c for c in df.columns if c not in exclude_cols]

        logger.info(f"Created {len(self.feature_names)} features")
        logger.info(f"Final dataset shape: {df.shape}")

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        logger.debug("Adding price features...")

        # Returns
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Candlestick features
        df["high_low_range"] = df["high"] - df["low"]
        df["open_close_range"] = abs(df["close"] - df["open"])
        df["body_size"] = abs(df["close"] - df["open"])
        df["upper_shadow"] = df["high"] - df[["close", "open"]].max(axis=1)
        df["lower_shadow"] = df[["close", "open"]].min(axis=1) - df["low"]

        # Direction (1 for up, 0 for down)
        df["direction"] = (df["close"] > df["open"]).astype(int)

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicator features."""
        logger.debug("Adding momentum features...")

        # RSI
        df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        df["rsi_7"] = ta.momentum.RSIIndicator(df["close"], window=7).rsi()

        # MACD
        macd = ta.trend.MACD(df["close"], window_fast=12, window_slow=26, window_sign=9)
        if macd is not None:
            df["macd"] = macd["MACD_12_26_9"]
            df["macd_signal"] = macd["MACDs_12_26_9"]
            df["macd_hist"] = macd["MACDh_12_26_9"]

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"], window=14, smooth_window=3)
        if stoch is not None:
            df["stoch_k"] = stoch["STOCHk_14_3_3"]
            df["stoch_d"] = stoch["STOCHd_14_3_3"]

        # Williams %R
        df["williams_r"] = ta.momentum.WilliamsRIndicator(df["high"], df["low"], df["close"], lbp=14).williams_r()

        # CCI
        df["cci_20"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=20).cci()

        # Momentum
        df["momentum_10"] = df["close"] - df["close"].shift(10)

        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicator features."""
        logger.debug("Adding trend features...")

        # Simple Moving Averages
        df["sma_10"] = df["close"].rolling(window=10).mean()
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()

        # Exponential Moving Averages
        df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

        # Price relative to EMAs
        df["price_vs_ema20"] = (df["close"] - df["ema_20"]) / df["ema_20"]
        df["price_vs_ema50"] = (df["close"] - df["ema_50"]) / df["ema_50"]

        # ADX (Average Directional Index)
        adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
        if adx is not None:
            df["adx"] = adx["ADX_14"]
            df["adx_pos"] = adx["DMP_14"]
            df["adx_neg"] = adx["DMN_14"]

        # EMA crossover signals
        df["ema_cross_signal"] = np.where(
            df["ema_10"] > df["ema_20"], 1,
            np.where(df["ema_10"] < df["ema_20"], -1, 0)
        )

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicator features."""
        logger.debug("Adding volatility features...")

        # ATR (Average True Range)
        atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
        if atr is not None:
            df["atr_14"] = atr
        df["atr_7"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=7).average_true_range()

        # ATR ratio (current ATR vs rolling mean)
        df["atr_ratio"] = df["atr_14"] / df["atr_14"].rolling(50).mean()

        # Bollinger Bands
        bbands = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
        if bbands is not None:
            df["bollinger_upper"] = bbands["BBU_20_2.0"]
            df["bollinger_middle"] = bbands["BBM_20_2.0"]
            df["bollinger_lower"] = bbands["BBL_20_2.0"]
            df["bollinger_width"] = (
                df["bollinger_upper"] - df["bollinger_lower"]
            ) / df["bollinger_middle"]
            df["bollinger_pct"] = (
                (df["close"] - df["bollinger_lower"]) /
                (df["bollinger_upper"] - df["bollinger_lower"])
            )

        # Keltner Channels
        keltner = ta.volatility.KeltnerChannel(df["high"], df["low"], df["close"], window=20, window_atr=10)
        if keltner is not None:
            df["keltner_upper"] = keltner["KCUe_20_2"]
            df["keltner_lower"] = keltner["KCLe_20_2"]
            df["keltner_width"] = (
                df["keltner_upper"] - df["keltner_lower"]
            ) / df["close"]

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        logger.debug("Adding volume features...")

        # Volume SMA and ratio
        df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

        # OBV (On Balance Volume)
        df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()

        # CMF (Chaikin Money Flow)
        df["cmf_20"] = ta.volume.ChaikinMoneyFlowIndicator(df["high"], df["low"], df["close"], df["volume"], window=20).chaikin_money_flow()

        return df

    def _add_lag_features(self, df: pd.DataFrame, lags: List[int] = None) -> pd.DataFrame:
        """Add lagged price features."""
        logger.debug("Adding lag features...")

        if lags is None:
            lags = [1, 2, 3, 5, 10]

        for lag in lags:
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
            df[f"returns_lag_{lag}"] = df["returns"].shift(lag)
            df[f"rsi_lag_{lag}"] = df["rsi_14"].shift(lag)

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based cyclical features."""
        logger.debug("Adding time features...")

        # Extract datetime components
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month

        # Cyclical encoding for hour (24-hour cycle)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Cyclical encoding for day of week
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Session flags (Forex market sessions)
        df["is_asian"] = ((df["hour"] >= 0) & (df["hour"] < 8)).astype(int)
        df["is_london"] = ((df["hour"] >= 8) & (df["hour"] < 16)).astype(int)
        df["is_ny"] = ((df["hour"] >= 13) & (df["hour"] < 21)).astype(int)

        # London-NY overlap (highest volatility period)
        df["is_london_ny_overlap"] = ((df["hour"] >= 13) & (df["hour"] < 16)).astype(int)

        return df

    def create_target(
        self,
        df: pd.DataFrame,
        lookahead: int = 5,
        threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Create target variable for classification.

        Args:
            df: DataFrame with close prices
            lookahead: Number of bars to look ahead
            threshold: Minimum price change to consider (0 = any direction)

        Returns:
            DataFrame with 'target' column added
        """
        # Calculate future returns
        future_returns = df["close"].shift(-lookahead) / df["close"] - 1

        # Create binary target: 1 if price goes up, 0 if down
        if threshold > 0:
            df["target"] = np.where(future_returns > threshold, 1,
                                   np.where(future_returns < -threshold, 0, np.nan))
        else:
            df["target"] = (future_returns > 0).astype(int)

        # Drop rows where target is NaN (end of dataset)
        df = df.dropna(subset=["target"])

        # Convert to integer
        df["target"] = df["target"].astype(int)

        logger.info(f"Target distribution:\n{df['target'].value_counts(normalize=True)}")

        return df

    def get_feature_importance_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a report of feature correlations with target.

        Args:
            df: DataFrame with features and target

        Returns:
            DataFrame with correlation values
        """
        if "target" not in df.columns:
            raise ValueError("DataFrame must contain 'target' column")

        correlations = df[self.feature_names + ["target"]].corr()["target"].drop("target")
        correlations = correlations.abs().sort_values(ascending=False)

        return pd.DataFrame({
            "feature": correlations.index,
            "abs_correlation": correlations.values
        })
