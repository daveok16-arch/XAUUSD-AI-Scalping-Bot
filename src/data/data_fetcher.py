"""
XAUUSD Data Fetcher Module

Handles fetching historical XAUUSD data from multiple sources.
Optimized for Kaggle environment with local data paths.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)


class XAUUSDDataFetcher:
    """Fetch and manage XAUUSD historical price data."""

    # XAUUSD Yahoo Finance ticker symbol
    SYMBOL = "GC=F"  # Gold Futures
    # Alternative: "XAUUSD=X" or "GLD"

    # Kaggle dataset paths (adjust as needed)
    KAGGLE_INPUT_PATH = "/kaggle/input"
    KAGGLE_OUTPUT_PATH = "/kaggle/working"

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the data fetcher.

        Args:
            data_dir: Directory to store/load data. Auto-detects Kaggle environment.
        """
        self.is_kaggle = self._detect_kaggle()

        if data_dir:
            self.data_dir = Path(data_dir)
        elif self.is_kaggle:
            self.data_dir = Path(self.KAGGLE_OUTPUT_PATH) / "data"
        else:
            self.data_dir = Path("data")

        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Kaggle environment: {self.is_kaggle}")

    def _detect_kaggle(self) -> bool:
        """Detect if running in Kaggle environment."""
        return os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ

    def fetch_historical_data(
        self,
        period: str = "5y",
        interval: str = "1h",
        save: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical XAUUSD data from Yahoo Finance.

        Args:
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            save: Whether to save fetched data to disk

        Returns:
            DataFrame with OHLCV data
        """
        cache_file = self.data_dir / f"xauusd_{interval}_{period}.csv"

        # Try to load from cache first
        if cache_file.exists():
            logger.info(f"Loading cached data from {cache_file}")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            logger.info(f"Loaded {len(df)} rows from cache")
            return df

        logger.info(f"Fetching XAUUSD data: period={period}, interval={interval}")

        try:
            ticker = yf.Ticker(self.SYMBOL)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                raise ValueError("No data fetched from Yahoo Finance")

            # Clean column names
            df.columns = df.columns.str.lower().str.replace(" ", "_")

            # Remove timezone info from index for consistency
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            logger.info(f"Fetched {len(df)} rows of data")
            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

            if save:
                df.to_csv(cache_file)
                logger.info(f"Saved data to {cache_file}")

            return df

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise

    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load data from a CSV file.

        Args:
            filepath: Path to CSV file

        Returns:
            DataFrame with OHLCV data
        """
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare raw OHLCV data.

        Args:
            df: Raw DataFrame with OHLCV columns

        Returns:
            Cleaned DataFrame
        """
        # Ensure required columns exist
        required_cols = ["open", "high", "low", "close", "volume"]
        df_cols_lower = [c.lower() for c in df.columns]

        for col in required_cols:
            if col not in df_cols_lower:
                raise ValueError(f"Missing required column: {col}")

        # Standardize column names
        rename_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in required_cols:
                rename_map[col] = col_lower

        df = df.rename(columns=rename_map)

        # Select only required columns
        df = df[required_cols].copy()

        # Handle missing values
        initial_rows = len(df)
        df = df.dropna()
        dropped = initial_rows - len(df)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows with missing values")

        # Remove zero volume rows (non-trading periods)
        df = df[df["volume"] > 0]

        # Sort by index (datetime)
        df = df.sort_index()

        logger.info(f"Prepared data: {len(df)} rows")
        return df

    def get_train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets (time-based, no shuffling).

        Args:
            df: Full dataset
            test_size: Fraction of data for testing

        Returns:
            Tuple of (train_df, test_df)
        """
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        logger.info(f"Train set: {len(train_df)} rows ({train_df.index.min()} to {train_df.index.max()})")
        logger.info(f"Test set: {len(test_df)} rows ({test_df.index.max()} to {test_df.index.max()})")

        return train_df, test_df
