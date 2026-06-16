"""
Test suite for XAUUSD Scalping Bot pipeline.

Run tests:
    python -m pytest tests/test_pipeline.py -v
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_fetcher import XAUUSDDataFetcher
from src.features.feature_engineering import FeatureEngineer
from src.signals.signal_generator import SignalGenerator, SignalThresholds


class TestDataFetcher(unittest.TestCase):
    """Test data fetching module."""

    def setUp(self):
        self.fetcher = XAUUSDDataFetcher(data_dir="test_data")

    def test_kaggle_detection(self):
        """Test Kaggle environment detection."""
        is_kaggle = self.fetcher.is_kaggle
        self.assertIsInstance(is_kaggle, bool)

    def test_prepare_data(self):
        """Test data preparation."""
        # Create sample data
        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        df = pd.DataFrame({
            "open": np.random.randn(100) + 2300,
            "high": np.random.randn(100) + 2310,
            "low": np.random.randn(100) + 2290,
            "close": np.random.randn(100) + 2305,
            "volume": np.random.randint(1000, 10000, 100)
        }, index=dates)

        prepared = self.fetcher.prepare_data(df)
        self.assertEqual(len(prepared), 100)
        self.assertIn("open", prepared.columns)
        self.assertIn("close", prepared.columns)

    def test_train_test_split(self):
        """Test train-test split."""
        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        df = pd.DataFrame({
            "open": np.random.randn(100) + 2300,
            "high": np.random.randn(100) + 2310,
            "low": np.random.randn(100) + 2290,
            "close": np.random.randn(100) + 2305,
            "volume": np.random.randint(1000, 10000, 100)
        }, index=dates)

        train, test = self.fetcher.get_train_test_split(df, test_size=0.2)
        self.assertEqual(len(train), 80)
        self.assertEqual(len(test), 20)


class TestFeatureEngineer(unittest.TestCase):
    """Test feature engineering module."""

    def setUp(self):
        self.fe = FeatureEngineer()

    def test_create_all_features(self):
        """Test feature creation."""
        dates = pd.date_range("2024-01-01", periods=200, freq="1h")
        np.random.seed(42)
        df = pd.DataFrame({
            "open": np.cumsum(np.random.randn(200) * 0.5) + 2300,
            "high": np.cumsum(np.random.randn(200) * 0.5) + 2310,
            "low": np.cumsum(np.random.randn(200) * 0.5) + 2290,
            "close": np.cumsum(np.random.randn(200) * 0.5) + 2305,
            "volume": np.random.randint(1000, 10000, 200)
        }, index=dates)

        df_features = self.fe.create_all_features(df)

        # Should have more columns than input
        self.assertGreater(len(df_features.columns), 5)

        # Should not have NaN in all rows (some NaN from rolling windows is OK)
        self.assertGreater(len(df_features), 0)

    def test_create_target(self):
        """Test target variable creation."""
        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        df = pd.DataFrame({
            "close": np.cumsum(np.random.randn(100) * 0.5) + 2300
        }, index=dates)

        df_with_target = self.fe.create_target(df, lookahead=5)

        self.assertIn("target", df_with_target.columns)
        self.assertEqual(df_with_target["target"].nunique(), 2)
        self.assertTrue(set(df_with_target["target"].unique()).issubset({0, 1}))


class TestSignalGenerator(unittest.TestCase):
    """Test signal generation module."""

    def setUp(self):
        self.sg = SignalGenerator()

    def test_buy_signal(self):
        """Test BUY signal generation."""
        signal = self.sg.generate_signal(
            prediction=1,
            probability=0.75,
            current_price=2300.0,
            atr=5.0,
            features={"rsi_14": 60, "macd": 0.5}
        )

        self.assertIsNotNone(signal)
        self.assertEqual(signal.signal_type, "BUY")
        self.assertGreater(signal.confidence, 0.6)
        self.assertLess(signal.stop_loss, signal.entry_price)
        self.assertGreater(signal.take_profit, signal.entry_price)

    def test_sell_signal(self):
        """Test SELL signal generation."""
        signal = self.sg.generate_signal(
            prediction=0,
            probability=0.25,  # Low probability = SELL
            current_price=2300.0,
            atr=5.0,
            features={"rsi_14": 40, "macd": -0.5}
        )

        self.assertIsNotNone(signal)
        self.assertEqual(signal.signal_type, "SELL")
        self.assertGreater(signal.stop_loss, signal.entry_price)
        self.assertLess(signal.take_profit, signal.entry_price)

    def test_no_signal(self):
        """Test when no signal should be generated."""
        signal = self.sg.generate_signal(
            prediction=1,
            probability=0.50,  # Too uncertain
            current_price=2300.0,
            atr=5.0,
            features={}
        )

        self.assertIsNone(signal)

    def test_signal_statistics(self):
        """Test signal statistics."""
        # Generate some signals
        for i in range(5):
            self.sg.generate_signal(
                prediction=1,
                probability=0.7,
                current_price=2300.0 + i,
                atr=5.0,
                features={}
            )

        stats = self.sg.get_signal_statistics()
        self.assertEqual(stats["total_signals"], 5)
        self.assertEqual(stats["buy_signals"], 5)
        self.assertEqual(stats["sell_signals"], 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""

    def test_full_pipeline(self):
        """Test the complete pipeline end-to-end."""
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=300, freq="1h")
        df = pd.DataFrame({
            "open": np.cumsum(np.random.randn(300) * 0.5) + 2300,
            "high": np.cumsum(np.random.randn(300) * 0.5) + 2310,
            "low": np.cumsum(np.random.randn(300) * 0.5) + 2290,
            "close": np.cumsum(np.random.randn(300) * 0.5) + 2305,
            "volume": np.random.randint(1000, 10000, 300)
        }, index=dates)

        # Ensure high > low
        df["high"] = df[["open", "close", "high"]].max(axis=1) + 1
        df["low"] = df[["open", "close", "low"]].min(axis=1) - 1

        # Step 1: Feature engineering
        fe = FeatureEngineer()
        df_features = fe.create_all_features(df)
        df_features = fe.create_target(df_features, lookahead=5)

        self.assertGreater(len(df_features.columns), 10)
        self.assertIn("target", df_features.columns)

        # Step 2: Prepare data
        exclude_cols = ["open", "high", "low", "close", "volume", "target"]
        feature_cols = [c for c in df_features.columns if c not in exclude_cols]

        df_clean = df_features.dropna()
        self.assertGreater(len(df_clean), 50)

        # Step 3: Generate signals
        sg = SignalGenerator()
        for i in range(min(10, len(df_clean))):
            signal = sg.generate_signal(
                prediction=np.random.randint(0, 2),
                probability=np.random.uniform(0.3, 0.8),
                current_price=df_clean["close"].iloc[i],
                atr=df_clean["atr_14"].iloc[i] if "atr_14" in df_clean.columns else 5.0,
                features={col: df_clean[col].iloc[i] for col in feature_cols[:5]}
            )

        stats = sg.get_signal_statistics()
        self.assertIn("total_signals", stats)


if __name__ == "__main__":
    unittest.main()
