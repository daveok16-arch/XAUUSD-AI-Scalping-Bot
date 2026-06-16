"""
XAUUSD Scalping Signal Generator

Generates trading signals from ML model predictions with risk management.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Represents a trading signal."""
    timestamp: datetime
    signal_type: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    features_used: Dict[str, float]


@dataclass
class SignalThresholds:
    """Thresholds for signal generation."""
    buy_threshold: float = 0.65  # Probability threshold for BUY
    sell_threshold: float = 0.35  # Probability threshold for SELL
    min_confidence: float = 0.60  # Minimum confidence for any signal
    atr_multiplier_sl: float = 1.5  # ATR multiplier for stop loss
    atr_multiplier_tp: float = 3.0  # ATR multiplier for take profit
    min_atr: float = 0.1  # Minimum ATR value


class SignalGenerator:
    """Generate trading signals from ML predictions."""

    def __init__(self, thresholds: Optional[SignalThresholds] = None):
        """
        Initialize signal generator.

        Args:
            thresholds: Signal generation thresholds
        """
        self.thresholds = thresholds or SignalThresholds()
        self.signal_history: List[TradingSignal] = []

    def generate_signal(
        self,
        prediction: int,
        probability: float,
        current_price: float,
        atr: float,
        features: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> Optional[TradingSignal]:
        """
        Generate a trading signal from model prediction.

        Args:
            prediction: Model prediction (0=down, 1=up)
            probability: Probability of prediction
            current_price: Current market price
            atr: Average True Range for position sizing
            features: Dictionary of current feature values
            timestamp: Signal timestamp

        Returns:
            TradingSignal if conditions met, None otherwise
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Determine signal type
        if probability >= self.thresholds.buy_threshold:
            signal_type = "BUY"
        elif probability <= self.thresholds.sell_threshold:
            signal_type = "SELL"
        else:
            return None

        # Check minimum confidence
        confidence = probability if prediction == 1 else (1 - probability)
        if confidence < self.thresholds.min_confidence:
            return None

        # Calculate stop loss and take profit
        atr_value = max(atr, self.thresholds.min_atr)

        if signal_type == "BUY":
            stop_loss = current_price - (atr_value * self.thresholds.atr_multiplier_sl)
            take_profit = current_price + (atr_value * self.thresholds.atr_multiplier_tp)
        else:  # SELL
            stop_loss = current_price + (atr_value * self.thresholds.atr_multiplier_sl)
            take_profit = current_price - (atr_value * self.thresholds.atr_multiplier_tp)

        # Calculate risk-reward ratio
        risk = abs(current_price - stop_loss)
        reward = abs(take_profit - current_price)
        risk_reward = reward / risk if risk > 0 else 0

        signal = TradingSignal(
            timestamp=timestamp,
            signal_type=signal_type,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward,
            features_used=features
        )

        self.signal_history.append(signal)

        logger.info(
            f"Signal generated: {signal_type} | "
            f"Confidence: {confidence:.2%} | "
            f"Entry: {current_price:.2f} | "
            f"SL: {stop_loss:.2f} | "
            f"TP: {take_profit:.2f} | "
            f"R:R = 1:{risk_reward:.1f}"
        )

        return signal

    def batch_generate_signals(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        feature_cols: List[str]
    ) -> List[TradingSignal]:
        """
        Generate signals for a batch of predictions.

        Args:
            df: DataFrame with price data and features
            predictions: Array of predictions
            probabilities: Array of probabilities
            feature_cols: List of feature column names

        Returns:
            List of TradingSignal objects
        """
        signals = []

        for i in range(len(predictions)):
            if "atr_14" not in df.columns:
                logger.warning("ATR column not found, skipping signal generation")
                break

            features = {col: df[col].iloc[i] for col in feature_cols[:5]}  # Top 5 features

            signal = self.generate_signal(
                prediction=predictions[i],
                probability=probabilities[i],
                current_price=df["close"].iloc[i],
                atr=df["atr_14"].iloc[i],
                features=features,
                timestamp=df.index[i]
            )

            if signal:
                signals.append(signal)

        logger.info(f"Generated {len(signals)} signals from {len(predictions)} predictions")
        return signals

    def get_signal_statistics(self) -> Dict:
        """
        Get statistics about generated signals.

        Returns:
            Dictionary of signal statistics
        """
        if not self.signal_history:
            return {"total_signals": 0}

        buy_signals = [s for s in self.signal_history if s.signal_type == "BUY"]
        sell_signals = [s for s in self.signal_history if s.signal_type == "SELL"]

        return {
            "total_signals": len(self.signal_history),
            "buy_signals": len(buy_signals),
            "sell_signals": len(sell_signals),
            "avg_confidence": np.mean([s.confidence for s in self.signal_history]),
            "avg_risk_reward": np.mean([s.risk_reward_ratio for s in self.signal_history]),
            "recent_signals": len([
                s for s in self.signal_history
                if (datetime.now() - s.timestamp).days < 7
            ])
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert signal history to DataFrame."""
        if not self.signal_history:
            return pd.DataFrame()

        records = []
        for signal in self.signal_history:
            records.append({
                "timestamp": signal.timestamp,
                "signal_type": signal.signal_type,
                "confidence": signal.confidence,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "risk_reward_ratio": signal.risk_reward_ratio,
            })

        return pd.DataFrame(records)

    def clear_history(self) -> None:
        """Clear signal history."""
        self.signal_history = []
