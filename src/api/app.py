"""
Flask API for XAUUSD Scalping Signal Bot

Deployable on Render with endpoints for signal generation and model info.
"""

import json
import logging
import os
from src.signal_worker import init_worker
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_fetcher import XAUUSDDataFetcher
from src.features.feature_engineering import FeatureEngineer
from src.models.model_trainer import XAUUSDModelTrainer
from src.signals.signal_generator import SignalGenerator, SignalThresholds

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

try:
    init_worker()
except Exception as e:
    logger.warning(f"Telegram worker failed: {e}")

CORS(app)

# Global model and components
MODEL = None
SCALER = None
FEATURE_ENGINEER = None
SIGNAL_GENERATOR = None
FEATURE_NAMES = []


def load_model():
    """Load the trained model on startup."""
    global MODEL, SCALER, FEATURE_ENGINEER, SIGNAL_GENERATOR, FEATURE_NAMES

    try:
        # Check for model in multiple locations
        model_paths = [
            os.environ.get("MODEL_PATH"),
            "/app/models/xauusd_best_model.joblib",
            "./models/xauusd_best_model.joblib",
            "../models/xauusd_best_model.joblib",
            "./xauusd_best_model.joblib",
            "/app/models/xauusd_xgboost_model.joblib",
            "./models/xauusd_xgboost_model.joblib",
        ]

        model_path = None
        for path in model_paths:
            if path and os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            logger.warning("No trained model found. API will run in info-only mode.")
            return False

        # Load model
        trainer = XAUUSDModelTrainer()
        trainer.load_model(model_path)

        MODEL = trainer.model
        SCALER = trainer.scaler
        FEATURE_NAMES = trainer.feature_names

        # Initialize components
        FEATURE_ENGINEER = FeatureEngineer()
        SIGNAL_GENERATOR = SignalGenerator()

        logger.info(f"Model loaded successfully from {model_path}")
        logger.info(f"Available features: {len(FEATURE_NAMES)}")

        return True

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


@app.route("/", methods=["GET"])
def home():
    """Root endpoint with API info."""
    return jsonify({
        "name": "XAUUSD Scalping Signal Bot API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": MODEL is not None,
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "/": "API information",
            "/health": "Health check",
            "/signal": "Get current trading signal",
            "/predict": "Predict from provided data",
            "/model-info": "Model information",
            "/features": "Available features",
            "/historical-signals": "Get historical signals"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "timestamp": datetime.now().isoformat()
    })


@app.route("/signal", methods=["GET"])
def get_signal():
    """Get current XAUUSD trading signal."""
    if MODEL is None:
        return jsonify({
            "error": "Model not loaded",
            "message": "The ML model is not available. Please train and deploy a model first."
        }), 503

    try:
        # Fetch recent data
        fetcher = XAUUSDDataFetcher()
        df = fetcher.fetch_historical_data(period="5d", interval="1h")

        if df.empty or len(df) < 50:
            return jsonify({
                "error": "Insufficient data",
                "message": "Not enough data to generate signal"
            }), 400

        # Prepare features
        df = FEATURE_ENGINEER.create_all_features(df)

        # Create target (won't be used for prediction)
        df = FEATURE_ENGINEER.create_target(df, lookahead=5)

        # Get latest row
        latest = df.iloc[-1:]

        # Extract features
        available_features = [f for f in FEATURE_NAMES if f in latest.columns]
        X = latest[available_features].values

        # Scale and predict
        X_scaled = SCALER.transform(X)
        prediction = MODEL.predict(X_scaled)[0]
        probability = MODEL.predict_proba(X_scaled)[0][1]

        # Generate signal
        atr_value = latest["atr_14"].values[0] if "atr_14" in latest.columns else 1.0
        current_price = latest["close"].values[0]

        features_dict = {f: latest[f].values[0] for f in available_features[:10]}

        signal = SIGNAL_GENERATOR.generate_signal(
            prediction=prediction,
            probability=probability,
            current_price=current_price,
            atr=atr_value,
            features=features_dict,
            timestamp=df.index[-1]
        )

        if signal is None:
            return jsonify({
                "signal": "HOLD",
                "confidence": max(probability, 1 - probability),
                "current_price": current_price,
                "message": "No clear trading signal at this time",
                "timestamp": df.index[-1].isoformat()
            })

        return jsonify({
            "signal": signal.signal_type,
            "confidence": signal.confidence,
            "current_price": signal.entry_price,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "risk_reward_ratio": signal.risk_reward_ratio,
            "timestamp": signal.timestamp.isoformat(),
            "features": features_dict
        })

    except Exception as e:
        logger.error(f"Error generating signal: {e}")
        return jsonify({
            "error": "Signal generation failed",
            "message": str(e)
        }), 500


@app.route("/predict", methods=["POST"])
def predict():
    """Predict from provided OHLCV data."""
    if MODEL is None:
        return jsonify({
            "error": "Model not loaded",
            "message": "The ML model is not available"
        }), 503

    try:
        data = request.get_json()

        if not data or "ohlcv" not in data:
            return jsonify({
                "error": "Invalid input",
                "message": "Please provide 'ohlcv' data"
            }), 400

        # Convert to DataFrame
        ohlcv = data["ohlcv"]
        df = pd.DataFrame(ohlcv)
        df.index = pd.to_datetime(df["timestamp"])
        df = df.drop("timestamp", axis=1, errors="ignore")

        # Generate features
        df = FEATURE_ENGINEER.create_all_features(df)

        # Get latest row
        latest = df.iloc[-1:]
        available_features = [f for f in FEATURE_NAMES if f in latest.columns]

        if not available_features:
            return jsonify({
                "error": "Feature mismatch",
                "message": "Provided data does not contain required features"
            }), 400

        X = latest[available_features].values
        X_scaled = SCALER.transform(X)

        prediction = MODEL.predict(X_scaled)[0]
        probability = MODEL.predict_proba(X_scaled)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability_up": float(probability),
            "probability_down": float(1 - probability),
            "signal": "BUY" if prediction == 1 else "SELL",
            "timestamp": df.index[-1].isoformat()
        })

    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({
            "error": "Prediction failed",
            "message": str(e)
        }), 500


@app.route("/model-info", methods=["GET"])
def model_info():
    """Get model information."""
    if MODEL is None:
        return jsonify({
            "status": "not_loaded",
            "message": "Model not loaded"
        })

    model_type = type(MODEL).__name__

    info = {
        "status": "loaded",
        "model_type": model_type,
        "features_count": len(FEATURE_NAMES),
        "features": FEATURE_NAMES[:20],  # First 20 features
    }

    # Add model-specific info
    if hasattr(MODEL, "n_estimators"):
        info["n_estimators"] = MODEL.n_estimators
    if hasattr(MODEL, "max_depth"):
        info["max_depth"] = MODEL.max_depth
    if hasattr(MODEL, "learning_rate"):
        info["learning_rate"] = MODEL.learning_rate

    return jsonify(info)


@app.route("/features", methods=["GET"])
def get_features():
    """Get list of available features."""
    return jsonify({
        "features": FEATURE_NAMES,
        "count": len(FEATURE_NAMES),
        "categories": {
            "price": [f for f in FEATURE_NAMES if any(x in f for x in ["return", "body", "shadow", "range", "direction"])],
            "momentum": [f for f in FEATURE_NAMES if any(x in f for x in ["rsi", "macd", "stoch", "williams", "cci", "momentum"])],
            "trend": [f for f in FEATURE_NAMES if any(x in f for x in ["sma", "ema", "adx"])],
            "volatility": [f for f in FEATURE_NAMES if any(x in f for x in ["atr", "bollinger", "keltner"])],
            "volume": [f for f in FEATURE_NAMES if any(x in f for x in ["volume", "obv", "cmf"])],
            "time": [f for f in FEATURE_NAMES if any(x in f for x in ["hour", "dow", "month", "session"])],
        }
    })


@app.route("/historical-signals", methods=["GET"])
def historical_signals():
    """Get historical signal statistics."""
    if SIGNAL_GENERATOR is None:
        return jsonify({
            "total_signals": 0,
            "message": "No signals generated yet"
        })

    stats = SIGNAL_GENERATOR.get_signal_statistics()
    return jsonify(stats)


# Initialize model on startup
with app.app_context():
    load_model()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
