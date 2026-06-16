"""
XAUUSD Scalping Model Training Script

Train ML models for XAUUSD scalping signal generation.
Optimized for Kaggle environment with proper error handling.

Usage:
    python -m src.train --model xgboost --period 5y --interval 1h
"""

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler

from src.data.data_fetcher import XAUUSDDataFetcher
from src.features.feature_engineering import FeatureEngineer
from src.models.model_trainer import XAUUSDModelTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train XAUUSD Scalping ML Model"
    )
    parser.add_argument(
        "--model", type=str, default="xgboost",
        choices=["xgboost", "lightgbm", "random_forest", "gradient_boosting", "logistic_regression"],
        help="Model type to train"
    )
    parser.add_argument(
        "--period", type=str, default="5y",
        help="Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max)"
    )
    parser.add_argument(
        "--interval", type=str, default="1h",
        help="Data interval (1m, 2m, 5m, 15m, 30m, 60m, 1h, 1d, 1wk, 1mo)"
    )
    parser.add_argument(
        "--lookahead", type=int, default=5,
        help="Number of bars to look ahead for target"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data for testing"
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Enable hyperparameter tuning"
    )
    parser.add_argument(
        "--cv-splits", type=int, default=5,
        help="Number of cross-validation splits"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for saved models"
    )
    parser.add_argument(
        "--kaggle", action="store_true",
        help="Run in Kaggle mode"
    )

    return parser.parse_args()


def train_model(args) -> dict:
    """
    Main training pipeline.

    Args:
        args: Parsed command line arguments

    Returns:
        Dictionary with training results
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("XAUUSD Scalping Model Training")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Period: {args.period}, Interval: {args.interval}")
    logger.info(f"Lookahead: {args.lookahead} bars")
    logger.info(f"Test size: {args.test_size}")
    logger.info(f"Hyperparameter tuning: {args.tune}")
    logger.info(f"Started at: {start_time}")

    try:
        # Step 1: Fetch data
        logger.info("\n[Step 1/6] Fetching historical data...")
        fetcher = XAUUSDDataFetcher(data_dir=args.output_dir)
        df_raw = fetcher.fetch_historical_data(
            period=args.period,
            interval=args.interval
        )

        if df_raw.empty or len(df_raw) < 100:
            raise ValueError(f"Insufficient data: only {len(df_raw)} rows")

        logger.info(f"Data fetched: {len(df_raw)} rows")

        # Step 2: Feature engineering
        logger.info("\n[Step 2/6] Engineering features...")
        fe = FeatureEngineer()
        df_features = fe.create_all_features(df_raw)
        df_features = fe.create_target(
            df_features,
            lookahead=args.lookahead
        )

        if df_features.empty:
            raise ValueError("Feature engineering produced no data")

        logger.info(f"Features created: {df_features.shape}")

        # Step 3: Prepare train/test split
        logger.info("\n[Step 3/6] Preparing data split...")
        exclude_cols = ["open", "high", "low", "close", "volume", "target"]
        feature_cols = [c for c in df_features.columns if c not in exclude_cols]

        X = df_features[feature_cols].values
        y = df_features["target"].values

        split_idx = int(len(df_features) * (1 - args.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        logger.info(f"Train: {X_train.shape[0]} samples")
        logger.info(f"Test: {X_test.shape[0]} samples")
        logger.info(f"Features: {X_train.shape[1]}")
        logger.info(f"Target distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

        # Step 4: Train model
        logger.info(f"\n[Step 4/6] Training {args.model} model...")
        trainer = XAUUSDModelTrainer(
            model_type=args.model,
            tune_hyperparams=args.tune
        )
        trainer.scaler = scaler
        trainer.feature_names = feature_cols

        train_metrics = trainer.train(X_train_scaled, y_train, X_test_scaled, y_test)

        # Step 5: Evaluate
        logger.info("\n[Step 5/6] Evaluating model...")
        test_metrics = trainer.evaluate(X_test_scaled, y_test)

        # Cross-validation
        logger.info(f"\n[Step 5.5/6] {args.cv_splits}-Fold Cross-Validation...")
        cv_results = trainer.cross_validate(X_train_scaled, y_train, n_splits=args.cv_splits)

        # Step 6: Save model
        logger.info("\n[Step 6/6] Saving model...")
        model_path = trainer.save_model()

        # Save scaler separately
        scaler_path = str(Path(model_path).parent / "scaler.joblib")
        joblib.dump(scaler, scaler_path)

        # Save feature list
        features_path = str(Path(model_path).parent / "feature_list.txt")
        with open(features_path, "w") as f:
            f.write("\n".join(feature_cols))

        # Feature importance
        importance_df = trainer.get_feature_importance()
        if importance_df is not None:
            importance_path = str(Path(model_path).parent / "feature_importance.csv")
            importance_df.to_csv(importance_path, index=False)
            logger.info(f"\nTop 10 important features:")
            logger.info(importance_df.head(10).to_string(index=False))

        # Training summary
        duration = (datetime.now() - start_time).total_seconds()

        summary = {
            "status": "success",
            "model_type": args.model,
            "training_date": datetime.now().isoformat(),
            "duration_seconds": duration,
            "data": {
                "period": args.period,
                "interval": args.interval,
                "total_samples": len(df_features),
                "n_features": len(feature_cols),
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            },
            "metrics": {
                "train": train_metrics,
                "test": test_metrics,
                "cv": cv_results
            },
            "model_path": model_path,
            "scaler_path": scaler_path,
            "features_path": features_path
        }

        # Save summary
        summary_path = str(Path(model_path).parent / "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration:.1f} seconds")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test F1: {test_metrics['f1_score']:.4f}")
        logger.info(f"Test AUC: {test_metrics['auc']:.4f}")
        logger.info(f"Model saved to: {model_path}")

        return summary

    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())

        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def main():
    """Main entry point."""
    args = parse_args()

    # Auto-detect Kaggle
    if os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        args.kaggle = True
        if args.output_dir is None:
            args.output_dir = "/kaggle/working/models"

    # Set default output dir
    if args.output_dir is None:
        args.output_dir = "models"

    os.makedirs(args.output_dir, exist_ok=True)

    # Run training
    result = train_model(args)

    # Print summary
    if result["status"] == "success":
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Model: {result['model_type']}")
        print(f"Test Accuracy: {result['metrics']['test']['accuracy']:.4f}")
        print(f"Test F1: {result['metrics']['test']['f1_score']:.4f}")
        print(f"Test AUC: {result['metrics']['test']['auc']:.4f}")
        print(f"Model Path: {result['model_path']}")
        print("=" * 60)

        sys.exit(0)
    else:
        print(f"\nTraining failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
