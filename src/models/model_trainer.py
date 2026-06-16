"""
Advanced ML Model Training Pipeline for XAUUSD Scalping

Supports multiple algorithms with hyperparameter tuning.
Optimized for Kaggle environment with proper cross-validation.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)


class XAUUSDModelTrainer:
    """Train and evaluate ML models for XAUUSD scalping signals."""

    # Available models and their default hyperparameters
    MODEL_CONFIGS = {
        "xgboost": {
            "class": xgb.XGBClassifier,
            "params": {
                "n_estimators": 200,
                "max_depth": 7,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "random_state": 42,
                "n_jobs": -1,
                "use_label_encoder": False,
            },
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 7, 9],
                "learning_rate": [0.05, 0.1, 0.2],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.9],
            }
        },
        "lightgbm": {
            "class": lgb.LGBMClassifier,
            "params": {
                "n_estimators": 200,
                "max_depth": 7,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "binary",
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": -1,
            },
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 7, 9],
                "learning_rate": [0.05, 0.1, 0.2],
                "num_leaves": [31, 50, 100],
            }
        },
        "random_forest": {
            "class": RandomForestClassifier,
            "params": {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": -1,
                "class_weight": "balanced",
            },
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }
        },
        "gradient_boosting": {
            "class": GradientBoostingClassifier,
            "params": {
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "random_state": 42,
            },
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1, 0.2],
            }
        },
        "logistic_regression": {
            "class": LogisticRegression,
            "params": {
                "max_iter": 1000,
                "random_state": 42,
                "class_weight": "balanced",
                "n_jobs": -1,
            },
            "param_grid": {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"],
            }
        }
    }

    def __init__(
        self,
        model_type: str = "xgboost",
        models_dir: Optional[str] = None,
        tune_hyperparams: bool = False
    ):
        """
        Initialize model trainer.

        Args:
            model_type: Type of model to train
            models_dir: Directory to save trained models
            tune_hyperparams: Whether to perform hyperparameter tuning
        """
        if model_type not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available: {list(self.MODEL_CONFIGS.keys())}")

        self.model_type = model_type
        self.model_config = self.MODEL_CONFIGS[model_type]
        self.tune_hyperparams = tune_hyperparams
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []

        # Setup model directory
        if models_dir:
            self.models_dir = Path(models_dir)
        elif os.path.exists("/kaggle"):
            self.models_dir = Path("/kaggle/working/models")
        else:
            self.models_dir = Path("models")

        self.models_dir.mkdir(parents=True, exist_ok=True)

    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for training.

        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            fit_scaler: Whether to fit the scaler

        Returns:
            Tuple of (X, y) arrays
        """
        if "target" not in df.columns:
            raise ValueError("DataFrame must contain 'target' column")

        if feature_cols is None:
            # Auto-detect feature columns
            exclude = ["open", "high", "low", "close", "volume", "target"]
            feature_cols = [c for c in df.columns if c not in exclude]

        self.feature_names = feature_cols

        X = df[feature_cols].values
        y = df["target"].values

        # Scale features
        if fit_scaler:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        return X, y

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Training metrics dictionary
        """
        logger.info(f"Training {self.model_type} model...")

        # Calculate class weights for imbalanced data
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight_dict = {c: w for c, w in zip(classes, class_weights)}

        logger.info(f"Class distribution: {np.bincount(y_train)}")
        logger.info(f"Class weights: {class_weight_dict}")

        # Get model class and parameters
        model_class = self.model_config["class"]
        params = self.model_config["params"].copy()

        # Add class weight for supported models
        if self.model_type in ["xgboost", "lightgbm"]:
            params["scale_pos_weight"] = class_weight_dict.get(1, 1.0) / class_weight_dict.get(0, 1.0)

        # Hyperparameter tuning
        if self.tune_hyperparams and self.model_config["param_grid"]:
            logger.info("Starting hyperparameter tuning with TimeSeriesSplit...")
            tscv = TimeSeriesSplit(n_splits=3)

            grid_search = GridSearchCV(
                estimator=model_class(**params),
                param_grid=self.model_config["param_grid"],
                cv=tscv,
                scoring="f1",
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_

            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        else:
            # Direct training
            if X_val is not None and y_val is not None:
                # Train with early stopping
                if self.model_type == "xgboost":
                    self.model = model_class(**params)
                    self.model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=50,
                        verbose=False
                    )
                elif self.model_type == "lightgbm":
                    self.model = model_class(**params)
                    self.model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(50, verbose=False)]
                    )
                else:
                    self.model = model_class(**params)
                    self.model.fit(X_train, y_train)
            else:
                self.model = model_class(**params)
                self.model.fit(X_train, y_train)

        # Training metrics
        train_preds = self.model.predict(X_train)
        train_proba = self.model.predict_proba(X_train)[:, 1]

        metrics = {
            "train_accuracy": accuracy_score(y_train, train_preds),
            "train_precision": precision_score(y_train, train_preds, zero_division=0),
            "train_recall": recall_score(y_train, train_preds, zero_division=0),
            "train_f1": f1_score(y_train, train_preds, zero_division=0),
            "train_auc": roc_auc_score(y_train, train_proba),
        }

        logger.info("Training metrics:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        return metrics

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test data.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Evaluation metrics dictionary
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        logger.info("Evaluating model on test set...")

        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1_score": f1_score(y_test, predictions, zero_division=0),
            "auc": roc_auc_score(y_test, probabilities),
        }

        logger.info("Test metrics:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        # Classification report
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, predictions, target_names=["DOWN", "UP"]))

        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        logger.info(f"\nConfusion Matrix:\n{cm}")

        return metrics

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5
    ) -> Dict:
        """
        Perform time-series cross-validation.

        Args:
            X: Features
            y: Labels
            n_splits: Number of CV splits

        Returns:
            Cross-validation metrics
        """
        logger.info(f"Performing {n_splits}-fold time-series cross-validation...")

        tscv = TimeSeriesSplit(n_splits=n_splits)

        cv_scores = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "auc": []
        }

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"Fold {fold + 1}/{n_splits}")

            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Train model on fold
            model = self.model_config["class"](**self.model_config["params"])
            model.fit(X_train_fold, y_train_fold)

            # Evaluate
            preds = model.predict(X_val_fold)
            proba = model.predict_proba(X_val_fold)[:, 1]

            cv_scores["accuracy"].append(accuracy_score(y_val_fold, preds))
            cv_scores["precision"].append(precision_score(y_val_fold, preds, zero_division=0))
            cv_scores["recall"].append(recall_score(y_val_fold, preds, zero_division=0))
            cv_scores["f1"].append(f1_score(y_val_fold, preds, zero_division=0))
            cv_scores["auc"].append(roc_auc_score(y_val_fold, proba))

        # Calculate mean and std
        cv_results = {}
        for metric, scores in cv_scores.items():
            cv_results[f"cv_{metric}_mean"] = np.mean(scores)
            cv_results[f"cv_{metric}_std"] = np.std(scores)

        logger.info("Cross-validation results:")
        for k, v in cv_results.items():
            logger.info(f"  {k}: {v:.4f}")

        return cv_results

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance from trained model.

        Returns:
            DataFrame with feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importance = np.abs(self.model.coef_[0])
        else:
            logger.warning("Model does not support feature importance")
            return None

        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False)

        return importance_df

    def save_model(self, filename: Optional[str] = None) -> str:
        """
        Save trained model and scaler.

        Args:
            filename: Base filename for saved model

        Returns:
            Path to saved model file
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        if filename is None:
            filename = f"xauusd_{self.model_type}_model"

        model_path = self.models_dir / f"{filename}.joblib"
        scaler_path = self.models_dir / f"{filename}_scaler.joblib"
        features_path = self.models_dir / f"{filename}_features.txt"

        # Save model
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")

        # Save scaler
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

        # Save feature names
        with open(features_path, "w") as f:
            f.write("\n".join(self.feature_names))
        logger.info(f"Feature names saved to {features_path}")

        return str(model_path)

    def load_model(self, filepath: str) -> None:
        """
        Load a trained model.

        Args:
            filepath: Path to saved model
        """
        self.model = joblib.load(filepath)

        # Try to load scaler
        scaler_path = filepath.replace(".joblib", "_scaler.joblib")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

        # Try to load feature names
        features_path = filepath.replace(".joblib", "_features.txt")
        if os.path.exists(features_path):
            with open(features_path, "r") as f:
                self.feature_names = [line.strip() for line in f]

        logger.info(f"Model loaded from {filepath}")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.

        Args:
            X: Feature array

        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        return predictions, probabilities
