"""
model.py
Time-Aware Random Forest Regression Model for Retail SKU Demand Forecasting.
Includes temporal cross-validation, baseline comparisons (Naive, Persistence, Ridge),
comprehensive retail metrics (WMAPE, MAPE, MAE, RMSE, R2, Bias), and feature importance.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes industry-standard retail demand forecasting metrics.
    Includes WMAPE (Weighted Mean Absolute Percentage Error) and Forecast Bias.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Clip predictions to non-negative (demand cannot be negative)
    y_pred = np.clip(y_pred, a_min=0, a_max=None)

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    # MAPE: avoid division by zero using small epsilon
    epsilon = 1e-5
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, epsilon))) * 100.0)

    # WMAPE: Weighted MAPE = sum(|actual - pred|) / sum(actual) * 100
    sum_actual = float(np.sum(y_true))
    wmape = float((np.sum(np.abs(y_true - y_pred)) / max(sum_actual, epsilon)) * 100.0)

    # Forecast Bias: sum(pred - actual) / sum(actual) * 100
    bias = float((np.sum(y_pred - y_true) / max(sum_actual, epsilon)) * 100.0)

    return {
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "wmape_pct": round(wmape, 2),
        "mape_pct": round(mape, 2),
        "r2_score": round(r2, 4),
        "forecast_bias_pct": round(bias, 2)
    }


class RetailDemandForecaster:
    """
    Time-aware Random Forest Regressor for Weekly SKU Demand Forecasting.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 12,
        min_samples_split: int = 4,
        min_samples_leaf: int = 2,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.baseline_ridge = Ridge(alpha=1.0)
        self.feature_columns: List[str] = []
        self.categorical_columns = ["store_id", "category", "store_type"]
        self.target_column = "target_next_week_demand"
        self.categories_mapping: Dict[str, Any] = {}
        self.is_fitted = False

    def prepare_feature_matrix(
        self,
        df: pd.DataFrame,
        is_training: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Encodes categorical variables and extracts numerical feature matrix X and target y.
        Encodings are calibrated exclusively on training data to prevent leakage.
        """
        data = df.copy()

        # Identify numerical feature columns
        candidate_numeric = [
            "demand_lag_1w", "demand_lag_2w", "demand_lag_3w", "demand_lag_4w",
            "demand_lag_8w", "demand_lag_52w",
            "demand_rolling_mean_2w", "demand_rolling_mean_4w", "demand_rolling_mean_8w", "demand_rolling_mean_12w",
            "demand_rolling_std_4w", "demand_rolling_std_8w",
            "demand_rolling_min_4w", "demand_rolling_max_4w",
            "demand_ema_4w", "demand_momentum_2_8",
            "avg_unit_price", "avg_discount_pct", "max_discount_pct",
            "promo_days_count", "holiday_days_count",
            "target_next_week_is_promo", "target_next_week_holiday",
            "calendar_week", "calendar_month", "calendar_quarter",
            "sin_week", "cos_week", "sin_month", "cos_month",
            "price_ratio_to_base", "unit_cost", "is_perishable", "shelf_life_days"
        ]

        numeric_features = [col for col in candidate_numeric if col in data.columns]

        # One-hot encode categoricals deterministically
        for cat_col in self.categorical_columns:
            if cat_col in data.columns:
                if is_training:
                    unique_vals = sorted(data[cat_col].dropna().unique().tolist())
                    self.categories_mapping[cat_col] = unique_vals
                else:
                    unique_vals = self.categories_mapping.get(cat_col, [])
                
                for val in unique_vals:
                    dummy_name = f"{cat_col}_{val}"
                    data[dummy_name] = (data[cat_col] == val).astype(float)
                    if dummy_name not in numeric_features:
                        numeric_features.append(dummy_name)

        if is_training:
            self.feature_columns = numeric_features

        X = data[self.feature_columns].fillna(0.0)
        y = data[self.target_column] if self.target_column in data.columns else None
        return X, y

    def temporal_train_test_split(
        self,
        df: pd.DataFrame,
        test_weeks: int = 12
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits data strictly by time. The last `test_weeks` form the holdout test set.
        """
        df_sorted = df.sort_values("week_start").reset_index(drop=True)
        unique_weeks = sorted(df_sorted["week_start"].unique())

        if len(unique_weeks) <= test_weeks:
            raise ValueError(f"Not enough weeks ({len(unique_weeks)}) to hold out {test_weeks} weeks.")

        cutoff_date = unique_weeks[-test_weeks]
        train_df = df_sorted[df_sorted["week_start"] < cutoff_date].copy()
        test_df = df_sorted[df_sorted["week_start"] >= cutoff_date].copy()

        print(f"Temporal Split: Train period {train_df['week_start'].min().strftime('%Y-%m-%d')} to {train_df['week_start'].max().strftime('%Y-%m-%d')} ({len(train_df):,} rows)")
        print(f"Temporal Split: Test period  {test_df['week_start'].min().strftime('%Y-%m-%d')} to {test_df['week_start'].max().strftime('%Y-%m-%d')} ({len(test_df):,} rows)")

        return train_df, test_df

    def train_and_evaluate(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Trains Random Forest and baseline models, then evaluates out-of-time test metrics.
        """
        X_train, y_train = self.prepare_feature_matrix(train_df, is_training=True)
        X_test, y_test = self.prepare_feature_matrix(test_df, is_training=False)

        print(f"Training Random Forest model on {X_train.shape[0]} rows with {X_train.shape[1]} features...")
        self.model.fit(X_train, y_train)
        self.baseline_ridge.fit(X_train, y_train)
        self.is_fitted = True

        # Predictions
        rf_pred = np.clip(self.model.predict(X_test), 0, None)
        ridge_pred = np.clip(self.baseline_ridge.predict(X_test), 0, None)
        
        # Baselines
        naive_rolling_pred = test_df["demand_rolling_mean_4w"].values
        persistence_lag1_pred = test_df["demand_lag_1w"].values

        # Evaluation
        rf_metrics = calculate_metrics(y_test, rf_pred)
        ridge_metrics = calculate_metrics(y_test, ridge_pred)
        naive_metrics = calculate_metrics(y_test, naive_rolling_pred)
        persistence_metrics = calculate_metrics(y_test, persistence_lag1_pred)

        # Feature Importance
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": importances
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        results = {
            "random_forest": rf_metrics,
            "baseline_ridge": ridge_metrics,
            "baseline_4w_rolling_mean": naive_metrics,
            "baseline_lag1_persistence": persistence_metrics,
            "top_features": feature_importance_df.head(15).to_dict(orient="records"),
            "test_predictions": test_df[["store_id", "sku_id", "week_start"]].copy()
        }
        results["test_predictions"]["actual_demand"] = y_test.values
        results["test_predictions"]["rf_forecast"] = np.round(rf_pred, 1)
        results["test_predictions"]["residual"] = np.round(y_test.values - rf_pred, 1)

        print("\n=== Model Evaluation on Out-of-Time Test Horizon ===")
        print(f"Random Forest Regressor : WMAPE = {rf_metrics['wmape_pct']}% | RMSE = {rf_metrics['rmse']} | R2 = {rf_metrics['r2_score']}")
        print(f"Ridge Regression        : WMAPE = {ridge_metrics['wmape_pct']}% | RMSE = {ridge_metrics['rmse']} | R2 = {ridge_metrics['r2_score']}")
        print(f"Naive 4-Week Rolling    : WMAPE = {naive_metrics['wmape_pct']}% | RMSE = {naive_metrics['rmse']} | R2 = {naive_metrics['r2_score']}")
        print(f"Lag-1 Persistence       : WMAPE = {persistence_metrics['wmape_pct']}% | RMSE = {persistence_metrics['rmse']} | R2 = {persistence_metrics['r2_score']}")

        return results

    def save(self, model_dir: str = "models"):
        """Saves trained model weights and feature schema."""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "rf_demand_model.joblib")
        metadata_path = os.path.join(model_dir, "model_metadata.json")

        joblib.dump(self.model, model_path)
        with open(metadata_path, "w") as f:
            json.dump({
                "feature_columns": self.feature_columns,
                "categorical_columns": self.categorical_columns,
                "categories_mapping": self.categories_mapping,
                "params": {
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "min_samples_split": self.min_samples_split,
                    "min_samples_leaf": self.min_samples_leaf
                }
            }, f, indent=2)
        print(f"Model saved to {model_path} and metadata to {metadata_path}")

    @classmethod
    def load(cls, model_dir: str = "models"):
        """Loads fitted forecaster from disk."""
        model_path = os.path.join(model_dir, "rf_demand_model.joblib")
        metadata_path = os.path.join(model_dir, "model_metadata.json")

        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Model artifacts not found in {model_dir}")

        with open(metadata_path, "r") as f:
            meta = json.load(f)

        instance = cls(**meta.get("params", {}))
        instance.model = joblib.load(model_path)
        instance.feature_columns = meta["feature_columns"]
        instance.categorical_columns = meta["categorical_columns"]
        instance.categories_mapping = meta["categories_mapping"]
        instance.is_fitted = True
        return instance
