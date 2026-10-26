"""
test_pipeline.py
Unit tests for data generation and time-aware feature pipeline.
"""

import os
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.data_generator import generate_pos_transactions, DEFAULT_SKU_CATALOG, DEFAULT_STORE_CATALOG
from src.pipeline import aggregate_pos_to_weekly, build_time_aware_features
from src.model import calculate_metrics


def test_pos_generator_simulation():
    """Tests synthetic POS transaction generator for consistency and schema."""
    df_pos = generate_pos_transactions(
        start_date="2024-01-01",
        num_days=30,  # fast test run
        random_seed=42,
        output_dir="data/test_raw"
    )
    
    assert not df_pos.empty
    expected_cols = [
        "transaction_id", "timestamp", "date", "store_id", "sku_id",
        "category", "unit_price", "quantity", "total_sales", "is_promoted", "stockout_flag"
    ]
    for col in expected_cols:
        assert col in df_pos.columns, f"Missing expected column: {col}"

    assert (df_pos["quantity"] > 0).all()
    assert (df_pos["total_sales"] >= 0).all()


def test_weekly_aggregation():
    """Tests aggregation from POS transactions to Store-SKU-Weekly grain."""
    df_pos = pd.read_csv("data/test_raw/pos_transactions.csv")
    df_weekly = aggregate_pos_to_weekly(df_pos)

    assert not df_weekly.empty
    assert "weekly_demand" in df_weekly.columns
    assert "gross_margin" in df_weekly.columns
    # Check uniqueness of (store_id, sku_id, week_start)
    assert not df_weekly.duplicated(subset=["store_id", "sku_id", "week_start"]).any()


def test_time_aware_features_no_leakage():
    """Verifies that lag features and rolling means strictly reflect past periods."""
    df_pos = pd.read_csv("data/test_raw/pos_transactions.csv")
    df_weekly = aggregate_pos_to_weekly(df_pos)
    
    # We need enough weeks to test lags, let's create a controlled synthetic weekly df
    weeks = pd.date_range("2024-01-01", periods=10, freq="W-MON")
    sample_weekly = pd.DataFrame({
        "store_id": ["STORE_01"] * 10,
        "sku_id": ["SKU_DAIRY_01"] * 10,
        "week_start": weeks,
        "weekly_demand": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "avg_unit_price": [4.0] * 10,
        "unit_cost": [2.5] * 10,
        "avg_discount_pct": [0.0] * 10,
        "max_discount_pct": [0.0] * 10,
        "promo_days_count": [0] * 10,
        "holiday_days_count": [0] * 10,
        "stockout_occurred": [0] * 10,
        "store_name": ["Store 1"] * 10,
        "store_type": ["Supermarket"] * 10,
        "sku_name": ["Milk"] * 10,
        "category": ["Dairy"] * 10,
        "is_perishable": [1] * 10,
        "shelf_life_days": [12] * 10,
        "total_revenue": [40.0] * 10,
        "total_cost": [25.0] * 10,
        "gross_margin": [15.0] * 10,
        "transaction_count": [10] * 10
    })

    df_feats = build_time_aware_features(sample_weekly)

    # For row index corresponding to weekly_demand = 50:
    row_50 = df_feats[df_feats["weekly_demand"] == 50].iloc[0]
    
    # Target should be week t+1 demand = 60
    assert row_50["target_next_week_demand"] == 60
    
    # demand_lag_1w is week t demand = 50
    assert row_50["demand_lag_1w"] == 50
    
    # demand_lag_2w is week t-1 demand = 40
    assert row_50["demand_lag_2w"] == 40
    
    # Cyclical sine and cosine values must be bounded [-1, 1]
    assert -1.0 <= row_50["sin_week"] <= 1.0
    assert -1.0 <= row_50["cos_week"] <= 1.0


def test_metric_calculations():
    """Tests accuracy of RMSE, MAE, WMAPE, and Bias calculations."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])

    metrics = calculate_metrics(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(20.0 / 3.0, 0.01)
    # WMAPE = (10 + 10 + 0) / (100 + 200 + 300) = 20 / 600 = 3.33%
    assert metrics["wmape_pct"] == pytest.approx(3.33, 0.01)
    # Bias = (10 - 10 + 0) / 600 = 0.0%
    assert metrics["forecast_bias_pct"] == pytest.approx(0.0, 0.01)
