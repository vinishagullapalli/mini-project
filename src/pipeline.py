"""
pipeline.py
Structured Time-Aware Data Processing and Feature Engineering Pipeline.
Transforms raw Point-of-Sale (POS) transaction logs into store-SKU weekly grain
with non-leaking lagged features, rolling statistical windows, calendar encodings,
and promotional dynamics for retail demand forecasting.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional


def load_raw_pos_data(filepath: str = "data/raw/pos_transactions.csv") -> pd.DataFrame:
    """Loads POS transaction logs with parsed datetime."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"POS transactions file not found at {filepath}. Please run data_generator first.")
    
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def aggregate_pos_to_weekly(df_pos: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates granular POS transactions to Store-SKU-Weekly grain.
    Weeks start on Monday to mirror standard retail operating cycles.
    """
    df = df_pos.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    # Map each transaction date to its Monday week-commencing date (fast vectorized)
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.day_of_week, unit='D')
    df["week_start"] = pd.to_datetime(df["week_start"].dt.date)
    
    # Aggregate POS records per (store_id, sku_id, week_start)
    weekly_agg = df.groupby(["store_id", "sku_id", "week_start"]).agg(
        total_quantity=("quantity", "sum"),
        total_revenue=("total_sales", "sum"),
        total_cost=("total_cost", "sum"),
        avg_unit_price=("unit_price", "mean"),
        max_discount_pct=("discount_pct", "max"),
        avg_discount_pct=("discount_pct", "mean"),
        promo_days_count=("is_promoted", lambda s: int(df.loc[s.index[s > 0], "date"].nunique()) if (s > 0).any() else 0),
        holiday_days_count=("is_holiday", lambda s: int(df.loc[s.index[s > 0], "date"].nunique()) if (s > 0).any() else 0),
        stockout_occurred=("stockout_flag", "max"),
        transaction_count=("transaction_id", "count")
    ).reset_index()

    # Merge static Store & SKU metadata
    store_meta_cols = [c for c in ["store_id", "store_name", "store_type"] if c in df.columns]
    sku_meta_cols = [c for c in ["sku_id", "sku_name", "category", "unit_cost", "is_perishable", "shelf_life_days"] if c in df.columns]

    if store_meta_cols:
        store_meta = df[store_meta_cols].drop_duplicates()
        weekly_df = weekly_agg.merge(store_meta, on="store_id", how="left")
    else:
        weekly_df = weekly_agg

    if sku_meta_cols:
        sku_meta = df[sku_meta_cols].drop_duplicates()
        weekly_df = weekly_df.merge(sku_meta, on="sku_id", how="left")

    # Rename total_quantity to weekly_demand
    weekly_df = weekly_df.rename(columns={"total_quantity": "weekly_demand"})
    weekly_df["gross_margin"] = weekly_df["total_revenue"] - weekly_df["total_cost"]

    # Sort strictly by time for time-series integrity
    weekly_df = weekly_df.sort_values(["store_id", "sku_id", "week_start"]).reset_index(drop=True)
    return weekly_df


def build_time_aware_features(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs time-aware features strictly honoring temporal boundaries.
    Prevents lookahead bias by computing lags and rolling metrics exclusively
    on historical observations.
    
    Features built:
    - Target: demand at week t+1 (target_next_week_demand)
    - Historical Lags: t (lag 1), t-1 (lag 2), t-2 (lag 3), t-3 (lag 4), t-7 (lag 8), t-51 (lag 52)
    - Rolling Statistics: 2, 4, 8, and 12-week rolling mean, std, min, max
    - Momentum / Demand Trend: short-term rolling mean vs long-term rolling mean
    - Cyclical Calendar Encodings: sine and cosine transforms for week of year and month
    - Promotional & Pricing Context: upcoming promotional plan for t+1
    """
    df = weekly_df.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values(["store_id", "sku_id", "week_start"]).reset_index(drop=True)

    grouped = df.groupby(["store_id", "sku_id"])

    # 1. Target Variable: Demand in Coming Week (t+1)
    df["target_next_week_demand"] = grouped["weekly_demand"].shift(-1)
    
    # Also capture if week t+1 is scheduled for promotion (common known input in retail planning)
    df["target_next_week_is_promo"] = grouped["avg_discount_pct"].shift(-1).apply(lambda x: 1 if x > 0 else 0)
    df["target_next_week_holiday"] = grouped["holiday_days_count"].shift(-1).apply(lambda x: 1 if x > 0 else 0)

    # 2. Historical Lag Features (Demand in weeks t, t-1, t-2, t-3, t-7, t-51)
    # At forecasting time for week t+1, current week t demand is observed as lag_1w
    df["demand_lag_1w"] = df["weekly_demand"]
    df["demand_lag_2w"] = grouped["weekly_demand"].shift(1)
    df["demand_lag_3w"] = grouped["weekly_demand"].shift(2)
    df["demand_lag_4w"] = grouped["weekly_demand"].shift(3)
    df["demand_lag_8w"] = grouped["weekly_demand"].shift(7)
    df["demand_lag_52w"] = grouped["weekly_demand"].shift(51) # Year-over-year same week

    # 3. Rolling Window Statistics computed over observed history [t, t-1, t-2, ...]
    # We use closed='left' on shift or rolling directly on current & past observed demand
    df["demand_rolling_mean_2w"] = grouped["weekly_demand"].transform(lambda s: s.rolling(2, min_periods=1).mean())
    df["demand_rolling_mean_4w"] = grouped["weekly_demand"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    df["demand_rolling_mean_8w"] = grouped["weekly_demand"].transform(lambda s: s.rolling(8, min_periods=1).mean())
    df["demand_rolling_mean_12w"] = grouped["weekly_demand"].transform(lambda s: s.rolling(12, min_periods=1).mean())

    df["demand_rolling_std_4w"] = grouped["weekly_demand"].transform(lambda s: s.rolling(4, min_periods=2).std()).fillna(0.0)
    df["demand_rolling_std_8w"] = grouped["weekly_demand"].transform(lambda s: s.rolling(8, min_periods=2).std()).fillna(0.0)

    df["demand_rolling_min_4w"] = grouped["weekly_demand"].transform(lambda s: s.rolling(4, min_periods=1).min())
    df["demand_rolling_max_4w"] = grouped["weekly_demand"].transform(lambda s: s.rolling(4, min_periods=1).max())

    # Exponential Moving Average (EMA) to give more weight to recent sales
    df["demand_ema_4w"] = grouped["weekly_demand"].transform(lambda s: s.ewm(span=4, adjust=False).mean())

    # 4. Momentum & Demand Velocity Indicators
    # Ratio of short-term velocity (2-week) to medium-term baseline (8-week)
    df["demand_momentum_2_8"] = df["demand_rolling_mean_2w"] / (df["demand_rolling_mean_8w"] + 1e-4)

    # Price Ratio relative to SKU average
    df["price_ratio_to_base"] = df["avg_unit_price"] / (df["unit_cost"] * 1.5 + 1e-4)

    # 5. Calendar and Cyclical Seasonal Encodings
    df["calendar_week"] = df["week_start"].dt.isocalendar().week.astype(int)
    df["calendar_month"] = df["week_start"].dt.month
    df["calendar_quarter"] = df["week_start"].dt.quarter
    df["calendar_year"] = df["week_start"].dt.year

    # Trigonometric transformation preserving circular continuity (e.g. week 52 is adjacent to week 1)
    df["sin_week"] = np.sin(2 * np.pi * df["calendar_week"] / 52.14)
    df["cos_week"] = np.cos(2 * np.pi * df["calendar_week"] / 52.14)
    df["sin_month"] = np.sin(2 * np.pi * df["calendar_month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["calendar_month"] / 12.0)

    # Drop the final row per group where target_next_week_demand is NaN (the unobserved future week)
    # and drop early warmup periods where essential lags (e.g., 4-week lag) are unavailable
    df_clean = df.dropna(subset=["target_next_week_demand", "demand_lag_4w"]).copy()

    # Fill year-over-year lag 52w if dataset is < 52 weeks or early weeks with 8-week mean
    if df_clean["demand_lag_52w"].isna().any():
        df_clean["demand_lag_52w"] = df_clean["demand_lag_52w"].fillna(df_clean["demand_rolling_mean_8w"])

    return df_clean


def run_feature_pipeline(
    raw_pos_path: str = "data/raw/pos_transactions.csv",
    output_dir: str = "data/processed"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes the end-to-end data processing and feature engineering pipeline.
    Saves processed weekly records and feature matrix to output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading raw POS transaction logs from {raw_pos_path}...")
    df_pos = load_raw_pos_data(raw_pos_path)

    print("Aggregating POS logs to Store-SKU-Weekly grain...")
    df_weekly = aggregate_pos_to_weekly(df_pos)
    weekly_save_path = os.path.join(output_dir, "weekly_store_sku_demand.csv")
    df_weekly.to_csv(weekly_save_path, index=False)
    print(f"Weekly aggregated demand saved to {weekly_save_path} ({len(df_weekly):,} rows)")

    print("Generating time-aware lagged, rolling, and seasonal feature sets...")
    df_features = build_time_aware_features(df_weekly)
    features_save_path = os.path.join(output_dir, "demand_features_time_aware.csv")
    df_features.to_csv(features_save_path, index=False)
    print(f"Time-aware feature dataset saved to {features_save_path} ({len(df_features):,} rows)")

    return df_weekly, df_features


if __name__ == "__main__":
    run_feature_pipeline()
