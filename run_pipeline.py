"""
run_pipeline.py
Complete End-to-End Orchestrator for Retail Demand Forecasting & Replenishment.
Executes:
1. POS Transaction Log Simulation (data_generator.py)
2. Time-Aware Feature Engineering & Leakage-Free Aggregation (pipeline.py)
3. Random Forest Model Training, Baselines, & Evaluation (model.py)
4. Store Replenishment Plan Generation & Safety Stock Optimization (replenishment.py)
5. Groq AI Supply Chain Briefing (ai_advisor.py)
6. Publication-grade Report Visualizations (matplotlib/seaborn)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from src.data_generator import generate_pos_transactions
from src.pipeline import run_feature_pipeline
from src.model import RetailDemandForecaster
from src.replenishment import StoreReplenishmentPlanner
from src.ai_advisor import RetailAIAdvisor


def generate_visualizations(
    results: dict,
    test_df: pd.DataFrame,
    sku_catalog: pd.DataFrame,
    reports_dir: str = "reports"
):
    """Generates professional diagnostic plots for model evaluation and replenishment."""
    os.makedirs(reports_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")

    # 1. Feature Importance Plot
    top_features = pd.DataFrame(results["top_features"])
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_features, x="importance", y="feature", hue="feature", palette="viridis", legend=False)
    plt.title("Top 15 Most Influential Time-Aware Features (Random Forest MDI)", fontsize=14, weight="bold")
    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Engineered Feature", fontsize=12)
    plt.tight_layout()
    feat_plot_path = os.path.join(reports_dir, "feature_importance.png")
    plt.savefig(feat_plot_path, dpi=300)
    plt.close()
    print(f"Feature importance visualization saved to {feat_plot_path}")

    # 2. Model Comparison Metrics Chart
    comparison_data = [
        {"Model": "Random Forest Regressor", "WMAPE (%)": results["random_forest"]["wmape_pct"], "RMSE": results["random_forest"]["rmse"], "R2 Score": results["random_forest"]["r2_score"]},
        {"Model": "Ridge Regression", "WMAPE (%)": results["baseline_ridge"]["wmape_pct"], "RMSE": results["baseline_ridge"]["rmse"], "R2 Score": results["baseline_ridge"]["r2_score"]},
        {"Model": "Naive 4W Rolling Mean", "WMAPE (%)": results["baseline_4w_rolling_mean"]["wmape_pct"], "RMSE": results["baseline_4w_rolling_mean"]["rmse"], "R2 Score": results["baseline_4w_rolling_mean"]["r2_score"]},
        {"Model": "Lag-1 Persistence", "WMAPE (%)": results["baseline_lag1_persistence"]["wmape_pct"], "RMSE": results["baseline_lag1_persistence"]["rmse"], "R2 Score": results["baseline_lag1_persistence"]["r2_score"]}
    ]
    df_comp = pd.DataFrame(comparison_data)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=df_comp, x="Model", y="WMAPE (%)", hue="Model", palette="coolwarm", ax=ax[0], legend=False)
    ax[0].set_title("Weighted MAPE Comparison (Lower is Better)", fontsize=13, weight="bold")
    ax[0].set_ylabel("WMAPE (%)", fontsize=11)
    ax[0].tick_params(axis='x', rotation=25)

    sns.barplot(data=df_comp, x="Model", y="R2 Score", hue="Model", palette="mako", ax=ax[1], legend=False)
    ax[1].set_title("Out-of-Time R2 Score Comparison (Higher is Better)", fontsize=13, weight="bold")
    ax[1].set_ylabel("R2 Score", fontsize=11)
    ax[1].tick_params(axis='x', rotation=25)
    plt.tight_layout()
    metrics_plot_path = os.path.join(reports_dir, "model_benchmark_comparison.png")
    plt.savefig(metrics_plot_path, dpi=300)
    plt.close()
    print(f"Benchmark comparison visualization saved to {metrics_plot_path}")

    # 3. Time-Series Actual vs Forecast Plot for a Sample Store-SKU
    preds = results["test_predictions"]
    sample_store = preds["store_id"].iloc[0]
    sample_sku = preds["sku_id"].iloc[0]
    sample_slice = preds[(preds["store_id"] == sample_store) & (preds["sku_id"] == sample_sku)].sort_values("week_start")

    if not sample_slice.empty:
        plt.figure(figsize=(12, 5))
        plt.plot(sample_slice["week_start"], sample_slice["actual_demand"], marker="o", label="Actual POS Demand", color="#1f77b4", linewidth=2.5)
        plt.plot(sample_slice["week_start"], sample_slice["rf_forecast"], marker="s", linestyle="--", label="Random Forest Forecast (t+1)", color="#d62728", linewidth=2.0)
        plt.title(f"Out-of-Time Forecast vs Actuals: {sample_store} | {sample_sku}", fontsize=14, weight="bold")
        plt.xlabel("Week Commencing", fontsize=12)
        plt.ylabel("Weekly Units Demanded", fontsize=12)
        plt.legend(frameon=True)
        plt.tight_layout()
        ts_plot_path = os.path.join(reports_dir, "forecast_vs_actual_timeseries.png")
        plt.savefig(ts_plot_path, dpi=300)
        plt.close()
        print(f"Time-series forecast plot saved to {ts_plot_path}")


def main():
    print("=" * 70)
    print("RETAIL DEMAND FORECASTING & STORE REPLENISHMENT PIPELINE")
    print("=" * 70)

    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 1. POS Transaction Log Simulation
    pos_file = "data/raw/pos_transactions.csv"
    if not os.path.exists(pos_file):
        df_pos = generate_pos_transactions(start_date="2024-01-01", num_days=730, output_dir="data/raw")
    else:
        print(f"Existing POS transaction data found at {pos_file}.")

    # 2. Time-Aware Pipeline
    df_weekly, df_features = run_feature_pipeline(raw_pos_path=pos_file, output_dir="data/processed")

    # 3. Model Training & Evaluation
    forecaster = RetailDemandForecaster(n_estimators=150, max_depth=12, random_state=42)
    train_df, test_df = forecaster.temporal_train_test_split(df_features, test_weeks=12)
    eval_results = forecaster.train_and_evaluate(train_df, test_df)

    # Persist Model
    forecaster.save("models")

    # 4. Store Replenishment Planning
    print("\n--- Generating Actionable Store Replenishment Plan ---")
    sku_catalog = pd.read_csv("data/raw/sku_catalog.csv")
    planner = StoreReplenishmentPlanner(
        default_lead_time_days=3,
        default_review_period_days=7,
        default_service_level=0.95
    )

    # Generate replenishment schedule on the latest horizon
    latest_week = test_df["week_start"].max()
    latest_predictions = eval_results["test_predictions"][eval_results["test_predictions"]["week_start"] == latest_week]
    
    replenishment_plan = planner.generate_plan_for_dataset(
        predictions_df=latest_predictions,
        sku_catalog=sku_catalog,
        default_stock_ratio=0.35, # Simulating stock ready for reorder check
        service_level=0.95,
        lead_time_days=3
    )

    plan_path = "reports/replenishment_schedule.csv"
    replenishment_plan.to_csv(plan_path, index=False)
    print(f"Replenishment plan saved to {plan_path} ({len(replenishment_plan)} SKU lines)")

    # 5. Visualizations
    generate_visualizations(eval_results, test_df, sku_catalog, reports_dir="reports")

    # 6. Groq AI Executive Briefing
    print("\n--- Generating Groq AI Retail Copilot Executive Briefing ---")
    advisor = RetailAIAdvisor()
    summary_stats = {
        "total_skus": len(replenishment_plan),
        "reorder_count": int(replenishment_plan["needs_reorder"].sum()),
        "total_order_cost": float(replenishment_plan["order_total_cost"].sum()),
        "total_expected_margin": float(replenishment_plan["expected_margin"].sum()),
        "stockout_alerts": int(replenishment_plan["imminent_stockout_risk"].sum()),
        "perishability_alerts": int(replenishment_plan["perishability_risk"].sum())
    }
    ai_briefing = advisor.generate_store_summary(summary_stats)
    print(ai_briefing)

    # Save summary report to JSON
    summary_report = {
        "evaluation_metrics": {
            "random_forest": eval_results["random_forest"],
            "ridge": eval_results["baseline_ridge"],
            "naive_4w": eval_results["baseline_4w_rolling_mean"],
            "persistence": eval_results["baseline_lag1_persistence"]
        },
        "replenishment_summary": summary_stats,
        "ai_executive_briefing": ai_briefing
    }
    with open("reports/pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    print("\n[SUCCESS] Pipeline executed successfully! All artifacts and reports are ready.")


if __name__ == "__main__":
    main()
