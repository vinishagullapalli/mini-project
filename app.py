"""
app.py
Streamlit Web Dashboard for Retail Demand Forecasting & Store Replenishment Planner.
Features interactive time-aware regression diagnostics, real-time safety stock / ROP simulation,
automated purchase order generation, and Groq-powered AI retail advisory.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

from src.model import RetailDemandForecaster
from src.replenishment import StoreReplenishmentPlanner
from src.ai_advisor import RetailAIAdvisor

load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Retail Demand Forecasting & Replenishment Planner",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-urgent {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-ok {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_project_data():
    """Loads processed features, predictions, and catalog."""
    features_path = "data/processed/demand_features_time_aware.csv"
    sku_path = "data/raw/sku_catalog.csv"
    store_path = "data/raw/store_catalog.csv"
    summary_path = "reports/pipeline_summary.json"

    if not os.path.exists(features_path) or not os.path.exists(sku_path):
        return None, None, None, None

    df_features = pd.read_csv(features_path)
    df_features["week_start"] = pd.to_datetime(df_features["week_start"])
    df_sku = pd.read_csv(sku_path)
    df_store = pd.read_csv(store_path)

    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)

    return df_features, df_sku, df_store, summary


def render_header():
    st.markdown('<div class="main-title">🛒 Retail Demand Forecasting & Store Replenishment Planner</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">A Random Forest Time-Aware Regression Model generating actionable inventory, safety stock, and reorder decisions.</div>', unsafe_allow_html=True)


def main():
    render_header()
    df_features, df_sku, df_store, summary = load_project_data()

    if df_features is None:
        st.warning("⚠️ Data and model artifacts not found. Please click below to generate data and train the Random Forest pipeline.")
        if st.button("🚀 Run Complete Training Pipeline Now", type="primary"):
            with st.spinner("Executing simulation, time-aware feature pipeline, and model training..."):
                import run_pipeline
                run_pipeline.main()
            st.success("Pipeline executed successfully! Please reload the page.")
            st.rerun()
        return

    # Sidebar Controls
    st.sidebar.header("⚙️ Replenishment Parameters")

    # Store Selector
    store_list = df_store["store_id"].tolist()
    store_labels = {row["store_id"]: f"{row['store_name']} ({row['store_type']})" for _, row in df_store.iterrows()}
    selected_store = st.sidebar.selectbox("🏬 Select Store Location:", store_list, format_func=lambda x: store_labels.get(x, x))

    # Category Selector
    category_list = ["All Categories"] + sorted(df_sku["category"].unique().tolist())
    selected_category = st.sidebar.selectbox("📦 Filter Product Category:", category_list)

    # Replenishment Policy Sliders
    st.sidebar.subheader("📐 Inventory Policy Settings")
    service_level = st.sidebar.slider("Target Service Level (Cycle SLA):", min_value=0.80, max_value=0.99, value=0.95, step=0.01, help="Higher service level increases safety stock buffer to prevent out-of-stock events.")
    lead_time_days = st.sidebar.slider("Supplier Lead Time (Days):", min_value=1, max_value=14, value=3, step=1, help="Lead time between placing purchase order and delivery at store dock.")
    case_pack_size = st.sidebar.selectbox("Master Carton Case Pack:", [1, 4, 6, 12, 24], index=2, help="Orders are rounded up to the nearest case pack multiple.")
    stock_simulation_ratio = st.sidebar.slider("Current Stock Simulation Ratio:", min_value=0.1, max_value=1.5, value=0.35, step=0.05, help="Simulated current on-hand inventory as fraction of weekly demand.")

    # Groq API status
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    st.sidebar.markdown("---")
    if groq_api_key:
        st.sidebar.success("⚡ Groq LLM Copilot: Connected")
    else:
        st.sidebar.info("ℹ️ Groq API: Operating in Analytical Mode")

    # Filter Data for Selected Store
    df_store_data = df_features[df_features["store_id"] == selected_store].copy()
    if selected_category != "All Categories":
        df_store_data = df_store_data[df_store_data["category"] == selected_category]

    latest_week = df_store_data["week_start"].max()
    latest_snapshot = df_store_data[df_store_data["week_start"] == latest_week].copy()

    # Initialize Planner
    planner = StoreReplenishmentPlanner(
        default_lead_time_days=lead_time_days,
        default_service_level=service_level,
        default_case_pack_size=case_pack_size
    )

    # Calculate Replenishment Plan
    plan_df = planner.generate_plan_for_dataset(
        predictions_df=latest_snapshot,
        sku_catalog=df_sku,
        default_stock_ratio=stock_simulation_ratio,
        service_level=service_level,
        lead_time_days=lead_time_days
    )

    # Dashboard Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🛒 Store Replenishment Planner",
        "📊 Demand Trends & Forecasts",
        "🧠 Time-Aware Model & Feature Drivers",
        "🤖 AI Retail Copilot"
    ])

    # ---------------- TAB 1: Replenishment Planner ----------------
    with tab1:
        st.subheader("📋 Actionable Store Replenishment Schedule (Next Week: t+1)")
        st.write(f"Showing optimal procurement recommendations for **{store_labels.get(selected_store, selected_store)}** for the coming week commencing **{latest_week.strftime('%Y-%m-%d')}**.")

        # Metric Tiles
        total_forecast_demand = plan_df["forecasted_weekly_demand"].sum()
        total_reorder_skus = int(plan_df["needs_reorder"].sum())
        total_order_qty = int(plan_df["recommended_order_qty"].sum())
        total_spend = plan_df["order_total_cost"].sum()
        total_margin = plan_df["expected_margin"].sum()
        stockout_count = int(plan_df["imminent_stockout_risk"].sum())

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Forecast Demand</div><div class="metric-value">{total_forecast_demand:,.0f}</div><div>units next week</div></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">SKUs to Reorder</div><div class="metric-value">{total_reorder_skus} / {len(plan_df)}</div><div>trigger ROP</div></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Purchase Units</div><div class="metric-value">{total_order_qty:,}</div><div>pack-adjusted</div></div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Capital Required</div><div class="metric-value">${total_spend:,.2f}</div><div>PO commitment</div></div>""", unsafe_allow_html=True)
        with col5:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Stockout Risk</div><div class="metric-value" style="color: {'#DC2626' if stockout_count > 0 else '#16A34A'};">{stockout_count}</div><div>imminent alerts</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Replenishment Data Table
        display_cols = [
            "sku_id", "sku_name", "category", "forecasted_weekly_demand",
            "current_on_hand", "safety_stock", "reorder_point",
            "needs_reorder", "recommended_order_qty",
            "order_total_cost", "expected_margin",
            "imminent_stockout_risk", "perishability_risk"
        ]
        
        table_df = plan_df[display_cols].copy()
        table_df = table_df.rename(columns={
            "sku_id": "SKU ID",
            "sku_name": "Product Name",
            "category": "Category",
            "forecasted_weekly_demand": "Forecast (t+1)",
            "current_on_hand": "On-Hand",
            "safety_stock": "Safety Stock",
            "reorder_point": "ROP",
            "needs_reorder": "Order?",
            "recommended_order_qty": "Order Qty",
            "order_total_cost": "PO Cost ($)",
            "expected_margin": "Exp. Margin ($)",
            "imminent_stockout_risk": "Stockout Risk",
            "perishability_risk": "Perishability Alert"
        })

        st.dataframe(
            table_df.style.format({
                "Forecast (t+1)": "{:.1f}",
                "PO Cost ($)": "${:,.2f}",
                "Exp. Margin ($)": "${:,.2f}"
            }),
            use_container_width=True
        )

        # Download Button
        csv_data = plan_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Replenishment Purchase Order (CSV)",
            data=csv_data,
            file_name=f"replenishment_order_{selected_store}_{latest_week.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

        st.markdown("---")
        # Interactive SKU Simulation
        st.subheader("🎯 Single SKU Replenishment Simulator")
        sku_selected = st.selectbox("Select SKU for detailed breakdown:", plan_df["sku_id"].tolist(), format_func=lambda s: f"{s} - {plan_df[plan_df['sku_id']==s]['sku_name'].iloc[0]}")
        selected_row = plan_df[plan_df["sku_id"] == sku_selected].iloc[0]

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.metric("Forecast Weekly Demand", f"{selected_row['forecasted_weekly_demand']:.1f} units")
            st.metric("Current On-Hand Inventory", f"{selected_row['current_on_hand']} units")
            st.metric("Lead Time Demand", f"{selected_row['lead_time_demand']:.1f} units")
        with sc2:
            st.metric("Dynamic Safety Stock (SS)", f"{selected_row['safety_stock']} units")
            st.metric("Reorder Point (ROP)", f"{selected_row['reorder_point']} units")
            st.metric("Order-Up-To Target (S)", f"{selected_row['order_up_to_target']} units")
        with sc3:
            st.metric("Recommended Order Qty", f"{selected_row['recommended_order_qty']} units")
            st.metric("Total PO Cost", f"${selected_row['order_total_cost']:,.2f}")
            status_text = "🚨 CRITICAL STOCKOUT IMMINENT" if selected_row['imminent_stockout_risk'] else ("ORDER REQUIRED" if selected_row['needs_reorder'] else "HEALTHY INVENTORY")
            st.metric("Inventory Status", status_text)

    # ---------------- TAB 2: Demand Trends & Forecasts ----------------
    with tab2:
        st.subheader("📈 SKU Historical Demand & Forward Forecast")
        sku_trend_choice = st.selectbox("Select SKU to view historical demand trajectory:", df_sku["sku_id"].tolist(), format_func=lambda s: f"{s} - {df_sku[df_sku['sku_id']==s]['sku_name'].iloc[0]}")
        
        sku_history = df_features[(df_features["store_id"] == selected_store) & (df_features["sku_id"] == sku_trend_choice)].sort_values("week_start")

        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(sku_history["week_start"], sku_history["weekly_demand"], label="Historical Observed Demand (t)", color="#2563EB", lw=2)
        ax.plot(sku_history["week_start"], sku_history["demand_rolling_mean_4w"], label="4-Week Rolling Baseline", color="#F59E0B", linestyle=":", lw=1.8)
        
        # Highlight promotional weeks
        promo_points = sku_history[sku_history["avg_discount_pct"] > 0]
        if not promo_points.empty:
            ax.scatter(promo_points["week_start"], promo_points["weekly_demand"], color="#DC2626", s=40, zorder=5, label="Promotion Active")

        ax.set_title(f"Demand Velocity & Promotional Uplifts: {selected_store} - {sku_trend_choice}", fontsize=13, weight="bold")
        ax.set_ylabel("Weekly Demand (Units)", fontsize=11)
        ax.set_xlabel("Week", fontsize=11)
        ax.legend()
        st.pyplot(fig)
        plt.close()

    # ---------------- TAB 3: Model Diagnostics & Drivers ----------------
    with tab3:
        st.subheader("🧠 Time-Aware Random Forest vs Industry Baselines")
        
        if summary and "evaluation_metrics" in summary:
            metrics = summary["evaluation_metrics"]
            metrics_table = [
                {"Model": "Random Forest Time-Aware Regressor", "WMAPE (%)": f"{metrics['random_forest']['wmape_pct']}%", "RMSE": metrics['random_forest']['rmse'], "MAE": metrics['random_forest']['mae'], "R² Score": metrics['random_forest']['r2_score']},
                {"Model": "Ridge Linear Regression", "WMAPE (%)": f"{metrics['ridge']['wmape_pct']}%", "RMSE": metrics['ridge']['rmse'], "MAE": metrics['ridge']['mae'], "R² Score": metrics['ridge']['r2_score']},
                {"Model": "Naive 4-Week Rolling Mean", "WMAPE (%)": f"{metrics['naive_4w']['wmape_pct']}%", "RMSE": metrics['naive_4w']['rmse'], "MAE": metrics['naive_4w']['mae'], "R² Score": metrics['naive_4w']['r2_score']},
                {"Model": "Lag-1 Persistence", "WMAPE (%)": f"{metrics['persistence']['wmape_pct']}%", "RMSE": metrics['persistence']['rmse'], "MAE": metrics['persistence']['mae'], "R² Score": metrics['persistence']['r2_score']}
            ]
            st.table(pd.DataFrame(metrics_table))

        # Show saved Feature Importance plot if exists
        feat_img_path = "reports/feature_importance.png"
        if os.path.exists(feat_img_path):
            st.image(feat_img_path, caption="Top Time-Aware Features Driving SKU Demand (MDI Score)", use_container_width=True)

        benchmark_img_path = "reports/model_benchmark_comparison.png"
        if os.path.exists(benchmark_img_path):
            st.image(benchmark_img_path, caption="Model Benchmark Comparison across Time Horizons", use_container_width=True)

    # ---------------- TAB 4: AI Retail Copilot ----------------
    with tab4:
        st.subheader("🤖 Groq AI Supply Chain Copilot")
        st.write("Leveraging Groq's high-speed inference engine (GPT-OSS-120B / Qwen) to interpret model forecasts, analyze supply chain risks, and provide executive procurement narratives.")

        advisor = RetailAIAdvisor(api_key=groq_api_key)

        col_ai1, col_ai2 = st.columns([1, 1])
        with col_ai1:
            st.markdown("#### 🏢 Store Replenishment Synthesis")
            if st.button("Generate Store-Wide Executive Briefing", type="primary"):
                summary_stats = {
                    "total_skus": len(plan_df),
                    "reorder_count": int(plan_df["needs_reorder"].sum()),
                    "total_order_cost": float(plan_df["order_total_cost"].sum()),
                    "total_expected_margin": float(plan_df["expected_margin"].sum()),
                    "stockout_alerts": int(plan_df["imminent_stockout_risk"].sum()),
                    "perishability_alerts": int(plan_df["perishability_risk"].sum())
                }
                with st.spinner("Groq AI is analyzing replenishment trade-offs..."):
                    briefing = advisor.generate_store_summary(summary_stats)
                st.markdown(briefing)

        with col_ai2:
            st.markdown("#### 🔍 SKU Decision Deep-Dive")
            target_sku_id = st.selectbox("Select SKU for AI Rationale:", plan_df["sku_id"].tolist(), key="ai_sku_selector")
            target_item = plan_df[plan_df["sku_id"] == target_sku_id].iloc[0]

            if st.button("Generate SKU Rationale & Stockout Mitigation"):
                with st.spinner(f"Consulting Groq AI for {target_item['sku_name']}..."):
                    rationale = advisor.analyze_replenishment_plan(
                        store_id=selected_store,
                        store_name=store_labels.get(selected_store, selected_store),
                        sku_id=target_sku_id,
                        sku_name=target_item["sku_name"],
                        category=target_item["category"],
                        forecast_demand=target_item["forecasted_weekly_demand"],
                        current_on_hand=target_item["current_on_hand"],
                        lead_time_days=lead_time_days,
                        safety_stock=target_item["safety_stock"],
                        reorder_point=target_item["reorder_point"],
                        recommended_order=target_item["recommended_order_qty"],
                        order_cost=target_item["order_total_cost"],
                        expected_margin=target_item["expected_margin"],
                        stockout_risk=target_item["imminent_stockout_risk"],
                        perishability_risk=target_item["perishability_risk"]
                    )
                st.markdown(rationale)


if __name__ == "__main__":
    main()
