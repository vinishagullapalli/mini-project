# A Random Forest Time-Aware Regression Model for Retail Demand Forecasting & Store Replenishment

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Groq AI](https://img.shields.io/badge/Groq%20LLM-Inference-green.svg)](https://groq.com/)

---

## 📌 Project Overview

- **Domain:** Machine Learning & Predictive Analytics
- **Title:** *A Random Forest time-aware regression model for Retail demand forecasting*
- **Keywords:** `retail`, `demand`, `forecasting`, `random`, `forest`, `time`, `replenishment`, `safety-stock`
- **Repository:** [https://github.com/vinishagullapalli/mini-project.git](https://github.com/vinishagullapalli/mini-project.git)

### Abstract
This project develops a **Random Forest time-aware regression model** to forecast SKU-level demand for the coming week ($t+1$), addressing the fundamental challenges of retail demand volatility through a rigorous, data-driven methodology. Granular **Point-of-Sale (POS) transaction logs** form the empirical foundation of the dataset, which is processed through a structured temporal pipeline before model building and evaluation. Crucially, predictions are translated directly into action via an automated **Store Replenishment Planner**, computing dynamic safety stocks, reorder points (ROP), and pack-adjusted purchase orders to eliminate stockouts while minimizing working capital and perishability waste.

---

## 🏗️ System Architecture & Pipeline Workflow

```
[POS Transaction Logs (929k+ Txns)]
                 │
                 ▼
[Store-SKU Weekly Aggregator (pipeline.py)]
                 │
                 ▼
[Time-Aware Feature Engineering]
  ├── Non-leaking Lags (t, t-1, t-2, t-3, t-7, t-51)
  ├── Rolling Windows (Mean, Std, Min, Max over 2, 4, 8, 12 weeks)
  ├── Demand Momentum & Price Elasticity Ratios
  ├── Cyclical Calendar Encodings (sin/cos for week & month)
  └── Forward Promotional & Holiday Awareness
                 │
                 ▼
[Strict Temporal Train/Test Split (No Lookahead Leakage)]
                 │
                 ▼
[Random Forest Time-Aware Regressor vs Industry Baselines]
  ├── Random Forest (150 Trees, Depth 12, WMAPE = 15.69%)
  ├── Ridge Linear Regression Baseline (WMAPE = 16.73%)
  ├── 4-Week Rolling Mean Baseline (WMAPE = 21.02%)
  └── Lag-1 Persistence Baseline (WMAPE = 27.83%)
                 │
                 ▼
[Store Replenishment Optimization Engine]
  ├── Dynamic Safety Stock: SS = Z_α * σ_RMSE * sqrt((L + R)/7)
  ├── Reorder Point: ROP = Lead_Time_Demand + SS
  ├── Order-Up-To Target (S) & Case Pack Rounding
  └── Stockout & Perishability Spoilage Risk Flags
                 │
         ┌───────┴───────┐
         ▼               ▼
[Streamlit Dashboard]  [Groq AI Retail Copilot]
```

---

## 📐 Mathematical & Inventory Formulation

### 1. Weighted Mean Absolute Percentage Error (WMAPE)
In retail, traditional MAPE breaks down when demand volume is low or zero. We utilize **WMAPE**, the gold standard in retail supply chains:
$$\text{WMAPE} = \frac{\sum_{i=1}^{N} |y_i - \hat{y}_i|}{\sum_{i=1}^{N} y_i} \times 100\%$$

### 2. Cyclical Calendar Transforms
To maintain boundary continuity between week 52 and week 1:
$$\text{sin\_week} = \sin\left(\frac{2\pi \cdot \text{week}}{52.14}\right), \quad \text{cos\_week} = \cos\left(\frac{2\pi \cdot \text{week}}{52.14}\right)$$

### 3. Dynamic Safety Stock Formulation
Safety stock buffers against demand uncertainty during the order protection interval ($L + R$):
$$\text{Safety Stock (SS)} = \lceil Z_{\alpha} \times \sigma_{\text{RMSE}} \times \sqrt{\frac{L + R}{7}} \rceil$$
Where:
- $Z_{\alpha}$: Standard normal quantile for target cycle service level $\alpha$ (e.g., $Z_{0.95} = 1.645$).
- $\sigma_{\text{RMSE}}$: Standard error of the Random Forest forecast model.
- $L$: Supplier lead time in days.
- $R$: Store replenishment review cycle (7 days).

### 4. Reorder Point (ROP) and Order Quantity (ROQ)
$$\text{Lead Time Demand (LTD)} = \hat{D}_{t+1} \times \left(\frac{L}{7}\right)$$
$$\text{ROP} = \lceil \text{LTD} + \text{SS} \rceil$$
$$\text{Net Inventory} = I_{\text{on\_hand}} + I_{\text{in\_transit}}$$
$$\text{Recommended Order Quantity (ROQ)} = \begin{cases} \lceil \frac{S - \text{Net Inventory}}{\text{Case Pack}} \rceil \times \text{Case Pack} & \text{if Net Inventory} \le \text{ROP} \\ 0 & \text{otherwise} \end{cases}$$

---

## 📊 Empirical Model Benchmark Results

Evaluated on an **out-of-time holdout test set** (October – December):

| Model Architecture | WMAPE (%) | RMSE (Units) | MAE (Units) | $R^2$ Score | Forecast Bias (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Regressor (Ours)** | **15.69%** | **114.08** | **78.21** | **0.6181** | **-0.84%** |
| Ridge Linear Regression Baseline | 16.73% | 116.86 | 83.45 | 0.5992 | +1.12% |
| Naive 4-Week Rolling Mean Baseline | 21.02% | 119.67 | 104.78 | 0.5797 | -3.42% |
| Lag-1 Persistence Baseline | 27.83% | 156.84 | 138.74 | 0.2781 | -5.10% |

> **Key Finding:** The Random Forest time-aware model reduces forecast error by **43.6%** compared to lag-1 persistence and **25.4%** compared to standard moving averages, significantly cutting safety stock capital requirements.

---

## 🗂️ Project Repository Structure

```
mini project/
├── data/
│   ├── raw/
│   │   ├── pos_transactions.csv        # 929,418 simulated POS transactions
│   │   ├── sku_catalog.csv             # Product taxonomy, costs, perishability
│   │   └── store_catalog.csv           # Multi-format store network definitions
│   └── processed/
│       ├── weekly_store_sku_demand.csv # Aggregated weekly demand
│       └── demand_features_time_aware.csv # 47 engineered time-aware features
├── models/
│   ├── rf_demand_model.joblib          # Persisted Random Forest model weights
│   └── model_metadata.json             # Feature schema and hyperparameters
├── reports/
│   ├── feature_importance.png          # Top 15 MDI feature drivers
│   ├── model_benchmark_comparison.png  # WMAPE & R2 comparison chart
│   ├── forecast_vs_actual_timeseries.png # Time-series validation tracking
│   ├── replenishment_schedule.csv      # Complete actionable store PO schedule
│   └── pipeline_summary.json           # Comprehensive JSON run results
├── src/
│   ├── __init__.py
│   ├── data_generator.py               # POS transaction generation engine
│   ├── pipeline.py                     # Aggregation, lagging & rolling features
│   ├── model.py                        # RF regression, baselines & evaluation
│   ├── replenishment.py                # Safety stock, ROP, PO calculator
│   └── ai_advisor.py                   # Groq LLM retail copilot integration
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py                # Feature pipeline & metric verification
│   └── test_replenishment.py           # Inventory formulation unit tests
├── app.py                              # Interactive Streamlit Web Dashboard
├── run_pipeline.py                     # End-to-end command line execution runner
├── requirements.txt                    # Project dependencies
├── .env.example                        # Groq API environment configuration
├── .gitignore                          # Git ignore definitions
└── README.md                           # Comprehensive documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Windows / macOS / Linux

### 2. Installation
Clone the repository and install required packages:
```bash
git clone https://github.com/vinishagullapalli/mini-project.git
cd mini-project
pip install -r requirements.txt
```

### 3. Environment Configuration (Groq API)
Create a `.env` file from `.env.example`:
```bash
cp .env.example .env
```
Add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Execute the End-to-End Pipeline
Run the full training, evaluation, and replenishment pipeline:
```bash
python run_pipeline.py
```

### 5. Run Unit Tests
Verify model and pipeline integrity:
```bash
pytest tests/ -v
```

### 6. Launch the Interactive Web Dashboard
Start the Streamlit application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🖥️ Web Application Features

1. **Store Replenishment Planner**:
   - Filter by store location (Downtown Supermarket, Suburban Hypermarket, Express Mart).
   - Dynamic service level slider ($80\%$ to $99\%$) and supplier lead-time adjuster ($1$ to $14$ days).
   - Real-time Safety Stock, Reorder Point (ROP), and Recommended Order Quantity (ROQ).
   - Stockout risk alerts (🚨) and Perishability spoilage alerts (⚠️).
   - Single-click CSV export for direct procurement ERP ingestion.
2. **Demand Trends & Forecasts**:
   - Historical sales curves, 4-week moving baseline, and promotional uplifts.
3. **Model Diagnostics & Driver Analysis**:
   - Out-of-time benchmark metrics comparison table.
   - Top 15 time-aware feature importance visualization.
4. **AI Retail Copilot (Powered by Groq LLM)**:
   - Automated executive briefings and tactical purchase order justifications.
   - Dynamic SKU rationale and supplier lead-time risk mitigation.

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
