# Project FORESIGHT

## Demand & Inventory Intelligence Dashboard

Project FORESIGHT is an end-to-end demand forecasting and inventory intelligence solution developed for **NorthBay Living**, a D2C home and lifestyle brand.

The project uses historical sales, product, calendar, promotion, pricing and inventory information to forecast future SKU-level demand and identify inventory risks.

The solution combines:

* Data preparation and cleaning
* Exploratory Data Analysis
* Feature engineering
* Seasonal-naive baseline forecasting
* Time-based rolling-origin backtesting
* Machine learning demand forecasting
* 8-week recursive forecasting
* Inventory risk scoring
* Replenishment recommendations
* Interactive Streamlit dashboard
* FastAPI scoring service

---

# 1. Project Overview

### Project Name

**Project FORESIGHT — Demand & Inventory Intelligence**

### Client

**NorthBay Living**

### Domain

Demand Forecasting, Inventory Analytics, Machine Learning and Decision Support

### Project Objective

The primary objective of Project FORESIGHT is to forecast future demand at SKU level and convert those forecasts into useful inventory planning insights.

The system helps identify:

* Products with potential stockout risk
* Products requiring replenishment
* Products requiring monitoring
* Products with excess inventory
* Demand patterns and variability
* SKU-level forecasting performance

---

# 2. Business Problem

NorthBay Living manages multiple products and needs better visibility into future demand and inventory requirements.

Traditional inventory planning can lead to:

* Stockouts
* Overstocking
* Excess inventory
* Poor replenishment timing
* Increased inventory holding costs
* Lost sales opportunities
* Inefficient purchasing decisions
* Difficulty prioritizing high-risk products

Project FORESIGHT addresses these challenges by combining demand forecasting with inventory risk analysis.

The overall decision-support workflow is:

```text
Historical Business Data
        ↓
Data Cleaning & Integration
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Demand Forecasting
        ↓
Inventory Risk Analysis
        ↓
Replenishment Recommendations
        ↓
Interactive Dashboard
```

---

# 3. Project Objectives

The major objectives are:

1. Analyze historical sales and inventory data.
2. Clean and integrate multiple business datasets.
3. Perform exploratory data analysis.
4. Identify demand patterns and seasonality.
5. Create time-series and business-related features.
6. Establish a seasonal-naive forecasting baseline.
7. Train a machine learning forecasting model.
8. Perform leakage-safe rolling-origin backtesting.
9. Generate an 8-week SKU-level demand forecast.
10. Calculate inventory risk.
11. Generate replenishment recommendations.
12. Build an interactive Streamlit dashboard.
13. Provide a FastAPI scoring service.
14. Present business findings through project documentation.

---

# 4. Technology Stack

| Technology       | Purpose                      |
| ---------------- | ---------------------------- |
| Python           | Core programming language    |
| Pandas           | Data processing and analysis |
| NumPy            | Numerical computation        |
| Scikit-learn     | Machine learning             |
| Joblib           | Model serialization          |
| Streamlit        | Interactive dashboard        |
| Plotly           | Interactive visualizations   |
| FastAPI          | Prediction/scoring API       |
| Uvicorn          | API server                   |
| Jupyter Notebook | EDA and experimentation      |
| Git / GitHub     | Version control              |

---

# 5. Dataset

Project FORESIGHT uses four primary raw datasets.

```text
data/raw/
├── sales_daily.csv
├── sku_master.csv
├── calendar.csv
└── inventory_snapshots.csv
```

---

## 5.1 Sales Data

File:

```text
data/raw/sales_daily.csv
```

The sales dataset contains historical daily sales information.

Important fields include:

```text
date
sku_id
units_sold
revenue
unit_price
promo_flag
```

---

## 5.2 SKU Master

File:

```text
data/raw/sku_master.csv
```

The SKU master contains product-level information such as:

```text
sku_id
category
subcategory
launch_date
unit_cost
list_price
```

---

## 5.3 Calendar Data

File:

```text
data/raw/calendar.csv
```

The calendar dataset provides time and business-event information used during feature engineering.

It supports information related to:

* Dates
* Holidays
* Promotions
* Seasonal patterns

---

## 5.4 Inventory Snapshots

File:

```text
data/raw/inventory_snapshots.csv
```

Important fields include:

```text
date
sku_id
on_hand_units
on_order_units
lead_time_days
reorder_point
```

These fields are used for inventory risk analysis and replenishment planning.

---

# 6. Data Preparation

The data preparation pipeline performs:

* Data loading
* Column standardization
* Date conversion
* Missing-value handling
* Duplicate checking
* Dataset integration
* SKU-level processing
* Weekly demand aggregation
* Inventory data preparation

The processed analysis-ready dataset contains:

```text
Rows    : 133,373
Columns : 30
SKUs    : 200
Period  : 2022–2023
```

The main processed dataset is:

```text
data/processed/analysis_ready.csv
```

---

# 7. Data Quality

The project includes data-quality documentation at:

```text
reports/data_quality_report.md
```

A cleaning summary is available at:

```text
reports/cleaning_summary.csv
```

The data preparation process checks for:

* Duplicate records
* Missing values
* Invalid data types
* Invalid dates
* Inconsistent SKU identifiers
* Numeric conversion issues

---

# 8. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand:

* Overall demand
* SKU-level demand
* Demand variability
* Revenue patterns
* Product performance
* Seasonality
* Promotion effects
* Holiday effects
* Inventory levels
* Forecast errors

Detailed findings are documented in:

```text
eda_insights.md
```

An additional report is available at:

```text
reports/eda_insights.md
```

---

# 9. Feature Engineering

The forecasting model uses **20 features**.

The features are divided into:

1. Lag features
2. Rolling-demand features
3. Calendar features
4. Business features

---

## 9.1 Lag Features

The model uses:

```text
lag_1
lag_2
lag_4
lag_8
lag_13
lag_26
lag_52
```

These features represent previous demand observations.

They allow the model to learn both short-term and longer-term demand patterns.

For example:

```text
lag_1  → recent demand
lag_4  → approximately four periods earlier
lag_13 → approximately one quarter earlier
lag_52 → approximately one year earlier
```

---

# 10. Rolling Features

The following rolling features are used:

```text
rolling_mean_4
rolling_mean_8
rolling_mean_13
rolling_std_8
```

These features help represent:

* Recent average demand
* Medium-term demand
* Longer-term demand
* Demand volatility

Rolling statistics allow the model to understand both demand level and demand variation.

---

# 11. Calendar Features

The model uses:

```text
month_num
quarter
week_of_year
sin_week
cos_week
```

The cyclic features:

```text
sin_week
cos_week
```

represent the seasonal position of the week in a continuous way.

These features help the model capture recurring seasonal patterns.

---

# 12. Business Features

The model uses:

```text
promo_ratio
holiday_ratio
unit_cost
list_price
```

These features provide additional business and commercial context.

### promo_ratio

Represents the proportion of the relevant period affected by promotions.

### holiday_ratio

Represents the proportion of the relevant period affected by holidays.

### unit_cost

Represents the cost associated with the product.

### list_price

Represents the product list price.

---

# 13. Complete Model Feature List

The final forecasting model uses:

```text
1.  lag_1
2.  lag_2
3.  lag_4
4.  lag_8
5.  lag_13
6.  lag_26
7.  lag_52
8.  rolling_mean_4
9.  rolling_mean_8
10. rolling_mean_13
11. rolling_std_8
12. month_num
13. quarter
14. week_of_year
15. sin_week
16. cos_week
17. promo_ratio
18. holiday_ratio
19. unit_cost
20. list_price
```

---

# 14. Forecasting Model

The machine learning model used is:

```text
HistGradientBoostingRegressor
```

The model configuration is:

```text
max_iter          = 300
learning_rate     = 0.05
max_leaf_nodes    = 31
min_samples_leaf  = 20
l2_regularization = 1.0
random_state      = 42
```

The model is saved as:

```text
models/forecast_model.joblib
```

---

# 15. Forecasting Methodology

Project FORESIGHT uses a time-based forecasting methodology.

No random train/test split is used.

Instead, the project uses **rolling-origin backtesting**.

The concept is:

```text
Fold 1

TRAIN TRAIN TRAIN TRAIN | TEST TEST TEST TEST
                         ↑
                    Future period


Fold 2

TRAIN TRAIN TRAIN TRAIN TRAIN | TEST TEST TEST TEST
                              ↑
                         Future period
```

This approach is more appropriate for time-series forecasting because the model is always evaluated on data occurring after the training period.

---

# 16. Seasonal-Naive Baseline

A seasonal-naive baseline is used for comparison.

The baseline uses historical seasonal demand information as a simple forecasting approach.

The project uses:

```text
Seasonal Period = 52 weeks
```

The purpose of the baseline is to determine whether the machine learning model provides useful improvement over a simpler forecasting method.

---

# 17. Rolling-Origin Backtesting

The forecasting pipeline uses:

```text
Forecast Horizon = 8 weeks
Backtest Folds    = 4
```

The four completed backtest folds are stored in:

```text
data/processed/backtest_results.csv
```

The average results across the four folds are approximately:

| Metric | Seasonal-Naive | FORESIGHT Model |
| ------ | -------------: | --------------: |
| WAPE   |         1.2017 |          0.1792 |

Average WAPE improvement across the four rolling-origin folds:

```text
85.10%
```

This indicates that the model substantially outperformed the seasonal-naive baseline across the completed rolling-origin backtests.

---

# 18. Model Comparison

An additional model comparison artifact is available at:

```text
data/processed/model_comparison.csv
```

It reports:

| Model                         |    MAE |   RMSE |
| ----------------------------- | -----: | -----: |
| Naive Baseline                | 2.4543 | 4.7533 |
| HistGradientBoostingRegressor | 1.6823 | 2.9350 |

Lower MAE and RMSE indicate smaller prediction errors.

The HistGradientBoostingRegressor therefore performs better than the reported naive baseline on these metrics.

---

# 19. Forecast Output

The forecasting pipeline generates an 8-week forecast for each SKU.

Output file:

```text
data/processed/forecast_output.csv
```

Main columns:

```text
sku_id
forecast_date
forecast_units
```

Current forecast coverage:

```text
SKUs               : 200
Forecast Horizon   : 8 weeks
Forecast Records   : 1,600
```

The forecast provides a forward-looking estimate of expected SKU-level demand.

---

# 20. Recursive Forecasting

The forecasting system generates future demand recursively.

The future forecast uses:

* Historical lag information
* Rolling demand information
* Calendar information
* Product information
* Promotion assumptions
* Holiday assumptions

When future promotion or holiday information is unavailable, the current forecasting process uses:

```text
promo_ratio   = 0
holiday_ratio = 0
```

This is an important limitation and should be considered when interpreting future forecasts.

---

# 21. Inventory Risk Analysis

Forecast demand is combined with inventory information to identify potential inventory risks.

The risk analysis considers:

```text
Current Inventory
+
On-Order Inventory
+
Forecast Demand
+
Lead Time
+
Reorder Point
+
Demand Variability
        ↓
Inventory Risk
        ↓
Recommended Action
```

The inventory-risk output is:

```text
data/processed/inventory_risk.csv
```

---

# 22. Inventory Risk Metrics

The inventory risk dataset contains information such as:

```text
inventory_units
forecast_8w_units
average_weekly_demand
peak_weekly_demand
forecast_std
days_of_cover
weeks_of_cover
stockout_gap_units
excess_inventory_units
stockout_probability_score
stockout_score
overstock_score
risk_score
stockout_risk
overstock_risk
risk_level
safety_stock_units
target_inventory_units
recommended_replenishment_units
inventory_value
recommended_action
```

---

# 23. Risk Levels

The current inventory-risk output contains:

```text
MEDIUM : 155 SKUs
HIGH   : 45 SKUs
```

The system also calculates stockout and overstock risk separately.

The risk level is intended as a decision-support indicator rather than an automatic purchase decision.

---

# 24. Replenishment Recommendations

The system generates business-oriented recommendations.

Examples include:

```text
MONITOR STOCK
consider planned replenishment
```

```text
REPLENISH SOON
increase replenishment priority
```

```text
URGENT REPLENISHMENT
prioritize purchase/order immediately
```

The system can also identify products where inventory reduction or clearance may be appropriate.

Examples include:

```text
REDUCE INVENTORY
slow purchasing and consider promotion
```

and:

```text
SEVERE OVERSTOCK
reduce purchasing and consider clearance
```

The recommendation should be treated as decision support and reviewed alongside actual business constraints.

---

# 25. Inventory Recommendation Summary

The current inventory-risk output contains the following recommendation counts:

| Recommendation       | SKUs |
| -------------------- | ---: |
| Monitor Stock        |   79 |
| Replenish Soon       |   61 |
| Urgent Replenishment |   45 |
| Reduce Inventory     |   14 |
| Severe Overstock     |    1 |

These categories help prioritize inventory-planning activities.

---

# 26. Streamlit Dashboard

The project includes an interactive Streamlit dashboard located at:

```text
app/app.py
```

The dashboard provides:

* Project overview
* SKU-level analysis
* Demand forecast visualization
* Inventory information
* Inventory risk analysis
* Risk classification
* Replenishment recommendations
* Forecast and inventory metrics
* Interactive Plotly charts

The dashboard is designed to convert machine learning outputs into business-friendly insights.

---

# 27. FastAPI Scoring Service

The project also includes a FastAPI service:

```text
service/main.py
```

The API provides endpoints for accessing forecasting and inventory-scoring information.

Main endpoints include:

```text
/
 /health
 /info
 /score/{sku_id}
 /forecast/{sku_id}
 /scores
 /reload
 /docs
```

### Root Endpoint

```text
GET /
```

Provides project and API information.

### Health Endpoint

```text
GET /health
```

Checks whether the forecast and risk datasets are available.

### SKU Score

```text
GET /score/{sku_id}
```

Returns forecast and risk information for an individual SKU.

### SKU Forecast

```text
GET /forecast/{sku_id}
```

Provides forecast information for a selected SKU.

### Scores

```text
GET /scores
```

Provides scoring information across SKUs.

### Reload

```text
GET /reload
```

Reloads the required datasets.

### API Documentation

FastAPI automatically provides interactive API documentation at:

```text
/docs
```

when the service is running.

---

# 28. Running the FastAPI Service

From the project root:

```powershell
uvicorn service.main:app --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

Interactive documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 29. Project Structure

```text
foresight/
│
├── app/
│   ├── app.py
│   └── app1.py
│
├── data/
│   ├── raw/
│   │   ├── sales_daily.csv
│   │   ├── sku_master.csv
│   │   ├── calendar.csv
│   │   └── inventory_snapshots.csv
│   │
│   └── processed/
│       ├── analysis_ready.csv
│       ├── backtest_results.csv
│       ├── baseline_results.csv
│       ├── forecast_metrics.csv
│       ├── forecast_model.pkl
│       ├── forecast_output.csv
│       ├── inventory_risk.csv
│       ├── inventory_snapshots.csv
│       ├── model_comparison.csv
│       ├── model_evaluation.csv
│       ├── sku_error_analysis.csv
│       ├── sku_forecast_metrics.csv
│       └── weekly_demand.csv
│
├── models/
│   └── forecast_model.joblib
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_model.ipynb
│
├── reports/
│   ├── cleaning_summary.csv
│   ├── data_quality_report.md
│   ├── eda_insights.md
│   └── risk_summary.csv
│
├── service/
│   └── main.py
│
├── src/
│   ├── forecast.py
│   ├── pipeline.py
│   ├── risk.py
│   └── tempCodeRunnerFile.py
│
├── utils/
│   └── api_client.py
│
├── eda_insights.md
├── executive_readout.md
├── README.md
├── requirements.txt
└── .gitignore
```

> `venv/`, Python cache files and other local development files should not be uploaded to GitHub.

---

# 30. Important Project Files

### `app/app.py`

Main Streamlit dashboard application.

### `src/pipeline.py`

Handles data preparation and processing.

### `src/forecast.py`

Contains forecasting feature engineering, model training, rolling-origin backtesting and forecast generation.

### `src/risk.py`

Contains inventory-risk analysis and risk calculations.

### `service/main.py`

Contains the FastAPI scoring service.

### `models/forecast_model.joblib`

Serialized forecasting model.

### `data/processed/forecast_output.csv`

8-week SKU-level forecast output.

### `data/processed/inventory_risk.csv`

Inventory risk and recommendation output.

### `eda_insights.md`

Detailed exploratory-data-analysis findings.

### `executive_readout.md`

Business-oriented executive summary.

---

# 31. Installation

Clone or download the project and open a terminal in the project root.

```powershell
cd C:\Projects\foresight
```

Create a virtual environment:

```powershell
python -m venv venv
```

Activate it:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

# 32. Requirements

The project dependencies are listed in:

```text
requirements.txt
```

Main packages include:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
streamlit
fastapi
uvicorn
joblib
openpyxl
```

---

# 33. Run the Data Pipeline

From the project root:

```powershell
python src/pipeline.py
```

This prepares the data required by the forecasting workflow.

---

# 34. Run the Forecasting Pipeline

Run:

```powershell
python src/forecast.py
```

The forecasting pipeline performs:

```text
Feature Engineering
        ↓
Model Training
        ↓
Rolling-Origin Backtesting
        ↓
Model Evaluation
        ↓
8-Week Forecast Generation
```

---

# 35. Run the Streamlit Dashboard

From the project root:

```powershell
python -m streamlit run app/app.py
```

The local dashboard normally runs at:

```text
http://localhost:8501
```

Open the displayed URL in a web browser.

---

# 36. Run the FastAPI Service

Use:

```powershell
uvicorn service.main:app --reload
```

The API normally runs at:

```text
http://127.0.0.1:8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

to access the interactive API documentation.

---

# 37. Model Verification

The trained model is stored at:

```text
models/forecast_model.joblib
```

It can be inspected using:

```powershell
python -c "import joblib; m=joblib.load('./models/forecast_model.joblib'); print(type(m)); print('Features:', m.n_features_in_)"
```

The expected model is:

```text
HistGradientBoostingRegressor
```

with:

```text
20 features
```

---

# 38. Forecast Verification

The forecast output can be checked using:

```powershell
python -c "import pandas as pd; f=pd.read_csv('data/processed/forecast_output.csv'); print(f.shape); print(f.columns.tolist()); print('SKUs:', f['sku_id'].nunique())"
```

Expected output structure:

```text
sku_id
forecast_date
forecast_units
```

---

# 39. Business Value

Project FORESIGHT can support NorthBay Living in:

### Demand Planning

Provides an 8-week forward view of expected demand.

### Stockout Prevention

Helps identify products where expected demand may exceed available inventory.

### Overstock Management

Highlights products where inventory may be high relative to expected demand.

### Replenishment Prioritization

Helps prioritize products requiring replenishment.

### Inventory Visibility

Provides a centralized view of inventory and forecast information.

### Data-Driven Decisions

Converts historical data into actionable decision-support information.

---

# 40. Key Results

The project produced an 8-week forecast for:

```text
200 SKUs
```

with:

```text
1,600 forecast records
```

The rolling-origin backtest used:

```text
4 folds
8-week forecast horizon
```

Average rolling-origin WAPE:

```text
Seasonal-Naive : 1.2017
FORESIGHT      : 0.1792
```

Average WAPE improvement:

```text
85.10%
```

An additional model-comparison artifact reports:

```text
Naive Baseline
MAE  = 2.4543
RMSE = 4.7533

HistGradientBoostingRegressor
MAE  = 1.6823
RMSE = 2.9350
```

The inventory-risk analysis covers:

```text
200 SKUs
```

with:

```text
MEDIUM risk = 155
HIGH risk   = 45
```

---

# 41. Limitations

The current solution has several limitations.

### Future Promotions

Future promotion information may not be known.

When unavailable, future promotion ratio is initialized to zero.

### Future Holidays

Future holiday information may not be available in the input data.

### SKU-Level Variation

Forecast accuracy can vary between different SKUs.

### MAPE Sensitivity

MAPE can become unstable for very low or zero demand values.

### Inventory Data Dependency

Inventory recommendations depend on the accuracy and freshness of inventory snapshots.

### Business Constraints

Supplier capacity, purchase-order constraints, budget limitations and other operational constraints are not fully represented by the forecasting model.

---

# 42. Future Improvements

Possible future improvements include:

1. Hyperparameter optimization.
2. Advanced time-series models.
3. Improved intermittent-demand handling.
4. Prediction intervals.
5. Better promotion forecasting.
6. Better holiday and event features.
7. Automated model retraining.
8. Model-performance monitoring.
9. Data-drift monitoring.
10. Forecast-vs-actual monitoring.
11. Real-time inventory integration.
12. Automated purchase-order recommendations.
13. Advanced SKU segmentation.
14. Explainable AI.
15. Production cloud infrastructure.

---

# 43. Production Roadmap

A future production architecture could follow:

```text
Business Data Sources
        ↓
Automated Data Pipeline
        ↓
Data Quality Checks
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Model Validation
        ↓
Forecast API
        ↓
Inventory Risk Engine
        ↓
Business Dashboard
        ↓
Decision Makers
```

---

# 44. Project Status

The current project includes:

* [x] Raw data preparation
* [x] Data integration
* [x] Data quality analysis
* [x] Exploratory Data Analysis
* [x] Lag features
* [x] Rolling features
* [x] Calendar features
* [x] Business features
* [x] Seasonal-naive baseline
* [x] Machine learning forecasting model
* [x] Time-based rolling-origin backtesting
* [x] Model evaluation
* [x] 8-week forecasting
* [x] Inventory risk analysis
* [x] Replenishment recommendations
* [x] Streamlit dashboard
* [x] FastAPI scoring service
* [x] EDA documentation
* [x] Executive readout
* [x] README documentation
* [ ] Automated model monitoring
* [ ] Automated retraining
* [ ] Advanced hyperparameter optimization

---

# 45. Deployment

The Streamlit dashboard can be deployed using a cloud hosting service such as Streamlit Community Cloud.

Recommended deployment entry point:

```text
app/app.py
```

Before cloud deployment, ensure that all files required at runtime by the Streamlit application are available in the deployed repository or through an appropriate external storage mechanism.

In particular, verify the availability of:

```text
data/processed/weekly_demand.csv
data/processed/forecast_output.csv
data/processed/inventory_risk.csv
models/forecast_model.joblib
```

The final live deployment URL can be added below:

```text
Live Dashboard:
<ADD-YOUR-DEPLOYED-STREAMLIT-URL-HERE>
```

---

# 46. Project Links

Add the final submission links after deployment:

```text
Source Code:
<ADD-YOUR-GITHUB-REPOSITORY-LINK>

Live Deployment:
<ADD-YOUR-STREAMLIT-DEPLOYMENT-LINK>

Demo Video:
<ADD-YOUR-DEMO-VIDEO-LINK>

Feedback Video:
<ADD-YOUR-FEEDBACK-VIDEO-LINK>

Project Report:
<ADD-YOUR-PROJECT-REPORT-LINK>
```

---

# 47. Conclusion

Project FORESIGHT provides an end-to-end demand forecasting and inventory intelligence workflow.

The solution combines:

```text
Data Preparation
       ↓
Exploratory Data Analysis
       ↓
Feature Engineering
       ↓
Baseline Forecasting
       ↓
Machine Learning
       ↓
Time-Based Backtesting
       ↓
8-Week Demand Forecast
       ↓
Inventory Risk Analysis
       ↓
Replenishment Recommendations
       ↓
Interactive Dashboard
       ↓
Business Decision Support
```

The project demonstrates how historical business data can be transformed into demand forecasts and inventory insights.

The HistGradientBoostingRegressor model improves upon the evaluated baseline metrics, while the rolling-origin backtesting framework provides a time-aware approach for evaluating forecasting performance.

The final dashboard makes the outputs accessible to business users and provides a foundation for future automation and production deployment.

---

# Project FORESIGHT

## Demand & Inventory Intelligence

**Client:** NorthBay Living

**Forecast Horizon:** 8 Weeks

**SKUs:** 200

**Model:** HistGradientBoostingRegressor

**Model Features:** 20

**Dashboard:** Streamlit

**API:** FastAPI

**Primary Outputs:** Demand Forecast + Inventory Risk + Replenishment Recommendations
