# Project FORESIGHT

## Demand & Inventory Intelligence

### Executive Readout

**Client:** NorthBay Living
**Project:** Project FORESIGHT — Demand & Inventory Intelligence
**Domain:** Demand Forecasting, Inventory Analytics and Machine Learning
**Forecast Horizon:** 8 Weeks
**SKUs Covered:** 200
**Forecasting Model:** HistGradientBoostingRegressor
**Dashboard:** Streamlit
**API:** FastAPI

---

# 1. Executive Summary

Project FORESIGHT is an end-to-end demand forecasting and inventory intelligence solution developed for NorthBay Living.

The objective of the project is to transform historical sales, product, calendar and inventory data into forward-looking demand forecasts and actionable inventory insights.

The solution combines:

* Historical demand analysis
* Feature engineering
* Time-based forecasting
* Machine learning
* Rolling-origin backtesting
* Inventory risk scoring
* Replenishment recommendations
* Interactive business visualization

The overall workflow is:

```text
Historical Business Data
        ↓
Data Preparation
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Forecasting Model
        ↓
8-Week Demand Forecast
        ↓
Inventory Risk Analysis
        ↓
Business Recommendations
        ↓
Decision Support Dashboard
```

The final solution produces an 8-week demand forecast for 200 SKUs and combines those forecasts with inventory conditions to help prioritize replenishment and inventory-management activities.

---

# 2. Business Problem

NorthBay Living needs better visibility into future product demand and inventory requirements.

Without a forecasting and inventory intelligence system, businesses may face:

* Stockouts
* Overstocking
* Excess inventory
* Poor replenishment timing
* Increased holding costs
* Lost sales
* Inefficient purchasing decisions
* Difficulty identifying high-priority SKUs

Project FORESIGHT addresses this problem by connecting demand forecasting with inventory risk analysis.

---

# 3. Business Objectives

The main objectives were:

1. Understand historical SKU-level demand.
2. Prepare and integrate business datasets.
3. Identify important demand patterns.
4. Build a machine learning forecasting model.
5. Compare the model with a seasonal-naive baseline.
6. Evaluate the model using time-based validation.
7. Generate an 8-week forecast.
8. Identify inventory risks.
9. Prioritize replenishment requirements.
10. Present results through an interactive dashboard.

---

# 4. Data Overview

The solution uses four main datasets:

```text
sales_daily.csv
sku_master.csv
calendar.csv
inventory_snapshots.csv
```

The integrated analysis-ready dataset contains:

```text
Rows    : 133,373
Columns : 30
SKUs    : 200
Period  : 2022–2023
```

The datasets provide information about:

* Historical sales
* Products
* Categories
* Prices
* Costs
* Promotions
* Holidays
* Calendar periods
* On-hand inventory
* On-order inventory
* Lead time
* Reorder points

---

# 5. Forecasting Approach

The project uses a machine learning forecasting approach based on:

```text
HistGradientBoostingRegressor
```

The model uses 20 engineered features.

These features include:

### Historical Demand

```text
lag_1
lag_2
lag_4
lag_8
lag_13
lag_26
lag_52
```

### Rolling Demand

```text
rolling_mean_4
rolling_mean_8
rolling_mean_13
rolling_std_8
```

### Calendar

```text
month_num
quarter
week_of_year
sin_week
cos_week
```

### Business

```text
promo_ratio
holiday_ratio
unit_cost
list_price
```

These features allow the model to use recent demand, longer-term patterns, seasonality and product/business information.

---

# 6. Validation Strategy

Forecasting validation is performed using a time-based rolling-origin approach.

No random train/test split is used.

The project uses:

```text
Backtest Folds     : 4
Forecast Horizon   : 8 weeks
Seasonal Period    : 52 weeks
```

The approach ensures that the model is trained using historical information and evaluated on later periods.

This is more appropriate for demand forecasting because future information should not be used to predict the past.

---

# 7. Forecasting Results

The rolling-origin backtest results show that the machine learning model outperformed the seasonal-naive baseline.

Average results across the four completed folds:

| Metric | Seasonal-Naive | FORESIGHT Model |
| ------ | -------------: | --------------: |
| WAPE   |         1.2017 |          0.1792 |

Average WAPE improvement:

```text
85.10%
```

This indicates a substantial reduction in weighted forecast error compared with the seasonal-naive approach during the rolling-origin backtests.

---

# 8. Additional Model Evaluation

The project also contains a model-comparison evaluation using MAE and RMSE.

| Model                         |    MAE |   RMSE |
| ----------------------------- | -----: | -----: |
| Naive Baseline                | 2.4543 | 4.7533 |
| HistGradientBoostingRegressor | 1.6823 | 2.9350 |

The machine learning model has lower MAE and RMSE than the reported naive baseline.

This indicates that the model provides better prediction accuracy on this evaluation.

---

# 9. Forecast Output

The forecasting pipeline generates:

```text
200 SKUs
×
8 Weeks
=
1,600 Forecast Records
```

The forecast output contains:

```text
sku_id
forecast_date
forecast_units
```

This provides an 8-week forward view of expected SKU-level demand.

---

# 10. Inventory Intelligence

Forecasting is combined with inventory information to identify products that may require action.

The risk analysis considers:

```text
On-Hand Inventory
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
```

The resulting information is used to calculate:

* Stockout risk
* Overstock risk
* Overall risk level
* Safety stock
* Target inventory
* Recommended replenishment quantity
* Business action

---

# 11. Inventory Risk Results

The current inventory-risk output covers:

```text
200 SKUs
```

Risk levels are:

| Risk Level | SKUs |
| ---------- | ---: |
| MEDIUM     |  155 |
| HIGH       |   45 |

The risk analysis indicates that a significant number of SKUs require inventory monitoring or replenishment prioritization.

---

# 12. Recommended Actions

The system generates business-oriented recommendations.

Current recommendation distribution:

| Recommended Action   | SKUs |
| -------------------- | ---: |
| Monitor Stock        |   79 |
| Replenish Soon       |   61 |
| Urgent Replenishment |   45 |
| Reduce Inventory     |   14 |
| Severe Overstock     |    1 |

### Monitor Stock

Used when inventory should be monitored and planned replenishment may be appropriate.

### Replenish Soon

Used when replenishment priority should be increased.

### Urgent Replenishment

Used when the product has a high stockout-related risk and should receive immediate attention.

### Reduce Inventory

Used when inventory is relatively high and purchasing or promotional action may be appropriate.

### Severe Overstock

Used when inventory reduction or clearance should be considered.

---

# 13. Business Impact

Project FORESIGHT can support several business activities.

## Demand Planning

The 8-week forecast provides visibility into expected future demand.

## Replenishment Prioritization

High-risk products can be prioritized for purchasing review.

## Stockout Prevention

Products with insufficient inventory relative to forecast demand can be identified earlier.

## Overstock Management

Products with excess inventory can be reviewed for reduced purchasing or promotional action.

## Inventory Visibility

The dashboard provides a centralized view of forecast and inventory conditions.

## Data-Driven Decision Making

The solution converts historical business data into structured decision-support information.

---

# 14. Dashboard

The project includes an interactive Streamlit dashboard.

The dashboard allows users to explore:

* SKU-level forecasts
* Forecast trends
* Inventory information
* Risk levels
* Stockout risk
* Overstock risk
* Recommended replenishment
* Business recommendations

The dashboard is designed for users who may not need to interact directly with the underlying machine learning code.

---

# 15. API Service

A FastAPI scoring service is included in the project.

Main API capabilities include:

```text
Health Check
Project Information
SKU Forecast
SKU Risk Score
Multiple SKU Scores
Dataset Reload
Interactive API Documentation
```

The API provides a foundation for integrating the forecasting and inventory intelligence system with other applications.

---

# 16. Key Findings

### Finding 1 — SKU-level forecasting is important

Demand varies across products, so inventory planning should consider SKU-level demand rather than relying only on aggregate demand.

### Finding 2 — Historical demand provides strong predictive information

Lag and rolling features allow the model to capture recent and historical demand patterns.

### Finding 3 — Seasonal information is useful

Calendar and cyclic weekly features provide additional information about recurring demand behavior.

### Finding 4 — Machine learning improves on the evaluated baseline

The rolling-origin backtests show a large WAPE improvement over the seasonal-naive baseline.

The MAE/RMSE evaluation also reports lower errors for the HistGradientBoostingRegressor.

### Finding 5 — Forecasting should be combined with inventory information

A demand forecast becomes more useful when it is evaluated against current inventory, on-order units, lead time and reorder point.

### Finding 6 — Inventory decisions require prioritization

The inventory-risk output allows business users to focus attention on high-risk and replenishment-priority products.

---

# 17. Business Decision Framework

The recommended business workflow is:

```text
Review Forecast
      ↓
Check Current Inventory
      ↓
Check On-Order Inventory
      ↓
Check Lead Time
      ↓
Check Reorder Point
      ↓
Review Risk Level
      ↓
Review Recommended Quantity
      ↓
Business Approval
      ↓
Replenishment / Inventory Action
```

The system is intended as a decision-support tool.

Final purchasing decisions should also consider:

* Supplier constraints
* Purchase-order status
* Budget
* Warehouse capacity
* Business strategy
* Promotion plans
* Supplier lead-time changes

---

# 18. Limitations

The current solution has several limitations.

### Future Promotion Information

Future promotional activity may not always be known.

When future information is unavailable, the forecasting process uses:

```text
promo_ratio = 0
```

### Future Holiday Information

Future holiday information may not always be available.

When unavailable, the forecasting process uses:

```text
holiday_ratio = 0
```

### SKU-Level Accuracy

Forecast performance can vary between products.

### MAPE Sensitivity

MAPE can become unstable when actual demand is zero or very small.

### Inventory Data

Risk recommendations depend on the accuracy and freshness of inventory snapshots.

### Operational Constraints

The current model does not fully incorporate supplier capacity, budget constraints, warehouse capacity or purchase-order constraints.

---

# 19. Future Improvements

Future versions could include:

1. Hyperparameter optimization.
2. Advanced time-series models.
3. Intermittent-demand forecasting.
4. Prediction intervals.
5. Improved promotion forecasting.
6. Better holiday and event data.
7. Automated model retraining.
8. Forecast monitoring.
9. Data-drift monitoring.
10. Actual-vs-forecast monitoring.
11. Real-time inventory integration.
12. Automated purchase-order recommendations.
13. Advanced SKU segmentation.
14. Explainable AI.
15. Production-scale cloud deployment.

---

# 20. Overall Project Architecture

The complete solution can be summarized as:

```text
                    DATA SOURCES
                         |
                         v
              Data Preparation
                         |
                         v
              Data Quality Checks
                         |
                         v
           Exploratory Data Analysis
                         |
                         v
             Feature Engineering
                         |
                         v
              Seasonal Baseline
                         |
                         v
        HistGradientBoostingRegressor
                         |
                         v
          Rolling-Origin Backtesting
                         |
                         v
              8-Week Forecast
                         |
                         v
             Inventory Risk Engine
                         |
                         v
          Replenishment Recommendations
                         |
              +----------+----------+
              |                     |
              v                     v
        Streamlit Dashboard     FastAPI Service
              |                     |
              +----------+----------+
                         |
                         v
                Business Decisions
```

---

# 21. Project Deliverables

The project includes the following major deliverables:

### Data Pipeline

```text
src/pipeline.py
```

### Forecasting Pipeline

```text
src/forecast.py
```

### Inventory Risk Analysis

```text
src/risk.py
```

### Trained Model

```text
models/forecast_model.joblib
```

### Forecast Output

```text
data/processed/forecast_output.csv
```

### Inventory Risk Output

```text
data/processed/inventory_risk.csv
```

### Streamlit Dashboard

```text
app/app.py
```

### FastAPI Service

```text
service/main.py
```

### EDA Documentation

```text
eda_insights.md
```

### Executive Readout

```text
executive_readout.md
```

---

# 22. Final Conclusion

Project FORESIGHT demonstrates an end-to-end approach for transforming historical business data into demand forecasts and inventory intelligence.

The project combines:

```text
Historical Data
      ↓
Data Preparation
      ↓
EDA
      ↓
Feature Engineering
      ↓
Forecasting
      ↓
Time-Based Validation
      ↓
8-Week Demand Forecast
      ↓
Inventory Risk Analysis
      ↓
Replenishment Recommendations
      ↓
Interactive Dashboard
```

The HistGradientBoostingRegressor model provides improved performance over the evaluated baseline, while rolling-origin backtesting provides a time-aware validation framework.

The resulting forecasts and inventory-risk outputs can help NorthBay Living prioritize products that require replenishment, monitoring or inventory reduction.

The Streamlit dashboard provides an accessible interface for business users, while the FastAPI service provides a foundation for future system integration.

Overall, Project FORESIGHT demonstrates how machine learning and data analytics can be applied to demand planning and inventory decision support.

---

## Final Project Summary

| Item                     | Result                          |
| ------------------------ | ------------------------------- |
| Client                   | NorthBay Living                 |
| Domain                   | Demand & Inventory Intelligence |
| SKUs                     | 200                             |
| Forecast Horizon         | 8 Weeks                         |
| Forecast Records         | 1,600                           |
| Model                    | HistGradientBoostingRegressor   |
| Model Features           | 20                              |
| Backtest Folds           | 4                               |
| Baseline                 | Seasonal-Naive                  |
| Average Backtest WAPE    | 0.1792                          |
| Baseline Average WAPE    | 1.2017                          |
| Average WAPE Improvement | 85.10%                          |
| Inventory Risk SKUs      | 200                             |
| Medium Risk              | 155                             |
| High Risk                | 45                              |
| Dashboard                | Streamlit                       |
| API                      | FastAPI                         |
