# EDA Insights — Project FORESIGHT

## 1. Project Overview

Project FORESIGHT is a Demand & Inventory Intelligence solution developed for
NorthBay Living, a D2C home and lifestyle brand.

The objective is to use historical sales, product, calendar, promotion, and
inventory information to understand demand patterns and support better
inventory planning.

---

## 2. Dataset Overview

The processed analysis dataset contains:

- Rows: 133,373
- Columns: 30
- SKUs: 200
- Historical period: 2022–2023

The main analytical fields include:

- date
- sku_id
- units_sold
- revenue
- unit_price
- promo_flag
- category
- subcategory
- launch_date
- unit_cost
- list_price
- calendar features
- holiday information
- promotion information
- inventory information

---

## 3. Demand Analysis

The historical data contains daily demand observations for multiple SKUs.

Demand varies significantly across products. Some SKUs have relatively low
daily demand, while others have substantially higher demand.

This variation makes SKU-level forecasting important because a single overall
demand model or average would not adequately represent every product.

---

## 4. Time-Based Features

The analysis includes several calendar features:

- year
- month
- week of year
- quarter
- day of month
- day of week
- weekend indicator

These features help the forecasting model capture recurring seasonal and
calendar-related demand patterns.

Cyclic weekly features were also created:

- sin_week
- cos_week

These represent the cyclical nature of the week number.

---

## 5. Lag Features

The forecasting model uses historical demand as predictive information.

The following lag features were created:

- lag_1
- lag_2
- lag_4
- lag_8
- lag_13
- lag_26
- lag_52

These represent demand from previous periods and allow the model to learn
short-term and longer-term demand patterns.

---

## 6. Rolling Demand Features

Rolling demand statistics were created to capture recent demand trends.

Features include:

- rolling_mean_4
- rolling_mean_8
- rolling_mean_13
- rolling_std_8

Rolling means represent recent average demand, while rolling standard
deviation provides information about demand variability.

These features are particularly useful for identifying changing demand levels
and unstable demand patterns.

---

## 7. Promotion and Holiday Features

Business-event features were included in the model:

- promo_ratio
- holiday_ratio

These features represent the proportion of the relevant period affected by
promotional or holiday activity.

Future forecasts currently assume unknown future promotion and holiday events
as zero unless future event information is supplied.

---

## 8. Product Pricing Features

The model uses:

- unit_cost
- list_price

These features provide product-level commercial information that can help
capture differences in demand between products.

---

## 9. Inventory Information

The processed dataset also contains inventory-related fields:

- on_hand_units
- on_order_units
- lead_time_days
- reorder_point
- available_inventory_units

These fields support the inventory-risk and replenishment analysis.

---

## 10. Forecasting Model

The selected forecasting model is:

`HistGradientBoostingRegressor`

Model configuration:

- learning_rate = 0.05
- max_iter = 300
- l2_regularization = 1.0
- random_state = 42

The model uses 20 features.

### Model Features

1. lag_1
2. lag_2
3. lag_4
4. lag_8
5. lag_13
6. lag_26
7. lag_52
8. rolling_mean_4
9. rolling_mean_8
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

---

## 11. Model Performance

The forecasting model achieved:

| Metric | Result |
|---|---:|
| MAE | 1.6823 |
| RMSE | 2.9350 |
| MAPE | 46.45% |

MAE of 1.6823 means that the model's predictions differ from actual demand
by approximately 1.68 units on average.

RMSE is higher than MAE because RMSE gives greater weight to larger
prediction errors.

MAPE should be interpreted carefully because demand values close to zero can
produce very large percentage errors.

---

## 12. Baseline Comparison

A naive baseline was created using the previous observed demand as the
prediction.

Baseline performance:

| Metric | Naive Baseline |
|---|---:|
| MAE | 2.4543 |
| RMSE | 4.7533 |

Model performance:

| Metric | FORESIGHT Model |
|---|---:|
| MAE | 1.6823 |
| RMSE | 2.9350 |

The forecasting model performs better than the naive baseline on both MAE and
RMSE.

This indicates that using lag, rolling, calendar, business, and pricing
features provides additional predictive value over simply using the previous
demand value.

---

## 13. Forecast Output

The forecasting pipeline generates an 8-week forecast for each SKU.

Current forecast output contains:

- 200 SKUs
- 8 forecast dates per SKU
- 1,600 forecast observations

Output fields:

- sku_id
- forecast_date
- forecast_units

---

## 14. Inventory Risk Analysis

Inventory risk analysis combines demand forecasts with inventory information.

The current risk output contains:

- 200 SKUs
- MEDIUM risk: 155 SKUs
- HIGH risk: 45 SKUs

The system generates replenishment recommendations such as:

- URGENT REPLENISHMENT
- REPLENISH SOON
- MONITOR STOCK

These recommendations are intended to help prioritize purchasing and
inventory planning activities.

---

## 15. Key Business Insights

### Insight 1 — SKU-level forecasting is important

Demand varies considerably between SKUs. Therefore, inventory decisions
should be made at SKU level rather than using a single aggregate demand
estimate.

### Insight 2 — Recent demand history is important

Lag and rolling-demand features allow the model to use recent demand behavior
when producing forecasts.

### Insight 3 — Seasonal patterns should be considered

Weekly, monthly, quarterly, and cyclic calendar features help the model
represent recurring demand patterns.

### Insight 4 — Forecasting can improve over a naive approach

The FORESIGHT model achieved lower MAE and RMSE than the naive previous-demand
baseline.

### Insight 5 — Forecasting and inventory planning should work together

Demand forecasts become more useful when combined with current inventory,
on-order stock, lead time, and reorder-point information.

### Insight 6 — Low-volume products require careful evaluation

MAPE can become unstable for products or periods with very low or zero
demand. MAE and RMSE should therefore also be considered when evaluating
forecast quality.

---

## 16. Limitations

1. Future promotional activity is currently assumed to be zero when unknown.
2. Future holiday information is not explicitly supplied to the forecasting
   process.
3. MAPE is sensitive to zero and near-zero demand.
4. Forecast accuracy can vary significantly between individual SKUs.
5. Inventory recommendations depend on the quality and freshness of inventory
   data.

---

## 17. Recommended Improvements

Future versions can improve the system by adding:

- Known future promotions
- Holiday calendars
- Price changes
- Stockout indicators
- More advanced hyperparameter tuning
- SKU-level model evaluation
- Prediction intervals
- Automated model retraining
- Forecast monitoring
- Actual-vs-forecast tracking

---

## 18. Conclusion

The exploratory analysis established the foundation for Project FORESIGHT by
combining historical demand, time-based patterns, business events, pricing,
and inventory information.

The HistGradientBoostingRegressor model improved upon the naive baseline and
produced 8-week forecasts for 200 SKUs.

The resulting forecasts can be combined with inventory risk analysis to support
replenishment prioritization and data-driven inventory planning.