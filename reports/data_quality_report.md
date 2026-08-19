# Project FORESIGHT
## Data Quality & Cleaning Report

**Client:** NorthBay Living  
**Project:** Demand & Inventory Intelligence  
**Phase:** Week 1 — Data Foundation  

## 1. Dataset Summary

| Dataset | Raw Rows | Clean Rows | Columns |
|---|---:|---:|---:|
| sales_daily | 133,433 | 133,373 | 6 |
| sku_master | 200 | 200 | 6 |
| calendar | 730 | 730 | 6 |
| inventory_snapshots | 18,992 | 18,992 | 6 |

## 2. Important Data-Quality Findings

### Negative `units_sold`: 15 records

> Negative unit quantities were identified as invalid sales quantities in the supplied simulated extract. Since the project models demand rather than returns/refunds, these values were treated as invalid demand observations and replaced with zero after validation.

### Exact duplicate sales records: 60

> Exact duplicate sales records were removed because retaining them would artificially inflate demand and revenue estimates.

### Missing `promo_event`: 654 records

> Missing promotional events were replaced with `No Event`. A null promotional event is interpreted as no named promotion being recorded rather than as a broken sales record.

## 3. Cleaning Decisions

| Dataset | Issue | Count | Treatment | Reason |
|---|---|---:|---|---|
| sales_daily | Invalid dates | 0 | Removed rows | A sales record requires a valid transaction date. |
| sales_daily | Missing SKU IDs | 0 | Reported; rows retained for validation | SKU is required to identify product demand. |
| sales_daily | Exact duplicate sales records | 60 | Removed exact duplicates | Duplicates would artificially inflate demand and revenue. |
| sales_daily | Negative units_sold | 15 | Replaced with 0 | FORESIGHT forecasts demand rather than returns/refunds. Negative quantities are invalid demand observations. |
| sales_daily | Missing units_sold | 3,934 | Replaced with 0 | Missing demand observations are treated as zero recorded demand for this simulated extract. |
| sales_daily | Negative revenue | 15 | Set to missing and reconstructed where possible | Negative revenue is not valid for the demand and revenue modelling used in FORESIGHT. |
| sales_daily | Missing revenue | 3,949 | Reconstructed 3949 records using units_sold × unit_price | Revenue can be derived from demand and selling price when both are available. |
| sales_daily | Missing unit_price | 0 | Reported | Price is needed for revenue reconstruction and business impact calculations. |
| sales_daily | Negative unit_price | 0 | Set to missing | Selling price cannot be negative. |
| sales_daily | Missing promo_flag | 0 | Missing values replaced with 0 | 0 represents no recorded promotion. |
| sku_master | Missing/invalid launch_date | 0 | Retained as missing | Launch date is descriptive information and should not be fabricated. |
| sku_master | Negative unit_cost | 0 | Set to missing | Product cost cannot be negative. |
| sku_master | Negative list_price | 0 | Set to missing | Product list price cannot be negative. |
| sku_master | Duplicate SKU IDs | 0 | Kept first record | sku_id is the primary key of sku_master. |
| calendar | Invalid dates | 0 | Removed rows | Calendar requires a valid date. |
| calendar | Duplicate calendar dates | 0 | Kept first record | Calendar date acts as the primary key for the daily calendar. |
| calendar | Missing is_holiday | 0 | Replaced with 0 | 0 represents no recorded holiday. |
| calendar | Missing promo_event | 654 | Replaced with 'No Event' | Null promotional event means no named promotion was recorded. |
| inventory_snapshots | Invalid dates | 0 | Removed rows | Inventory snapshots require valid dates. |
| inventory_snapshots | Negative on_hand_units | 0 | Set to 0 | Physical stock cannot be negative. |
| inventory_snapshots | Negative on_order_units | 0 | Set to 0 | Ordered stock cannot be negative. |
| inventory_snapshots | Negative lead_time_days | 0 | Set to missing | Lead time cannot be negative. |
| inventory_snapshots | Negative reorder_point | 0 | Set to 0 | Reorder point cannot be negative. |
| inventory_snapshots | Duplicate inventory snapshots | 0 | Kept first record | Each SKU should have one inventory position per snapshot date. |
| inventory_snapshots | Missing lead_time_days | 379 | Filled 379 values using SKU median, then overall median | Lead time is required for stockout risk calculation. |
| inventory_snapshots | Missing on_hand_units | 0 | Reported | Current stock position is required for inventory risk scoring. |
| inventory_snapshots | Missing on_order_units | 0 | Replaced with 0 | No on-order quantity is interpreted as no stock currently on order. |
| inventory_snapshots | Missing reorder_point | 0 | Reported | Reorder point is an important input to replenishment decisions. |

## 4. Analysis-Ready Dataset

- Rows: **133,373**
- Columns: **30**
- Unique SKUs: **200**
- Start date: **2022-01-01**
- End date: **2023-12-31**

## 5. Weekly Demand Dataset

- Rows: **19,191**
- Unique SKUs: **200**
- Weeks: **105**

## 6. Final Missing-Value Check

| Column | Missing Values |
|---|---:|
| lead_time_days | 429 |
| on_order_units | 429 |
| on_hand_units | 429 |
| reorder_point | 429 |

## 7. Validation Checks

- Negative units remaining: **0**
- Negative revenue remaining: **0**
- Exact duplicate rows remaining: **0**

## 8. Business Impact of Cleaning

The cleaning process prevents invalid or duplicated records from distorting demand forecasts, revenue estimates and inventory-risk decisions. This is important because the FORESIGHT model will use historical demand to recommend whether NorthBay should reorder, monitor or clear products.

## 9. Reproducibility

All cleaning and transformation steps are coded in `src/pipeline.py`. No manual spreadsheet cleaning is required. Running the pipeline from the project root regenerates the processed datasets and quality report from the raw extracts.
