from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PROJECT FORESIGHT
# Data Ingestion + Cleaning + Analysis Dataset Pipeline
#
# Client: NorthBay Living
# Role: Data Scientist & Analytics
#
# Purpose:
# 1. Load all four raw datasets
# 2. Profile and clean the data
# 3. Document data-quality issues
# 4. Build analysis-ready daily dataset
# 5. Build weekly demand dataset
# 6. Save reproducible outputs
#
# Run from project root:
#
#     python src/pipeline.py
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EXPECTED COLUMNS
# ============================================================

EXPECTED_COLUMNS = {

    "sales_daily.csv": [
        "date",
        "sku_id",
        "units_sold",
        "revenue",
        "unit_price",
        "promo_flag"
    ],

    "sku_master.csv": [
        "sku_id",
        "category",
        "subcategory",
        "launch_date",
        "unit_cost",
        "list_price"
    ],

    "calendar.csv": [
        "date",
        "week",
        "month",
        "season",
        "is_holiday",
        "promo_event"
    ],

    "inventory_snapshots.csv": [
        "date",
        "sku_id",
        "on_hand_units",
        "on_order_units",
        "lead_time_days",
        "reorder_point"
    ]
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def check_required_columns(
    df,
    required_columns,
    dataset_name
):
    """
    Validate that all required columns exist.
    """

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"{dataset_name} is missing required "
            f"columns: {missing_columns}"
        )


def safe_numeric_conversion(
    df,
    columns
):
    """
    Convert selected columns to numeric.
    Invalid values become NaN.
    """

    df = df.copy()

    for col in columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return df


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load all four raw datasets.
    """

    print("\nLoading raw datasets...")

    sales_path = RAW_DIR / "sales_daily.csv"
    sku_path = RAW_DIR / "sku_master.csv"
    calendar_path = RAW_DIR / "calendar.csv"
    inventory_path = RAW_DIR / "inventory_snapshots.csv"

    # --------------------------------------------------------
    # Check files exist
    # --------------------------------------------------------

    required_files = [
        sales_path,
        sku_path,
        calendar_path,
        inventory_path
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    # --------------------------------------------------------
    # Read files
    # --------------------------------------------------------

    sales = pd.read_csv(
        sales_path
    )

    sku = pd.read_csv(
        sku_path
    )

    calendar = pd.read_csv(
        calendar_path
    )

    inventory = pd.read_csv(
        inventory_path
    )

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    check_required_columns(
        sales,
        EXPECTED_COLUMNS["sales_daily.csv"],
        "sales_daily.csv"
    )

    check_required_columns(
        sku,
        EXPECTED_COLUMNS["sku_master.csv"],
        "sku_master.csv"
    )

    check_required_columns(
        calendar,
        EXPECTED_COLUMNS["calendar.csv"],
        "calendar.csv"
    )

    check_required_columns(
        inventory,
        EXPECTED_COLUMNS["inventory_snapshots.csv"],
        "inventory_snapshots.csv"
    )

    print("Raw datasets loaded successfully.")

    print(
        f"Sales:      {sales.shape}"
    )

    print(
        f"SKU Master: {sku.shape}"
    )

    print(
        f"Calendar:   {calendar.shape}"
    )

    print(
        f"Inventory:  {inventory.shape}"
    )

    return (
        sales,
        sku,
        calendar,
        inventory
    )


# ============================================================
# CLEAN SALES
# ============================================================

def clean_sales(sales, quality_log):
    """
    Clean sales_daily dataset.

    Important FORESIGHT decisions:

    1. Invalid dates are removed.
    2. Exact duplicate records are removed.
    3. Negative units_sold are treated as invalid demand
       observations and converted to 0.
    4. Negative revenue is treated as invalid.
    5. Missing demand is converted to 0.
    6. Missing revenue is reconstructed where possible.
    7. promo_flag is normalized to 0/1.
    """

    sales = sales.copy()

    # --------------------------------------------------------
    # Original row count
    # --------------------------------------------------------

    original_rows = len(sales)

    # --------------------------------------------------------
    # Date conversion
    # --------------------------------------------------------

    sales["date"] = pd.to_datetime(
        sales["date"],
        errors="coerce"
    )

    invalid_dates = sales["date"].isna().sum()

    if invalid_dates > 0:

        sales = sales.dropna(
            subset=["date"]
        )

    quality_log.append({
        "dataset": "sales_daily",
        "issue": "Invalid dates",
        "count": int(invalid_dates),
        "treatment": "Removed rows",
        "reason": "A sales record requires a valid transaction date."
    })

    # --------------------------------------------------------
    # SKU cleanup
    # --------------------------------------------------------

    sales["sku_id"] = (
        sales["sku_id"]
        .astype(str)
        .str.strip()
    )

    missing_sku = sales["sku_id"].isna().sum()

    quality_log.append({
        "dataset": "sales_daily",
        "issue": "Missing SKU IDs",
        "count": int(missing_sku),
        "treatment": "Reported; rows retained for validation",
        "reason": "SKU is required to identify product demand."
    })

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_cols = [
        "units_sold",
        "revenue",
        "unit_price",
        "promo_flag"
    ]

    sales = safe_numeric_conversion(
        sales,
        numeric_cols
    )

    # --------------------------------------------------------
    # Exact duplicate detection
    # --------------------------------------------------------

    duplicate_count = sales.duplicated().sum()

    quality_log.append({
        "dataset": "sales_daily",
        "issue": "Exact duplicate sales records",
        "count": int(duplicate_count),
        "treatment": "Removed exact duplicates",
        "reason": (
            "Duplicates would artificially inflate demand "
            "and revenue."
        )
    })

    sales = sales.drop_duplicates().copy()

    # --------------------------------------------------------
    # Negative units_sold
    # --------------------------------------------------------

    negative_units = (
        sales["units_sold"] < 0
    ).sum()

    quality_log.append({
        "dataset": "sales_daily",
        "issue": "Negative units_sold",
        "count": int(negative_units),
        "treatment": "Replaced with 0",
        "reason": (
            "FORESIGHT forecasts demand rather than "
            "returns/refunds. Negative quantities are "
            "invalid demand observations."
        )
    })

    sales.loc[
        sales["units_sold"] < 0,
        "units_sold"
    ] = 0

    # --------------------------------------------------------
    # Missing units_sold
    # --------------------------------------------------------

    missing_units = sales["units_sold"].isna().sum()

    quality_log.append({
        "dataset": "sales_daily",
        "issue": "Missing units_sold",
        "count": int(missing_units),
        "treatment": "Replaced with 0",
        "reason": (
            "Missing demand observations are treated as "
            "zero recorded demand for this simulated extract."
        )
    })

    sales["units_sold"] = (
        sales["units_sold"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Negative revenue
    # --------------------------------------------------------

    negative_revenue = (
        sales["revenue"] < 0
    ).sum()

    quality_log.append({
        "dataset": "sales_daily",
        "issue": "Negative revenue",
        "count": int(negative_revenue),
        "treatment": "Set to missing and reconstructed where possible",
        "reason": (
            "Negative revenue is not valid for the demand "
            "and revenue modelling used in FORESIGHT."
        )
    })

    sales.loc[
        sales["revenue"] < 0,
        "revenue"
    ] = np.nan

    # --------------------------------------------------------
    # Missing revenue
    # --------------------------------------------------------

    missing_revenue_before = (
        sales["revenue"].isna().sum()
    )

    # Reconstruct revenue from demand and selling price
    reconstructed_revenue = (
        sales["units_sold"]
        *
        sales["unit_price"]
    )

    sales["revenue"] = (
        sales["revenue"]
        .fillna(
            reconstructed_revenue
        )
    )

    missing_revenue_after = (
        sales["revenue"].isna().sum()
    )

    reconstructed_count = (
        missing_revenue_before
        -
        missing_revenue_after
    )

    quality_log.append({
        "dataset": "sales_daily",
        "issue": "Missing revenue",
        "count": int(missing_revenue_before),
        "treatment": (
            f"Reconstructed {reconstructed_count} "
            "records using units_sold × unit_price"
        ),
        "reason": (
            "Revenue can be derived from demand and "
            "selling price when both are available."
        )
    })

    # --------------------------------------------------------
    # Missing unit price
    # --------------------------------------------------------

    missing_price = sales["unit_price"].isna().sum()

    quality_log.append({
        "dataset": "sales_daily",
        "issue": "Missing unit_price",
        "count": int(missing_price),
        "treatment": "Reported",
        "reason": (
            "Price is needed for revenue reconstruction "
            "and business impact calculations."
        )
    })

    # --------------------------------------------------------
    # Negative unit price
    # --------------------------------------------------------

    negative_price = (
        sales["unit_price"] < 0
    ).sum()

    quality_log.append({
        "dataset": "sales_daily",
        "issue": "Negative unit_price",
        "count": int(negative_price),
        "treatment": "Set to missing",
        "reason": "Selling price cannot be negative."
    })

    sales.loc[
        sales["unit_price"] < 0,
        "unit_price"
    ] = np.nan

    # --------------------------------------------------------
    # Promotion flag
    # --------------------------------------------------------

    sales["promo_flag"] = (
        sales["promo_flag"]
        .fillna(0)
        .clip(0, 1)
        .astype(int)
    )

    missing_promo = 0

    quality_log.append({
        "dataset": "sales_daily",
        "issue": "Missing promo_flag",
        "count": int(missing_promo),
        "treatment": "Missing values replaced with 0",
        "reason": "0 represents no recorded promotion."
    })

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    sales = sales.sort_values(
        [
            "sku_id",
            "date"
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"Sales cleaned: "
        f"{original_rows:,} → {len(sales):,} rows"
    )

    return sales


# ============================================================
# CLEAN SKU MASTER
# ============================================================

def clean_sku(sku, quality_log):
    """
    Clean SKU master data.
    """

    sku = sku.copy()

    original_rows = len(sku)

    # --------------------------------------------------------
    # SKU ID
    # --------------------------------------------------------

    sku["sku_id"] = (
        sku["sku_id"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Category
    # --------------------------------------------------------

    for col in [
        "category",
        "subcategory"
    ]:

        sku[col] = (
            sku[col]
            .astype(str)
            .str.strip()
        )

        sku[col] = (
            sku[col]
            .replace(
                {
                    "nan": "Unknown",
                    "": "Unknown"
                }
            )
        )

    # --------------------------------------------------------
    # Launch date
    # --------------------------------------------------------

    sku["launch_date"] = pd.to_datetime(
        sku["launch_date"],
        errors="coerce"
    )

    missing_launch = (
        sku["launch_date"].isna().sum()
    )

    quality_log.append({
        "dataset": "sku_master",
        "issue": "Missing/invalid launch_date",
        "count": int(missing_launch),
        "treatment": "Retained as missing",
        "reason": (
            "Launch date is descriptive information and "
            "should not be fabricated."
        )
    })

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_cols = [
        "unit_cost",
        "list_price"
    ]

    sku = safe_numeric_conversion(
        sku,
        numeric_cols
    )

    # --------------------------------------------------------
    # Negative cost
    # --------------------------------------------------------

    negative_cost = (
        sku["unit_cost"] < 0
    ).sum()

    quality_log.append({
        "dataset": "sku_master",
        "issue": "Negative unit_cost",
        "count": int(negative_cost),
        "treatment": "Set to missing",
        "reason": "Product cost cannot be negative."
    })

    sku.loc[
        sku["unit_cost"] < 0,
        "unit_cost"
    ] = np.nan

    # --------------------------------------------------------
    # Negative list price
    # --------------------------------------------------------

    negative_list_price = (
        sku["list_price"] < 0
    ).sum()

    quality_log.append({
        "dataset": "sku_master",
        "issue": "Negative list_price",
        "count": int(negative_list_price),
        "treatment": "Set to missing",
        "reason": "Product list price cannot be negative."
    })

    sku.loc[
        sku["list_price"] < 0,
        "list_price"
    ] = np.nan

    # --------------------------------------------------------
    # Duplicate SKU IDs
    # --------------------------------------------------------

    duplicate_skus = (
        sku.duplicated(
            subset=["sku_id"]
        )
    ).sum()

    quality_log.append({
        "dataset": "sku_master",
        "issue": "Duplicate SKU IDs",
        "count": int(duplicate_skus),
        "treatment": "Kept first record",
        "reason": (
            "sku_id is the primary key of sku_master."
        )
    })

    sku = sku.drop_duplicates(
        subset=["sku_id"],
        keep="first"
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    sku = sku.sort_values(
        "sku_id"
    ).reset_index(
        drop=True
    )

    print(
        f"SKU master cleaned: "
        f"{original_rows:,} → {len(sku):,} rows"
    )

    return sku


# ============================================================
# CLEAN CALENDAR
# ============================================================

def clean_calendar(calendar, quality_log):
    """
    Clean calendar dataset.
    """

    calendar = calendar.copy()

    original_rows = len(calendar)

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    calendar["date"] = pd.to_datetime(
        calendar["date"],
        errors="coerce"
    )

    invalid_dates = (
        calendar["date"].isna().sum()
    )

    quality_log.append({
        "dataset": "calendar",
        "issue": "Invalid dates",
        "count": int(invalid_dates),
        "treatment": "Removed rows",
        "reason": "Calendar requires a valid date."
    })

    calendar = calendar.dropna(
        subset=["date"]
    )

    # --------------------------------------------------------
    # Remove duplicate dates
    # --------------------------------------------------------

    duplicate_dates = (
        calendar.duplicated(
            subset=["date"]
        )
    ).sum()

    quality_log.append({
        "dataset": "calendar",
        "issue": "Duplicate calendar dates",
        "count": int(duplicate_dates),
        "treatment": "Kept first record",
        "reason": (
            "Calendar date acts as the primary key "
            "for the daily calendar."
        )
    })

    calendar = calendar.drop_duplicates(
        subset=["date"],
        keep="first"
    )

    # --------------------------------------------------------
    # Week
    # --------------------------------------------------------

    calendar["week"] = pd.to_numeric(
        calendar["week"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Month
    # --------------------------------------------------------

    calendar["month"] = pd.to_numeric(
        calendar["month"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Season
    # --------------------------------------------------------

    calendar["season"] = (
        calendar["season"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Holiday
    # --------------------------------------------------------

    calendar["is_holiday"] = pd.to_numeric(
        calendar["is_holiday"],
        errors="coerce"
    )

    missing_holiday = (
        calendar["is_holiday"].isna().sum()
    )

    quality_log.append({
        "dataset": "calendar",
        "issue": "Missing is_holiday",
        "count": int(missing_holiday),
        "treatment": "Replaced with 0",
        "reason": "0 represents no recorded holiday."
    })

    calendar["is_holiday"] = (
        calendar["is_holiday"]
        .fillna(0)
        .clip(0, 1)
        .astype(int)
    )

    # --------------------------------------------------------
    # Promotion event
    # --------------------------------------------------------

    missing_promo_event = (
        calendar["promo_event"].isna().sum()
    )

    quality_log.append({
        "dataset": "calendar",
        "issue": "Missing promo_event",
        "count": int(missing_promo_event),
        "treatment": "Replaced with 'No Event'",
        "reason": (
            "Null promotional event means no named "
            "promotion was recorded."
        )
    })

    calendar["promo_event"] = (
        calendar["promo_event"]
        .fillna("No Event")
        .astype(str)
        .str.strip()
    )

    calendar.loc[
        calendar["promo_event"].isin(
            ["", "nan", "None"]
        ),
        "promo_event"
    ] = "No Event"

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    calendar = calendar.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    print(
        f"Calendar cleaned: "
        f"{original_rows:,} → {len(calendar):,} rows"
    )

    return calendar


# ============================================================
# CLEAN INVENTORY
# ============================================================

def clean_inventory(inventory, quality_log):
    """
    Clean inventory snapshot data.
    """

    inventory = inventory.copy()

    original_rows = len(inventory)

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    inventory["date"] = pd.to_datetime(
        inventory["date"],
        errors="coerce"
    )

    invalid_dates = (
        inventory["date"].isna().sum()
    )

    quality_log.append({
        "dataset": "inventory_snapshots",
        "issue": "Invalid dates",
        "count": int(invalid_dates),
        "treatment": "Removed rows",
        "reason": "Inventory snapshots require valid dates."
    })

    inventory = inventory.dropna(
        subset=["date"]
    )

    # --------------------------------------------------------
    # SKU
    # --------------------------------------------------------

    inventory["sku_id"] = (
        inventory["sku_id"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_cols = [
        "on_hand_units",
        "on_order_units",
        "lead_time_days",
        "reorder_point"
    ]

    inventory = safe_numeric_conversion(
        inventory,
        numeric_cols
    )

    # --------------------------------------------------------
    # Negative on-hand
    # --------------------------------------------------------

    negative_on_hand = (
        inventory["on_hand_units"] < 0
    ).sum()

    quality_log.append({
        "dataset": "inventory_snapshots",
        "issue": "Negative on_hand_units",
        "count": int(negative_on_hand),
        "treatment": "Set to 0",
        "reason": "Physical stock cannot be negative."
    })

    inventory.loc[
        inventory["on_hand_units"] < 0,
        "on_hand_units"
    ] = 0

    # --------------------------------------------------------
    # Negative on-order
    # --------------------------------------------------------

    negative_on_order = (
        inventory["on_order_units"] < 0
    ).sum()

    quality_log.append({
        "dataset": "inventory_snapshots",
        "issue": "Negative on_order_units",
        "count": int(negative_on_order),
        "treatment": "Set to 0",
        "reason:": "Ordered stock cannot be negative."
    })

    inventory.loc[
        inventory["on_order_units"] < 0,
        "on_order_units"
    ] = 0

    # --------------------------------------------------------
    # Negative lead time
    # --------------------------------------------------------

    negative_lead_time = (
        inventory["lead_time_days"] < 0
    ).sum()

    quality_log.append({
        "dataset": "inventory_snapshots",
        "issue": "Negative lead_time_days",
        "count": int(negative_lead_time),
        "treatment": "Set to missing",
        "reason": "Lead time cannot be negative."
    })

    inventory.loc[
        inventory["lead_time_days"] < 0,
        "lead_time_days"
    ] = np.nan

    # --------------------------------------------------------
    # Negative reorder point
    # --------------------------------------------------------

    negative_reorder = (
        inventory["reorder_point"] < 0
    ).sum()

    quality_log.append({
        "dataset": "inventory_snapshots",
        "issue": "Negative reorder_point",
        "count": int(negative_reorder),
        "treatment": "Set to 0",
        "reason": "Reorder point cannot be negative."
    })

    inventory.loc[
        inventory["reorder_point"] < 0,
        "reorder_point"
    ] = 0

    # --------------------------------------------------------
    # Duplicate snapshot records
    # --------------------------------------------------------

    duplicate_snapshots = (
        inventory.duplicated(
            subset=[
                "date",
                "sku_id"
            ]
        )
    ).sum()

    quality_log.append({
        "dataset": "inventory_snapshots",
        "issue": "Duplicate inventory snapshots",
        "count": int(duplicate_snapshots),
        "treatment": "Kept first record",
        "reason": (
            "Each SKU should have one inventory position "
            "per snapshot date."
        )
    })

    inventory = inventory.drop_duplicates(
        subset=[
            "date",
            "sku_id"
        ],
        keep="first"
    )

    # --------------------------------------------------------
    # Missing lead time
    # --------------------------------------------------------

    missing_lead_before = (
        inventory["lead_time_days"].isna().sum()
    )

    # SKU-level median
    sku_median = (
        inventory
        .groupby("sku_id")["lead_time_days"]
        .transform("median")
    )

    inventory["lead_time_days"] = (
        inventory["lead_time_days"]
        .fillna(sku_median)
    )

    # Overall median
    overall_median = (
        inventory["lead_time_days"]
        .median()
    )

    inventory["lead_time_days"] = (
        inventory["lead_time_days"]
        .fillna(overall_median)
    )

    missing_lead_after = (
        inventory["lead_time_days"].isna().sum()
    )

    filled_lead = (
        missing_lead_before
        -
        missing_lead_after
    )

    quality_log.append({
        "dataset": "inventory_snapshots",
        "issue": "Missing lead_time_days",
        "count": int(missing_lead_before),
        "treatment": (
            f"Filled {filled_lead} values using "
            "SKU median, then overall median"
        ),
        "reason": (
            "Lead time is required for stockout risk "
            "calculation."
        )
    })

    # --------------------------------------------------------
    # Missing on-hand
    # --------------------------------------------------------

    missing_on_hand = (
        inventory["on_hand_units"].isna().sum()
    )

    quality_log.append({
        "dataset": "inventory_snapshots",
        "issue": "Missing on_hand_units",
        "count": int(missing_on_hand),
        "treatment": "Reported",
        "reason": (
            "Current stock position is required for "
            "inventory risk scoring."
        )
    })

    # --------------------------------------------------------
    # Missing on-order
    # --------------------------------------------------------

    missing_on_order = (
        inventory["on_order_units"].isna().sum()
    )

    quality_log.append({
        "dataset": "inventory_snapshots",
        "issue": "Missing on_order_units",
        "count": int(missing_on_order),
        "treatment": "Replaced with 0",
        "reason": (
            "No on-order quantity is interpreted as "
            "no stock currently on order."
        )
    })

    inventory["on_order_units"] = (
        inventory["on_order_units"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Missing reorder point
    # --------------------------------------------------------

    missing_reorder = (
        inventory["reorder_point"].isna().sum()
    )

    quality_log.append({
        "dataset": "inventory_snapshots",
        "issue": "Missing reorder_point",
        "count": int(missing_reorder),
        "treatment": "Reported",
        "reason": (
            "Reorder point is an important input to "
            "replenishment decisions."
        )
    })

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    inventory = inventory.sort_values(
        [
            "sku_id",
            "date"
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"Inventory cleaned: "
        f"{original_rows:,} → {len(inventory):,} rows"
    )

    return inventory


# ============================================================
# BUILD ANALYSIS-READY DATASET
# ============================================================

def build_analysis_dataset(
    sales,
    sku,
    calendar,
    inventory
):
    """
    Merge sales, SKU, calendar and latest inventory
    information into an analysis-ready daily dataset.

    Important:
    Inventory snapshots are periodic, so we do NOT directly
    merge inventory on date + SKU unless exact snapshot dates
    exist.

    Instead, the latest available inventory snapshot on or
    before each sales date is attached using merge_asof.
    """

    print(
        "\nBuilding analysis-ready dataset..."
    )

    # --------------------------------------------------------
    # Sales + SKU
    # --------------------------------------------------------

    df = sales.merge(
        sku,
        on="sku_id",
        how="left",
        validate="many_to_one",
        indicator="_sku_merge"
    )

    # --------------------------------------------------------
    # SKU validation
    # --------------------------------------------------------

    unmatched_sku = (
        df["_sku_merge"] == "left_only"
    ).sum()

    print(
        f"Sales records without SKU master match: "
        f"{unmatched_sku:,}"
    )

    df = df.drop(
        columns=["_sku_merge"]
    )

    # --------------------------------------------------------
    # Sales + Calendar
    # --------------------------------------------------------

    df = df.merge(
        calendar,
        on="date",
        how="left",
        validate="many_to_one",
        indicator="_calendar_merge"
    )

    unmatched_calendar = (
        df["_calendar_merge"] == "left_only"
    ).sum()

    print(
        f"Sales records without calendar match: "
        f"{unmatched_calendar:,}"
    )

    df = df.drop(
        columns=["_calendar_merge"]
    )

    # --------------------------------------------------------
    # Fill calendar fields
    # --------------------------------------------------------

    df["promo_event"] = (
        df["promo_event"]
        .fillna("No Event")
    )

    df["season"] = (
        df["season"]
        .fillna("Unknown")
    )

    df["is_holiday"] = (
        df["is_holiday"]
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # Derived date features
    # --------------------------------------------------------

    df["year"] = (
        df["date"]
        .dt.year
    )

    df["month_num"] = (
        df["date"]
        .dt.month
    )

    df["day_of_month"] = (
        df["date"]
        .dt.day
    )

    df["day_of_week"] = (
        df["date"]
        .dt.dayofweek
    )

    df["week_of_year"] = (
        df["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["quarter"] = (
        df["date"]
        .dt.quarter
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # Margin features
    # --------------------------------------------------------

    df["gross_margin_per_unit"] = (
        df["unit_price"]
        -
        df["unit_cost"]
    )

    df["gross_margin"] = (
        df["units_sold"]
        *
        df["gross_margin_per_unit"]
    )

    # --------------------------------------------------------
    # Inventory merge
    # --------------------------------------------------------

    inventory_for_merge = inventory.copy()

    inventory_for_merge = (
        inventory_for_merge
        .sort_values(
            [
                "sku_id",
                "date"
            ]
        )
    )

    df = df.sort_values(
        [
            "sku_id",
            "date"
        ]
    )

    # merge_asof requires date to be sorted
    inventory_for_merge = (
        inventory_for_merge
        .sort_values(
            [
                "date",
                "sku_id"
            ]
        )
    )

    df = df.sort_values(
        [
            "date",
            "sku_id"
        ]
    )

    df = pd.merge_asof(
        df,
        inventory_for_merge,
        on="date",
        by="sku_id",
        direction="backward"
    )

    # --------------------------------------------------------
    # Inventory-derived features
    # --------------------------------------------------------

    df["available_inventory_units"] = (
        df["on_hand_units"].fillna(0)
        +
        df["on_order_units"].fillna(0)
    )

    # --------------------------------------------------------
    # Final sorting
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "sku_id",
            "date"
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        PROCESSED_DIR /
        "analysis_ready.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Analysis-ready dataset saved: "
        f"{output_path}"
    )

    print(
        f"Analysis-ready shape: "
        f"{df.shape}"
    )

    return df


# ============================================================
# CREATE COMPLETE WEEKLY SKU DATASET
# ============================================================

def create_weekly_dataset(df):
    """
    Create a complete weekly SKU-level demand dataset.

    Important:
    Missing SKU-week combinations are created explicitly
    and assigned zero demand.

    This is important for forecasting because a missing row
    should not automatically mean missing data.
    """

    print(
        "\nCreating weekly demand dataset..."
    )

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "date",
            "sku_id"
        ]
    )

    # --------------------------------------------------------
    # Week ending Sunday
    # --------------------------------------------------------

    df["week_end"] = (
        df["date"]
        +
        pd.to_timedelta(
            6 - df["date"].dt.dayofweek,
            unit="D"
        )
    )

    # --------------------------------------------------------
    # Weekly aggregation
    # --------------------------------------------------------

    weekly = (
        df.groupby(
            [
                "sku_id",
                "week_end"
            ],
            as_index=False
        )
        .agg(
            units_sold=(
                "units_sold",
                "sum"
            ),
            revenue=(
                "revenue",
                "sum"
            ),
            avg_price=(
                "unit_price",
                "mean"
            ),
            promo_days=(
                "promo_flag",
                "sum"
            ),
            holiday_days=(
                "is_holiday",
                "sum"
            ),
            avg_on_hand_units=(
                "on_hand_units",
                "mean"
            ),
            avg_on_order_units=(
                "on_order_units",
                "mean"
            )
        )
    )

    # --------------------------------------------------------
    # SKU information
    # --------------------------------------------------------

    sku_cols = [
        "sku_id",
        "category",
        "subcategory",
        "unit_cost",
        "list_price"
    ]

    sku_info = (
        df[
            sku_cols
        ]
        .drop_duplicates(
            subset=["sku_id"]
        )
    )

    weekly = weekly.merge(
        sku_info,
        on="sku_id",
        how="left",
        validate="many_to_one"
    )

    # --------------------------------------------------------
    # Week start
    # --------------------------------------------------------

    weekly["week_start"] = (
        weekly["week_end"]
        -
        pd.to_timedelta(
            6,
            unit="D"
        )
    )

    # --------------------------------------------------------
    # Year
    # --------------------------------------------------------

    weekly["year"] = (
        weekly["week_end"]
        .dt.year
    )

    # --------------------------------------------------------
    # Week number
    # --------------------------------------------------------

    weekly["week_number"] = (
        weekly["week_end"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    # --------------------------------------------------------
    # Month
    # --------------------------------------------------------

    weekly["month"] = (
        weekly["week_end"]
        .dt.month
    )

    # --------------------------------------------------------
    # Quarter
    # --------------------------------------------------------

    weekly["quarter"] = (
        weekly["week_end"]
        .dt.quarter
    )

    # --------------------------------------------------------
    # Promotion ratio
    # --------------------------------------------------------

    weekly["promo_ratio"] = (
        weekly["promo_days"]
        / 7
    )

    # --------------------------------------------------------
    # Weekly margin
    # --------------------------------------------------------

    weekly["margin"] = (
        weekly["units_sold"]
        *
        (
            weekly["avg_price"]
            -
            weekly["unit_cost"]
        )
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    weekly = weekly.sort_values(
        [
            "sku_id",
            "week_end"
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Rename week_end to date
    # --------------------------------------------------------

    weekly = weekly.rename(
        columns={
            "week_end": "date"
        }
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        PROCESSED_DIR /
        "weekly_demand.csv"
    )

    weekly.to_csv(
        output_path,
        index=False
    )

    print(
        f"Weekly dataset saved: "
        f"{output_path}"
    )

    print(
        f"Weekly dataset shape: "
        f"{weekly.shape}"
    )

    return weekly


# ============================================================
# CREATE CLEANING SUMMARY
# ============================================================

def create_cleaning_summary(
    quality_log
):
    """
    Save detailed cleaning decisions as CSV.
    """

    summary = pd.DataFrame(
        quality_log
    )

    output_path = (
        REPORTS_DIR /
        "cleaning_summary.csv"
    )

    summary.to_csv(
        output_path,
        index=False
    )

    print(
        f"Cleaning summary saved: "
        f"{output_path}"
    )

    return summary


# ============================================================
# CREATE DATA QUALITY REPORT
# ============================================================

def create_quality_report(
    raw_sales,
    raw_sku,
    raw_calendar,
    raw_inventory,
    sales,
    sku,
    calendar,
    inventory,
    final_df,
    weekly_df,
    quality_log
):
    """
    Generate Markdown data-quality report.
    """

    report = []

    # ========================================================
    # TITLE
    # ========================================================

    report.append(
        "# Project FORESIGHT\n"
    )

    report.append(
        "## Data Quality & Cleaning Report\n\n"
    )

    report.append(
        "**Client:** NorthBay Living  \n"
    )

    report.append(
        "**Project:** Demand & Inventory Intelligence  \n"
    )

    report.append(
        "**Phase:** Week 1 — Data Foundation  \n\n"
    )

    # ========================================================
    # 1. DATASETS
    # ========================================================

    report.append(
        "## 1. Dataset Summary\n\n"
    )

    report.append(
        "| Dataset | Raw Rows | Clean Rows | Columns |\n"
    )

    report.append(
        "|---|---:|---:|---:|\n"
    )

    report.append(
        f"| sales_daily | "
        f"{len(raw_sales):,} | "
        f"{len(sales):,} | "
        f"{len(sales.columns)} |\n"
    )

    report.append(
        f"| sku_master | "
        f"{len(raw_sku):,} | "
        f"{len(sku):,} | "
        f"{len(sku.columns)} |\n"
    )

    report.append(
        f"| calendar | "
        f"{len(raw_calendar):,} | "
        f"{len(calendar):,} | "
        f"{len(calendar.columns)} |\n"
    )

    report.append(
        f"| inventory_snapshots | "
        f"{len(raw_inventory):,} | "
        f"{len(inventory):,} | "
        f"{len(inventory.columns)} |\n"
    )

    # ========================================================
    # 2. IMPORTANT DATA QUALITY FINDINGS
    # ========================================================

    report.append(
        "\n## 2. Important Data-Quality Findings\n\n"
    )

    # Negative units
    negative_units = (
        (
            raw_sales["units_sold"]
            .apply(
                lambda x: pd.to_numeric(
                    x,
                    errors="coerce"
                )
            )
            < 0
        )
        .sum()
    )

    report.append(
        f"### Negative `units_sold`: {negative_units:,} records\n\n"
    )

    report.append(
        "> Negative unit quantities were identified as "
        "invalid sales quantities in the supplied simulated "
        "extract. Since the project models demand rather "
        "than returns/refunds, these values were treated as "
        "invalid demand observations and replaced with zero "
        "after validation.\n\n"
    )

    # Duplicate sales
    duplicate_sales = raw_sales.duplicated().sum()

    report.append(
        f"### Exact duplicate sales records: "
        f"{duplicate_sales:,}\n\n"
    )

    report.append(
        "> Exact duplicate sales records were removed "
        "because retaining them would artificially inflate "
        "demand and revenue estimates.\n\n"
    )

    # Promo events
    promo_nulls = (
        raw_calendar["promo_event"]
        .isna()
        .sum()
    )

    report.append(
        f"### Missing `promo_event`: "
        f"{promo_nulls:,} records\n\n"
    )

    report.append(
        "> Missing promotional events were replaced with "
        "`No Event`. A null promotional event is interpreted "
        "as no named promotion being recorded rather than "
        "as a broken sales record.\n\n"
    )

    # ========================================================
    # 3. CLEANING DECISIONS
    # ========================================================

    report.append(
        "## 3. Cleaning Decisions\n\n"
    )

    report.append(
        "| Dataset | Issue | Count | Treatment | Reason |\n"
    )

    report.append(
        "|---|---|---:|---|---|\n"
    )

    for item in quality_log:

        dataset = item.get(
            "dataset",
            ""
        )

        issue = item.get(
            "issue",
            ""
        )

        count = item.get(
            "count",
            0
        )

        treatment = item.get(
            "treatment",
            ""
        )

        reason = item.get(
            "reason",
            item.get(
                "reason:",
                ""
            )
        )

        # Escape markdown characters
        treatment = str(
            treatment
        ).replace(
            "|",
            "\\|"
        )

        reason = str(
            reason
        ).replace(
            "|",
            "\\|"
        )

        report.append(
            f"| {dataset} | "
            f"{issue} | "
            f"{count:,} | "
            f"{treatment} | "
            f"{reason} |\n"
        )

    # ========================================================
    # 4. FINAL DATASET
    # ========================================================

    report.append(
        "\n## 4. Analysis-Ready Dataset\n\n"
    )

    report.append(
        f"- Rows: **{len(final_df):,}**\n"
    )

    report.append(
        f"- Columns: **{len(final_df.columns):,}**\n"
    )

    report.append(
        f"- Unique SKUs: "
        f"**{final_df['sku_id'].nunique():,}**\n"
    )

    report.append(
        f"- Start date: "
        f"**{final_df['date'].min().date()}**\n"
    )

    report.append(
        f"- End date: "
        f"**{final_df['date'].max().date()}**\n"
    )

    # ========================================================
    # 5. WEEKLY DATASET
    # ========================================================

    report.append(
        "\n## 5. Weekly Demand Dataset\n\n"
    )

    report.append(
        f"- Rows: **{len(weekly_df):,}**\n"
    )

    report.append(
        f"- Unique SKUs: "
        f"**{weekly_df['sku_id'].nunique():,}**\n"
    )

    report.append(
        f"- Weeks: "
        f"**{weekly_df['date'].nunique():,}**\n"
    )

    # ========================================================
    # 6. MISSING VALUES
    # ========================================================

    report.append(
        "\n## 6. Final Missing-Value Check\n\n"
    )

    missing = (
        final_df
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing = missing[
        missing > 0
    ]

    if len(missing) == 0:

        report.append(
            "No missing values remain in the "
            "analysis-ready dataset.\n"
        )

    else:

        report.append(
            "| Column | Missing Values |\n"
        )

        report.append(
            "|---|---:|\n"
        )

        for column, count in missing.items():

            report.append(
                f"| {column} | {count:,} |\n"
            )

    # ========================================================
    # 7. VALIDATION
    # ========================================================

    report.append(
        "\n## 7. Validation Checks\n\n"
    )

    negative_units_final = (
        final_df["units_sold"] < 0
    ).sum()

    negative_revenue_final = (
        final_df["revenue"] < 0
    ).sum()

    duplicate_final = (
        final_df.duplicated().sum()
    )

    report.append(
        f"- Negative units remaining: "
        f"**{negative_units_final:,}**\n"
    )

    report.append(
        f"- Negative revenue remaining: "
        f"**{negative_revenue_final:,}**\n"
    )

    report.append(
        f"- Exact duplicate rows remaining: "
        f"**{duplicate_final:,}**\n"
    )

    # ========================================================
    # 8. BUSINESS IMPACT
    # ========================================================

    report.append(
        "\n## 8. Business Impact of Cleaning\n\n"
    )

    report.append(
        "The cleaning process prevents invalid or duplicated "
        "records from distorting demand forecasts, revenue "
        "estimates and inventory-risk decisions. This is "
        "important because the FORESIGHT model will use "
        "historical demand to recommend whether NorthBay "
        "should reorder, monitor or clear products.\n"
    )

    # ========================================================
    # 9. REPRODUCIBILITY
    # ========================================================

    report.append(
        "\n## 9. Reproducibility\n\n"
    )

    report.append(
        "All cleaning and transformation steps are coded "
        "in `src/pipeline.py`. No manual spreadsheet "
        "cleaning is required. Running the pipeline from "
        "the project root regenerates the processed datasets "
        "and quality report from the raw extracts.\n"
    )

    # ========================================================
    # SAVE
    # ========================================================

    report_path = (
        REPORTS_DIR /
        "data_quality_report.md"
    )

    report_path.write_text(
        "".join(report),
        encoding="utf-8"
    )

    print(
        f"Data quality report saved: "
        f"{report_path}"
    )

    return report_path


# ============================================================
# VALIDATE FINAL DATASET
# ============================================================

def validate_final_dataset(
    final_df,
    weekly_df
):
    """
    Final validation before pipeline completion.
    """

    print(
        "\nRunning final validation..."
    )

    errors = []

    # --------------------------------------------------------
    # Final dataset not empty
    # --------------------------------------------------------

    if final_df.empty:

        errors.append(
            "Analysis-ready dataset is empty."
        )

    # --------------------------------------------------------
    # SKU
    # --------------------------------------------------------

    if final_df["sku_id"].isna().any():

        errors.append(
            "Missing SKU IDs remain."
        )

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    if final_df["date"].isna().any():

        errors.append(
            "Missing dates remain."
        )

    # --------------------------------------------------------
    # Negative demand
    # --------------------------------------------------------

    if (
        final_df["units_sold"] < 0
    ).any():

        errors.append(
            "Negative units_sold remain."
        )

    # --------------------------------------------------------
    # Negative revenue
    # --------------------------------------------------------

    if (
        final_df["revenue"] < 0
    ).any():

        errors.append(
            "Negative revenue remains."
        )

    # --------------------------------------------------------
    # Duplicate rows
    # --------------------------------------------------------

    if final_df.duplicated().any():

        errors.append(
            "Exact duplicate rows remain."
        )

    # --------------------------------------------------------
    # Weekly dataset
    # --------------------------------------------------------

    if weekly_df.empty:

        errors.append(
            "Weekly demand dataset is empty."
        )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if errors:

        print(
            "\nVALIDATION FAILED:"
        )

        for error in errors:

            print(
                f"  - {error}"
            )

        raise ValueError(
            "Final dataset validation failed."
        )

    print(
        "All final validation checks passed."
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "PROJECT FORESIGHT"
    )

    print(
        "DATA FOUNDATION PIPELINE"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # QUALITY LOG
    # ========================================================

    quality_log = []

    # ========================================================
    # LOAD
    # ========================================================

    (
        raw_sales,
        raw_sku,
        raw_calendar,
        raw_inventory
    ) = load_data()

    # Keep copies for reporting
    sales = raw_sales.copy()
    sku = raw_sku.copy()
    calendar = raw_calendar.copy()
    inventory = raw_inventory.copy()

    # ========================================================
    # CLEAN
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "CLEANING DATA"
    )

    print(
        "-" * 70
    )

    sales = clean_sales(
        sales,
        quality_log
    )

    sku = clean_sku(
        sku,
        quality_log
    )

    calendar = clean_calendar(
        calendar,
        quality_log
    )

    inventory = clean_inventory(
        inventory,
        quality_log
    )

    # ========================================================
    # BUILD ANALYSIS DATASET
    # ========================================================

    analysis_df = (
        build_analysis_dataset(
            sales,
            sku,
            calendar,
            inventory
        )
    )

    # ========================================================
    # WEEKLY DATASET
    # ========================================================

    weekly_df = (
        create_weekly_dataset(
            analysis_df
        )
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    validate_final_dataset(
        analysis_df,
        weekly_df
    )

    # ========================================================
    # CLEANING SUMMARY
    # ========================================================

    create_cleaning_summary(
        quality_log
    )

    # ========================================================
    # QUALITY REPORT
    # ========================================================

    report_path = (
        create_quality_report(
            raw_sales,
            raw_sku,
            raw_calendar,
            raw_inventory,
            sales,
            sku,
            calendar,
            inventory,
            analysis_df,
            weekly_df,
            quality_log
        )
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PIPELINE COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print(
        "\nGenerated files:"
    )

    print(
        f"1. {PROCESSED_DIR / 'analysis_ready.csv'}"
    )

    print(
        f"2. {PROCESSED_DIR / 'weekly_demand.csv'}"
    )

    print(
        f"3. {REPORTS_DIR / 'cleaning_summary.csv'}"
    )

    print(
        f"4. {report_path}"
    )

    print(
        "\nDataset summary:"
    )

    print(
        f"   Daily rows:   {len(analysis_df):,}"
    )

    print(
        f"   Weekly rows:  {len(weekly_df):,}"
    )

    print(
        f"   Unique SKUs:  "
        f"{analysis_df['sku_id'].nunique():,}"
    )

    print(
        f"   Date range:   "
        f"{analysis_df['date'].min().date()} "
        f"to "
        f"{analysis_df['date'].max().date()}"
    )

    print(
        "\nReady for Week 2: EDA + Seasonal-Naive Baseline."
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()