from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT FORESIGHT
# Demand & Inventory Intelligence
# Inventory Risk Engine
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORT_DIR = BASE_DIR / "reports"

FORECAST_FILE = PROCESSED_DIR / "forecast_output.csv"
INVENTORY_FILE = DATA_DIR / "raw" / "inventory_snapshots.csv"

RISK_OUTPUT_PATH = PROCESSED_DIR / "inventory_risk.csv"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

FORECAST_HORIZON = 8

# Risk thresholds
STOCKOUT_COVER_WEEKS = 2
LOW_COVER_WEEKS = 4
HIGH_COVER_WEEKS = 12

# Safety stock assumptions
SAFETY_STOCK_WEEKS = 2

# Risk score weights
STOCKOUT_WEIGHT = 0.60
OVERSTOCK_WEIGHT = 0.40


# ============================================================
# 3. LOAD FORECAST DATA
# ============================================================

def load_forecast():
    """
    Load the 8-week SKU-level demand forecast.
    """

    print("\nLoading forecast data...")

    if not FORECAST_FILE.exists():
        raise FileNotFoundError(
            f"Forecast file not found:\n{FORECAST_FILE}"
        )

    forecast = pd.read_csv(
        FORECAST_FILE,
        parse_dates=["forecast_date"]
    )

    required_columns = [
        "sku_id",
        "forecast_date",
        "forecast_units"
    ]

    missing = [
        col
        for col in required_columns
        if col not in forecast.columns
    ]

    if missing:
        raise ValueError(
            f"Forecast file is missing columns: {missing}"
        )

    forecast = forecast.sort_values(
        ["sku_id", "forecast_date"]
    ).reset_index(drop=True)

    forecast["forecast_units"] = (
        pd.to_numeric(
            forecast["forecast_units"],
            errors="coerce"
        )
        .fillna(0)
        .clip(lower=0)
    )

    print(
        f"Forecast rows: {len(forecast):,}"
    )

    print(
        f"Forecast SKUs: "
        f"{forecast['sku_id'].nunique():,}"
    )

    return forecast


# ============================================================
# 4. LOAD INVENTORY DATA
# ============================================================

def load_inventory():
    """
    Load the latest inventory snapshot for every SKU.
    """

    print("\nLoading inventory snapshots...")

    if not INVENTORY_FILE.exists():
        raise FileNotFoundError(
            f"Inventory file not found:\n{INVENTORY_FILE}"
        )

    inventory = pd.read_csv(
        INVENTORY_FILE
    )

    print(
        f"Inventory rows loaded: "
        f"{len(inventory):,}"
    )

    print(
        "Inventory columns:"
    )

    print(
        inventory.columns.tolist()
    )

    return inventory


# ============================================================
# 5. STANDARDIZE INVENTORY COLUMNS
# ============================================================

def standardize_inventory_columns(inventory):
    """
    Standardize common inventory column names.
    """

    inventory = inventory.copy()

    rename_map = {}

    for col in inventory.columns:

        normalized = (
            str(col)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        if normalized in [
            "sku",
            "sku_id",
            "product_id"
        ]:

            rename_map[col] = "sku_id"

        elif normalized in [
            "inventory",
            "inventory_units",
            "stock",
            "stock_units",
            "on_hand",
            "on_hand_units",
            "current_inventory",
            "current_stock"
        ]:

            rename_map[col] = "inventory_units"

        elif normalized in [
            "date",
            "snapshot_date",
            "inventory_date"
        ]:

            rename_map[col] = "snapshot_date"

        elif normalized in [
            "unit_cost",
            "cost"
        ]:

            rename_map[col] = "unit_cost"

        elif normalized in [
            "category"
        ]:

            rename_map[col] = "category"

        elif normalized in [
            "subcategory",
            "sub_category"
        ]:

            rename_map[col] = "subcategory"

    inventory = inventory.rename(
        columns=rename_map
    )

    return inventory


# ============================================================
# 6. VALIDATE INVENTORY DATA
# ============================================================

def validate_inventory(inventory):
    """
    Validate the inventory dataset.
    """

    required = [
        "sku_id",
        "inventory_units"
    ]

    missing = [
        col
        for col in required
        if col not in inventory.columns
    ]

    if missing:

        raise ValueError(
            "\nInventory file is missing required "
            f"columns: {missing}\n\n"
            "Expected at minimum:\n"
            "- sku_id\n"
            "- inventory_units"
        )

    inventory = inventory.copy()

    inventory["inventory_units"] = (
        pd.to_numeric(
            inventory["inventory_units"],
            errors="coerce"
        )
        .fillna(0)
        .clip(lower=0)
    )

    inventory["sku_id"] = (
        inventory["sku_id"]
        .astype(str)
        .str.strip()
    )

    return inventory


# ============================================================
# 7. GET LATEST INVENTORY SNAPSHOT
# ============================================================

def get_latest_inventory(inventory):
    """
    If multiple inventory snapshots exist for each SKU,
    keep only the latest snapshot.
    """

    inventory = inventory.copy()

    if "snapshot_date" in inventory.columns:

        inventory["snapshot_date"] = pd.to_datetime(
            inventory["snapshot_date"],
            errors="coerce"
        )

        inventory = (
            inventory
            .sort_values(
                ["sku_id", "snapshot_date"]
            )
            .groupby(
                "sku_id",
                as_index=False
            )
            .tail(1)
        )

    else:

        inventory = (
            inventory
            .drop_duplicates(
                subset=["sku_id"],
                keep="last"
            )
        )

    inventory = inventory.reset_index(
        drop=True
    )

    print(
        f"Latest inventory SKUs: "
        f"{inventory['sku_id'].nunique():,}"
    )

    return inventory


# ============================================================
# 8. AGGREGATE FORECAST BY SKU
# ============================================================

def aggregate_forecast(forecast):
    """
    Calculate total 8-week forecast demand
    for every SKU.
    """

    print(
        "\nCalculating 8-week SKU demand..."
    )

    sku_forecast = (
        forecast
        .groupby("sku_id", as_index=False)
        .agg(
            forecast_8w_units=(
                "forecast_units",
                "sum"
            ),
            average_weekly_demand=(
                "forecast_units",
                "mean"
            ),
            peak_weekly_demand=(
                "forecast_units",
                "max"
            ),
            minimum_weekly_demand=(
                "forecast_units",
                "min"
            )
        )
    )

    return sku_forecast


# ============================================================
# 9. CALCULATE DEMAND VARIABILITY
# ============================================================

def calculate_demand_variability(forecast):
    """
    Calculate forecast variability for every SKU.
    """

    variability = (
        forecast
        .groupby("sku_id")["forecast_units"]
        .agg(
            forecast_std="std"
        )
        .reset_index()
    )

    variability["forecast_std"] = (
        variability["forecast_std"]
        .fillna(0)
    )

    return variability


# ============================================================
# 10. CALCULATE DAYS OF COVER
# ============================================================

def calculate_days_of_cover(df):
    """
    Calculate how many days current inventory
    can cover based on forecast demand.
    """

    df = df.copy()

    daily_demand = (
        df["average_weekly_demand"] / 7
    )

    df["days_of_cover"] = np.where(
        daily_demand > 0,

        df["inventory_units"]
        / daily_demand,

        np.inf
    )

    return df


# ============================================================
# 11. CALCULATE WEEKS OF COVER
# ============================================================

def calculate_weeks_of_cover(df):
    """
    Calculate inventory coverage in weeks.
    """

    df = df.copy()

    df["weeks_of_cover"] = np.where(
        df["average_weekly_demand"] > 0,

        (
            df["inventory_units"]
            / df["average_weekly_demand"]
        ),

        np.inf
    )

    return df


# ============================================================
# 12. CALCULATE STOCKOUT RISK
# ============================================================

def calculate_stockout_risk(df):
    """
    Calculate stockout risk based on inventory coverage.
    """

    df = df.copy()

    df["stockout_gap_units"] = (
        df["forecast_8w_units"]
        - df["inventory_units"]
    )

    df["stockout_gap_units"] = (
        df["stockout_gap_units"]
        .clip(lower=0)
    )

    df["stockout_probability_score"] = np.where(

        df["average_weekly_demand"] <= 0,

        0,

        np.clip(
            (
                df["forecast_8w_units"]
                - df["inventory_units"]
            )
            /
            df["forecast_8w_units"].replace(
                0,
                np.nan
            ),

            0,
            1
        )
    )

    df["stockout_probability_score"] = (
        df["stockout_probability_score"]
        .fillna(0)
    )

    def classify_stockout(row):

        weeks = row["weeks_of_cover"]
        gap = row["stockout_gap_units"]

        if gap > 0 and weeks <= 2:

            return "VERY HIGH"

        elif gap > 0 and weeks <= 4:

            return "HIGH"

        elif gap > 0 or weeks <= 8:

            return "MEDIUM"

        else:

            return "LOW"

    df["stockout_risk"] = (
        df.apply(
            classify_stockout,
            axis=1
        )
    )

    return df


# ============================================================
# 13. CALCULATE OVERSTOCK RISK
# ============================================================

def calculate_overstock_risk(df):
    """
    Calculate overstock risk.
    """

    df = df.copy()

    df["excess_inventory_units"] = np.where(

        df["inventory_units"]
        >
        df["forecast_8w_units"],

        (
            df["inventory_units"]
            - df["forecast_8w_units"]
        ),

        0
    )

    def classify_overstock(weeks):

        if np.isinf(weeks):

            return "LOW"

        if weeks > 12:

            return "VERY HIGH"

        elif weeks > 8:

            return "HIGH"

        elif weeks > 4:

            return "MEDIUM"

        else:

            return "LOW"

    df["overstock_risk"] = (
        df["weeks_of_cover"]
        .apply(classify_overstock)
    )

    return df


# ============================================================
# 14. CALCULATE RISK SCORES
# ============================================================

def calculate_risk_scores(df):
    """
    Convert stockout and overstock conditions
    into numerical risk scores.
    """

    df = df.copy()

    stockout_score_map = {
        "LOW": 10,
        "MEDIUM": 40,
        "HIGH": 70,
        "VERY HIGH": 100
    }

    overstock_score_map = {
        "LOW": 10,
        "MEDIUM": 40,
        "HIGH": 70,
        "VERY HIGH": 100
    }

    df["stockout_score"] = (
        df["stockout_risk"]
        .map(stockout_score_map)
        .fillna(0)
    )

    df["overstock_score"] = (
        df["overstock_risk"]
        .map(overstock_score_map)
        .fillna(0)
    )

    df["risk_score"] = (
        (
            df["stockout_score"]
            * STOCKOUT_WEIGHT
        )
        +
        (
            df["overstock_score"]
            * OVERSTOCK_WEIGHT
        )
    )

    df["risk_score"] = (
        df["risk_score"]
        .clip(0, 100)
    )

    return df


# ============================================================
# 15. CLASSIFY OVERALL RISK
# ============================================================

def classify_overall_risk(df):

    df = df.copy()

    def classify(score):

        if score >= 75:

            return "CRITICAL"

        elif score >= 50:

            return "HIGH"

        elif score >= 25:

            return "MEDIUM"

        else:

            return "LOW"

    df["risk_level"] = (
        df["risk_score"]
        .apply(classify)
    )

    return df


# ============================================================
# 16. RECOMMEND BUSINESS ACTION
# ============================================================

def recommend_action(df):

    df = df.copy()

    def action(row):

        stockout = row["stockout_risk"]
        overstock = row["overstock_risk"]
        weeks = row["weeks_of_cover"]

        if stockout == "VERY HIGH":

            return (
                "URGENT REPLENISHMENT - "
                "prioritize purchase/order immediately"
            )

        if stockout == "HIGH":

            return (
                "REPLENISH SOON - "
                "increase replenishment priority"
            )

        if stockout == "MEDIUM":

            return (
                "MONITOR STOCK - "
                "consider planned replenishment"
            )

        if overstock == "VERY HIGH":

            return (
                "SEVERE OVERSTOCK - "
                "reduce purchasing and consider clearance"
            )

        if overstock == "HIGH":

            return (
                "REDUCE INVENTORY - "
                "slow purchasing and consider promotion"
            )

        if overstock == "MEDIUM":

            return (
                "MONITOR OVERSTOCK - "
                "review future purchases"
            )

        if weeks <= 4:

            return (
                "MONITOR - "
                "inventory coverage is relatively low"
            )

        return (
            "NORMAL - "
            "continue regular inventory monitoring"
        )

    df["recommended_action"] = (
        df.apply(
            action,
            axis=1
        )
    )

    return df


# ============================================================
# 17. CALCULATE RECOMMENDED REPLENISHMENT
# ============================================================

def calculate_replenishment(df):

    df = df.copy()

    df["safety_stock_units"] = (
        df["average_weekly_demand"]
        * SAFETY_STOCK_WEEKS
    )

    df["target_inventory_units"] = (
        df["forecast_8w_units"]
        +
        df["safety_stock_units"]
    )

    df["recommended_replenishment_units"] = (
        df["target_inventory_units"]
        -
        df["inventory_units"]
    ).clip(lower=0)

    return df


# ============================================================
# 18. CALCULATE INVENTORY VALUE
# ============================================================

def calculate_inventory_value(df):

    df = df.copy()

    if "unit_cost" in df.columns:

        df["unit_cost"] = (
            pd.to_numeric(
                df["unit_cost"],
                errors="coerce"
            )
            .fillna(0)
        )

        df["inventory_value"] = (
            df["inventory_units"]
            * df["unit_cost"]
        )

        df["excess_inventory_value"] = (
            df["excess_inventory_units"]
            * df["unit_cost"]
        )

    else:

        df["inventory_value"] = 0

        df["excess_inventory_value"] = 0

    return df


# ============================================================
# 19. BUILD RISK DATASET
# ============================================================

def build_risk_dataset(
    forecast,
    inventory
):

    print(
        "\nBuilding inventory risk dataset..."
    )

    sku_forecast = aggregate_forecast(
        forecast
    )

    variability = calculate_demand_variability(
        forecast
    )

    result = sku_forecast.merge(
        variability,
        on="sku_id",
        how="left"
    )

    result = result.merge(
        inventory,
        on="sku_id",
        how="left",
        suffixes=(
            "",
            "_inventory"
        )
    )

    missing_inventory = (
        result["inventory_units"]
        .isna()
        .sum()
    )

    if missing_inventory > 0:

        print(
            f"WARNING: "
            f"{missing_inventory} SKUs have no inventory record."
        )

        result["inventory_units"] = (
            result["inventory_units"]
            .fillna(0)
        )

    result = calculate_days_of_cover(
        result
    )

    result = calculate_weeks_of_cover(
        result
    )

    result = calculate_stockout_risk(
        result
    )

    result = calculate_overstock_risk(
        result
    )

    result = calculate_risk_scores(
        result
    )

    result = classify_overall_risk(
        result
    )

    result = calculate_replenishment(
        result
    )

    result = calculate_inventory_value(
        result
    )

    result = recommend_action(
        result
    )

    return result


# ============================================================
# 20. FORMAT FINAL OUTPUT
# ============================================================

def format_output(df):

    df = df.copy()

    preferred_columns = [

        "sku_id",

        "category",
        "subcategory",

        "inventory_units",

        "forecast_8w_units",
        "average_weekly_demand",
        "peak_weekly_demand",

        "forecast_std",

        "days_of_cover",
        "weeks_of_cover",

        "stockout_gap_units",
        "excess_inventory_units",

        "stockout_probability_score",

        "stockout_score",
        "overstock_score",
        "risk_score",

        "stockout_risk",
        "overstock_risk",
        "risk_level",

        "safety_stock_units",
        "target_inventory_units",
        "recommended_replenishment_units",

        "inventory_value",
        "excess_inventory_value",

        "recommended_action"
    ]

    available = [
        col
        for col in preferred_columns
        if col in df.columns
    ]

    remaining = [
        col
        for col in df.columns
        if col not in available
    ]

    df = df[
        available + remaining
    ]

    return df


# ============================================================
# 21. VALIDATE RISK OUTPUT
# ============================================================

def validate_output(df):

    print(
        "\n" + "=" * 60
    )

    print(
        "RISK OUTPUT VALIDATION"
    )

    print(
        "=" * 60
    )

    print(
        f"Risk rows: {len(df):,}"
    )

    print(
        f"Risk SKUs: "
        f"{df['sku_id'].nunique():,}"
    )

    duplicates = (
        df["sku_id"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate SKU rows: {duplicates}"
    )

    if duplicates > 0:

        raise ValueError(
            "Duplicate SKU records found."
        )

    negative_inventory = (
        df["inventory_units"] < 0
    ).sum()

    print(
        f"Negative inventory values: "
        f"{negative_inventory}"
    )

    if negative_inventory > 0:

        raise ValueError(
            "Negative inventory values found."
        )

    negative_forecast = (
        df["forecast_8w_units"] < 0
    ).sum()

    print(
        f"Negative forecast values: "
        f"{negative_forecast}"
    )

    if negative_forecast > 0:

        raise ValueError(
            "Negative forecast values found."
        )

    invalid_scores = (
        (df["risk_score"] < 0)
        |
        (df["risk_score"] > 100)
    ).sum()

    print(
        f"Invalid risk scores: "
        f"{invalid_scores}"
    )

    if invalid_scores > 0:

        raise ValueError(
            "Risk score outside 0-100 range."
        )

    missing_levels = (
        df["risk_level"]
        .isna()
        .sum()
    )

    print(
        f"Missing risk levels: "
        f"{missing_levels}"
    )

    if missing_levels > 0:

        raise ValueError(
            "Missing risk levels found."
        )

    print(
        "\nRisk output validation PASSED."
    )


# ============================================================
# 22. PRINT RISK SUMMARY
# ============================================================

def print_summary(df):

    print(
        "\n" + "=" * 60
    )

    print(
        "INVENTORY RISK SUMMARY"
    )

    print(
        "=" * 60
    )

    print(
        f"Total SKUs: "
        f"{df['sku_id'].nunique():,}"
    )

    print(
        f"Total inventory units: "
        f"{df['inventory_units'].sum():,.2f}"
    )

    print(
        f"Total 8-week forecast demand: "
        f"{df['forecast_8w_units'].sum():,.2f}"
    )

    print(
        f"Potential stockout gap: "
        f"{df['stockout_gap_units'].sum():,.2f}"
    )

    print(
        f"Excess inventory units: "
        f"{df['excess_inventory_units'].sum():,.2f}"
    )

    print(
        "\nRisk Level Distribution:"
    )

    risk_distribution = (
        df["risk_level"]
        .value_counts()
        .reindex(
            [
                "CRITICAL",
                "HIGH",
                "MEDIUM",
                "LOW"
            ],
            fill_value=0
        )
    )

    for level, count in (
        risk_distribution.items()
    ):

        print(
            f"{level:10s}: {count:>5}"
        )

    print(
        "\nStockout Risk Distribution:"
    )

    stockout_distribution = (
        df["stockout_risk"]
        .value_counts()
        .reindex(
            [
                "VERY HIGH",
                "HIGH",
                "MEDIUM",
                "LOW"
            ],
            fill_value=0
        )
    )

    for level, count in (
        stockout_distribution.items()
    ):

        print(
            f"{level:10s}: {count:>5}"
        )

    print(
        "\nOverstock Risk Distribution:"
    )

    overstock_distribution = (
        df["overstock_risk"]
        .value_counts()
        .reindex(
            [
                "VERY HIGH",
                "HIGH",
                "MEDIUM",
                "LOW"
            ],
            fill_value=0
        )
    )

    for level, count in (
        overstock_distribution.items()
    ):

        print(
            f"{level:10s}: {count:>5}"
        )

    print(
        "\nTop 10 highest-risk SKUs:"
    )

    top_risk = (
        df[
            [
                "sku_id",
                "risk_score",
                "risk_level",
                "inventory_units",
                "forecast_8w_units",
                "weeks_of_cover",
                "recommended_action"
            ]
        ]
        .sort_values(
            "risk_score",
            ascending=False
        )
        .head(10)
    )

    print(
        top_risk.to_string(
            index=False
        )
    )

    print(
        "\n" + "=" * 60
    )


# ============================================================
# 23. CREATE RISK SUMMARY CSV
# ============================================================

def create_risk_summary(df):
    """
    Create a business-level inventory risk summary
    and save it as reports/risk_summary.csv.
    """

    print(
        "\nCreating risk summary report..."
    )

    total_skus = (
        df["sku_id"].nunique()
    )

    total_inventory = (
        df["inventory_units"].sum()
    )

    total_forecast = (
        df["forecast_8w_units"].sum()
    )

    stockout_gap = (
        df["stockout_gap_units"].sum()
    )

    excess_inventory = (
        df["excess_inventory_units"].sum()
    )

    # --------------------------------------------------------
    # Risk level counts
    # --------------------------------------------------------

    risk_counts = (
        df["risk_level"]
        .value_counts()
        .to_dict()
    )

    # --------------------------------------------------------
    # Stockout risk counts
    # --------------------------------------------------------

    stockout_counts = (
        df["stockout_risk"]
        .value_counts()
        .to_dict()
    )

    # --------------------------------------------------------
    # Overstock risk counts
    # --------------------------------------------------------

    overstock_counts = (
        df["overstock_risk"]
        .value_counts()
        .to_dict()
    )

    # --------------------------------------------------------
    # Create summary dataframe
    # --------------------------------------------------------

    summary = pd.DataFrame({

        "metric": [

            "Total SKUs",

            "Total Inventory Units",

            "Total 8-Week Forecast Demand",

            "Potential 8-Week Inventory Coverage Gap",

            "Excess Inventory Units",

            "Critical Risk SKUs",

            "High Risk SKUs",

            "Medium Risk SKUs",

            "Low Risk SKUs",

            "Stockout Very High SKUs",

            "Stockout High SKUs",

            "Stockout Medium SKUs",

            "Stockout Low SKUs",

            "Overstock Very High SKUs",

            "Overstock High SKUs",

            "Overstock Medium SKUs",

            "Overstock Low SKUs"

        ],

        "value": [

            total_skus,

            round(total_inventory, 2),

            round(total_forecast, 2),

            round(stockout_gap, 2),

            round(excess_inventory, 2),

            risk_counts.get(
                "CRITICAL",
                0
            ),

            risk_counts.get(
                "HIGH",
                0
            ),

            risk_counts.get(
                "MEDIUM",
                0
            ),

            risk_counts.get(
                "LOW",
                0
            ),

            stockout_counts.get(
                "VERY HIGH",
                0
            ),

            stockout_counts.get(
                "HIGH",
                0
            ),

            stockout_counts.get(
                "MEDIUM",
                0
            ),

            stockout_counts.get(
                "LOW",
                0
            ),

            overstock_counts.get(
                "VERY HIGH",
                0
            ),

            overstock_counts.get(
                "HIGH",
                0
            ),

            overstock_counts.get(
                "MEDIUM",
                0
            ),

            overstock_counts.get(
                "LOW",
                0
            )

        ]

    })

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    output_path = (
        REPORT_DIR /
        "risk_summary.csv"
    )

    summary.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nRisk summary saved to:\n{output_path}"
    )

    return output_path


# ============================================================
# 24. MAIN PIPELINE
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "PROJECT FORESIGHT"
    )

    print(
        "Demand & Inventory Intelligence"
    )

    print(
        "Inventory Risk Engine"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    print(
        "\n[1/6] Loading forecast..."
    )

    forecast = load_forecast()

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    print(
        "\n[2/6] Loading inventory snapshots..."
    )

    inventory = load_inventory()

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    print(
        "\n[3/6] Preparing inventory data..."
    )

    inventory = standardize_inventory_columns(
        inventory
    )

    inventory = validate_inventory(
        inventory
    )

    inventory = get_latest_inventory(
        inventory
    )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    print(
        "\n[4/6] Calculating inventory risk..."
    )

    risk_df = build_risk_dataset(
        forecast,
        inventory
    )

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    print(
        "\n[5/6] Formatting risk output..."
    )

    risk_df = format_output(
        risk_df
    )

    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    print(
        "\n[6/6] Validating and saving..."
    )

    validate_output(
        risk_df
    )

    # --------------------------------------------------------
    # Save detailed risk dataset
    # --------------------------------------------------------

    risk_df.to_csv(
        RISK_OUTPUT_PATH,
        index=False
    )

    print(
        f"\nRisk output saved to:"
    )

    print(
        RISK_OUTPUT_PATH
    )

    # --------------------------------------------------------
    # Print terminal summary
    # --------------------------------------------------------

    print_summary(
        risk_df
    )

    # --------------------------------------------------------
    # Create risk summary CSV
    # --------------------------------------------------------

    risk_summary_path = create_risk_summary(
        risk_df
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "INVENTORY RISK PIPELINE COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        "\nGenerated files:"
    )

    print(
        f"1. {RISK_OUTPUT_PATH}"
    )

    print(
        f"2. {risk_summary_path}"
    )

    print(
        "=" * 70
    )


# ============================================================
# 25. RUN
# ============================================================

if __name__ == "__main__":
    main()