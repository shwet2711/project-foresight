
# a67b5b #ffcc99
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PROJECT FORESIGHT
# Demand & Inventory Intelligence Dashboard
#
# Run from project root:
#     python -m streamlit run dashboard/app.py
#
# Current architecture:
#   weekly_demand.csv
#        ↓
#   HistGradientBoostingRegressor
#        ↓
#   forecast_output.csv
#        ↓
#   inventory_risk.csv
#        ↓
#   Streamlit Dashboard
#
# This version DOES NOT use the old XGBoost prediction pipeline.
# ============================================================


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PROJECT FORESIGHT",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
ASSETS_DIR = BASE_DIR / "assets"

WEEKLY_FILE = PROCESSED_DIR / "weekly_demand.csv"
FORECAST_FILE = PROCESSED_DIR / "forecast_output.csv"
FORECAST_MODEL_FILE = PROCESSED_DIR / "forecast_model.pkl"
BACKTEST_FILE = PROCESSED_DIR / "backtest_results.csv"

RISK_FILE = PROCESSED_DIR / "inventory_risk.csv"

INVENTORY_RAW_FILE = RAW_DIR / "inventory_snapshots.csv"
INVENTORY_PROCESSED_FILE = PROCESSED_DIR / "inventory_snapshots.csv"

RISK_SUMMARY_FILE = REPORTS_DIR / "risk_summary.csv"

LOGO_FILE = ASSETS_DIR / "logo.png"


# ============================================================
# MODEL FEATURES FROM CURRENT FORECASTING PIPELINE
# ============================================================

FORECAST_FEATURES = [
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_8",
    "lag_13",
    "lag_26",
    "lag_52",
    "rolling_mean_4",
    "rolling_mean_8",
    "rolling_mean_13",
    "rolling_std_8",
    "month_num",
    "quarter",
    "week_of_year",
    "sin_week",
    "cos_week",
    "promo_ratio",
    "holiday_ratio",
    "unit_cost",
    "list_price",
]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main {
            background: #f8fafc;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        .project-title {
            font-size: 54px;
            font-weight: 800;
            margin-bottom: 0;
        }

        .project-subtitle {
            font-size: 20px;
            font-weight: 600;
            margin-top: 2px;
        }

        .project-description {
            font-size: 14px;
            opacity: 0.72;
            margin-top: 6px;
            margin-bottom: 14px;
        }

        div[data-testid="metric-container"] {
            background: white;
            border: 1px solid #e5e7eb;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }

        section[data-testid="stSidebar"] {
            background: #556b2f;
        }

        section[data-testid="stSidebar"] * {
            color: #cfa0a0!important;
        }

        section[data-testid="stSidebar"] .stMultiSelect span,
        section[data-testid="stSidebar"] .stMultiSelect label,
        section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] span,
        section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
            color: rgb(49, 51, 63)!important;
        }

        section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
            background: rgba(255, 255, 255, 0.08) !important;
            border-color: rgba(255, 255, 255, 0.2) !important;
        }

        .info-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 16px;
            margin: 8px 0;
        }

        .risk-card {
            background: #fff7ed;
            border: 1px solid #fed7aa;
            border-radius: 12px;
            padding: 16px;
            margin: 8px 0;
        }

        .success-card {
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 12px;
            padding: 16px;
            margin: 8px 0;
        }

        .footer {
            text-align: center;
            opacity: 0.6;
            font-size: 13px;
            padding: 25px 0 5px 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_columns(df):
    """Standardize dataframe column names."""
    if df is None:
        return None

    df = df.copy()

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    return df


def clean_dataframe(df):
    """Clean common PROJECT FORESIGHT columns."""
    if df is None:
        return None

    df = clean_columns(df)

    date_columns = [
        "date",
        "forecast_date",
    ]

    for column in date_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
            )

    if "sku_id" in df.columns:
        df["sku_id"] = (
            df["sku_id"]
            .astype(str)
            .str.strip()
        )

    return df


@st.cache_data(show_spinner=False)
def read_csv_file(path_string):
    """Read a CSV safely."""
    path = Path(path_string)

    if not path.exists():
        return None

    try:
        return clean_dataframe(
            pd.read_csv(path)
        )
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_forecast_model(path_string):
    """Load current HistGradientBoosting forecast model."""
    path = Path(path_string)

    if not path.exists():
        return None

    try:
        return joblib.load(path)
    except Exception:
        return None


def safe_number(value, default=0.0):
    try:
        number = float(value)

        if np.isfinite(number):
            return number

        return default

    except (TypeError, ValueError):
        return default


def format_number(value, decimals=0):
    return f"{safe_number(value):,.{decimals}f}"


def format_money(value):
    value = safe_number(value)

    if abs(value) >= 1_000_000_000:
        return f"₹{value / 1_000_000_000:.2f}B"

    if abs(value) >= 1_000_000:
        return f"₹{value / 1_000_000:.2f}M"

    if abs(value) >= 1_000:
        return f"₹{value / 1_000:.2f}K"

    return f"₹{value:,.0f}"


def make_chart(fig, height=420):
    fig.update_layout(
        template="plotly_white",
        height=height,
        title_x=0.5,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    return fig


def numeric_columns(df, columns):
    """Convert selected columns to numeric without failing."""
    if df is None:
        return None

    df = df.copy()

    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            ).fillna(0)

    return df


def recommendation(row):
    """Create business recommendation from inventory-risk data."""

    risk_level = str(
        row.get("risk_level", "")
    ).upper()

    stockout_risk = str(
        row.get("stockout_risk", "")
    ).upper()

    overstock_risk = str(
        row.get("overstock_risk", "")
    ).upper()

    stockout_gap = safe_number(
        row.get("stockout_gap_units", 0)
    )

    excess_units = safe_number(
        row.get("excess_inventory_units", 0)
    )

    reorder_units = safe_number(
        row.get("recommended_replenishment_units", 0)
    )

    if (
        risk_level == "CRITICAL"
        or stockout_risk in {"CRITICAL", "HIGH"}
    ):
        return (
            f"Urgent replenishment required. "
            f"Potential stockout gap: "
            f"{stockout_gap:,.0f} units."
        )

    if risk_level == "HIGH":
        return (
            f"Prioritize replenishment. "
            f"Suggested quantity: "
            f"{reorder_units:,.0f} units."
        )

    if (
        overstock_risk in {"HIGH", "CRITICAL"}
        and excess_units > 0
    ):
        return (
            f"Reduce excess inventory. "
            f"Potential excess: "
            f"{excess_units:,.0f} units."
        )

    if reorder_units > 0:
        return (
            f"Plan replenishment of approximately "
            f"{reorder_units:,.0f} units."
        )

    return (
        "Monitor demand and inventory. "
        "No urgent action is required."
    )


def inventory_value_from_snapshot(
    inventory_df,
    weekly_df,
):
    """
    Estimate latest inventory value.

    Inventory snapshots contain units, while SKU cost information
    is available in weekly_demand.csv.
    """

    if (
        inventory_df is None
        or inventory_df.empty
    ):
        return 0.0

    latest = inventory_df.copy()

    if "date" in latest.columns:
        latest = latest.sort_values("date")

        latest = (
            latest
            .groupby("sku_id", as_index=False)
            .tail(1)
        )

    if "on_hand_units" not in latest.columns:
        return 0.0

    latest["on_hand_units"] = pd.to_numeric(
        latest["on_hand_units"],
        errors="coerce",
    ).fillna(0)

    if "unit_cost" in latest.columns:
        latest["unit_cost"] = pd.to_numeric(
            latest["unit_cost"],
            errors="coerce",
        ).fillna(0)

        return safe_number(
            (
                latest["on_hand_units"]
                * latest["unit_cost"]
            ).sum()
        )

    if (
        weekly_df is None
        or "unit_cost" not in weekly_df.columns
    ):
        return 0.0

    sku_cost = (
        weekly_df[
            ["sku_id", "unit_cost"]
        ]
        .copy()
        .drop_duplicates("sku_id")
    )

    sku_cost["unit_cost"] = pd.to_numeric(
        sku_cost["unit_cost"],
        errors="coerce",
    ).fillna(0)

    latest = latest.merge(
        sku_cost,
        on="sku_id",
        how="left",
        suffixes=("", "_weekly"),
    )

    cost_column = (
        "unit_cost"
        if "unit_cost" in latest.columns
        else "unit_cost_weekly"
    )

    if cost_column not in latest.columns:
        return 0.0

    return safe_number(
        (
            latest["on_hand_units"]
            * pd.to_numeric(
                latest[cost_column],
                errors="coerce",
            ).fillna(0)
        ).sum()
    )


def get_summary_value(
    summary_df,
    metric_name,
    default=0.0,
):
    """Get a numeric metric from risk_summary.csv."""

    if (
        summary_df is None
        or summary_df.empty
        or "metric" not in summary_df.columns
        or "value" not in summary_df.columns
    ):
        return default

    match = summary_df[
        summary_df["metric"]
        .astype(str)
        .str.strip()
        .str.lower()
        == metric_name.strip().lower()
    ]

    if match.empty:
        return default

    return safe_number(
        match.iloc[0]["value"],
        default,
    )


def risk_color_text(level):
    level = str(level).upper()

    if level == "CRITICAL":
        return "🔴 CRITICAL"

    if level == "HIGH":
        return "🟠 HIGH"

    if level == "MEDIUM":
        return "🟡 MEDIUM"

    if level == "LOW":
        return "🟢 LOW"

    return level


def apply_risk_filters(
    risk_df,
    selected_sku,
    selected_risk_levels,
    selected_stockout,
    selected_overstock,
):
    """Apply risk filters safely."""

    result = risk_df.copy()

    if (
        selected_risk_levels
        and "risk_level" in result.columns
    ):
        result = result[
            result["risk_level"].isin(
                selected_risk_levels
            )
        ]

    if (
        selected_stockout
        and "stockout_risk" in result.columns
    ):
        result = result[
            result["stockout_risk"].isin(
                selected_stockout
            )
        ]

    if (
        selected_overstock
        and "overstock_risk" in result.columns
    ):
        result = result[
            result["overstock_risk"].isin(
                selected_overstock
            )
        ]

    if (
        selected_sku != "All"
        and "sku_id" in result.columns
    ):
        result = result[
            result["sku_id"].astype(str)
            == str(selected_sku)
        ]

    return result


# ============================================================
# LOAD DATA
# ============================================================

weekly = read_csv_file(
    str(WEEKLY_FILE)
)

forecast = read_csv_file(
    str(FORECAST_FILE)
)

risk = read_csv_file(
    str(RISK_FILE)
)

inventory = read_csv_file(
    str(INVENTORY_RAW_FILE)
)

if (
    inventory is None
    or inventory.empty
):
    inventory = read_csv_file(
        str(INVENTORY_PROCESSED_FILE)
    )

risk_summary = read_csv_file(
    str(RISK_SUMMARY_FILE)
)

backtest = read_csv_file(
    str(BACKTEST_FILE)
)

forecast_model = load_forecast_model(
    str(FORECAST_MODEL_FILE)
)


# ============================================================
# MAIN DATA VALIDATION
# ============================================================

if weekly is None or weekly.empty:

    st.error(
        "PROJECT FORESIGHT weekly demand data was not found."
    )

    st.code(
        str(WEEKLY_FILE)
    )

    st.info(
        "Run the project data pipeline first so that "
        "data/processed/weekly_demand.csv is created."
    )

    st.stop()


required_weekly = [
    "date",
    "sku_id",
    "units_sold",
]

missing_weekly = [
    column
    for column in required_weekly
    if column not in weekly.columns
]

if missing_weekly:

    st.error(
        "weekly_demand.csv is missing required columns."
    )

    st.write(
        "Missing columns:"
    )

    st.code(
        "\n".join(missing_weekly)
    )

    st.stop()


# ============================================================
# CLEAN WEEKLY DATA
# ============================================================

weekly["units_sold"] = pd.to_numeric(
    weekly["units_sold"],
    errors="coerce",
).fillna(0)

if "revenue" in weekly.columns:
    weekly["revenue"] = pd.to_numeric(
        weekly["revenue"],
        errors="coerce",
    ).fillna(0)

if "date" in weekly.columns:
    weekly = weekly.dropna(
        subset=["date"]
    ).copy()

    weekly["year"] = (
        weekly["date"]
        .dt.year
        .astype(int)
    )

    weekly["month"] = (
        weekly["date"]
        .dt.month
    )


# ============================================================
# CLEAN FORECAST
# ============================================================

if forecast is not None:

    required_forecast = [
        "sku_id",
        "forecast_date",
        "forecast_units",
    ]

    missing_forecast = [
        column
        for column in required_forecast
        if column not in forecast.columns
    ]

    if missing_forecast:
        st.warning(
            "forecast_output.csv is missing required columns: "
            + ", ".join(missing_forecast)
        )

        forecast = None

    else:

        forecast["forecast_units"] = pd.to_numeric(
            forecast["forecast_units"],
            errors="coerce",
        ).fillna(0).clip(lower=0)

        forecast["forecast_date"] = pd.to_datetime(
            forecast["forecast_date"],
            errors="coerce",
        )

        forecast["sku_id"] = (
            forecast["sku_id"]
            .astype(str)
            .str.strip()
        )

        forecast = forecast.dropna(
            subset=[
                "forecast_date",
                "sku_id",
            ]
        ).copy()


# ============================================================
# CLEAN INVENTORY
# ============================================================

if inventory is not None:

    inventory["sku_id"] = (
        inventory["sku_id"]
        .astype(str)
        .str.strip()
    )

    if "date" in inventory.columns:
        inventory["date"] = pd.to_datetime(
            inventory["date"],
            errors="coerce",
        )

    for column in [
        "on_hand_units",
        "on_order_units",
        "lead_time_days",
        "reorder_point",
    ]:

        if column in inventory.columns:

            inventory[column] = pd.to_numeric(
                inventory[column],
                errors="coerce",
            ).fillna(0)


# ============================================================
# CLEAN RISK DATA
# ============================================================

if risk is not None:

    risk["sku_id"] = (
        risk["sku_id"]
        .astype(str)
        .str.strip()
    )

    risk["risk_level"] = (
        risk["risk_level"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    for column in [
        "risk_score",
        "inventory_units",
        "forecast_8w_units",
        "weeks_of_cover",
        "stockout_gap_units",
        "excess_inventory_units",
        "recommended_replenishment_units",
    ]:

        if column in risk.columns:

            risk[column] = pd.to_numeric(
                risk[column],
                errors="coerce",
            ).fillna(0)

    for column in [
        "inventory_units",
        "forecast_8w_units",
        "stockout_gap_units",
        "excess_inventory_units",
        "recommended_replenishment_units",
    ]:

        if column in risk.columns:
            risk[column] = risk[column].clip(
                lower=0
            )

    for column in [
        "stockout_risk",
        "overstock_risk",
    ]:

        if column in risk.columns:

            risk[column] = (
                risk[column]
                .astype(str)
                .str.strip()
                .str.upper()
            )


# ============================================================
# CLEAN RISK SUMMARY
# ============================================================

if risk_summary is not None:

    if "value" in risk_summary.columns:

        risk_summary["value"] = pd.to_numeric(
            risk_summary["value"],
            errors="coerce",
        ).fillna(0)


# ============================================================
# SIDEBAR
# ============================================================

if LOGO_FILE.exists():

    st.sidebar.image(
        str(LOGO_FILE),
        width=180,
    )

st.sidebar.markdown(
    "## ⚙️ Dashboard Controls"
)

st.sidebar.caption(
    "Use these filters to investigate "
    "sales, demand, forecasts and inventory risk."
)

st.sidebar.markdown("---")


# ------------------------------------------------------------
# Category filter
# ------------------------------------------------------------

categories = []

if "category" in weekly.columns:

    categories = sorted(
        weekly["category"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

selected_category = st.sidebar.selectbox(
    "Category",
    ["All"] + categories,
)


# ------------------------------------------------------------
# Year filter
# ------------------------------------------------------------

years = sorted(
    weekly["year"]
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)

selected_year = st.sidebar.selectbox(
    "Year",
    ["All"] + years,
)


# ------------------------------------------------------------
# Date range
# ------------------------------------------------------------

selected_date_range = None

if (
    "date" in weekly.columns
    and not weekly["date"].dropna().empty
):

    min_date = weekly["date"].min().date()
    max_date = weekly["date"].max().date()

    selected_date_range = st.sidebar.date_input(
        "📅 Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )


# ------------------------------------------------------------
# SKU filter
# ------------------------------------------------------------

sku_options = sorted(
    weekly["sku_id"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_sku = st.sidebar.selectbox(
    "🏷️ SKU",
    ["All"] + sku_options,
)


# ------------------------------------------------------------
# Risk filters
# ------------------------------------------------------------

selected_risk_levels = []
selected_stockout = []
selected_overstock = []

if risk is not None and not risk.empty:

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📦 Inventory Risk")

    risk_levels = sorted(
        risk["risk_level"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_risk_levels = st.sidebar.multiselect(
        "Risk Level",
        risk_levels,
        default=risk_levels,
    )

    if "stockout_risk" in risk.columns:

        stockout_levels = sorted(
            risk["stockout_risk"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_stockout = st.sidebar.multiselect(
            "Stockout Risk",
            stockout_levels,
            default=stockout_levels,
        )

    if "overstock_risk" in risk.columns:

        overstock_levels = sorted(
            risk["overstock_risk"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_overstock = st.sidebar.multiselect(
            "Overstock Risk",
            overstock_levels,
            default=overstock_levels,
        )


# ============================================================
# APPLY WEEKLY FILTERS
# ============================================================

filtered_weekly = weekly.copy()

if (
    selected_category != "All"
    and "category" in filtered_weekly.columns
):

    filtered_weekly = filtered_weekly[
        filtered_weekly["category"]
        .astype(str)
        == str(selected_category)
    ]


if (
    selected_year != "All"
    and "year" in filtered_weekly.columns
):

    filtered_weekly = filtered_weekly[
        filtered_weekly["year"]
        == int(selected_year)
    ]


if (
    selected_date_range is not None
    and "date" in filtered_weekly.columns
):

    if (
        isinstance(
            selected_date_range,
            (tuple, list),
        )
        and len(selected_date_range) == 2
    ):

        start_date = pd.to_datetime(
            selected_date_range[0]
        )

        end_date = (
            pd.to_datetime(
                selected_date_range[1]
            )
            + pd.Timedelta(days=1)
        )

        filtered_weekly = filtered_weekly[
            (
                filtered_weekly["date"]
                >= start_date
            )
            &
            (
                filtered_weekly["date"]
                < end_date
            )
        ]


if selected_sku != "All":

    filtered_weekly = filtered_weekly[
        filtered_weekly["sku_id"]
        .astype(str)
        == str(selected_sku)
    ]


# ============================================================
# APPLY RISK FILTERS
# ============================================================

filtered_risk = None

if risk is not None:

    filtered_risk = apply_risk_filters(
        risk,
        selected_sku,
        selected_risk_levels,
        selected_stockout,
        selected_overstock,
    )

    # Category and year do not exist in inventory_risk.csv.
    # Therefore use the weekly dataset to identify matching SKUs.
    matching_skus = set(
        filtered_weekly["sku_id"]
        .astype(str)
        .unique()
        .tolist()
    )

    if selected_category != "All" or selected_year != "All":

        filtered_risk = filtered_risk[
            filtered_risk["sku_id"]
            .astype(str)
            .isin(matching_skus)
        ]


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="project-title">
        📈 PROJECT FORESIGHT
    </div>

    <div class="project-subtitle">
        🚀 Demand & Inventory Intelligence
    </div>

    <div class="project-description">
        Sales Intelligence • 8-Week Forecasting • Inventory Risk •
        Replenishment • SKU Intelligence • Business Insights
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# TABS
# ============================================================

(
    tab_dashboard,
    tab_forecast,
    tab_risk,
    tab_ai,
    tab_insights,
    tab_sku,
    tab_data,
) = st.tabs(
    [
        "📊 Dashboard",
        "📈 Forecast",
        "📦 Inventory Risk",
        "🤖 AI Prediction",
        "🎯 AI Insights",
        "🔍 SKU Intelligence",
        "📋 Data & Downloads",
    ]
)


# ============================================================
# TAB 1 — EXECUTIVE DASHBOARD
# ============================================================

with tab_dashboard:

    st.header(
        "📊 Executive Dashboard"
    )

    if filtered_weekly.empty:

        st.warning(
            "No weekly demand data matches the selected filters."
        )

    else:

        revenue = 0.0

        if "revenue" in filtered_weekly.columns:

            revenue = safe_number(
                filtered_weekly["revenue"].sum()
            )

        units = safe_number(
            filtered_weekly["units_sold"].sum()
        )

        sku_count = (
            filtered_weekly["sku_id"]
            .nunique()
        )

        avg_weekly_units = (
            safe_number(
                filtered_weekly
                .groupby("date")["units_sold"]
                .sum()
                .mean()
            )
            if "date" in filtered_weekly.columns
            else 0.0
        )

        inventory_units = 0.0
        high_critical = 0

        if (
            filtered_risk is not None
            and not filtered_risk.empty
        ):

            if "inventory_units" in filtered_risk.columns:

                inventory_units = safe_number(
                    filtered_risk[
                        "inventory_units"
                    ].sum()
                )

            if "risk_level" in filtered_risk.columns:

                high_critical = int(
                    filtered_risk[
                        "risk_level"
                    ]
                    .isin(
                        ["HIGH", "CRITICAL"]
                    )
                    .sum()
                )

        inventory_value = inventory_value_from_snapshot(
            inventory,
            weekly,
        )

        if (
            selected_sku != "All"
            and inventory is not None
            and not inventory.empty
        ):

            sku_inventory = inventory[
                inventory["sku_id"].astype(str)
                == str(selected_sku)
            ]

            inventory_value = inventory_value_from_snapshot(
                sku_inventory,
                weekly,
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "💰 Revenue",
                format_money(revenue),
            )

        with col2:
            st.metric(
                "📦 Units Sold",
                format_number(units),
            )

        with col3:
            st.metric(
                "🏷️ SKUs",
                format_number(sku_count),
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "📈 Avg Weekly Demand",
                format_number(avg_weekly_units),
            )

        with col2:
            st.metric(
                "💵 Inventory Value",
                format_money(inventory_value),
            )

        with col3:
            st.metric(
                "⚠️ High/Critical SKUs",
                format_number(high_critical),
            )

        st.divider()

        # --------------------------------------------------------
        # Revenue trend
        # --------------------------------------------------------

        if (
            "date" in filtered_weekly.columns
            and "revenue" in filtered_weekly.columns
        ):

            monthly = (
                filtered_weekly
                .assign(
                    period=filtered_weekly["date"]
                    .dt.to_period("M")
                )
                .groupby("period", as_index=False)
                ["revenue"]
                .sum()
            )

            monthly["month"] = (
                monthly["period"]
                .astype(str)
            )

            fig = px.line(
                monthly,
                x="month",
                y="revenue",
                markers=True,
                title="📈 Monthly Revenue Trend",
            )

            st.plotly_chart(
                make_chart(fig),
                use_container_width=True,
            )

        # --------------------------------------------------------
        # Category and SKU charts
        # --------------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            if (
                "category" in filtered_weekly.columns
                and "revenue" in filtered_weekly.columns
            ):

                category_revenue = (
                    filtered_weekly
                    .groupby("category", as_index=False)
                    ["revenue"]
                    .sum()
                    .sort_values(
                        "revenue",
                        ascending=False,
                    )
                )

                fig = px.bar(
                    category_revenue,
                    x="category",
                    y="revenue",
                    color="revenue",
                    title="📊 Revenue by Category",
                )

                st.plotly_chart(
                    make_chart(fig),
                    use_container_width=True,
                )

        with col2:

            if (
                "sku_id" in filtered_weekly.columns
                and "units_sold" in filtered_weekly.columns
            ):

                top_skus = (
                    filtered_weekly
                    .groupby("sku_id", as_index=False)
                    ["units_sold"]
                    .sum()
                    .nlargest(10, "units_sold")
                    .sort_values(
                        "units_sold"
                    )
                )

                fig = px.bar(
                    top_skus,
                    x="units_sold",
                    y="sku_id",
                    orientation="h",
                    color="units_sold",
                    title="🏆 Top 10 SKUs by Demand",
                )

                st.plotly_chart(
                    make_chart(fig),
                    use_container_width=True,
                )

        # --------------------------------------------------------
        # Demand trend
        # --------------------------------------------------------

        if (
            "date" in filtered_weekly.columns
            and "units_sold" in filtered_weekly.columns
        ):

            demand = (
                filtered_weekly
                .groupby("date", as_index=False)
                ["units_sold"]
                .sum()
            )

            fig = px.line(
                demand,
                x="date",
                y="units_sold",
                markers=True,
                title="📦 Historical Weekly Demand",
            )

            st.plotly_chart(
                make_chart(fig),
                use_container_width=True,
            )

        # --------------------------------------------------------
        # Revenue distribution
        # --------------------------------------------------------

        if (
            "category" in filtered_weekly.columns
            and "revenue" in filtered_weekly.columns
        ):

            category_revenue = (
                filtered_weekly
                .groupby("category", as_index=False)
                ["revenue"]
                .sum()
            )

            fig = px.pie(
                category_revenue,
                names="category",
                values="revenue",
                hole=0.55,
                title="🍩 Revenue Distribution",
            )

            st.plotly_chart(
                make_chart(fig, 450),
                use_container_width=True,
            )

        with st.expander(
            "🔍 View Filtered Weekly Dataset"
        ):

            st.dataframe(
                filtered_weekly.head(1000),
                use_container_width=True,
                hide_index=True,
            )

        st.download_button(
            "📥 Download Filtered Weekly Demand",
            filtered_weekly.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="filtered_weekly_demand.csv",
            mime="text/csv",
        )


# ============================================================
# TAB 2 — FORECAST
# ============================================================

with tab_forecast:

    st.header(
        "📈 8-Week Demand Forecast"
    )

    if forecast is None or forecast.empty:

        st.warning(
            "forecast_output.csv was not found."
        )

        st.code(
            str(FORECAST_FILE)
        )

        st.info(
            "Run the current forecasting pipeline first."
        )

    else:

        forecast_view = forecast.copy()

        if selected_sku != "All":

            forecast_view = forecast_view[
                forecast_view["sku_id"]
                .astype(str)
                == str(selected_sku)
            ]

        if forecast_view.empty:

            st.warning(
                "No forecast is available for the selected SKU."
            )

        else:

            total_forecast = safe_number(
                forecast_view[
                    "forecast_units"
                ].sum()
            )

            forecast_weeks = (
                forecast_view[
                    "forecast_date"
                ]
                .nunique()
            )

            avg_weekly_forecast = (
                safe_number(
                    forecast_view[
                        "forecast_units"
                    ].mean()
                )
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "📅 Forecast Weeks",
                    format_number(
                        forecast_weeks
                    ),
                )

            with col2:
                st.metric(
                    "🔮 Forecast Demand",
                    format_number(
                        total_forecast
                    ),
                )

            with col3:
                st.metric(
                    "📈 Avg Weekly Forecast",
                    format_number(
                        avg_weekly_forecast
                    ),
                )

            st.success(
                "Using the current PROJECT FORESIGHT "
                "8-week recursive SKU-level forecasting pipeline."
            )

            # ----------------------------------------------------
            # Overall forecast
            # ----------------------------------------------------

            overall_forecast = (
                forecast_view
                .groupby(
                    "forecast_date",
                    as_index=False,
                )["forecast_units"]
                .sum()
            )

            fig = px.line(
                overall_forecast,
                x="forecast_date",
                y="forecast_units",
                markers=True,
                title="🔮 8-Week Demand Forecast",
            )

            st.plotly_chart(
                make_chart(fig, 500),
                use_container_width=True,
            )

            # ----------------------------------------------------
            # SKU forecast comparison
            # ----------------------------------------------------

            if selected_sku == "All":

                sku_forecast = (
                    forecast_view
                    .groupby("sku_id", as_index=False)
                    ["forecast_units"]
                    .sum()
                    .nlargest(15, "forecast_units")
                    .sort_values(
                        "forecast_units"
                    )
                )

                fig = px.bar(
                    sku_forecast,
                    x="forecast_units",
                    y="sku_id",
                    orientation="h",
                    color="forecast_units",
                    title="🏆 Top 15 SKUs by 8-Week Forecast",
                )

                st.plotly_chart(
                    make_chart(fig, 520),
                    use_container_width=True,
                )

            # ----------------------------------------------------
            # Forecast table
            # ----------------------------------------------------

            st.subheader(
                "📋 Forecast Output"
            )

            st.dataframe(
                forecast_view.sort_values(
                    [
                        "forecast_date",
                        "sku_id",
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "📥 Download Filtered Forecast",
                forecast_view.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="forecast_output_filtered.csv",
                mime="text/csv",
            )


# ============================================================
# TAB 3 — INVENTORY RISK
# ============================================================

with tab_risk:

    st.header(
        "📦 Inventory Risk & Replenishment"
    )

    if risk is None or risk.empty:

        st.warning(
            "inventory_risk.csv was not found."
        )

        st.code(
            str(RISK_FILE)
        )

    elif filtered_risk is None or filtered_risk.empty:

        st.warning(
            "No inventory-risk records match the selected filters."
        )

    else:

        risk_view = filtered_risk.copy()

        inventory_units = (
            safe_number(
                risk_view["inventory_units"].sum()
            )
            if "inventory_units" in risk_view.columns
            else 0.0
        )

        forecast_units = (
            safe_number(
                risk_view["forecast_8w_units"].sum()
            )
            if "forecast_8w_units" in risk_view.columns
            else 0.0
        )

        stockout_gap = (
            safe_number(
                risk_view[
                    "stockout_gap_units"
                ]
                .clip(lower=0)
                .sum()
            )
            if "stockout_gap_units"
            in risk_view.columns
            else 0.0
        )

        excess_inventory = (
            safe_number(
                risk_view[
                    "excess_inventory_units"
                ]
                .clip(lower=0)
                .sum()
            )
            if "excess_inventory_units"
            in risk_view.columns
            else 0.0
        )

        replenishment = (
            safe_number(
                risk_view[
                    "recommended_replenishment_units"
                ]
                .clip(lower=0)
                .sum()
            )
            if "recommended_replenishment_units"
            in risk_view.columns
            else 0.0
        )

        high_critical = int(
            risk_view["risk_level"]
            .isin(
                ["HIGH", "CRITICAL"]
            )
            .sum()
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "📦 Inventory Units",
                format_number(
                    inventory_units
                ),
            )

        with col2:
            st.metric(
                "🔮 8-Week Forecast",
                format_number(
                    forecast_units
                ),
            )

        with col3:
            st.metric(
                "⚠️ High/Critical SKUs",
                format_number(
                    high_critical
                ),
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "🚨 Stockout Gap",
                format_number(
                    stockout_gap
                ),
            )

        with col2:
            st.metric(
                "📦 Excess Inventory",
                format_number(
                    excess_inventory
                ),
            )

        with col3:
            st.metric(
                "🔄 Replenishment",
                format_number(
                    replenishment
                ),
            )

        st.divider()

        # --------------------------------------------------------
        # Risk distribution
        # --------------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            if "risk_level" in risk_view.columns:

                risk_counts = (
                    risk_view[
                        "risk_level"
                    ]
                    .value_counts()
                    .reset_index()
                )

                risk_counts.columns = [
                    "risk_level",
                    "count",
                ]

                fig = px.bar(
                    risk_counts,
                    x="risk_level",
                    y="count",
                    color="risk_level",
                    title="🚦 Risk Level Distribution",
                )

                st.plotly_chart(
                    make_chart(fig),
                    use_container_width=True,
                )

        with col2:

            if "risk_score" in risk_view.columns:

                fig = px.histogram(
                    risk_view,
                    x="risk_score",
                    nbins=20,
                    title="📊 Risk Score Distribution",
                )

                st.plotly_chart(
                    make_chart(fig),
                    use_container_width=True,
                )

        # --------------------------------------------------------
        # Inventory vs forecast
        # --------------------------------------------------------

        if {
            "inventory_units",
            "forecast_8w_units",
            "risk_score",
            "sku_id",
        }.issubset(risk_view.columns):

            fig = px.scatter(
                risk_view,
                x="inventory_units",
                y="forecast_8w_units",
                size="risk_score",
                color="risk_level",
                hover_name="sku_id",
                title="📦 Inventory vs 8-Week Demand",
            )

            st.plotly_chart(
                make_chart(fig, 500),
                use_container_width=True,
            )

        # --------------------------------------------------------
        # Priority action list
        # --------------------------------------------------------

        action_view = risk_view.copy()

        action_view["recommended_action"] = (
            action_view.apply(
                recommendation,
                axis=1,
            )
        )

        if "risk_score" in action_view.columns:

            action_view = (
                action_view
                .sort_values(
                    "risk_score",
                    ascending=False,
                )
            )

        st.subheader(
            "🚨 Priority Action List"
        )

        display_columns = [
            "sku_id",
            "risk_score",
            "risk_level",
            "inventory_units",
            "forecast_8w_units",
            "weeks_of_cover",
            "stockout_gap_units",
            "excess_inventory_units",
            "recommended_replenishment_units",
            "stockout_risk",
            "overstock_risk",
            "recommended_action",
        ]

        display_columns = [
            column
            for column in display_columns
            if column in action_view.columns
        ]

        st.dataframe(
            action_view[
                display_columns
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "📥 Download Risk Actions",
            action_view.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="inventory_risk_actions.csv",
            mime="text/csv",
        )


# ============================================================
# TAB 4 — AI PREDICTION / MODEL INTELLIGENCE
# ============================================================

with tab_ai:

    st.header(
        "🤖 AI Forecast Model"
    )

    st.info(
        "This page uses the CURRENT PROJECT FORESIGHT "
        "forecasting architecture. The old XGBoost live-prediction "
        "logic has intentionally been removed."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        if forecast_model is not None:

            st.metric(
                "🤖 Model Status",
                "Loaded",
            )

        else:

            st.metric(
                "🤖 Model Status",
                "Not Loaded",
            )

    with col2:

        st.metric(
            "🧠 Algorithm",
            "HistGradientBoosting",
        )

    with col3:

        st.metric(
            "🔢 Features",
            len(FORECAST_FEATURES),
        )

    st.divider()

    st.subheader(
        "🧠 Current Model Architecture"
    )

    st.markdown(
        """
        **Forecasting method**

        - HistGradientBoostingRegressor
        - Weekly SKU-level forecasting
        - Time-based validation
        - Seasonal-naive baseline comparison
        - Rolling-origin backtesting
        - 8-week recursive forecasting
        - Non-negative demand predictions

        **Feature groups**

        - Lag features: 1, 2, 4, 8, 13, 26 and 52 weeks
        - Rolling demand: 4, 8 and 13 weeks
        - Rolling volatility: 8 weeks
        - Calendar: month, quarter and week of year
        - Seasonal encoding: sine and cosine week features
        - Business features: promotion and holiday ratios
        - Product features: unit cost and list price
        """
    )

    # ------------------------------------------------------------
    # Model file
    # ------------------------------------------------------------

    if FORECAST_MODEL_FILE.exists():

        st.success(
            f"Forecast model file found: "
            f"{FORECAST_MODEL_FILE}"
        )

    else:

        st.warning(
            "forecast_model.pkl was not found."
        )

    # ------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------

    st.subheader(
        "📊 Model Backtesting"
    )

    if backtest is None or backtest.empty:

        st.warning(
            "backtest_results.csv was not found."
        )

    else:

        st.dataframe(
            backtest,
            use_container_width=True,
            hide_index=True,
        )

        # Try to display WAPE comparison when columns exist.
        wape_columns = [
            column
            for column in [
                "baseline_wape",
                "model_wape",
            ]
            if column in backtest.columns
        ]

        if len(wape_columns) >= 1:

            chart_columns = [
                column
                for column in [
                    "fold",
                    "baseline_wape",
                    "model_wape",
                ]
                if column in backtest.columns
            ]

            if len(chart_columns) >= 2:

                plot_df = backtest[
                    chart_columns
                ].copy()

                id_column = (
                    "fold"
                    if "fold" in plot_df.columns
                    else None
                )

                if id_column is not None:

                    long_df = plot_df.melt(
                        id_vars=id_column,
                        var_name="model",
                        value_name="wape",
                    )

                    fig = px.line(
                        long_df,
                        x=id_column,
                        y="wape",
                        color="model",
                        markers=True,
                        title="📉 Backtesting WAPE",
                    )

                    st.plotly_chart(
                        make_chart(fig),
                        use_container_width=True,
                    )

    # ------------------------------------------------------------
    # Forecast validation
    # ------------------------------------------------------------

    if forecast is not None and not forecast.empty:

        forecast_per_sku = (
            forecast
            .groupby("sku_id")
            .size()
        )

        valid_8_week = bool(
            (forecast_per_sku == 8).all()
        )

        if valid_8_week:

            st.success(
                "Forecast validation: every SKU has "
                "exactly 8 forecast rows."
            )

        else:

            st.warning(
                "Some SKUs do not have exactly 8 forecast rows."
            )

        negative_forecast = bool(
            (
                forecast["forecast_units"]
                < 0
            ).any()
        )

        if not negative_forecast:

            st.success(
                "Forecast validation: no negative demand predictions."
            )

        else:

            st.warning(
                "Negative forecast values were detected."
            )


# ============================================================
# TAB 5 — AI BUSINESS INSIGHTS
# ============================================================

with tab_insights:

    st.header(
        "🎯 AI Business Insights"
    )

    insights = []

    if (
        "revenue" in filtered_weekly.columns
        and not filtered_weekly.empty
    ):

        revenue_value = safe_number(
            filtered_weekly[
                "revenue"
            ].sum()
        )

        insights.append(
            (
                "💰 Sales",
                f"Selected data generated "
                f"{format_money(revenue_value)} "
                f"in revenue.",
            )
        )

    if (
        "units_sold" in filtered_weekly.columns
        and not filtered_weekly.empty
    ):

        demand_value = safe_number(
            filtered_weekly[
                "units_sold"
            ].sum()
        )

        insights.append(
            (
                "📦 Demand",
                f"Observed demand is "
                f"{demand_value:,.0f} units.",
            )
        )

    if (
        "category" in filtered_weekly.columns
        and "revenue" in filtered_weekly.columns
        and not filtered_weekly.empty
    ):

        category_sales = (
            filtered_weekly
            .groupby("category")
            ["revenue"]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        if not category_sales.empty:

            insights.append(
                (
                    "🏆 Top Category",
                    f"{category_sales.index[0]} "
                    f"is the highest-revenue category.",
                )
            )

    if (
        forecast is not None
        and not forecast.empty
    ):

        forecast_for_insight = forecast.copy()

        if selected_sku != "All":

            forecast_for_insight = (
                forecast_for_insight[
                    forecast_for_insight[
                        "sku_id"
                    ].astype(str)
                    == str(selected_sku)
                ]
            )

        if not forecast_for_insight.empty:

            forecast_total = safe_number(
                forecast_for_insight[
                    "forecast_units"
                ].sum()
            )

            insights.append(
                (
                    "🔮 Future Demand",
                    f"Expected demand over the "
                    f"next 8 weeks is approximately "
                    f"{forecast_total:,.0f} units.",
                )
            )

    if (
        filtered_risk is not None
        and not filtered_risk.empty
    ):

        critical_count = int(
            filtered_risk[
                "risk_level"
            ]
            .eq("CRITICAL")
            .sum()
        )

        high_count = int(
            filtered_risk[
                "risk_level"
            ]
            .eq("HIGH")
            .sum()
        )

        insights.append(
            (
                "🚨 Inventory Risk",
                f"{critical_count} critical and "
                f"{high_count} high-risk SKUs "
                f"are present.",
            )
        )

        if "stockout_gap_units" in filtered_risk.columns:

            gap = safe_number(
                filtered_risk[
                    "stockout_gap_units"
                ]
                .clip(lower=0)
                .sum()
            )

            if gap > 0:

                insights.append(
                    (
                        "⚠️ Stockout Exposure",
                        f"Potential stockout exposure "
                        f"is {gap:,.0f} units.",
                    )
                )

        if "excess_inventory_units" in filtered_risk.columns:

            excess = safe_number(
                filtered_risk[
                    "excess_inventory_units"
                ]
                .clip(lower=0)
                .sum()
            )

            if excess > 0:

                insights.append(
                    (
                        "📦 Overstock",
                        f"Potential excess inventory "
                        f"is {excess:,.0f} units.",
                    )
                )

        if (
            "recommended_replenishment_units"
            in filtered_risk.columns
        ):

            replenish = safe_number(
                filtered_risk[
                    "recommended_replenishment_units"
                ]
                .clip(lower=0)
                .sum()
            )

            if replenish > 0:

                insights.append(
                    (
                        "🔄 Replenishment",
                        f"Recommended replenishment "
                        f"is approximately "
                        f"{replenish:,.0f} units.",
                    )
                )

    if not insights:

        st.info(
            "No insights are available for the selected filters."
        )

    else:

        for title, message in insights:

            st.markdown(
                f"""
                <div class="info-card">
                    <b>{title}</b><br>
                    {message}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ------------------------------------------------------------
    # Highest priority SKUs
    # ------------------------------------------------------------

    if (
        filtered_risk is not None
        and not filtered_risk.empty
    ):

        st.subheader(
            "🚨 Highest Priority SKUs"
        )

        priority = filtered_risk.copy()

        priority["AI Recommended Action"] = (
            priority.apply(
                recommendation,
                axis=1,
            )
        )

        if "risk_score" in priority.columns:

            priority = priority.sort_values(
                "risk_score",
                ascending=False,
            )

        priority_columns = [
            "sku_id",
            "risk_score",
            "risk_level",
            "weeks_of_cover",
            "stockout_gap_units",
            "excess_inventory_units",
            "recommended_replenishment_units",
            "AI Recommended Action",
        ]

        priority_columns = [
            column
            for column in priority_columns
            if column in priority.columns
        ]

        st.dataframe(
            priority[
                priority_columns
            ].head(20),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# TAB 6 — SKU INTELLIGENCE
# ============================================================

with tab_sku:

    st.header(
        "🔍 SKU Intelligence"
    )

    all_skus = sorted(
        weekly["sku_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not all_skus:

        st.warning(
            "No SKU data available."
        )

    else:

        default_index = 0

        if (
            selected_sku != "All"
            and selected_sku in all_skus
        ):

            default_index = all_skus.index(
                selected_sku
            )

        selected_detail_sku = st.selectbox(
            "Select SKU",
            all_skus,
            index=default_index,
        )

        sku_history = weekly[
            weekly["sku_id"].astype(str)
            == str(selected_detail_sku)
        ].copy()

        sku_revenue = (
            safe_number(
                sku_history[
                    "revenue"
                ].sum()
            )
            if "revenue" in sku_history.columns
            else 0.0
        )

        sku_units = safe_number(
            sku_history[
                "units_sold"
            ].sum()
        )

        average_price = (
            sku_revenue / sku_units
            if sku_units > 0
            else 0.0
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "🏷️ SKU",
                selected_detail_sku,
            )

        with col2:
            st.metric(
                "💰 Revenue",
                format_money(sku_revenue),
            )

        with col3:
            st.metric(
                "📦 Units",
                format_number(sku_units),
            )

        with col4:
            st.metric(
                "💵 Revenue / Unit",
                format_money(average_price),
            )

        # --------------------------------------------------------
        # Historical demand
        # --------------------------------------------------------

        if (
            "date" in sku_history.columns
            and "units_sold" in sku_history.columns
        ):

            history_chart = (
                sku_history
                .groupby(
                    "date",
                    as_index=False,
                )["units_sold"]
                .sum()
            )

            fig = px.line(
                history_chart,
                x="date",
                y="units_sold",
                markers=True,
                title=(
                    f"📈 Demand History — "
                    f"{selected_detail_sku}"
                ),
            )

            st.plotly_chart(
                make_chart(fig),
                use_container_width=True,
            )

        # --------------------------------------------------------
        # SKU forecast
        # --------------------------------------------------------

        if forecast is not None:

            sku_forecast = forecast[
                forecast["sku_id"].astype(str)
                == str(selected_detail_sku)
            ].copy()

            if not sku_forecast.empty:

                fig = px.line(
                    sku_forecast,
                    x="forecast_date",
                    y="forecast_units",
                    markers=True,
                    title=(
                        f"🔮 8-Week Forecast — "
                        f"{selected_detail_sku}"
                    ),
                )

                st.plotly_chart(
                    make_chart(fig),
                    use_container_width=True,
                )

        # --------------------------------------------------------
        # Risk information
        # --------------------------------------------------------

        if risk is not None:

            sku_risk = risk[
                risk["sku_id"].astype(str)
                == str(selected_detail_sku)
            ].copy()

            if not sku_risk.empty:

                risk_info = sku_risk.iloc[0]

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "🚦 Risk",
                        risk_color_text(
                            risk_info.get(
                                "risk_level",
                                "N/A",
                            )
                        ),
                    )

                with col2:
                    st.metric(
                        "📊 Risk Score",
                        format_number(
                            risk_info.get(
                                "risk_score",
                                0,
                            ),
                            2,
                        ),
                    )

                with col3:
                    st.metric(
                        "📦 Inventory",
                        format_number(
                            risk_info.get(
                                "inventory_units",
                                0,
                            )
                        ),
                    )

                with col4:
                    st.metric(
                        "📅 Weeks Cover",
                        format_number(
                            risk_info.get(
                                "weeks_of_cover",
                                0,
                            ),
                            2,
                        ),
                    )

                st.markdown(
                    f"""
                    <div class="risk-card">
                        <b>Recommended Action</b><br>
                        {recommendation(risk_info)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ------------------------------------------------
                # Risk gauge
                # ------------------------------------------------

                if "risk_score" in risk_info.index:

                    risk_score = safe_number(
                        risk_info[
                            "risk_score"
                        ]
                    )

                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=risk_score,
                            title={
                                "text": "Risk Score"
                            },
                            gauge={
                                "axis": {
                                    "range": [
                                        0,
                                        100,
                                    ]
                                }
                            },
                        )
                    )

                    fig.update_layout(
                        height=350
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                # ------------------------------------------------
                # Risk details
                # ------------------------------------------------

                detail_columns = [
                    "forecast_8w_units",
                    "stockout_gap_units",
                    "excess_inventory_units",
                    "recommended_replenishment_units",
                    "stockout_risk",
                    "overstock_risk",
                ]

                detail_columns = [
                    column
                    for column in detail_columns
                    if column in sku_risk.columns
                ]

                if detail_columns:

                    st.subheader(
                        "📋 Inventory Details"
                    )

                    st.dataframe(
                        sku_risk[
                            detail_columns
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

            else:

                st.info(
                    "No inventory-risk record exists "
                    f"for {selected_detail_sku}."
                )


# ============================================================
# TAB 7 — DATA & DOWNLOADS
# ============================================================

with tab_data:

    st.header(
        "📋 Data & Downloads"
    )

    st.subheader(
        "📁 PROJECT FORESIGHT Files"
    )

    file_status = pd.DataFrame(
        [
            {
                "File": "Weekly Demand",
                "Status": (
                    "Available"
                    if WEEKLY_FILE.exists()
                    else "Missing"
                ),
                "Path": str(WEEKLY_FILE),
            },
            {
                "File": "8-Week Forecast",
                "Status": (
                    "Available"
                    if FORECAST_FILE.exists()
                    else "Missing"
                ),
                "Path": str(FORECAST_FILE),
            },
            {
                "File": "Forecast Model",
                "Status": (
                    "Available"
                    if FORECAST_MODEL_FILE.exists()
                    else "Missing"
                ),
                "Path": str(FORECAST_MODEL_FILE),
            },
            {
                "File": "Backtest Results",
                "Status": (
                    "Available"
                    if BACKTEST_FILE.exists()
                    else "Missing"
                ),
                "Path": str(BACKTEST_FILE),
            },
            {
                "File": "Inventory Risk",
                "Status": (
                    "Available"
                    if RISK_FILE.exists()
                    else "Missing"
                ),
                "Path": str(RISK_FILE),
            },
            {
                "File": "Inventory Snapshots",
                "Status": (
                    "Available"
                    if (
                        INVENTORY_RAW_FILE.exists()
                        or INVENTORY_PROCESSED_FILE.exists()
                    )
                    else "Missing"
                ),
                "Path": str(
                    INVENTORY_RAW_FILE
                    if INVENTORY_RAW_FILE.exists()
                    else INVENTORY_PROCESSED_FILE
                ),
            },
            {
                "File": "Risk Summary",
                "Status": (
                    "Available"
                    if RISK_SUMMARY_FILE.exists()
                    else "Missing"
                ),
                "Path": str(RISK_SUMMARY_FILE),
            },
        ]
    )

    st.dataframe(
        file_status,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ------------------------------------------------------------
    # Downloads
    # ------------------------------------------------------------

    st.subheader(
        "📥 Download Results"
    )

    download_columns = st.columns(4)

    with download_columns[0]:

        if weekly is not None:

            st.download_button(
                "📥 Weekly Demand",
                weekly.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="weekly_demand.csv",
                mime="text/csv",
            )

    with download_columns[1]:

        if forecast is not None:

            st.download_button(
                "📥 Forecast",
                forecast.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="forecast_output.csv",
                mime="text/csv",
            )

    with download_columns[2]:

        if risk is not None:

            st.download_button(
                "📥 Inventory Risk",
                risk.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="inventory_risk.csv",
                mime="text/csv",
            )

    with download_columns[3]:

        if inventory is not None:

            st.download_button(
                "📥 Inventory Snapshots",
                inventory.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="inventory_snapshots.csv",
                mime="text/csv",
            )

    if risk_summary is not None:

        st.divider()

        st.subheader(
            "📊 Risk Summary"
        )

        st.dataframe(
            risk_summary,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "📥 Download Risk Summary",
            risk_summary.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="risk_summary.csv",
            mime="text/csv",
        )

    if backtest is not None:

        st.divider()

        st.subheader(
            "📈 Backtest Results"
        )

        st.dataframe(
            backtest,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "📥 Download Backtest Results",
            backtest.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="backtest_results.csv",
            mime="text/csv",
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        PROJECT FORESIGHT • Demand & Inventory Intelligence
        • 8-Week Forecasting • Inventory Risk • Decision Support
    </div>
    """,
    unsafe_allow_html=True,
)