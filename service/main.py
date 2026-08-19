# ============================================================
# cd C:\Projects\foresight
# uvicorn service.main:app --reload
# PROJECT FORESIGHT
# FastAPI Scoring Service
#
# Returns:
#   1. Forecast + risk for a given SKU
#   2. Forecast-only information for a SKU
#   3. Batch risk scores for all SKUs
#   4. API health/status information
#   5. Data reload support
#
# Run:
#   uvicorn service.main:app --reload
#
# Swagger:
#   http://127.0.0.1:8000/docs
# ============================================================

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Project structure:
#
# PROJECT FORESIGHT/
# ├── app/
# │   └── app.py
# ├── data/
# │   ├── raw/
# │   └── processed/
# │       ├── inventory_risk.csv
# │       └── forecast_output.csv
# ├── reports/
# │   └── risk_summary.csv
# └── service/
#     └── main.py

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

RISK_FILE = PROCESSED_DIR / "inventory_risk.csv"

FORECAST_FILE = PROCESSED_DIR / "forecast_output.csv"


# ============================================================
# 2. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="PROJECT FORESIGHT API",
    description=(
        "Demand and Inventory Intelligence "
        "Decision Support Scoring Service"
    ),
    version="2.0.0",
)


# ============================================================
# 3. GLOBAL DATAFRAMES
# ============================================================

risk = pd.DataFrame()

forecast = pd.DataFrame()


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataframe column names.
    """

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    return df


def clean_sku_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize SKU IDs.
    """

    df = df.copy()

    if "sku_id" in df.columns:

        df["sku_id"] = (
            df["sku_id"]
            .astype(str)
            .str.strip()
        )

    return df


def convert_for_json(value: Any):
    """
    Convert pandas / NumPy values into
    JSON-compatible Python values.
    """

    if value is None:
        return None

    # Handle pandas missing values
    try:

        if pd.isna(value):
            return None

    except (TypeError, ValueError):
        pass

    # Handle pandas Timestamp
    if isinstance(value, pd.Timestamp):

        return value.strftime("%Y-%m-%d")

    # Handle Python datetime/date
    if hasattr(value, "strftime"):

        try:

            return value.strftime("%Y-%m-%d")

        except Exception:
            pass

    # Handle NumPy scalar
    if hasattr(value, "item"):

        try:

            return value.item()

        except Exception:
            pass

    return value


def records_to_json_safe(
    dataframe: pd.DataFrame,
):
    """
    Convert DataFrame rows into
    JSON-safe dictionaries.
    """

    if dataframe.empty:
        return []

    records = dataframe.to_dict(
        orient="records"
    )

    cleaned_records = []

    for record in records:

        cleaned_record = {
            key: convert_for_json(value)
            for key, value in record.items()
        }

        cleaned_records.append(
            cleaned_record
        )

    return cleaned_records


# ============================================================
# 5. LOAD RISK DATA
# ============================================================

def load_risk_data() -> pd.DataFrame:
    """
    Load inventory risk data.

    Source:
        data/processed/inventory_risk.csv
    """

    if not RISK_FILE.exists():

        raise FileNotFoundError(
            f"Risk file not found: {RISK_FILE}"
        )

    risk_data = pd.read_csv(
        RISK_FILE
    )

    risk_data = clean_columns(
        risk_data
    )

    risk_data = clean_sku_column(
        risk_data
    )

    # --------------------------------------------------------
    # Standardize risk level
    # --------------------------------------------------------

    if "risk_level" in risk_data.columns:

        risk_data["risk_level"] = (
            risk_data["risk_level"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    # --------------------------------------------------------
    # Standardize stockout risk
    # --------------------------------------------------------

    if "stockout_risk" in risk_data.columns:

        risk_data["stockout_risk"] = (
            risk_data["stockout_risk"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    # --------------------------------------------------------
    # Standardize overstock risk
    # --------------------------------------------------------

    if "overstock_risk" in risk_data.columns:

        risk_data["overstock_risk"] = (
            risk_data["overstock_risk"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    # --------------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "risk_score",
        "inventory_units",
        "forecast_8w_units",
        "weeks_of_cover",
        "stockout_gap_units",
        "excess_inventory_units",
        "recommended_replenishment_units",
    ]

    for column in numeric_columns:

        if column in risk_data.columns:

            risk_data[column] = pd.to_numeric(
                risk_data[column],
                errors="coerce",
            )

    return risk_data


# ============================================================
# 6. LOAD FORECAST DATA
# ============================================================

def load_forecast_data() -> pd.DataFrame:
    """
    Load forecast data.

    Source:
        data/processed/forecast_output.csv
    """

    if not FORECAST_FILE.exists():

        raise FileNotFoundError(
            f"Forecast file not found: {FORECAST_FILE}"
        )

    forecast_data = pd.read_csv(
        FORECAST_FILE
    )

    forecast_data = clean_columns(
        forecast_data
    )

    forecast_data = clean_sku_column(
        forecast_data
    )

    # --------------------------------------------------------
    # Convert forecast date
    # --------------------------------------------------------

    if "forecast_date" in forecast_data.columns:

        forecast_data["forecast_date"] = (
            pd.to_datetime(
                forecast_data["forecast_date"],
                errors="coerce",
            )
        )

    # --------------------------------------------------------
    # Convert forecast units
    # --------------------------------------------------------

    if "forecast_units" in forecast_data.columns:

        forecast_data["forecast_units"] = (
            pd.to_numeric(
                forecast_data["forecast_units"],
                errors="coerce",
            )
        )

        # Prevent negative forecast values
        forecast_data["forecast_units"] = (
            forecast_data["forecast_units"]
            .clip(lower=0)
        )

    return forecast_data


# ============================================================
# 7. LOAD ALL DATA
# ============================================================

def load_all_data():
    """
    Load both risk and forecast datasets.
    """

    global risk
    global forecast

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    try:

        risk = load_risk_data()

        print(
            f"Risk dataset loaded: "
            f"{len(risk):,} rows"
        )

    except Exception as error:

        print(
            "WARNING: Risk dataset could not "
            "be loaded."
        )

        print(error)

        risk = pd.DataFrame()

    # --------------------------------------------------------
    # Forecast
    # --------------------------------------------------------

    try:

        forecast = load_forecast_data()

        print(
            f"Forecast dataset loaded: "
            f"{len(forecast):,} rows"
        )

    except Exception as error:

        print(
            "WARNING: Forecast dataset could not "
            "be loaded."
        )

        print(error)

        forecast = pd.DataFrame()


# ============================================================
# 8. LOAD DATA WHEN API STARTS
# ============================================================

load_all_data()


# ============================================================
# 9. REQUIRED COLUMN VALIDATION
# ============================================================

def validate_risk_columns():

    required_columns = [
        "sku_id",
        "risk_score",
        "risk_level",
        "inventory_units",
        "forecast_8w_units",
        "weeks_of_cover",
    ]

    missing = [
        column
        for column in required_columns
        if column not in risk.columns
    ]

    return missing


def validate_forecast_columns():

    required_columns = [
        "sku_id",
        "forecast_date",
        "forecast_units",
    ]

    missing = [
        column
        for column in required_columns
        if column not in forecast.columns
    ]

    return missing


# ============================================================
# 10. ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {

        "project": "PROJECT FORESIGHT",

        "service": (
            "Demand & Inventory Intelligence API"
        ),

        "status": "running",

        "version": "2.0.0",

        "endpoints": [

            "/",
            "/health",
            "/info",
            "/score/{sku_id}",
            "/forecast/{sku_id}",
            "/scores",
            "/reload",
            "/docs",

        ],
    }


# ============================================================
# 11. HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    risk_available = not risk.empty

    forecast_available = not forecast.empty

    if risk_available and forecast_available:

        status = "healthy"

    elif risk_available or forecast_available:

        status = "degraded"

    else:

        status = "unhealthy"

    return {

        "status": status,

        "risk_dataset": (
            "available"
            if risk_available
            else "unavailable"
        ),

        "forecast_dataset": (
            "available"
            if forecast_available
            else "unavailable"
        ),

        "risk_rows": len(risk),

        "forecast_rows": len(forecast),

    }


# ============================================================
# 12. SCORE SINGLE SKU
# ============================================================

@app.get("/score/{sku_id}")
def score_sku(sku_id: str):

    # --------------------------------------------------------
    # Validate datasets
    # --------------------------------------------------------

    if risk.empty:

        raise HTTPException(
            status_code=500,
            detail=(
                "Risk dataset is unavailable."
            ),
        )

    if forecast.empty:

        raise HTTPException(
            status_code=500,
            detail=(
                "Forecast dataset is unavailable."
            ),
        )

    # --------------------------------------------------------
    # Validate risk columns
    # --------------------------------------------------------

    missing_risk_columns = (
        validate_risk_columns()
    )

    if missing_risk_columns:

        raise HTTPException(
            status_code=500,
            detail=(
                "Risk dataset is missing columns: "
                + ", ".join(
                    missing_risk_columns
                )
            ),
        )

    # --------------------------------------------------------
    # Validate forecast columns
    # --------------------------------------------------------

    missing_forecast_columns = (
        validate_forecast_columns()
    )

    if missing_forecast_columns:

        raise HTTPException(
            status_code=500,
            detail=(
                "Forecast dataset is missing columns: "
                + ", ".join(
                    missing_forecast_columns
                )
            ),
        )

    # --------------------------------------------------------
    # Clean SKU
    # --------------------------------------------------------

    sku_id = sku_id.strip()

    if not sku_id:

        raise HTTPException(
            status_code=400,
            detail="SKU ID cannot be empty.",
        )

    # --------------------------------------------------------
    # Find forecast
    # --------------------------------------------------------

    forecast_data = forecast[
        forecast["sku_id"] == sku_id
    ].copy()

    # --------------------------------------------------------
    # Find risk
    # --------------------------------------------------------

    risk_data = risk[
        risk["sku_id"] == sku_id
    ].copy()

    # --------------------------------------------------------
    # SKU not found
    # --------------------------------------------------------

    if (
        forecast_data.empty
        and risk_data.empty
    ):

        raise HTTPException(
            status_code=404,
            detail=f"SKU {sku_id} not found.",
        )

    # ========================================================
    # FORECAST PROCESSING
    # ========================================================

    forecast_records = []

    if not forecast_data.empty:

        # Sort by date
        forecast_data = (
            forecast_data
            .sort_values(
                "forecast_date"
            )
        )

        preferred_columns = [
            "forecast_date",
            "forecast_units",
        ]

        available_columns = [
            column
            for column in preferred_columns
            if column in forecast_data.columns
        ]

        forecast_records = (
            records_to_json_safe(
                forecast_data[
                    available_columns
                ]
            )
        )

    # ========================================================
    # RISK PROCESSING
    # ========================================================

    risk_record = None

    if not risk_data.empty:

        risk_record = (
            records_to_json_safe(
                risk_data.head(1)
            )[0]
        )

    # ========================================================
    # FORECAST TOTAL
    # ========================================================

    forecast_total = 0.0

    if (
        not forecast_data.empty
        and "forecast_units"
        in forecast_data.columns
    ):

        forecast_total = (
            pd.to_numeric(
                forecast_data[
                    "forecast_units"
                ],
                errors="coerce",
            )
            .fillna(0)
            .clip(lower=0)
            .sum()
        )

    # ========================================================
    # RETURN RESPONSE
    # ========================================================

    return {

        "sku_id": sku_id,

        "forecast_8_week_total": float(
            forecast_total
        ),

        "forecast": forecast_records,

        "risk": risk_record,

    }


# ============================================================
# 13. FORECAST-ONLY ENDPOINT
# ============================================================

@app.get("/forecast/{sku_id}")
def forecast_sku(sku_id: str):

    # --------------------------------------------------------
    # Validate dataset
    # --------------------------------------------------------

    if forecast.empty:

        raise HTTPException(
            status_code=500,
            detail=(
                "Forecast dataset is unavailable."
            ),
        )

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    missing_columns = (
        validate_forecast_columns()
    )

    if missing_columns:

        raise HTTPException(
            status_code=500,
            detail=(
                "Forecast dataset is missing columns: "
                + ", ".join(
                    missing_columns
                )
            ),
        )

    # --------------------------------------------------------
    # Clean SKU
    # --------------------------------------------------------

    sku_id = sku_id.strip()

    if not sku_id:

        raise HTTPException(
            status_code=400,
            detail="SKU ID cannot be empty.",
        )

    # --------------------------------------------------------
    # Find SKU
    # --------------------------------------------------------

    sku_forecast = forecast[
        forecast["sku_id"] == sku_id
    ].copy()

    if sku_forecast.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Forecast not found for SKU "
                f"{sku_id}."
            ),
        )

    # --------------------------------------------------------
    # Sort forecast
    # --------------------------------------------------------

    sku_forecast = (
        sku_forecast
        .sort_values(
            "forecast_date"
        )
    )

    # --------------------------------------------------------
    # Select columns
    # --------------------------------------------------------

    preferred_columns = [
        "forecast_date",
        "forecast_units",
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in sku_forecast.columns
    ]

    records = records_to_json_safe(
        sku_forecast[
            available_columns
        ]
    )

    # --------------------------------------------------------
    # Calculate total
    # --------------------------------------------------------

    total_forecast = (
        pd.to_numeric(
            sku_forecast[
                "forecast_units"
            ],
            errors="coerce",
        )
        .fillna(0)
        .clip(lower=0)
        .sum()
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "sku_id": sku_id,

        "forecast_8_week_total": float(
            total_forecast
        ),

        "forecast": records,

    }


# ============================================================
# 14. BATCH SCORING
# ============================================================

@app.get("/scores")
def all_scores():

    if risk.empty:

        raise HTTPException(
            status_code=500,
            detail=(
                "Risk dataset is unavailable."
            ),
        )

    missing_columns = (
        validate_risk_columns()
    )

    if missing_columns:

        raise HTTPException(
            status_code=500,
            detail=(
                "Risk dataset is missing columns: "
                + ", ".join(
                    missing_columns
                )
            ),
        )

    return records_to_json_safe(
        risk
    )


# ============================================================
# 15. RELOAD DATA
# ============================================================

@app.post("/reload")
def reload_data():

    load_all_data()

    return {

        "status": "reloaded",

        "risk_rows": len(risk),

        "forecast_rows": len(forecast),

        "risk_file_exists": (
            RISK_FILE.exists()
        ),

        "forecast_file_exists": (
            FORECAST_FILE.exists()
        ),

    }


# ============================================================
# 16. API INFORMATION
# ============================================================

@app.get("/info")
def api_info():

    return {

        "project": "PROJECT FORESIGHT",

        "description": (
            "Demand & Inventory Intelligence "
            "Decision Support System"
        ),

        "version": "2.0.0",

        "base_directory": str(
            BASE_DIR
        ),

        "processed_directory": str(
            PROCESSED_DIR
        ),

        "risk_file": str(
            RISK_FILE
        ),

        "forecast_file": str(
            FORECAST_FILE
        ),

        "risk_file_exists": (
            RISK_FILE.exists()
        ),

        "forecast_file_exists": (
            FORECAST_FILE.exists()
        ),

        "risk_records": len(risk),

        "forecast_records": len(forecast),

        "risk_columns": (
            risk.columns.tolist()
            if not risk.empty
            else []
        ),

        "forecast_columns": (
            forecast.columns.tolist()
            if not forecast.empty
            else []
        ),

        "endpoints": {

            "home": "/",

            "health": "/health",

            "info": "/info",

            "single_sku": (
                "/score/{sku_id}"
            ),

            "forecast": (
                "/forecast/{sku_id}"
            ),

            "batch_scores": "/scores",

            "reload": "/reload",

            "swagger": "/docs",

        },

    }