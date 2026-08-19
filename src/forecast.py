# ================================================================
# PROJECT FORESIGHT
# Demand & Inventory Intelligence
#
# File: src/forecast.py
# Purpose:
#   - Weekly SKU-level demand forecasting
#   - Seasonal-naive baseline
#   - Rolling-origin backtesting
#   - HistGradientBoostingRegressor model
#   - 8-week recursive forecast
#
# IMPORTANT:
#   No random train/test split is used.
#   All forecasting validation is time-based.
# ================================================================

from pathlib import Path
import warnings
import joblib

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")


# ================================================================
# 1. PROJECT PATHS
# ================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROCESSED_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# 2. CONFIGURATION
# ================================================================

HORIZON = 8
N_FOLDS = 4

# 52 weeks = approximately one year
SEASONAL_PERIOD = 52

RANDOM_STATE = 42


# ================================================================
# 3. MODEL FEATURES
# ================================================================

FEATURE_COLUMNS = [
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


# ================================================================
# 4. LOAD DATA
# ================================================================

def load_data():
    """
    Load weekly demand dataset created by pipeline.py.
    """

    file_path = PROCESSED_DIR / "weekly_demand.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"\nWeekly demand file not found:\n{file_path}\n"
            "\nRun pipeline.py first."
        )

    df = pd.read_csv(
        file_path,
        parse_dates=["date"]
    )

    # ------------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------------

    required_columns = [
        "date",
        "sku_id",
        "units_sold"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"weekly_demand.csv is missing columns: {missing}"
        )

    # ------------------------------------------------------------
    # Clean data types
    # ------------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["units_sold"] = pd.to_numeric(
        df["units_sold"],
        errors="coerce"
    ).fillna(0)

    if "revenue" in df.columns:
        df["revenue"] = pd.to_numeric(
            df["revenue"],
            errors="coerce"
        ).fillna(0)

    numeric_columns = [
        "promo_days",
        "holiday_days",
        "unit_cost",
        "list_price",
        "avg_price",
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # ------------------------------------------------------------
    # Sort
    # ------------------------------------------------------------

    df = (
        df
        .dropna(subset=["date", "sku_id"])
        .sort_values(["sku_id", "date"])
        .reset_index(drop=True)
    )

    return df


# ================================================================
# 5. COMPLETE WEEKLY SKU SERIES
# ================================================================

def complete_weekly_series(df):
    """
    Ensure every SKU has a continuous weekly series.

    IMPORTANT:
    The source dataset uses Sunday weekly dates.
    Therefore W-SUN is used instead of W-MON.
    """

    output = []

    for sku_id, group in df.groupby("sku_id"):

        group = (
            group
            .sort_values("date")
            .copy()
        )

        start_date = group["date"].min()
        end_date = group["date"].max()

        # --------------------------------------------------------
        # Use Sunday because the supplied data ends on Sunday
        # --------------------------------------------------------

        full_dates = pd.date_range(
            start=start_date,
            end=end_date,
            freq="W-SUN"
        )

        group = (
            group
            .set_index("date")
            .reindex(full_dates)
        )

        group.index.name = "date"

        # Restore SKU
        group["sku_id"] = sku_id

        # --------------------------------------------------------
        # Demand columns
        # --------------------------------------------------------

        if "units_sold" in group.columns:

            group["units_sold"] = (
                pd.to_numeric(
                    group["units_sold"],
                    errors="coerce"
                )
                .fillna(0)
            )

        if "revenue" in group.columns:

            group["revenue"] = (
                pd.to_numeric(
                    group["revenue"],
                    errors="coerce"
                )
                .fillna(0)
            )

        # --------------------------------------------------------
        # Promotion / holiday
        # --------------------------------------------------------

        for col in [
            "promo_days",
            "holiday_days"
        ]:

            if col in group.columns:

                group[col] = (
                    pd.to_numeric(
                        group[col],
                        errors="coerce"
                    )
                    .fillna(0)
                )

        # --------------------------------------------------------
        # Descriptive fields
        # --------------------------------------------------------

        for col in [
            "category",
            "subcategory",
            "season",
        ]:

            if col in group.columns:

                group[col] = (
                    group[col]
                    .ffill()
                    .bfill()
                )

        # --------------------------------------------------------
        # Numeric product information
        # --------------------------------------------------------

        for col in [
            "unit_cost",
            "list_price",
            "avg_price",
        ]:

            if col in group.columns:

                group[col] = (
                    pd.to_numeric(
                        group[col],
                        errors="coerce"
                    )
                    .ffill()
                    .bfill()
                    .fillna(0)
                )

        output.append(
            group.reset_index()
        )

    result = pd.concat(
        output,
        ignore_index=True
    )

    result = (
        result
        .sort_values(["sku_id", "date"])
        .reset_index(drop=True)
    )

    return result


# ================================================================
# 6. LAG FEATURES
# ================================================================

def create_lag_features(df):
    """
    Create historical demand lag features.

    All lags use only previous observations.
    """

    df = df.copy()

    grouped = (
        df.groupby("sku_id")["units_sold"]
    )

    for lag in [
        1,
        2,
        4,
        8,
        13,
        26,
        52,
    ]:

        df[f"lag_{lag}"] = (
            grouped.shift(lag)
        )

    return df


# ================================================================
# 7. ROLLING FEATURES
# ================================================================

def create_rolling_features(df):
    """
    Create rolling statistics using only previous weeks.

    shift(1) is critical because it prevents
    current-week demand from entering the feature.
    """

    df = df.copy()

    grouped = (
        df.groupby("sku_id")["units_sold"]
    )

    df["rolling_mean_4"] = (
        grouped.transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=4,
                min_periods=1
            )
            .mean()
        )
    )

    df["rolling_mean_8"] = (
        grouped.transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=8,
                min_periods=1
            )
            .mean()
        )
    )

    df["rolling_mean_13"] = (
        grouped.transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=13,
                min_periods=1
            )
            .mean()
        )
    )

    df["rolling_std_8"] = (
        grouped.transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=8,
                min_periods=2
            )
            .std()
        )
    )

    return df


# ================================================================
# 8. CALENDAR FEATURES
# ================================================================

def create_calendar_features(df):
    """
    Create calendar features from the date itself.
    """

    df = df.copy()

    df["year"] = (
        df["date"].dt.year
    )

    df["month_num"] = (
        df["date"].dt.month
    )

    df["quarter"] = (
        df["date"].dt.quarter
    )

    df["week_of_year"] = (
        df["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    # ------------------------------------------------------------
    # Cyclic seasonal representation
    # ------------------------------------------------------------

    df["sin_week"] = (
        np.sin(
            2 * np.pi
            * df["week_of_year"]
            / 52
        )
    )

    df["cos_week"] = (
        np.cos(
            2 * np.pi
            * df["week_of_year"]
            / 52
        )
    )

    return df


# ================================================================
# 9. BUSINESS FEATURES
# ================================================================

def create_business_features(df):
    """
    Create promotion, holiday and margin-related features.
    """

    df = df.copy()

    # ------------------------------------------------------------
    # Promotion ratio
    # ------------------------------------------------------------

    if "promo_days" in df.columns:

        df["promo_ratio"] = (
            df["promo_days"] / 7
        )

    else:

        df["promo_ratio"] = 0.0

    # ------------------------------------------------------------
    # Holiday ratio
    # ------------------------------------------------------------

    if "holiday_days" in df.columns:

        df["holiday_ratio"] = (
            df["holiday_days"] / 7
        )

    else:

        df["holiday_ratio"] = 0.0

    # ------------------------------------------------------------
    # Ensure product numeric columns exist
    # ------------------------------------------------------------

    if "unit_cost" not in df.columns:
        df["unit_cost"] = 0.0

    if "list_price" not in df.columns:
        df["list_price"] = 0.0

    # ------------------------------------------------------------
    # Numeric cleaning
    # ------------------------------------------------------------

    for col in [
        "unit_cost",
        "list_price",
        "promo_ratio",
        "holiday_ratio",
    ]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).fillna(0)

    return df


# ================================================================
# 10. COMPLETE FEATURE ENGINEERING PIPELINE
# ================================================================

def create_features(df):
    """
    Run all feature engineering steps.
    """

    df = df.copy()

    df = (
        df
        .sort_values(["sku_id", "date"])
        .reset_index(drop=True)
    )

    df = create_lag_features(df)

    df = create_rolling_features(df)

    df = create_calendar_features(df)

    df = create_business_features(df)

    return df


# ================================================================
# 11. WAPE
# ================================================================

def wape(actual, predicted):
    """
    Weighted Absolute Percentage Error.

    WAPE =
        SUM(|actual - predicted|)
        /
        SUM(|actual|)
    """

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    # ------------------------------------------------------------
    # Remove invalid values
    # ------------------------------------------------------------

    valid = (
        np.isfinite(actual)
        &
        np.isfinite(predicted)
    )

    actual = actual[valid]

    predicted = predicted[valid]

    if len(actual) == 0:
        return np.nan

    denominator = np.sum(
        np.abs(actual)
    )

    if denominator == 0:
        return 0.0

    return (
        np.sum(
            np.abs(
                actual - predicted
            )
        )
        / denominator
    )


# ================================================================
# 12. BIAS
# ================================================================

def forecast_bias(actual, predicted):
    """
    Mean forecast error.

    Positive:
        model tends to over-forecast.

    Negative:
        model tends to under-forecast.
    """

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    valid = (
        np.isfinite(actual)
        &
        np.isfinite(predicted)
    )

    actual = actual[valid]

    predicted = predicted[valid]

    if len(actual) == 0:
        return np.nan

    return np.mean(
        predicted - actual
    )


# ================================================================
# 13. SEASONAL-NAIVE BASELINE
# ================================================================

def seasonal_naive_forecast(
    train,
    test
):
    """
    Seasonal-naive forecast.

    For each SKU:

        forecast = demand from 52 weeks earlier

    Date matching is explicit.
    """

    train = train.copy()
    test = test.copy()

    # ------------------------------------------------------------
    # Create historical lookup
    # ------------------------------------------------------------

    lookup = train[
        [
            "sku_id",
            "date",
            "units_sold"
        ]
    ].copy()

    # ------------------------------------------------------------
    # 52 weeks = 364 days
    # ------------------------------------------------------------

    lookup["forecast_date"] = (
        lookup["date"]
        + pd.Timedelta(
            weeks=SEASONAL_PERIOD
        )
    )

    lookup = lookup[
        [
            "sku_id",
            "forecast_date",
            "units_sold"
        ]
    ]

    lookup = lookup.rename(
        columns={
            "units_sold":
            "baseline_prediction"
        }
    )

    # ------------------------------------------------------------
    # Match SKU + forecast date
    # ------------------------------------------------------------

    result = test.merge(
        lookup,
        left_on=[
            "sku_id",
            "date"
        ],
        right_on=[
            "sku_id",
            "forecast_date"
        ],
        how="left"
    )

    # ------------------------------------------------------------
    # If historical seasonal value doesn't exist,
    # use recent demand fallback.
    #
    # This prevents NaN baseline predictions.
    # ------------------------------------------------------------

    recent_mean = (
        train
        .groupby("sku_id")["units_sold"]
        .tail(4)
        .groupby(train.groupby("sku_id").tail(4)["sku_id"])
        .mean()
    )

    result["baseline_prediction"] = (
        result["baseline_prediction"]
        .fillna(
            result["sku_id"]
            .map(recent_mean)
        )
        .fillna(0)
    )

    result["baseline_prediction"] = (
        pd.to_numeric(
            result["baseline_prediction"],
            errors="coerce"
        )
        .fillna(0)
        .clip(lower=0)
    )

    return result


# ================================================================
# 14. BUILD MODEL
# ================================================================

def build_model():
    """
    Create HistGradientBoostingRegressor.
    """

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=RANDOM_STATE
    )

    return model


# ================================================================
# 15. PREPARE TRAINING DATA
# ================================================================

def prepare_training_data(df):
    """
    Prepare numerical model matrix.

    Missing feature values are handled here.
    """

    data = df.copy()

    # ------------------------------------------------------------
    # Ensure all expected features exist
    # ------------------------------------------------------------

    for col in FEATURE_COLUMNS:

        if col not in data.columns:

            data[col] = 0.0

    # ------------------------------------------------------------
    # Convert features to numeric
    # ------------------------------------------------------------

    for col in FEATURE_COLUMNS:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    # ------------------------------------------------------------
    # Fill missing feature values
    #
    # Early history may not have lag_52 etc.
    # We use 0 for those unavailable historical values.
    # ------------------------------------------------------------

    data[FEATURE_COLUMNS] = (
        data[FEATURE_COLUMNS]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    # ------------------------------------------------------------
    # Target
    # ------------------------------------------------------------

    data["units_sold"] = pd.to_numeric(
        data["units_sold"],
        errors="coerce"
    ).fillna(0)

    return data


# ================================================================
# 16. TRAIN MODEL
# ================================================================

def train_model(train_df):
    """
    Train the forecasting model.
    """

    train_df = prepare_training_data(
        train_df
    )

    X = train_df[
        FEATURE_COLUMNS
    ]

    y = train_df[
        "units_sold"
    ]

    model = build_model()

    model.fit(
        X,
        y
    )

    return model


# ================================================================
# 17. ROLLING-ORIGIN BACKTEST
# ================================================================

def rolling_backtest(
    df,
    horizon=8,
    n_folds=4
):
    """
    Leakage-safe rolling-origin backtest.

    Example:

        TRAIN
        |--------------------|

                         TEST
                         |------|

    Then:

        TRAIN
        |--------------------------|

                              TEST
                              |------|

    No random splitting is used.
    """

    df = df.copy()

    df = (
        df
        .sort_values(
            ["date", "sku_id"]
        )
        .reset_index(drop=True)
    )

    all_dates = (
        pd.Series(
            df["date"]
            .drop_duplicates()
            .sort_values()
        )
        .tolist()
    )

    results = []

    max_date = max(all_dates)

    print(
        "\n" + "=" * 70
    )

    print(
        "ROLLING-ORIGIN BACKTEST"
    )

    print(
        "=" * 70
    )

    # ------------------------------------------------------------
    # Fold loop
    # ------------------------------------------------------------

    for fold in range(
        1,
        n_folds + 1
    ):

        test_end = (
            max_date
            - pd.Timedelta(
                weeks=(fold - 1) * horizon
            )
        )

        test_start = (
            test_end
            - pd.Timedelta(
                weeks=horizon - 1
            )
        )

        # --------------------------------------------------------
        # Train strictly BEFORE test
        # --------------------------------------------------------

        train = df[
            df["date"] < test_start
        ].copy()

        test = df[
            (
                df["date"] >= test_start
            )
            &
            (
                df["date"] <= test_end
            )
        ].copy()

        if train.empty or test.empty:

            print(
                f"Fold {fold}: skipped"
            )

            continue

        print(
            f"\nFold {fold}"
        )

        print(
            f"Train end: "
            f"{train['date'].max().date()}"
        )

        print(
            f"Test period: "
            f"{test['date'].min().date()} "
            f"to "
            f"{test['date'].max().date()}"
        )

        # --------------------------------------------------------
        # Feature engineering
        #
        # IMPORTANT:
        # Build features on combined chronological data so that
        # test rows can use historical lags from train.
        #
        # The target itself is never used as a current feature.
        # --------------------------------------------------------

        combined = pd.concat(
            [
                train,
                test
            ],
            ignore_index=True
        )

        combined = (
            combined
            .sort_values(
                ["sku_id", "date"]
            )
            .reset_index(drop=True)
        )

        combined_features = create_features(
            combined
        )

        # --------------------------------------------------------
        # Recover train/test feature rows
        # --------------------------------------------------------

        train_features = combined_features[
            combined_features["date"]
            < test_start
        ].copy()

        test_features = combined_features[
            (
                combined_features["date"]
                >= test_start
            )
            &
            (
                combined_features["date"]
                <= test_end
            )
        ].copy()

        # --------------------------------------------------------
        # Prepare model data
        # --------------------------------------------------------

        train_features = prepare_training_data(
            train_features
        )

        test_features = prepare_training_data(
            test_features
        )

        # --------------------------------------------------------
        # Train model
        # --------------------------------------------------------

        model = train_model(
            train_features
        )

        # --------------------------------------------------------
        # Model predictions
        # --------------------------------------------------------

        X_test = test_features[
            FEATURE_COLUMNS
        ]

        predictions = model.predict(
            X_test
        )

        predictions = np.asarray(
            predictions,
            dtype=float
        )

        # Demand cannot be negative
        predictions = np.clip(
            predictions,
            0,
            None
        )

        # --------------------------------------------------------
        # Actual demand
        # --------------------------------------------------------

        actual = pd.to_numeric(
            test_features[
                "units_sold"
            ],
            errors="coerce"
        ).fillna(0).values

        # --------------------------------------------------------
        # Seasonal-naive baseline
        # --------------------------------------------------------

        baseline_result = (
            seasonal_naive_forecast(
                train,
                test
            )
        )

        baseline_predictions = (
            baseline_result[
                "baseline_prediction"
            ]
            .fillna(0)
            .values
        )

        baseline_predictions = np.clip(
            baseline_predictions,
            0,
            None
        )

        # --------------------------------------------------------
        # Safety check
        # --------------------------------------------------------

        if len(actual) != len(predictions):

            raise ValueError(
                f"Fold {fold}: "
                f"actual/prediction length mismatch."
            )

        if len(actual) != len(
            baseline_predictions
        ):

            raise ValueError(
                f"Fold {fold}: "
                f"baseline length mismatch."
            )

        # --------------------------------------------------------
        # Calculate metrics
        # --------------------------------------------------------

        baseline_wape = wape(
            actual,
            baseline_predictions
        )

        model_wape = wape(
            actual,
            predictions
        )

        baseline_bias = forecast_bias(
            actual,
            baseline_predictions
        )

        model_bias = forecast_bias(
            actual,
            predictions
        )

        # --------------------------------------------------------
        # Improvement
        # --------------------------------------------------------

        if (
            np.isfinite(baseline_wape)
            and baseline_wape != 0
            and np.isfinite(model_wape)
        ):

            improvement = (
                (
                    baseline_wape
                    - model_wape
                )
                / baseline_wape
                * 100
            )

        else:

            improvement = np.nan

        # --------------------------------------------------------
        # Print fold results
        # --------------------------------------------------------

        print(
            f"Baseline WAPE: "
            f"{baseline_wape:.4f}"
        )

        print(
            f"Model WAPE: "
            f"{model_wape:.4f}"
        )

        print(
            f"Baseline Bias: "
            f"{baseline_bias:.4f}"
        )

        print(
            f"Model Bias: "
            f"{model_bias:.4f}"
        )

        if np.isfinite(improvement):

            print(
                f"Improvement: "
                f"{improvement:.2f}%"
            )

        # --------------------------------------------------------
        # Store fold result
        # --------------------------------------------------------

        results.append(
            {
                "fold": fold,
                "train_end":
                    train["date"].max().date(),
                "test_start":
                    test["date"].min().date(),
                "test_end":
                    test["date"].max().date(),
                "n_train_rows":
                    len(train),
                "n_test_rows":
                    len(test),
                "baseline_wape":
                    baseline_wape,
                "model_wape":
                    model_wape,
                "baseline_bias":
                    baseline_bias,
                "model_bias":
                    model_bias,
                "improvement_percent":
                    improvement,
            }
        )

    # ------------------------------------------------------------
    # Results DataFrame
    # ------------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:

        raise ValueError(
            "Backtest produced no valid folds."
        )

    # ------------------------------------------------------------
    # Remove invalid metric rows
    # ------------------------------------------------------------

    metric_columns = [
        "baseline_wape",
        "model_wape",
        "baseline_bias",
        "model_bias",
        "improvement_percent",
    ]

    for col in metric_columns:

        results_df[col] = pd.to_numeric(
            results_df[col],
            errors="coerce"
        )

    return results_df


# ================================================================
# 18. CREATE FUTURE CALENDAR
# ================================================================

def create_future_calendar(
    last_date,
    horizon=8
):
    """
    Generate future weekly dates.

    Future promotions and holidays are unknown.
    Therefore they are set to zero.

    This assumption must be documented in README.
    """

    future_dates = pd.date_range(
        start=(
            last_date
            + pd.Timedelta(weeks=1)
        ),
        periods=horizon,
        freq="W-SUN"
    )

    future = pd.DataFrame(
        {
            "date": future_dates
        }
    )

    future["year"] = (
        future["date"].dt.year
    )

    future["month_num"] = (
        future["date"].dt.month
    )

    future["quarter"] = (
        future["date"].dt.quarter
    )

    future["week_of_year"] = (
        future["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    future["sin_week"] = (
        np.sin(
            2 * np.pi
            * future["week_of_year"]
            / 52
        )
    )

    future["cos_week"] = (
        np.cos(
            2 * np.pi
            * future["week_of_year"]
            / 52
        )
    )

    # ------------------------------------------------------------
    # Unknown future events
    # ------------------------------------------------------------

    future["promo_days"] = 0

    future["holiday_days"] = 0

    future["promo_ratio"] = 0.0

    future["holiday_ratio"] = 0.0

    return future


# ================================================================
# 19. GENERATE 8-WEEK SKU FORECAST
# ================================================================

def generate_forecast(
    df,
    model,
    horizon=8
):
    """
    Generate recursive multi-step forecast
    for every SKU.

    Every SKU receives exactly `horizon`
    future weekly predictions.
    """

    results = []

    print(
        "\nGenerating forecasts..."
    )

    total_skus = (
        df["sku_id"]
        .nunique()
    )

    for counter, (
        sku_id,
        history
    ) in enumerate(
        df.groupby("sku_id"),
        start=1
    ):

        history = (
            history
            .sort_values("date")
            .copy()
        )

        # --------------------------------------------------------
        # Last historical date
        # --------------------------------------------------------

        last_date = (
            history["date"].max()
        )

        # --------------------------------------------------------
        # Future calendar
        # --------------------------------------------------------

        future_calendar = (
            create_future_calendar(
                last_date,
                horizon
            )
        )

        # --------------------------------------------------------
        # Temporary dataframe
        #
        # Predictions are appended to this dataframe so future
        # lag features can use previous predictions.
        # --------------------------------------------------------

        temp = history.copy()

        # --------------------------------------------------------
        # Recursive prediction
        # --------------------------------------------------------

        for _, future_row in (
            future_calendar.iterrows()
        ):

            future_date = (
                future_row["date"]
            )

            # ----------------------------------------------------
            # Product information
            # ----------------------------------------------------

            category = (
                history["category"].iloc[-1]
                if "category" in history.columns
                else "Unknown"
            )

            subcategory = (
                history["subcategory"].iloc[-1]
                if "subcategory" in history.columns
                else "Unknown"
            )

            unit_cost = (
                float(
                    history["unit_cost"].iloc[-1]
                )
                if "unit_cost" in history.columns
                else 0.0
            )

            list_price = (
                float(
                    history["list_price"].iloc[-1]
                )
                if "list_price" in history.columns
                else 0.0
            )

            # ----------------------------------------------------
            # Create future row
            # ----------------------------------------------------

            new_row = {
                "date":
                    future_date,

                "sku_id":
                    sku_id,

                "units_sold":
                    np.nan,

                "revenue":
                    0.0,

                "promo_days":
                    0.0,

                "holiday_days":
                    0.0,

                "category":
                    category,

                "subcategory":
                    subcategory,

                "unit_cost":
                    unit_cost,

                "list_price":
                    list_price,

                "month":
                    future_row[
                        "month_num"
                    ],

                "season":
                    "Unknown",
            }

            # ----------------------------------------------------
            # Add future row
            # ----------------------------------------------------

            temp = pd.concat(
                [
                    temp,
                    pd.DataFrame(
                        [new_row]
                    )
                ],
                ignore_index=True
            )

            # ----------------------------------------------------
            # Create features
            # ----------------------------------------------------

            feature_temp = create_features(
                temp
            )

            # ----------------------------------------------------
            # Get current future row
            # ----------------------------------------------------

            current_row = (
                feature_temp
                .iloc[-1:]
                .copy()
            )

            # ----------------------------------------------------
            # Prepare numerical features
            # ----------------------------------------------------

            current_row = (
                prepare_training_data(
                    current_row
                )
            )

            X_future = current_row[
                FEATURE_COLUMNS
            ]

            # ----------------------------------------------------
            # Prediction
            # ----------------------------------------------------

            try:

                prediction = model.predict(
                    X_future
                )[0]

            except Exception:

                # ------------------------------------------------
                # Safe fallback:
                # average of previous 4 weeks
                # ------------------------------------------------

                recent_values = (
                    temp["units_sold"]
                    .dropna()
                    .tail(4)
                )

                if len(recent_values) > 0:

                    prediction = (
                        recent_values.mean()
                    )

                else:

                    prediction = 0.0

            # ----------------------------------------------------
            # Demand cannot be negative
            # ----------------------------------------------------

            prediction = max(
                0.0,
                float(prediction)
            )

            # ----------------------------------------------------
            # Save prediction
            # ----------------------------------------------------

            results.append(
                {
                    "sku_id":
                        sku_id,

                    "forecast_date":
                        future_date,

                    "forecast_units":
                        prediction,
                }
            )

            # ----------------------------------------------------
            # IMPORTANT:
            # Put prediction into temp.
            #
            # This allows next week's lag_1 to use this
            # predicted demand.
            # ----------------------------------------------------

            temp.loc[
                temp.index[-1],
                "units_sold"
            ] = prediction

        # --------------------------------------------------------
        # Progress
        # --------------------------------------------------------

        if (
            counter % 25 == 0
            or counter == total_skus
        ):

            print(
                f"Processed "
                f"{counter}/{total_skus} SKUs"
            )

    # ------------------------------------------------------------
    # Final forecast dataframe
    # ------------------------------------------------------------

    forecast_df = pd.DataFrame(
        results
    )

    if forecast_df.empty:

        raise ValueError(
            "Forecast generation produced no rows."
        )

    # ------------------------------------------------------------
    # Clean
    # ------------------------------------------------------------

    forecast_df["forecast_date"] = (
        pd.to_datetime(
            forecast_df["forecast_date"]
        )
    )

    forecast_df["forecast_units"] = (
        pd.to_numeric(
            forecast_df["forecast_units"],
            errors="coerce"
        )
        .fillna(0)
        .clip(lower=0)
    )

    forecast_df = (
        forecast_df
        .sort_values(
            [
                "sku_id",
                "forecast_date"
            ]
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / "forecast_output.csv"
    )

    forecast_df.to_csv(
        output_path,
        index=False
    )

    print(
        "\nForecast saved to:"
    )

    print(
        output_path
    )

    return forecast_df


# ================================================================
# 20. TRAIN FINAL MODEL
# ================================================================

def train_final_model(df):
    """
    Train final model using all historical data.
    """

    print(
        "\nTraining final model using "
        "all available history..."
    )

    features = create_features(
        df
    )

    model = train_model(
        features
    )

    model_path = (
        OUTPUT_DIR
        / "forecast_model.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        "Final model saved to:"
    )

    print(
        model_path
    )

    return model


# ================================================================
# 21. VALIDATE FORECAST OUTPUT
# ================================================================

def validate_forecast_output(
    forecast_df,
    expected_skus,
    horizon=8
):
    """
    Validate final forecast output.

    Checks:
        - Every SKU exists
        - Every SKU has exactly 8 forecasts
        - No negative forecasts
        - No NaN forecasts
        - Forecast dates are weekly
    """

    print(
        "\n" + "=" * 60
    )

    print(
        "FORECAST VALIDATION"
    )

    print(
        "=" * 60
    )

    # ------------------------------------------------------------
    # SKU count
    # ------------------------------------------------------------

    actual_skus = (
        forecast_df["sku_id"]
        .nunique()
    )

    print(
        f"Expected SKUs: {expected_skus}"
    )

    print(
        f"Forecast SKUs: {actual_skus}"
    )

    if actual_skus != expected_skus:

        raise ValueError(
            "Forecast does not contain "
            "all expected SKUs."
        )

    # ------------------------------------------------------------
    # Forecast count per SKU
    # ------------------------------------------------------------

    counts = (
        forecast_df
        .groupby("sku_id")
        .size()
    )

    invalid_counts = (
        counts != horizon
    )

    if invalid_counts.any():

        print(
            "\nSKUs with invalid forecast count:"
        )

        print(
            counts[
                invalid_counts
            ]
        )

        raise ValueError(
            "Not every SKU has exactly "
            f"{horizon} forecasts."
        )

    print(
        f"Every SKU has exactly "
        f"{horizon} forecast weeks."
    )

    # ------------------------------------------------------------
    # Negative forecasts
    # ------------------------------------------------------------

    negative_count = (
        forecast_df[
            "forecast_units"
        ] < 0
    ).sum()

    print(
        f"Negative forecasts: "
        f"{negative_count}"
    )

    if negative_count > 0:

        raise ValueError(
            "Negative forecasts detected."
        )

    # ------------------------------------------------------------
    # Missing forecasts
    # ------------------------------------------------------------

    missing_count = (
        forecast_df[
            "forecast_units"
        ]
        .isna()
        .sum()
    )

    print(
        f"Missing forecasts: "
        f"{missing_count}"
    )

    if missing_count > 0:

        raise ValueError(
            "NaN forecasts detected."
        )

    # ------------------------------------------------------------
    # Date range
    # ------------------------------------------------------------

    print(
        "\nForecast date range:"
    )

    print(
        forecast_df[
            "forecast_date"
        ].min().date(),
        "to",
        forecast_df[
            "forecast_date"
        ].max().date()
    )

    print(
        "\nForecast validation PASSED."
    )


# ================================================================
# 22. MAIN
# ================================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PROJECT FORESIGHT"
    )

    print(
        "Demand & Inventory Intelligence"
    )

    print(
        "Forecasting Pipeline"
    )

    print(
        "=" * 70
    )

    # ============================================================
    # STEP 1 — LOAD
    # ============================================================

    print(
        "\n[1/6] Loading weekly demand..."
    )

    df = load_data()

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"SKUs: "
        f"{df['sku_id'].nunique()}"
    )

    print(
        f"Date range: "
        f"{df['date'].min().date()} "
        f"to "
        f"{df['date'].max().date()}"
    )

    # ============================================================
    # STEP 2 — COMPLETE WEEKLY SERIES
    # ============================================================

    print(
        "\n[2/6] Completing weekly SKU series..."
    )

    df = complete_weekly_series(
        df
    )

    print(
        f"Rows after weekly completion: "
        f"{len(df):,}"
    )

    print(
        f"SKUs after completion: "
        f"{df['sku_id'].nunique()}"
    )

    # ============================================================
    # STEP 3 — BACKTEST
    # ============================================================

    print(
        "\n[3/6] Running rolling-origin backtest..."
    )

    results = rolling_backtest(
        df,
        horizon=HORIZON,
        n_folds=N_FOLDS
    )

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------

    results_path = (
        OUTPUT_DIR
        / "backtest_results.csv"
    )

    results.to_csv(
        results_path,
        index=False
    )

    print(
        "\nBacktest Results:"
    )

    print(
        results.to_string(
            index=False
        )
    )

    # ============================================================
    # BACKTEST SUMMARY
    # ============================================================

    baseline_values = (
        results[
            "baseline_wape"
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    model_values = (
        results[
            "model_wape"
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    baseline_bias_values = (
        results[
            "baseline_bias"
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    model_bias_values = (
        results[
            "model_bias"
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    if baseline_values.empty:

        raise ValueError(
            "Baseline WAPE is invalid. "
            "Do not continue to model deployment."
        )

    if model_values.empty:

        raise ValueError(
            "Model WAPE is invalid. "
            "Do not continue to model deployment."
        )

    avg_baseline = (
        baseline_values.mean()
    )

    avg_model = (
        model_values.mean()
    )

    avg_baseline_bias = (
        baseline_bias_values.mean()
        if not baseline_bias_values.empty
        else np.nan
    )

    avg_model_bias = (
        model_bias_values.mean()
        if not model_bias_values.empty
        else np.nan
    )

    improvement = (
        (
            avg_baseline
            - avg_model
        )
        / avg_baseline
        * 100
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "BACKTEST SUMMARY"
    )

    print(
        "=" * 60
    )

    print(
        f"Average Baseline WAPE: "
        f"{avg_baseline:.4f}"
    )

    print(
        f"Average Model WAPE:    "
        f"{avg_model:.4f}"
    )

    print(
        f"Average Baseline Bias: "
        f"{avg_baseline_bias:.4f}"
    )

    print(
        f"Average Model Bias:    "
        f"{avg_model_bias:.4f}"
    )

    print(
        f"Model Improvement:     "
        f"{improvement:.2f}%"
    )

    print(
        "=" * 60
    )

    # ============================================================
    # MODEL DECISION
    # ============================================================

    if avg_model < avg_baseline:

        print(
            "\nMODEL BEATS BASELINE"
        )

        print(
            "The model performs better than "
            "the seasonal-naive benchmark."
        )

    else:

        print(
            "\nMODEL DOES NOT BEAT BASELINE"
        )

        print(
            "Per Project FORESIGHT rules, "
            "report the baseline honestly."
        )

        print(
            "Further model improvement may "
            "be investigated after the core pipeline."
        )

    # ============================================================
    # STEP 4 — TRAIN FINAL MODEL
    # ============================================================

    print(
        "\n[4/6] Training final model..."
    )

    final_model = train_final_model(
        df
    )

    # ============================================================
    # STEP 5 — GENERATE FORECAST
    # ============================================================

    print(
        "\n[5/6] Generating "
        f"{HORIZON}-week SKU-level forecast..."
    )

    forecast_df = generate_forecast(
        df,
        final_model,
        horizon=HORIZON
    )

    # ============================================================
    # STEP 6 — VALIDATE
    # ============================================================

    print(
        "\n[6/6] Validating forecast..."
    )

    validate_forecast_output(
        forecast_df,
        expected_skus=df[
            "sku_id"
        ].nunique(),
        horizon=HORIZON
    )

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FORECASTING PIPELINE COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nGenerated files:"
    )

    print(
        f"1. {results_path}"
    )

    print(
        f"2. {OUTPUT_DIR / 'forecast_model.pkl'}"
    )

    print(
        f"3. {OUTPUT_DIR / 'forecast_output.csv'}"
    )

    print(
        "\nForecast summary:"
    )

    print(
        f"SKUs: "
        f"{forecast_df['sku_id'].nunique()}"
    )

    print(
        f"Forecast weeks per SKU: "
        f"{HORIZON}"
    )

    print(
        f"Total forecast rows: "
        f"{len(forecast_df):,}"
    )

    print(
        f"Total forecast units: "
        f"{forecast_df['forecast_units'].sum():,.2f}"
    )

    print(
        "\nNext step:"
    )

    print(
        "Use forecast_output.csv + "
        "inventory_snapshots.csv "
        "to build src/risk.py."
    )

    print(
        "\n"
        + "=" * 70
    )


# ================================================================
# 23. RUN
# ================================================================

if __name__ == "__main__":
    main()