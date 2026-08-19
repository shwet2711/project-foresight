# ============================================================
# PROJECT FORESIGHT
# FastAPI Client
#
# Connects Streamlit frontend to FastAPI backend
#
# FastAPI:
# http://127.0.0.1:8000
# ============================================================

import requests


# ============================================================
# 1. API CONFIGURATION
# ============================================================

API_BASE_URL = "http://127.0.0.1:8000"

DEFAULT_TIMEOUT = 10


# ============================================================
# 2. GENERIC GET REQUEST
# ============================================================

def _get(endpoint: str, timeout: int = DEFAULT_TIMEOUT):

    try:

        response = requests.get(
            f"{API_BASE_URL}{endpoint}",
            timeout=timeout,
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.HTTPError as error:

        try:
            detail = response.json().get(
                "detail",
                str(error),
            )
        except Exception:
            detail = str(error)

        return {
            "error": detail,
            "status_code": response.status_code,
        }

    except requests.exceptions.ConnectionError:

        return {
            "error": (
                "Unable to connect to FastAPI. "
                "Make sure FastAPI is running on "
                "http://127.0.0.1:8000"
            ),
            "status_code": None,
        }

    except requests.exceptions.Timeout:

        return {
            "error": "FastAPI request timed out.",
            "status_code": None,
        }

    except requests.exceptions.RequestException as error:

        return {
            "error": str(error),
            "status_code": None,
        }


# ============================================================
# 3. HEALTH CHECK
# ============================================================

def check_api_health():

    return _get(
        "/health",
        timeout=5,
    )


# ============================================================
# 4. API INFORMATION
# ============================================================

def get_api_info():

    return _get(
        "/info",
        timeout=5,
    )


# ============================================================
# 5. GET ALL RISK SCORES
# ============================================================

def get_all_scores():

    return _get(
        "/scores",
        timeout=15,
    )


# ============================================================
# 6. GET SINGLE SKU SCORE + FORECAST
# ============================================================

def get_sku_score(sku_id: str):

    if not sku_id:

        return {
            "error": "SKU ID cannot be empty.",
            "status_code": 400,
        }

    sku_id = str(sku_id).strip()

    if not sku_id:

        return {
            "error": "SKU ID cannot be empty.",
            "status_code": 400,
        }

    return _get(
        f"/score/{sku_id}",
        timeout=10,
    )


# ============================================================
# 7. GET FORECAST FOR SINGLE SKU
# ============================================================

def get_sku_forecast(sku_id: str):

    if not sku_id:

        return {
            "error": "SKU ID cannot be empty.",
            "status_code": 400,
        }

    sku_id = str(sku_id).strip()

    if not sku_id:

        return {
            "error": "SKU ID cannot be empty.",
            "status_code": 400,
        }

    return _get(
        f"/forecast/{sku_id}",
        timeout=10,
    )


# ============================================================
# 8. RELOAD FASTAPI DATA
# ============================================================

def reload_api_data():

    try:

        response = requests.post(
            f"{API_BASE_URL}/reload",
            timeout=15,
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.HTTPError as error:

        try:
            detail = response.json().get(
                "detail",
                str(error),
            )
        except Exception:
            detail = str(error)

        return {
            "error": detail,
            "status_code": response.status_code,
        }

    except requests.exceptions.ConnectionError:

        return {
            "error": (
                "Unable to connect to FastAPI. "
                "Make sure the API is running."
            ),
            "status_code": None,
        }

    except requests.exceptions.Timeout:

        return {
            "error": "FastAPI reload request timed out.",
            "status_code": None,
        }

    except requests.exceptions.RequestException as error:

        return {
            "error": str(error),
            "status_code": None,
        }


# ============================================================
# 9. CHECK WHETHER API RESPONSE CONTAINS ERROR
# ============================================================

def api_has_error(data):

    return (
        isinstance(data, dict)
        and "error" in data
    )