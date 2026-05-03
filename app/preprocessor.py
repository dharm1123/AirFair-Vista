from __future__ import annotations

import calendar
import math
import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLAB_ROOT = Path("/content/drive/MyDrive/AirFair-Vista")
BASE_PATH = Path(os.environ.get("AIRFAIR_BASE_PATH", COLAB_ROOT if COLAB_ROOT.exists() else PROJECT_ROOT))

MODEL_PATH = str(BASE_PATH / "models" / "new_dataset_flight_price_prediction_pipeline.pkl")
RAW_DATA_PATH = BASE_PATH / "data" / "raw" / "Clean_Dataset.csv"
PROCESSED_DATA_PATH = BASE_PATH / "data" / "processed" / "new_flight_price_feature_engineered.csv"

NEW_MODEL_FEATURES = [
    "airline",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class",
    "duration",
    "days_left",
    "flight_freq",
    "airline_freq",
    "source_city_freq",
    "destination_city_freq",
    "route_freq",
    "duration_x_days_left",
    "duration_sq",
    "days_left_sq",
    "is_business",
    "is_non_stop",
]
MODEL_FEATURES = NEW_MODEL_FEATURES

NEW_CATEGORY_VALUES = {
    "airline": ["AirAsia", "Air_India", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"],
    "source_city": ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"],
    "departure_time": ["Afternoon", "Early_Morning", "Evening", "Late_Night", "Morning", "Night"],
    "stops": ["zero", "one", "two_or_more"],
    "arrival_time": ["Afternoon", "Early_Morning", "Evening", "Late_Night", "Morning", "Night"],
    "destination_city": ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"],
    "class": ["Business", "Economy"],
}

# UI constants aligned to the new dataset while preserving the app structure.
AIRLINES = NEW_CATEGORY_VALUES["airline"][:]
SOURCES = NEW_CATEGORY_VALUES["source_city"][:]
DESTINATIONS = NEW_CATEGORY_VALUES["destination_city"][:]
STOPS = NEW_CATEGORY_VALUES["stops"][:]
CLASSES = NEW_CATEGORY_VALUES["class"][:]
MONTHS = {3: "March", 4: "April", 5: "May", 6: "June"}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

VALID_ROUTES = {(src, dst) for src in SOURCES for dst in DESTINATIONS if src != dst}
VALID_DESTINATIONS = {src: [dst for dst in DESTINATIONS if dst != src] for src in SOURCES}
VALID_AIRLINE_STOPS = {
    "AirAsia": ["zero", "one", "two_or_more"],
    "Air_India": ["zero", "one", "two_or_more"],
    "GO_FIRST": ["zero", "one", "two_or_more"],
    "Indigo": ["zero", "one", "two_or_more"],
    "SpiceJet": ["zero", "one"],
    "Vistara": ["zero", "one", "two_or_more"],
}
MAX_PAX_BY_STOPS = {"zero": 9, "one": 9, "two_or_more": 6}

INDIAN_HOLIDAYS = {
    (3, 25): "Holi",
    (3, 29): "Good Friday",
    (4, 14): "Ambedkar Jayanti / Tamil New Year",
    (4, 17): "Ram Navami",
    (4, 19): "Mahavir Jayanti",
    (5, 1): "Labour Day",
    (5, 23): "Buddha Purnima",
    (6, 5): "Eid ul-Fitr",
}

CITY_COORDS = {
    "Bangalore": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Delhi": (28.7041, 77.1025),
    "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639),
    "Mumbai": (19.0760, 72.8777),
    # Legacy aliases are retained for compatibility with old saved session state.
    "Banglore": (12.9716, 77.5946),
    "New Delhi": (28.6139, 77.2090),
    "Cochin": (9.9312, 76.2673),
}

PRICE_MIN = 1105
PRICE_MAX = 123071
PRICE_AVG = 20890
PRICE_MED = 7425

NEW_AIRLINE_MEAN = {
    "AirAsia": 4091,
    "Air_India": 23507,
    "GO_FIRST": 5652,
    "Indigo": 5324,
    "SpiceJet": 6179,
    "Vistara": 30397,
}
NEW_STOPS_MEAN = {"zero": 9376, "one": 22901, "two_or_more": 14113}
NEW_CLASS_MEAN = {"Economy": 6572, "Business": 52540}
NEW_DEPARTURE_MEAN = {
    "Afternoon": 18179,
    "Early_Morning": 20371,
    "Evening": 21232,
    "Late_Night": 9295,
    "Morning": 21631,
    "Night": 23062,
}
NEW_SOURCE_MEAN = {
    "Bangalore": 21469,
    "Chennai": 21995,
    "Delhi": 18951,
    "Hyderabad": 20156,
    "Kolkata": 21746,
    "Mumbai": 21484,
}
NEW_DEST_MEAN = {
    "Bangalore": 21594,
    "Chennai": 21953,
    "Delhi": 18437,
    "Hyderabad": 20428,
    "Kolkata": 21960,
    "Mumbai": 21373,
}
NEW_ROUTE_MEAN = {
    ("Bangalore", "Chennai"): 23322,
    ("Bangalore", "Delhi"): 17723,
    ("Bangalore", "Hyderabad"): 21226,
    ("Bangalore", "Kolkata"): 23500,
    ("Bangalore", "Mumbai"): 23129,
    ("Chennai", "Bangalore"): 25082,
    ("Chennai", "Delhi"): 18982,
    ("Chennai", "Hyderabad"): 21591,
    ("Chennai", "Kolkata"): 22670,
    ("Chennai", "Mumbai"): 22766,
    ("Delhi", "Bangalore"): 17880,
    ("Delhi", "Chennai"): 19370,
    ("Delhi", "Hyderabad"): 17347,
    ("Delhi", "Kolkata"): 20566,
    ("Delhi", "Mumbai"): 19356,
    ("Hyderabad", "Bangalore"): 21347,
    ("Hyderabad", "Chennai"): 21848,
    ("Hyderabad", "Delhi"): 17244,
    ("Hyderabad", "Kolkata"): 20824,
    ("Hyderabad", "Mumbai"): 20081,
    ("Kolkata", "Bangalore"): 22745,
    ("Kolkata", "Chennai"): 23660,
    ("Kolkata", "Delhi"): 19422,
    ("Kolkata", "Hyderabad"): 21500,
    ("Kolkata", "Mumbai"): 22079,
    ("Mumbai", "Bangalore"): 23148,
    ("Mumbai", "Chennai"): 22782,
    ("Mumbai", "Delhi"): 18725,
    ("Mumbai", "Hyderabad"): 21004,
    ("Mumbai", "Kolkata"): 22379,
}

AIRLINE_TO_NEW = {
    "AirAsia": "AirAsia",
    "Air_India": "Air_India",
    "GO_FIRST": "GO_FIRST",
    "Indigo": "Indigo",
    "SpiceJet": "SpiceJet",
    "Vistara": "Vistara",
    # Legacy aliases kept so older session state still maps safely.
    "Air Asia": "AirAsia",
    "Air India": "Air_India",
    "GoAir": "GO_FIRST",
    "IndiGo": "Indigo",
    "Jet Airways": "Vistara",
    "Jet Airways Business": "Vistara",
    "Multiple Carriers": "Air_India",
    "Multiple Carriers Premium Economy": "Air_India",
    "TruJet": "Indigo",
    "Vistara Premium Economy": "Vistara",
}
CITY_TO_NEW = {
    "Banglore": "Bangalore",
    "Bangalore": "Bangalore",
    "Chennai": "Chennai",
    "Cochin": "Chennai",
    "Delhi": "Delhi",
    "New Delhi": "Delhi",
    "Hyderabad": "Hyderabad",
    "Kolkata": "Kolkata",
    "Mumbai": "Mumbai",
}
STOPS_TO_NEW = {
    "zero": "zero",
    "one": "one",
    "two_or_more": "two_or_more",
    "non-stop": "zero",
    "1 stop": "one",
    "2 stops": "two_or_more",
    "3 stops": "two_or_more",
    "4 stops": "two_or_more",
}

AIRLINE_MEAN_PRICE = {airline: NEW_AIRLINE_MEAN.get(airline, PRICE_AVG) for airline in AIRLINES}
AIRLINE_MEAN = AIRLINE_MEAN_PRICE.copy()
STOPS_MEAN = {old: NEW_STOPS_MEAN[STOPS_TO_NEW[old]] for old in STOPS}
_default_source_freq = 1 / max(len(SOURCES), 1)
_default_dest_freq = 1 / max(len(DESTINATIONS), 1)
_source_freq_fallback = {city: _default_source_freq for city in SOURCES}
_dest_freq_fallback = {city: _default_dest_freq for city in DESTINATIONS}
try:
    _freq_df = pd.read_csv(RAW_DATA_PATH, usecols=["source_city", "destination_city"])
    _source_vc = _freq_df["source_city"].value_counts(normalize=True)
    _dest_vc = _freq_df["destination_city"].value_counts(normalize=True)
    SOURCE_FREQ = {city: float(_source_vc.get(city, _source_freq_fallback[city])) for city in SOURCES}
    DEST_FREQ = {city: float(_dest_vc.get(city, _dest_freq_fallback[city])) for city in DESTINATIONS}
except Exception:
    SOURCE_FREQ = _source_freq_fallback.copy()
    DEST_FREQ = _dest_freq_fallback.copy()
SOURCE_MEAN_PRICE = {old: NEW_SOURCE_MEAN.get(CITY_TO_NEW[old], PRICE_AVG) for old in SOURCES}
ROUTE_MEAN = {
    (src, dst): NEW_ROUTE_MEAN.get((CITY_TO_NEW[src], CITY_TO_NEW[dst]), PRICE_AVG)
    for src in CITY_TO_NEW
    for dst in CITY_TO_NEW
    if src in CITY_COORDS and dst in CITY_COORDS and CITY_TO_NEW[src] != CITY_TO_NEW[dst]
}

DURATION_LOOKUP = {}
try:
    _dur_df = pd.read_csv(RAW_DATA_PATH, usecols=["source_city", "destination_city", "stops", "duration"])
    DURATION_LOOKUP = {
        (src, dst, stop): float(val)
        for (src, dst, stop), val in _dur_df.groupby(["source_city", "destination_city", "stops"])["duration"].mean().round(2).items()
    }
except Exception:
    DURATION_LOOKUP = {}
AVG_SPEED_BY_STOPS = {"zero": 756.8, "one": 198.9, "two_or_more": 116.8}


def _bucket_from_hour(hour: int) -> str:
    hour = int(hour) % 24
    if 4 <= hour < 8:
        return "Early_Morning"
    if 8 <= hour < 12:
        return "Morning"
    if 12 <= hour < 16:
        return "Afternoon"
    if 16 <= hour < 20:
        return "Evening"
    if 20 <= hour < 24:
        return "Night"
    return "Late_Night"


HOUR_MEAN = {hour: NEW_DEPARTURE_MEAN[_bucket_from_hour(hour)] for hour in range(24)}
MONTH_MEAN = {month: PRICE_AVG for month in range(1, 13)}
WEEKDAY_MEAN = {weekday: PRICE_AVG for weekday in range(7)}


def data_loading(path: Optional[os.PathLike[str] | str] = None) -> pd.DataFrame:
    """Load the Notebook 18 clean raw dataset."""

    data_path = Path(path) if path is not None else RAW_DATA_PATH
    df = pd.read_csv(data_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Notebook 18 non-leaky feature engineering."""

    df = df.copy()
    df["route"] = df["source_city"].astype(str) + " -> " + df["destination_city"].astype(str)
    for col in ["airline", "flight", "source_city", "destination_city", "route"]:
        freq = df[col].value_counts(normalize=True)
        df[f"{col}_freq"] = df[col].map(freq).astype(float)
    df["is_business"] = (df["class"].astype(str).str.lower() == "business").astype(int)
    df["is_non_stop"] = (df["stops"].astype(str).str.lower() == "zero").astype(int)
    df["duration_x_days_left"] = df["duration"] * df["days_left"]
    df["duration_sq"] = df["duration"] ** 2
    df["days_left_sq"] = df["days_left"] ** 2
    return df


def _normalise_path(path: Optional[os.PathLike[str] | str]) -> str:
    return str(Path(path) if path is not None else Path(MODEL_PATH))


@lru_cache(maxsize=4)
def load_artifact(path: Optional[str] = None) -> dict[str, Any]:
    """Load the Notebook 18 model artifact lazily."""

    artifact_path = Path(_normalise_path(path))
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}. Run Notebook 17 first.")
    import joblib

    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict) or "model" not in artifact or "features" not in artifact:
        raise ValueError("Expected a dictionary artifact with at least 'model' and 'features'.")
    return artifact


def artifact_exists(path: Optional[os.PathLike[str] | str] = None) -> bool:
    return Path(_normalise_path(path)).exists()


@lru_cache(maxsize=1)
def _frequency_maps_from_data() -> dict[str, dict[Any, float]]:
    """Fallback frequency maps from the new raw dataset when artifact is absent."""

    if not RAW_DATA_PATH.exists():
        return {col: {} for col in ["airline", "flight", "source_city", "destination_city", "route"]}
    df = data_loading(RAW_DATA_PATH)
    df["route"] = df["source_city"].astype(str) + " -> " + df["destination_city"].astype(str)
    return {
        col: df[col].value_counts(normalize=True).to_dict()
        for col in ["airline", "flight", "source_city", "destination_city", "route"]
    }


@lru_cache(maxsize=1)
def _category_values_from_data() -> dict[str, list[str]]:
    """Read category dropdown values from the clean raw dataset."""

    if not RAW_DATA_PATH.exists():
        return {key: values[:] for key, values in NEW_CATEGORY_VALUES.items()}
    df = data_loading(RAW_DATA_PATH)
    values: dict[str, list[str]] = {}
    for key, defaults in NEW_CATEGORY_VALUES.items():
        if key in df.columns:
            cats = sorted(df[key].dropna().astype(str).unique().tolist())
            values[key] = cats if cats else defaults[:]
        else:
            values[key] = defaults[:]
    return values


def _artifact_or_none() -> Optional[Mapping[str, Any]]:
    try:
        return load_artifact(MODEL_PATH)
    except Exception:
        return None


def get_category_values(artifact: Optional[Mapping[str, Any]] = None) -> dict[str, list[str]]:
    if artifact and isinstance(artifact.get("category_values"), Mapping):
        values = {str(k): sorted(str(x) for x in v) for k, v in artifact["category_values"].items()}
        for key, defaults in NEW_CATEGORY_VALUES.items():
            values.setdefault(key, defaults[:])
        return values
    return _category_values_from_data()


def get_model_metrics(artifact: Optional[Mapping[str, Any]] = None) -> dict[str, float | str]:
    if artifact and artifact.get("metrics"):
        try:
            best = min(artifact["metrics"], key=lambda row: float(row.get("RMSE", np.inf)))
            return {
                "model": str(best.get("model", "Stacking Ensemble")),
                "mae": float(best.get("MAE", 1106.53)),
                "rmse": float(best.get("RMSE", 2430.92)),
                "r2_price": float(best.get("R2_price", 0.988553)),
                "mape": float(best.get("MAPE", 0.069076)),
            }
        except Exception:
            pass
    return {"model": "Stacking Ensemble", "mae": 1106.53, "rmse": 2430.92, "r2_price": 0.988553, "mape": 0.069076}


def haversine_km(city1: str, city2: str) -> float:
    if city1 == city2:
        return 0.0
    c1 = CITY_COORDS.get(city1)
    c2 = CITY_COORDS.get(city2)
    if not c1 or not c2:
        return 0.0
    radius_km = 6371.0
    lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
    lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return round(2 * radius_km * math.asin(math.sqrt(a)), 1)


def is_indian_holiday(month: int, day: int) -> int:
    return 1 if (month, day) in INDIAN_HOLIDAYS else 0


def predict_duration(source: str, destination: str, stops: str) -> float:
    key = (source, destination, stops)
    if key in DURATION_LOOKUP:
        return DURATION_LOOKUP[key]
    dist = haversine_km(source, destination)
    speed = AVG_SPEED_BY_STOPS.get(stops, 300.0)
    if dist > 0 and speed > 0:
        return max(0.5, round(dist / speed, 1))
    return 10.2


def get_validation_errors(source, destination, airline, stops, passengers, dep_hour):
    """Preserve Notebook 14 UI validation behavior."""

    errors, warnings = [], []
    if source.lower() == destination.lower():
        errors.append("Source and destination cannot be the same city.")
    elif (source, destination) not in VALID_ROUTES:
        rev = (destination, source)
        hint = f" Try {destination} -> {source} instead." if rev in VALID_ROUTES else ""
        errors.append(f"Route {source} -> {destination} has no records in the dataset.{hint}")

    valid_stops_for_airline = VALID_AIRLINE_STOPS.get(airline, [])
    if stops not in valid_stops_for_airline:
        errors.append(
            f"{airline} does not operate {stops} flights in the dataset. "
            f"Valid options: {', '.join(valid_stops_for_airline)}."
        )

    max_pax = MAX_PAX_BY_STOPS.get(stops, 9)
    if passengers > max_pax:
        errors.append(f"{stops} flights support max {max_pax} passengers. Reduce to {max_pax} or choose fewer stops.")

    if dep_hour in [2, 3, 4]:
        warnings.append(f"Departure at {dep_hour:02d}:00 is a red-eye slot - low data confidence.")
    return errors, warnings


def _infer_cabin_class(airline: str) -> str:
    return "Business" if "Business" in str(airline) else "Economy"


def _days_left_from_ui(journey_month: int, journey_day: int) -> int:
    today = date.today()
    month = int(journey_month)
    max_day = calendar.monthrange(today.year, month)[1]
    day = min(max(int(journey_day), 1), max_day)
    candidate = date(today.year, month, day)
    if candidate < today:
        candidate = date(today.year + 1, month, min(day, calendar.monthrange(today.year + 1, month)[1]))
    return int(np.clip((candidate - today).days, 1, 49))


def _normalise_ui_city(city: str) -> str:
    return CITY_TO_NEW.get(str(city), str(city))


def build_features_from_ui(
    airline: str,
    source: str,
    destination: str,
    stops: str,
    flight_class: Optional[str],
    dep_hour: int,
    journey_month: int,
    journey_weekday: int,
    journey_day: int,
    duration_hours: float,
) -> dict[str, Any]:
    """Map unchanged Notebook 14 UI inputs into Notebook 18's model schema."""

    mapped_airline = AIRLINE_TO_NEW.get(airline, "Vistara")
    source_city = _normalise_ui_city(source)
    destination_city = _normalise_ui_city(destination)
    if source_city == destination_city:
        destination_city = next(city for city in NEW_CATEGORY_VALUES["destination_city"] if city != source_city)

    duration = max(float(duration_hours), 0.5)
    departure_time = _bucket_from_hour(int(dep_hour))
    arrival_time = _bucket_from_hour(int(round(dep_hour + duration)) % 24)
    mapped_stops = STOPS_TO_NEW.get(stops, "one")
    mapped_class = str(flight_class) if flight_class in NEW_CATEGORY_VALUES["class"] else _infer_cabin_class(airline)
    flight_code = f"UI-{mapped_airline}-{source_city[:3].upper()}-{destination_city[:3].upper()}"

    return {
        "airline": mapped_airline,
        "flight": flight_code,
        "source_city": source_city,
        "departure_time": departure_time,
        "stops": mapped_stops,
        "arrival_time": arrival_time,
        "destination_city": destination_city,
        "class": mapped_class,
        "duration": duration,
        "days_left": _days_left_from_ui(journey_month, journey_day),
    }


def _engineer_prediction_row(user_input: Mapping[str, Any], artifact: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
    row = pd.DataFrame([dict(user_input)])
    row["duration"] = pd.to_numeric(row["duration"], errors="coerce").fillna(0.5).clip(lower=0.5)
    row["days_left"] = pd.to_numeric(row["days_left"], errors="coerce").fillna(1).clip(lower=1)
    row["route"] = row["source_city"].astype(str) + " -> " + row["destination_city"].astype(str)

    frequency_maps = artifact.get("frequency_maps", {}) if artifact else _frequency_maps_from_data()
    for col in ["airline", "flight", "source_city", "destination_city", "route"]:
        mapping = frequency_maps.get(col, {}) if isinstance(frequency_maps, Mapping) else {}
        row[f"{col}_freq"] = row[col].map(mapping).fillna(0.0).astype(float)

    row["is_business"] = (row["class"].astype(str).str.lower() == "business").astype(int)
    row["is_non_stop"] = (row["stops"].astype(str).str.lower() == "zero").astype(int)
    row["duration_x_days_left"] = row["duration"] * row["days_left"]
    row["duration_sq"] = row["duration"] ** 2
    row["days_left_sq"] = row["days_left"] ** 2
    return row


def build_features(
    airline,
    source,
    destination,
    dep_hour,
    journey_month,
    journey_weekday,
    journey_day,
    duration_hours,
) -> pd.DataFrame:
    artifact = _artifact_or_none()
    features = list(artifact.get("features", MODEL_FEATURES)) if artifact else MODEL_FEATURES
    user_input = build_features_from_ui(
        airline,
        source,
        destination,
        "one",
        "Economy",
        dep_hour,
        journey_month,
        journey_weekday,
        journey_day,
        duration_hours,
    )
    row = _engineer_prediction_row(user_input, artifact)
    missing = [feature for feature in features if feature not in row.columns]
    if missing:
        raise ValueError(f"Could not build required model features: {missing}")
    return row[features]


def build_feature_matrix(combos: list[Mapping[str, Any]]) -> pd.DataFrame:
    artifact = _artifact_or_none()
    features = list(artifact.get("features", MODEL_FEATURES)) if artifact else MODEL_FEATURES
    rows = []
    for combo in combos:
        user_input = build_features_from_ui(
            combo["airline"],
            combo["source"],
            combo["destination"],
            combo.get("stops", "one"),
            combo.get("class", _infer_cabin_class(combo["airline"])),
            combo["dep_hour"],
            combo["journey_month"],
            combo["journey_weekday"],
            combo["journey_day"],
            combo["duration_hours"],
        )
        rows.append(_engineer_prediction_row(user_input, artifact))
    if not rows:
        return pd.DataFrame(columns=features)
    matrix = pd.concat(rows, ignore_index=True)
    missing = [feature for feature in features if feature not in matrix.columns]
    if missing:
        raise ValueError(f"Could not build required model features: {missing}")
    return matrix[features]


def batch_predict(model, combos: list[Mapping[str, Any]], passengers: int = 1, fallback_fn=None) -> list[float]:
    """Batch predict through the real new-dataset model.

    ``fallback_fn`` is kept only for old call-site compatibility. It is not
    used because App 18 must not fabricate prices when the model is missing.
    """

    if model is None:
        raise FileNotFoundError(
            f"Model artifact is required for prediction: {MODEL_PATH}. "
            "Run Notebook 17 first to create the new dataset model."
        )
    if not combos:
        return []

    x_batch = build_feature_matrix(combos)
    preds_log = model.predict(x_batch)
    return [round(max(float(np.expm1(pred)), 0.0) * int(passengers), 2) for pred in preds_log]


def validate_user_input(user_input: Mapping[str, Any], categories: Optional[Mapping[str, list[str]]] = None) -> list[str]:
    categories = categories or NEW_CATEGORY_VALUES
    required = [
        "airline",
        "flight",
        "source_city",
        "departure_time",
        "stops",
        "arrival_time",
        "destination_city",
        "class",
        "duration",
        "days_left",
    ]
    errors: list[str] = []
    for key in required:
        if key not in user_input or user_input[key] in (None, ""):
            errors.append(f"Missing value for {key}.")
    for key in ["airline", "source_city", "departure_time", "stops", "arrival_time", "destination_city", "class"]:
        if key in user_input and key in categories and str(user_input[key]) not in categories[key]:
            errors.append(f"{key} value '{user_input[key]}' was not present in the training categories.")
    if user_input.get("source_city") == user_input.get("destination_city"):
        errors.append("Source city and destination city must be different.")
    try:
        if float(user_input.get("duration", 0)) <= 0:
            errors.append("Duration must be greater than 0 hours.")
    except (TypeError, ValueError):
        errors.append("Duration must be numeric.")
    try:
        if int(user_input.get("days_left", 0)) < 1:
            errors.append("Days left must be at least 1.")
    except (TypeError, ValueError):
        errors.append("Days left must be a whole number.")
    return errors


def engineer_user_features(user_input: Mapping[str, Any], artifact: Mapping[str, Any]) -> pd.DataFrame:
    row = _engineer_prediction_row(user_input, artifact)
    features = artifact["features"]
    missing = [feature for feature in features if feature not in row.columns]
    if missing:
        raise ValueError(f"Could not build required model features: {missing}")
    return row[features]


def predict_new_flight_price(user_input: Mapping[str, Any], artifact_path: Optional[os.PathLike[str] | str] = None) -> float:
    artifact = load_artifact(_normalise_path(artifact_path))
    errors = validate_user_input(user_input, get_category_values(artifact))
    if errors:
        raise ValueError(" ".join(errors))
    feature_row = engineer_user_features(user_input, artifact)
    prediction_log = artifact["model"].predict(feature_row)[0]
    prediction_price = float(np.expm1(prediction_log))
    return round(max(prediction_price, 0.0), 2)


def predict_price(
    airline: str,
    flight: str,
    source_city: str,
    departure_time: str,
    stops: str,
    arrival_time: str,
    destination_city: str,
    flight_class: str,
    duration: float,
    days_left: int,
    artifact_path: Optional[os.PathLike[str] | str] = None,
) -> float:
    return predict_new_flight_price(
        {
            "airline": airline,
            "flight": flight,
            "source_city": source_city,
            "departure_time": departure_time,
            "stops": stops,
            "arrival_time": arrival_time,
            "destination_city": destination_city,
            "class": flight_class,
            "duration": duration,
            "days_left": days_left,
        },
        artifact_path=artifact_path,
    )


__all__ = [
    "AIRLINES",
    "SOURCES",
    "DESTINATIONS",
    "STOPS",
    "CLASSES",
    "MONTHS",
    "WEEKDAYS",
    "AIRLINE_MEAN_PRICE",
    "SOURCE_FREQ",
    "DEST_FREQ",
    "SOURCE_MEAN_PRICE",
    "PRICE_MIN",
    "PRICE_MAX",
    "PRICE_AVG",
    "PRICE_MED",
    "AIRLINE_MEAN",
    "STOPS_MEAN",
    "ROUTE_MEAN",
    "HOUR_MEAN",
    "MONTH_MEAN",
    "WEEKDAY_MEAN",
    "DURATION_LOOKUP",
    "AVG_SPEED_BY_STOPS",
    "CITY_COORDS",
    "VALID_ROUTES",
    "VALID_DESTINATIONS",
    "VALID_AIRLINE_STOPS",
    "MAX_PAX_BY_STOPS",
    "INDIAN_HOLIDAYS",
    "MODEL_PATH",
    "MODEL_FEATURES",
    "data_loading",
    "feature_engineering",
    "build_features_from_ui",
    "haversine_km",
    "is_indian_holiday",
    "predict_duration",
    "get_validation_errors",
    "build_features",
    "build_feature_matrix",
    "batch_predict",
    "load_artifact",
    "artifact_exists",
    "get_category_values",
    "get_model_metrics",
    "validate_user_input",
    "engineer_user_features",
    "predict_new_flight_price",
    "predict_price",
]
