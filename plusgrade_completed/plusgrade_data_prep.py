from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from datetime import datetime as dt
from category_encoders import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

np.random.seed(42)
# ------------------------------
# 1️⃣ DATE PROCESSING FUNCTIONS
# ------------------------------


def convert_to_datetime(df, columns):
    """Converts specified columns to datetime format."""
    if not columns:
        columns = [
            "mp_enrollment_date",
            "most_recent_flt",
            "most_recent_awd",
            "birth_dt_cd",
        ]
    df[columns] = df[columns].apply(pd.to_datetime, errors="coerce")
    return df


def calculate_age(df):
    """Calculates age from birth date."""
    df["age"] = dt.now().year - df["birth_dt_cd"].dt.year
    df["age"].fillna(df["age"].median(), inplace=True)
    return df


def calculate_tenure(df):
    """Calculates membership tenure in years."""
    df["tenure_yrs"] = dt.now().year - df["mp_enrollment_date"].dt.year
    return df


def calculate_days_since_last_event(df, reference_date="2025-02-20"):
    """Calculates days since last flight and award redemption."""
    reference_date = pd.Timestamp(reference_date, tz="UTC")
    df["days_since_last_flt"] = (reference_date - df["most_recent_flt"]).dt.days
    df["days_since_last_awd"] = (reference_date - df["most_recent_awd"]).dt.days
    return df


def process_dates(df):
    """Wrapper function to process all date-related features."""
    date_columns = [
        "mp_enrollment_date",
        "most_recent_flt",
        "most_recent_awd",
        "birth_dt_cd",
    ]
    df = convert_to_datetime(df, date_columns)
    df = calculate_age(df)
    df = calculate_tenure(df)
    df = calculate_days_since_last_event(df)

    # Drop original date columns
    df.drop(columns=date_columns, inplace=True)
    return df


# ------------------------------
# 2️⃣ FEATURE BINNING FUNCTIONS
# ------------------------------


def bin_balance(df):
    """Bins balance into Low, Medium, and High."""
    low_thresh = df["balance"].quantile(0.33)
    high_thresh = df["balance"].quantile(0.66)
    df["balance_binned"] = pd.cut(
        df["balance"],
        bins=[-float("inf"), low_thresh, high_thresh, float("inf")],
        labels=["Low", "Medium", "High"],
    )
    return df, {"Low": 0, "Medium": 1, "High": 2}


def bin_aag_yr1(df):
    """Bins aag_yr1 (Total Miles Earned in First Year)."""
    q1 = df.loc[df["aag_yr1"] > 0, "aag_yr1"].quantile(0.25)
    q3 = df.loc[df["aag_yr1"] > 0, "aag_yr1"].quantile(0.75)
    bins = [-np.inf, 0, q1, q3, np.inf]
    labels = ["Non-Earners", "Low-Earners", "Medium-Earners", "High-Earners"]
    df["aag_yr1_binned"] = pd.cut(df["aag_yr1"], bins=bins, labels=labels)
    return df, {
        "Non-Earners": 0,
        "Low-Earners": 1,
        "Medium-Earners": 2,
        "High-Earners": 3,
    }


def bin_features(df):
    """Wrapper function for binning balance and aag_yr1."""
    df, bal_map = bin_balance(df)
    df, aag_map = bin_aag_yr1(df)
    return df, bal_map, aag_map


# ------------------------------
# 3️⃣ SCALING & LOG TRANSFORMATION FUNCTIONS
# ------------------------------


def apply_log_transform(df, num_cols):
    """Applies log transformation to numerical columns."""
    df[num_cols] = np.log1p(df[num_cols])
    return df


def apply_scaling(df, num_cols):
    """Applies Min-Max scaling to numerical columns."""
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


# ------------------------------
# 4️⃣ CATEGORICAL ENCODING FUNCTIONS
# ------------------------------


def encode_ordinal_features(df, cols_to_encode, cat_map=None):
    """Applies ordinal encoding to categorical columns."""
    encoder = OrdinalEncoder(cols=cols_to_encode, mapping=cat_map)
    df[cols_to_encode] = encoder.fit_transform(df[cols_to_encode])
    return df


def encode_nominal_features(df, cols_to_encode):
    """Applies one-hot encoding to categorical columns."""
    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
    return df


# ------------------------------
# 5️⃣ COMPOSITE FEATURE CREATION
# ------------------------------


def create_nonflight_activity(df):
    """Creates a new feature combining non-flight activity variables."""
    non_flt_cols = ["affinity_spend_12mo", "nonflt_earn_12mo", "ptnr_miles_earned_12mo"]
    df["nonflight_activity"] = df[non_flt_cols].sum(axis=1)
    return df


def create_flight_activity(df):
    """Creates a new feature combining flight-related activity."""
    flt_cols = ["flt_base_12mo", "flt_promo_12mo", "flt_segs_12mo"]
    df["flight_activity"] = df[flt_cols].sum(axis=1)
    return df


def create_composite_features(df):
    """Wrapper function for composite feature creation."""
    df = create_nonflight_activity(df)
    df = create_flight_activity(df)
    return df


# ------------------------------
# 6️⃣ DATA CLEANING & PREPROCESSING
# ------------------------------


def map_current_tier(df):
    """Maps hashed tier values to readable categories."""
    tier_map = {
        "b455784ab53072773e72df494aa4b12ac467976447e250c694c11b6f1268e235": "Tier 1",
        "3953f7eea41d0899c63898d11f2ec9e4900d77fadbb725b465310e0595961ea5": "Tier 2",
        "a2738e371e35817c224c5bf458023da922bcb8ac7be4266acc3a3bf1c2fc04e3": "Tier 3",
        "4a3c6fb25c7b022b7e1b630e1781cf37c8bf76aae8a9cd8b376bb37b98e4dcda": "Tier 4",
        "104fed087a359478b41ae5be437bab10f61bccbe17843c43e00b894d905d3057": "Tier 5",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855": "Tier 6",
    }
    df["current_tier"] = df["current_tier"].map(tier_map)
    mapping = []
    mapping.append(
        {
            "col": "current_tier",
            "mapping": {
                "Tier 1": 0,
                "Tier 2": 1,
                "Tier 3": 2,
                "Tier 4": 3,
                "Tier 5": 4,
                "Tier 6": 5,
            },
        }
    )
    return df, mapping


def preprocess_data(
    df,
    num_cols,
    ordinal_cols,
    nominal_cols,
    log=True,
    scale=True,
    binning=True,
    cat=True,
    mapping=[],
):
    """Preprocesses data by applying transformations, binning, encoding, and feature engineering."""

    df = df.loc[df.balance >= 0]  # Remove negative balance

    # Apply binning
    if binning:
        df, bal_map, aag_map = bin_features(df)
        ordinal_cols.extend(["balance_binned", "aag_yr1_binned"])
        mapping.extend(
            [
                {"col": "balance_binned", "mapping": bal_map},
                {"col": "aag_yr1_binned", "mapping": aag_map},
            ]
        )
    else:
        mapping

    # Apply transformations
    if log:
        df = apply_log_transform(df, num_cols)
    if scale:
        df = apply_scaling(df, num_cols)
    if cat:
        df = encode_ordinal_features(df, ordinal_cols, mapping)
        df = encode_nominal_features(df, nominal_cols)

    df = create_composite_features(df)  # Create composite features
    return df


def clean_data(
    df, num_cols, ordinal_cols, nominal_cols, exclude_cols, cat=True, binning=True
):
    """Main wrapper function to clean and preprocess data."""
    df, mapping = map_current_tier(df)
    df = process_dates(df)
    df = preprocess_data(
        df,
        num_cols,
        ordinal_cols,
        nominal_cols,
        binning=binning,
        cat=cat,
        mapping=mapping,
    )
    return df[[col for col in df.columns if col not in exclude_cols]]
