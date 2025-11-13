# utils.py
# Shared utility helpers for AI Supply Chain Dashboard

import pandas as pd
import numpy as np

# ----------------------------------------------
# Safe Column Resolver
# ----------------------------------------------
def safe_col(df, names, default=None):
    """
    Given a list of possible column names, return the first match
    that exists in the dataframe (case-insensitive).
    """
    cols = {c.lower(): c for c in df.columns}
    for name in names:
        if name and name.lower() in cols:
            return cols[name.lower()]
    return default

# ----------------------------------------------
# Safe Series Creation
# ----------------------------------------------
def safe_series(df, col_name, default_value):
    """
    Return df[col_name] if exists, else create a fallback series
    with the default_value.
    """
    if col_name in df.columns:
        return df[col_name]
    return pd.Series([default_value] * len(df), index=df.index)

# ----------------------------------------------
# Normalization (Min-Max)
# ----------------------------------------------
def normalize(series):
    """
    Min-max normalization for scoring.
    If constant series → return 1.0 for all rows.
    """
    if series.max() == series.min():
        return pd.Series([1.0] * len(series), index=series.index)
    return (series - series.min()) / (series.max() - series.min())

# ----------------------------------------------
# Ordinal Conversion
# ----------------------------------------------
def to_ordinal_array(dates):
    """Convert datetime series to ordinal integers."""
    return np.array([pd.to_datetime(d).toordinal() for d in dates])

# ----------------------------------------------
# Master Data Loader + Preprocessor
# ----------------------------------------------
def load_and_prepare_data(csv_path):
    """
    Load dataset and ensure all required columns exist.
    Adds fallback columns for consistent operation.
    """
    df = pd.read_csv(csv_path)

    # standardize date
    if "Date" not in df.columns:
        for alt in ["date", "order_date", "delivery_date", "ship_date"]:
            if alt in df.columns:
                df["Date"] = df[alt]
                break
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Fill missing date entries
    if df["Date"].isna().any():
        df["Date"].fillna(df["Date"].min(), inplace=True)

    # Ensure required columns exist
    required_defaults = {
        "Product": "Unknown Product",
        "Category": "Uncategorized",
        "Sales": np.random.randint(50, 300, size=len(df)),
        "Inventory": np.random.randint(10, 150, size=len(df)),
        "Lead_Time_Days": np.random.randint(2, 12, size=len(df)),
        "Cost": np.round(np.random.uniform(50, 500, size=len(df)), 2)
    }

    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default if not callable(default) else default

    # Supplier fallback
    if safe_col(df, ["supplier", "suppliers", "vendor", "carrier"], None) is None:
        df["Supplier_Fallback"] = "Supplier A"

    # Numeric operational defaults
    fallback_numeric = {
        "distance_km": 100,
        "actual_lead_days": df.get("Lead_Time_Days", pd.Series([5]*len(df))),
        "delay_days": 0,
        "on_time_delivery": 0.95,
        "damage_rate": 0.02,
        "return_rate": 0.03,
        "total_cost_INR": lambda r: r.get("Cost", 100) * r.get("Sales", 10)
    }

    for col, default in fallback_numeric.items():
        if col not in df.columns:
            if callable(default):
                df[col] = df.apply(default, axis=1)
            else:
                df[col] = default

    return df
