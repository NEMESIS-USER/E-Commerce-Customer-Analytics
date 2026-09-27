"""
Preprocessing Module
Handles dataset loading, validation, missing value imputation, duplicate removal,
categorical encoding, and feature scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import warnings

# Suppress pandas future string warnings cleanly
warnings.filterwarnings("ignore", category=FutureWarning)


REQUIRED_COLUMNS = [
    "Customer_ID",
    "Age",
    "Gender",
    "City",
    "Product_Category",
    "Purchase_Amount",
    "Purchase_Frequency",
    "Discount_Used",
    "Customer_Rating",
    "Delivery_Days",
    "Payment_Method",
    "Customer_Segment",
]

NUMERIC_COLUMNS = [
    "Age",
    "Purchase_Amount",
    "Purchase_Frequency",
    "Customer_Rating",
    "Delivery_Days",
]

CATEGORICAL_COLUMNS = [
    "Customer_ID",
    "Gender",
    "City",
    "Product_Category",
    "Discount_Used",
    "Payment_Method",
    "Customer_Segment",
]


def validate_dataset(df):
    """
    Validates required schema, numeric/categorical columns, missing values, and duplicates.
    Does not mutate the original DataFrame.
    """
    report = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "missing_required": [],
        "extra_columns": [],
        "numeric_ok": [],
        "numeric_failed": [],
        "categorical_present": [],
        "missing_cells": 0,
        "missing_percentage": 0.0,
        "duplicate_rows": 0,
        "num_rows": 0,
        "num_columns": 0,
        "coerced_df": None,
    }

    if df is None or df.empty:
        report["is_valid"] = False
        report["errors"].append("Dataset is empty.")
        return report

    report["num_rows"] = int(len(df))
    report["num_columns"] = int(len(df.columns))
    report["extra_columns"] = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    report["missing_required"] = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if report["missing_required"]:
        report["is_valid"] = False
        report["errors"].append(
            "Missing required columns: " + ", ".join(report["missing_required"])
        )

    coerced = df.copy()
    for col in NUMERIC_COLUMNS:
        if col not in coerced.columns:
            continue
        converted = pd.to_numeric(coerced[col], errors="coerce")
        invalid_count = int(converted.isna().sum() - coerced[col].isna().sum())
        if invalid_count > 0:
            report["numeric_failed"].append(f"{col} ({invalid_count} non-numeric values)")
            report["warnings"].append(
                f"Column '{col}' has {invalid_count} values that could not be parsed as numeric."
            )
        else:
            report["numeric_ok"].append(col)
        coerced[col] = converted

    for col in CATEGORICAL_COLUMNS:
        if col in coerced.columns:
            report["categorical_present"].append(col)
        else:
            report["warnings"].append(f"Expected categorical column '{col}' is missing.")

    report["missing_cells"] = int(coerced.isnull().sum().sum())
    report["missing_percentage"] = round((report["missing_cells"] / coerced.size) * 100, 2) if coerced.size else 0.0
    report["duplicate_rows"] = int(coerced.duplicated().sum())
    report["coerced_df"] = coerced
    return report


def load_dataset(file_path_or_buffer):
    """
    Safely loads a CSV dataset into a pandas DataFrame.
    """
    try:
        if isinstance(file_path_or_buffer, str):
            df = pd.read_csv(file_path_or_buffer)
        else:
            df = pd.read_csv(file_path_or_buffer)
        return df, None
    except Exception as e:
        return None, str(e)


def get_dataset_summary(df):
    """
    Computes statistical and structural summary of the dataset.
    """
    if df is None or df.empty:
        return {}

    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    cat_cols = [c for c in df.columns if c not in num_cols]

    summary = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "total_cells": df.size,
        "missing_cells": int(df.isnull().sum().sum()),
        "missing_percentage": round((df.isnull().sum().sum() / df.size) * 100, 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_per_col": df.isnull().sum().to_dict(),
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
    }
    return summary


def preprocess_data(
    df,
    impute_strategy_num="Median",
    impute_strategy_cat="Mode",
    drop_duplicates=True,
    scale_numeric=False,
    scaler_type="StandardScaler"
):
    """
    Preprocesses the dataset without modifying the original DataFrame.
    Returns:
        - df_cleaned: Cleaned DataFrame with missing values handled and duplicates removed.
        - df_encoded: Machine-learning ready DataFrame with categorical features label-encoded.
        - encoders: Dictionary of fitted LabelEncoders for categorical columns.
        - scaler: Fitted scaler object if scale_numeric is True, else None.
        - stats: Summary before and after preprocessing.
    """
    if df is None or df.empty:
        return None, None, {}, None, {}

    stats_before = get_dataset_summary(df)

    # 1. Work on a deep copy
    df_clean = df.copy()

    # 2. Duplicate handling
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    # 3. Missing Value Handling
    num_cols = list(df_clean.select_dtypes(include=[np.number]).columns)
    cat_cols = [c for c in df_clean.columns if c not in num_cols]

    # Numerical imputation
    for col in num_cols:
        if df_clean[col].isnull().any():
            if impute_strategy_num == "Mean":
                val = df_clean[col].mean()
            elif impute_strategy_num == "Zero":
                val = 0
            else:  # Median default
                val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(val)

    # Categorical imputation
    for col in cat_cols:
        if df_clean[col].isnull().any():
            if impute_strategy_cat == "Mode" and not df_clean[col].mode().empty:
                val = df_clean[col].mode()[0]
            else:
                val = "Missing"
            df_clean[col] = df_clean[col].fillna(val)

    stats_after = get_dataset_summary(df_clean)

    # 4. Encoding for ML models
    df_encoded = df_clean.copy()
    encoders = {}

    for col in cat_cols:
        if col != "Customer_ID":  # Don't encode IDs as features
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le

    # 5. Scaling numeric columns (if requested)
    scaler = None
    if scale_numeric and len(num_cols) > 0:
        scaler = StandardScaler() if scaler_type == "StandardScaler" else MinMaxScaler()
        cols_to_scale = [c for c in num_cols if c != "Customer_ID"]
        if cols_to_scale:
            df_encoded[cols_to_scale] = scaler.fit_transform(df_encoded[cols_to_scale])

    stats = {
        "before": stats_before,
        "after": stats_after
    }

    return df_clean, df_encoded, encoders, scaler, stats
