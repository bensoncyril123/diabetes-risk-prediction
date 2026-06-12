"""Feature engineering utilities for the BRFSS 2015 diabetes health indicators dataset."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Diabetes_012"

TARGET_LABELS = {0: "No Diabetes", 1: "Prediabetes", 2: "Diabetes"}

# WHO BMI bands -> ordinal category (0=Underweight ... 3=Obese)
BMI_BINS = [0, 18.5, 25, 30, np.inf]
BMI_CATEGORY_LABELS = ["Underweight", "Normal", "Overweight", "Obese"]

# Composite feature groups, derived from EDA effect-size ranking
LIFESTYLE_POSITIVE = ["PhysActivity", "Fruits", "Veggies"]
LIFESTYLE_NEGATIVE = ["Smoker", "HvyAlcoholConsump"]  # 1 = risky behaviour present
COMORBIDITY_COLS = ["HighBP", "HighChol", "Stroke", "HeartDiseaseorAttack"]

# Only BMI gets an outlier cap. MentHlth/PhysHlth are 0-30 day counts where the
# tail (chronic poor health) is itself a strong diabetes signal (EDA section 5),
# so clipping them would remove ~13% of rows and destroy that signal.
BMI_CAP_IQR_MULTIPLIER = 3.0

# Precomputed Q3 + 3*IQR of BMI on the full raw training data (Q1=24, Q3=31, IQR=7).
# Fixed at inference time since a single row has no quantiles of its own.
BMI_CAP_VALUE = 52.0

# Final feature order expected by the saved model (raw columns + engineered columns).
RAW_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack",
    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
    "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]
FEATURES = RAW_FEATURES + ["BMICategory", "LifestyleScore", "ComorbidityScore"]


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def add_bmi_category(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ordinal BMI category column (0=Underweight, 1=Normal, 2=Overweight, 3=Obese)."""
    df = df.copy()
    df["BMICategory"] = pd.cut(
        df["BMI"], bins=BMI_BINS, labels=range(len(BMI_CATEGORY_LABELS))
    ).astype(int)
    return df


def add_lifestyle_score(df: pd.DataFrame) -> pd.DataFrame:
    """Composite 0-5 score: +1 for each healthy behaviour present.

    PhysActivity / Fruits / Veggies each contribute +1 when present (=1).
    Smoker / HvyAlcoholConsump each contribute +1 when ABSENT (=0).
    Higher = healthier lifestyle.
    """
    df = df.copy()
    positive = df[LIFESTYLE_POSITIVE].sum(axis=1)
    negative_absent = (1 - df[LIFESTYLE_NEGATIVE]).sum(axis=1)
    df["LifestyleScore"] = positive + negative_absent
    return df


def add_comorbidity_score(df: pd.DataFrame) -> pd.DataFrame:
    """Composite 0-4 count of pre-existing conditions: HighBP, HighChol, Stroke, HeartDiseaseorAttack."""
    df = df.copy()
    df["ComorbidityScore"] = df[COMORBIDITY_COLS].sum(axis=1)
    return df


def cap_bmi_outliers(df: pd.DataFrame, multiplier: float = BMI_CAP_IQR_MULTIPLIER) -> tuple[pd.DataFrame, float]:
    """Cap extreme BMI values at Q3 + multiplier*IQR (~52 at multiplier=3, affects ~0.7% of rows)."""
    df = df.copy()
    q1, q3 = df["BMI"].quantile([0.25, 0.75])
    upper = q3 + multiplier * (q3 - q1)
    df["BMI"] = df["BMI"].clip(upper=upper)
    return df, upper


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature engineering pipeline: BMI capping + BMI category +
    lifestyle score + comorbidity score."""
    df, _ = cap_bmi_outliers(df)
    df = add_bmi_category(df)
    df = add_lifestyle_score(df)
    df = add_comorbidity_score(df)
    return df


def engineer_single_row(raw: dict) -> pd.DataFrame:
    """Build a single-row, model-ready feature DataFrame from raw user input.

    `raw` must contain all columns in RAW_FEATURES. BMI is capped at the fixed
    BMI_CAP_VALUE (a single row has no quantiles of its own to derive a cap from).
    """
    df = pd.DataFrame([raw])[RAW_FEATURES]
    df["BMI"] = df["BMI"].clip(upper=BMI_CAP_VALUE)
    df = add_bmi_category(df)
    df = add_lifestyle_score(df)
    df = add_comorbidity_score(df)
    return df[FEATURES]


def get_feature_target_split(df: pd.DataFrame, target: str = TARGET) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float | None = None,
    random_state: int = 42,
):
    """Stratified train/test (or train/val/test) split, preserving the 84/2/14 class ratio.

    If val_size is None, returns (X_train, X_test, y_train, y_test).
    Otherwise returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    if val_size is None:
        return X_train, X_test, y_train, y_test

    val_fraction_of_train = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_fraction_of_train, stratify=y_train, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
