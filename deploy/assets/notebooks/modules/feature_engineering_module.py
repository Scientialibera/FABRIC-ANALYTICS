# Fabric Notebook
# feature_engineering_module.py - Bronze -> Silver feature table creation

import pandas as pd
import numpy as np


def aggregate_to_grain(df, date_column, grain_columns, target_column, feature_columns, frequency):
    """
    Aggregate a pandas DataFrame to the configured grain and frequency.
    Returns a DataFrame indexed by grain + period.
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df["_period"] = df[date_column].dt.to_period(frequency).dt.to_timestamp()

    group_cols = grain_columns + ["_period"]
    agg_spec = {target_column: "sum"}
    for col in feature_columns:
        if col in df.columns:
            agg_spec[col] = "mean"

    agg = df.groupby(group_cols, as_index=False).agg(agg_spec)
    agg = agg.rename(columns={"_period": date_column})
    return agg.sort_values(group_cols).reset_index(drop=True)


def add_lag_features(df, date_column, grain_columns, target_column, lags=None):
    """Add lag features for the target column within each grain group."""
    if lags is None:
        lags = [1, 2, 3, 6, 12]

    df = df.sort_values(grain_columns + [date_column]).reset_index(drop=True)
    for lag in lags:
        col_name = f"{target_column}_lag_{lag}"
        df[col_name] = df.groupby(grain_columns)[target_column].shift(lag)
    return df


def add_rolling_features(df, date_column, grain_columns, target_column, windows=None):
    """Add rolling mean/std features within each grain group."""
    if windows is None:
        windows = [3, 6, 12]

    df = df.sort_values(grain_columns + [date_column]).reset_index(drop=True)
    for w in windows:
        grouped = df.groupby(grain_columns)[target_column]
        df[f"{target_column}_roll_mean_{w}"] = grouped.transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f"{target_column}_roll_std_{w}"] = grouped.transform(lambda x: x.rolling(w, min_periods=1).std())
    return df


def add_calendar_features(df, date_column):
    """Add calendar-derived features from the date column."""
    df = df.copy()
    dt = pd.to_datetime(df[date_column])
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["year"] = dt.dt.year
    df["day_of_year"] = dt.dt.dayofyear
    df["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
    return df


def build_feature_table(df, date_column, grain_columns, target_column, feature_columns, frequency):
    """Full pipeline: aggregate -> lags -> rolling -> calendar."""
    agg = aggregate_to_grain(df, date_column, grain_columns, target_column, feature_columns, frequency)
    agg = add_lag_features(agg, date_column, grain_columns, target_column)
    agg = add_rolling_features(agg, date_column, grain_columns, target_column)
    agg = add_calendar_features(agg, date_column)
    return agg
