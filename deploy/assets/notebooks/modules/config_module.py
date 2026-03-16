# Fabric Notebook
# config_module.py - Shared configuration reader and lakehouse utilities

import json
import os


def get_notebook_params():
    """Read parameters injected by Fabric notebook run or set defaults."""
    params = {}
    for key in [
        "landing_lakehouse_id", "bronze_lakehouse_id",
        "silver_lakehouse_id", "gold_lakehouse_id",
        "source_lakehouse_id",
        "date_column", "frequency", "target_column",
        "grain_columns", "feature_columns", "feature_types",
        "test_split_ratio",
        "sarima_order", "sarima_seasonal_order",
        "xgboost_n_estimators", "xgboost_max_depth", "xgboost_learning_rate",
        "exp_smoothing_trend", "exp_smoothing_seasonal", "exp_smoothing_seasonal_periods",
        "experiment_name", "registered_model_prefix",
        "forecast_horizon", "output_table", "metrics_table",
        "source_tables",
    ]:
        params[key] = os.environ.get(key, "")
    return params


def parse_list_param(value):
    """Parse a JSON-encoded list or comma-separated string."""
    if not value:
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return [v.strip() for v in str(value).split(",") if v.strip()]


def parse_int_list_param(value):
    """Parse a JSON-encoded list of integers."""
    items = parse_list_param(value)
    return [int(x) for x in items]


def lakehouse_table_path(lakehouse_id, table_name):
    """Build the abfss path for a lakehouse delta table."""
    return f"abfss://workspace@onelake.dfs.fabric.microsoft.com/{lakehouse_id}/Tables/{table_name}"


def read_lakehouse_table(spark, lakehouse_id, table_name):
    """Read a delta table from a lakehouse."""
    path = lakehouse_table_path(lakehouse_id, table_name)
    return spark.read.format("delta").load(path)


def write_lakehouse_table(df, lakehouse_id, table_name, mode="overwrite"):
    """Write a Spark DataFrame as a delta table in a lakehouse."""
    path = lakehouse_table_path(lakehouse_id, table_name)
    df.write.format("delta").mode(mode).option("overwriteSchema", "true").save(path)
