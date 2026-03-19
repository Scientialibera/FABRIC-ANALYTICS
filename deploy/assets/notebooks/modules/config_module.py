# Fabric Notebook -- Module
# config_module.py - Shared configuration reader and lakehouse utilities

import json
import os


def _get_workspace_id() -> str:
    try:
        return spark.conf.get("trident.workspace.id")
    except Exception:
        pass
    return os.environ.get("WORKSPACE_ID", os.environ.get("fabric_workspace_id", ""))


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
    if isinstance(value, list):
        return value
    value = str(value).strip()
    if value.startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return [v.strip().strip("'\"") for v in value.split(",") if v.strip()]


def parse_int_list_param(value):
    """Parse a JSON-encoded list of integers."""
    items = parse_list_param(value)
    return [int(x) for x in items]


def lakehouse_table_path(lakehouse_id: str, table_name: str) -> str:
    ws = _get_workspace_id()
    return f"abfss://{ws}@onelake.dfs.fabric.microsoft.com/{lakehouse_id}/Tables/{table_name}"


def read_lakehouse_table(spark_session, lakehouse_id: str, table_name: str):
    path = lakehouse_table_path(lakehouse_id, table_name)
    return spark_session.read.format("delta").load(path)


def write_lakehouse_table(df, lakehouse_id: str, table_name: str, mode: str = "overwrite"):
    """Write as managed Delta table via saveAsTable for proper catalog registration."""
    try:
        if mode == "overwrite":
            spark.sql(f"DROP TABLE IF EXISTS `{table_name}`")
        df.write.format("delta").mode(mode).option("overwriteSchema", "true").saveAsTable(table_name)
    except Exception as e:
        print(f"  [warn] saveAsTable failed for '{table_name}', falling back to path write: {e}")
        path = lakehouse_table_path(lakehouse_id, table_name)
        df.write.format("delta").mode(mode).option("overwriteSchema", "true").save(path)
