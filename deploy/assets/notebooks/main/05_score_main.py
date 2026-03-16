# Fabric Notebook
# 05_score_main.py - Batch scoring: XGBoost via PREDICT, time-series via re-fit forecast

# %run ../modules/config_module
# %run ../modules/utils_module
# %run ../modules/scoring_module

import pandas as pd
from datetime import datetime

params = get_notebook_params()

silver_lakehouse_id = params["silver_lakehouse_id"]
gold_lakehouse_id = params["gold_lakehouse_id"]
date_column = params["date_column"]
target_column = params["target_column"]
grain_columns = parse_list_param(params["grain_columns"])
feature_columns = parse_list_param(params["feature_columns"])
forecast_horizon = int(params.get("forecast_horizon") or 6)
output_table = params.get("output_table") or "forecast_results"
metrics_table = params.get("metrics_table") or "model_metrics"
model_prefix = params.get("registered_model_prefix") or "demand_model"
sarima_order = tuple(parse_int_list_param(params.get("sarima_order") or "[1,1,1]"))
sarima_seasonal_order = tuple(parse_int_list_param(params.get("sarima_seasonal_order") or "[1,1,1,12]"))
ets_trend = params.get("exp_smoothing_trend") or "add"
ets_seasonal = params.get("exp_smoothing_seasonal") or "add"
ets_seasonal_periods = int(params.get("exp_smoothing_seasonal_periods") or 12)

if not silver_lakehouse_id or not gold_lakehouse_id:
    raise ValueError("silver_lakehouse_id and gold_lakehouse_id are required.")

print("[score] Loading feature table from silver lakehouse.")
spark_df = read_lakehouse_table(spark, silver_lakehouse_id, "feature_table")
pdf = spark_df.toPandas()
pdf = pdf.dropna(subset=[target_column]).reset_index(drop=True)
print(f"[score] Loaded {len(pdf)} rows.")

all_feature_cols = [c for c in pdf.columns if c not in [date_column, target_column] + grain_columns and pdf[c].dtype in ["float64", "int64", "float32", "int32"]]

all_forecasts = []
all_metrics = []

# --- XGBoost batch scoring ---
print("[score] Scoring XGBoost model.")
try:
    xgb_scored = score_xgboost(spark, pdf, f"{model_prefix}_xgboost", all_feature_cols)
    xgb_scored["model_type"] = "xgboost"
    xgb_scored["scored_at"] = datetime.utcnow().isoformat()
    all_forecasts.append(xgb_scored[[*grain_columns, date_column, target_column, "predicted", "model_type", "scored_at"]])

    xgb_metrics = build_metrics_summary(xgb_scored, target_column=target_column, pred_column="predicted", model_type="xgboost")
    if xgb_metrics:
        all_metrics.append(xgb_metrics)
    print(f"[score] XGBoost: {len(xgb_scored)} predictions.")
except Exception as e:
    print(f"[score] XGBoost scoring failed: {e}")

# --- SARIMA forecast ---
print("[score] Scoring SARIMA model.")
try:
    sarima_forecasts = score_timeseries_model(
        pdf, date_column, grain_columns, target_column,
        model_type="sarima",
        forecast_horizon=forecast_horizon,
        sarima_order=sarima_order,
        sarima_seasonal_order=sarima_seasonal_order,
    )
    if not sarima_forecasts.empty:
        all_forecasts.append(sarima_forecasts)
        print(f"[score] SARIMA: {len(sarima_forecasts)} forecast rows.")
except Exception as e:
    print(f"[score] SARIMA scoring failed: {e}")

# --- Exponential Smoothing forecast ---
print("[score] Scoring Exponential Smoothing model.")
try:
    ets_forecasts = score_timeseries_model(
        pdf, date_column, grain_columns, target_column,
        model_type="exp_smoothing",
        forecast_horizon=forecast_horizon,
        ets_trend=ets_trend,
        ets_seasonal=ets_seasonal,
        ets_seasonal_periods=ets_seasonal_periods,
    )
    if not ets_forecasts.empty:
        all_forecasts.append(ets_forecasts)
        print(f"[score] Exp Smoothing: {len(ets_forecasts)} forecast rows.")
except Exception as e:
    print(f"[score] Exp Smoothing scoring failed: {e}")

# --- Write combined results to Gold ---
if all_forecasts:
    combined = pd.concat(all_forecasts, ignore_index=True)
    combined_spark = spark.createDataFrame(combined)
    write_lakehouse_table(combined_spark, gold_lakehouse_id, output_table, mode="overwrite")
    print(f"[score] Wrote {len(combined)} rows to gold.{output_table}.")

if all_metrics:
    metrics_df = pd.DataFrame(all_metrics)
    metrics_spark = spark.createDataFrame(metrics_df)
    write_lakehouse_table(metrics_spark, gold_lakehouse_id, metrics_table, mode="overwrite")
    print(f"[score] Wrote {len(metrics_df)} metric rows to gold.{metrics_table}.")

print("[score] Complete.")
