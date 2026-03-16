# Fabric Notebook
# 04_train_exp_smoothing_main.py - Train Exponential Smoothing per grain

# %run ../modules/config_module
# %run ../modules/utils_module
# %run ../modules/train_exp_smoothing_module

params = get_notebook_params()

silver_lakehouse_id = params["silver_lakehouse_id"]
date_column = params["date_column"]
target_column = params["target_column"]
grain_columns = parse_list_param(params["grain_columns"])
test_split_ratio = float(params.get("test_split_ratio") or 0.2)
trend = params.get("exp_smoothing_trend") or "add"
seasonal = params.get("exp_smoothing_seasonal") or "add"
seasonal_periods = int(params.get("exp_smoothing_seasonal_periods") or 12)
experiment_name = params.get("experiment_name") or "demand_forecast"
model_prefix = params.get("registered_model_prefix") or "demand_model"

if not silver_lakehouse_id:
    raise ValueError("silver_lakehouse_id is required.")

print("[exp_smoothing_main] Loading feature table from silver lakehouse.")
spark_df = read_lakehouse_table(spark, silver_lakehouse_id, "feature_table")
pdf = spark_df.toPandas()
pdf = pdf.dropna(subset=[target_column]).reset_index(drop=True)
print(f"[exp_smoothing_main] Loaded {len(pdf)} rows.")

results_df, agg_metrics = train_exp_smoothing_per_grain(
    df=pdf,
    date_column=date_column,
    grain_columns=grain_columns,
    target_column=target_column,
    trend=trend,
    seasonal=seasonal,
    seasonal_periods=seasonal_periods,
    test_ratio=test_split_ratio,
    experiment_name=experiment_name,
    model_name=f"{model_prefix}_exp_smoothing",
)

if not results_df.empty:
    preds_spark = spark.createDataFrame(results_df)
    write_lakehouse_table(preds_spark, silver_lakehouse_id, "exp_smoothing_predictions", mode="overwrite")
    print(f"[exp_smoothing_main] Wrote {len(results_df)} prediction rows.")

print("[exp_smoothing_main] Complete.")
