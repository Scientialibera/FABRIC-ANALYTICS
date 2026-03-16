# Fabric Notebook
# 04_train_xgboost_main.py - Train XGBoost model on Silver feature table

# %run ../modules/config_module
# %run ../modules/utils_module
# %run ../modules/train_xgboost_module

params = get_notebook_params()

silver_lakehouse_id = params["silver_lakehouse_id"]
date_column = params["date_column"]
target_column = params["target_column"]
grain_columns = parse_list_param(params["grain_columns"])
feature_columns = parse_list_param(params["feature_columns"])
test_split_ratio = float(params.get("test_split_ratio") or 0.2)
n_estimators = int(params.get("xgboost_n_estimators") or 500)
max_depth = int(params.get("xgboost_max_depth") or 6)
learning_rate = float(params.get("xgboost_learning_rate") or 0.05)
experiment_name = params.get("experiment_name") or "demand_forecast"
model_prefix = params.get("registered_model_prefix") or "demand_model"

if not silver_lakehouse_id:
    raise ValueError("silver_lakehouse_id is required.")

print("[xgboost_main] Loading feature table from silver lakehouse.")
spark_df = read_lakehouse_table(spark, silver_lakehouse_id, "feature_table")
pdf = spark_df.toPandas()
print(f"[xgboost_main] Loaded {len(pdf)} rows.")

pdf = pdf.dropna(subset=[target_column]).reset_index(drop=True)

all_feature_cols = [c for c in pdf.columns if c not in [date_column, target_column] + grain_columns and pdf[c].dtype in ["float64", "int64", "float32", "int32"]]
print(f"[xgboost_main] Using {len(all_feature_cols)} numeric feature columns.")

train_df, test_df = time_split(pdf, date_column, test_split_ratio)
print(f"[xgboost_main] Train: {len(train_df)}, Test: {len(test_df)}")

model, train_preds, test_preds, test_metrics = train_xgboost(
    train_df=train_df,
    test_df=test_df,
    feature_cols=all_feature_cols,
    target_col=target_column,
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    experiment_name=experiment_name,
    model_name=f"{model_prefix}_xgboost",
)

import pandas as pd

test_df = test_df.copy()
test_df["predicted"] = test_preds
test_df["model_type"] = "xgboost"
preds_spark = spark.createDataFrame(test_df)
write_lakehouse_table(preds_spark, silver_lakehouse_id, "xgboost_predictions", mode="overwrite")

print("[xgboost_main] Complete.")
