# Fabric Notebook
# 06_aggregate_main.py - Gold roll-ups and warehouse-ready aggregate tables

# %run ../modules/config_module

from pyspark.sql import functions as F

params = get_notebook_params()

gold_lakehouse_id = params["gold_lakehouse_id"]
grain_columns = parse_list_param(params["grain_columns"])
output_table = params.get("output_table") or "forecast_results"

if not gold_lakehouse_id:
    raise ValueError("gold_lakehouse_id is required.")

print("[aggregate] Loading forecast results from gold lakehouse.")
forecast_df = read_lakehouse_table(spark, gold_lakehouse_id, output_table)
row_count = forecast_df.count()
print(f"[aggregate] Loaded {row_count} forecast rows.")

if row_count == 0:
    print("[aggregate] No forecast data to aggregate. Exiting.")
else:
    # --- Per-model aggregate by grain ---
    for model_type in ["xgboost", "sarima", "exp_smoothing"]:
        model_df = forecast_df.filter(F.col("model_type") == model_type)
        if model_df.count() == 0:
            continue

        if "predicted" in model_df.columns:
            agg_cols = [F.sum("predicted").alias("total_predicted"), F.count("predicted").alias("n_rows")]
            if model_type == "xgboost" and output_table:
                agg_cols.append(F.avg("predicted").alias("avg_predicted"))

            for level in range(len(grain_columns)):
                group_by = grain_columns[: level + 1]
                agg_df = model_df.groupBy(group_by).agg(*agg_cols)
                agg_df = agg_df.withColumn("model_type", F.lit(model_type))
                agg_df = agg_df.withColumn("aggregation_level", F.lit("|".join(group_by)))
                agg_df = agg_df.withColumn("aggregated_at", F.current_timestamp())

                table_name = f"agg_{model_type}_by_{'_'.join(group_by)}"
                write_lakehouse_table(agg_df, gold_lakehouse_id, table_name, mode="overwrite")
                print(f"[aggregate] Wrote {table_name} ({agg_df.count()} rows).")

    # --- Grand total across all models ---
    if "predicted" in forecast_df.columns:
        grand_agg = forecast_df.groupBy("model_type").agg(
            F.sum("predicted").alias("total_predicted"),
            F.avg("predicted").alias("avg_predicted"),
            F.count("predicted").alias("n_rows"),
        ).withColumn("aggregated_at", F.current_timestamp())
        write_lakehouse_table(grand_agg, gold_lakehouse_id, "agg_grand_total", mode="overwrite")
        print(f"[aggregate] Wrote agg_grand_total ({grand_agg.count()} rows).")

print("[aggregate] Complete.")
