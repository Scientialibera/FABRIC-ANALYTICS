# Fabric Notebook
# 03_feature_engineering_main.py - Bronze -> Silver feature table

# %run ../modules/config_module
# %run ../modules/feature_engineering_module

params = get_notebook_params()

bronze_lakehouse_id = params["bronze_lakehouse_id"]
silver_lakehouse_id = params["silver_lakehouse_id"]
date_column = params["date_column"]
frequency = params["frequency"] or "M"
target_column = params["target_column"]
grain_columns = parse_list_param(params["grain_columns"])
feature_columns = parse_list_param(params["feature_columns"])
source_tables = parse_list_param(params["source_tables"])

if not bronze_lakehouse_id:
    raise ValueError("bronze_lakehouse_id is required.")
if not silver_lakehouse_id:
    raise ValueError("silver_lakehouse_id is required.")
if not date_column or not target_column:
    raise ValueError("date_column and target_column are required.")

print(f"[feature_eng] Grain: {grain_columns}, Date: {date_column}, Target: {target_column}")
print(f"[feature_eng] Features: {feature_columns}, Frequency: {frequency}")

primary_table = source_tables[0] if source_tables else "orders"
print(f"[feature_eng] Reading primary table: {primary_table}")
spark_df = read_lakehouse_table(spark, bronze_lakehouse_id, primary_table)

all_needed = [date_column, target_column] + grain_columns + feature_columns
available = set(spark_df.columns)
missing = [c for c in all_needed if c not in available]
if missing:
    print(f"[feature_eng] WARNING: Missing columns {missing}, proceeding with available.")
    feature_columns = [c for c in feature_columns if c in available]

pdf = spark_df.toPandas()
print(f"[feature_eng] Loaded {len(pdf)} rows from bronze.")

feature_df = build_feature_table(pdf, date_column, grain_columns, target_column, feature_columns, frequency)
print(f"[feature_eng] Feature table: {len(feature_df)} rows, {len(feature_df.columns)} columns")

feature_spark = spark.createDataFrame(feature_df)
write_lakehouse_table(feature_spark, silver_lakehouse_id, "feature_table", mode="overwrite")
print("[feature_eng] Written feature_table to silver lakehouse.")

print("[feature_eng] Complete.")
