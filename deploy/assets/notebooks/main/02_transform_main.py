# Fabric Notebook
# 02_transform_main.py - Landing -> Bronze cleansing

# %run ../modules/config_module

from pyspark.sql import functions as F

params = get_notebook_params()

landing_lakehouse_id = params["landing_lakehouse_id"]
bronze_lakehouse_id = params["bronze_lakehouse_id"]
source_tables = parse_list_param(params["source_tables"])

if not landing_lakehouse_id:
    raise ValueError("landing_lakehouse_id is required.")
if not bronze_lakehouse_id:
    raise ValueError("bronze_lakehouse_id is required.")
if not source_tables:
    raise ValueError("source_tables list is empty.")

for table_name in source_tables:
    print(f"[transform] Cleaning table: {table_name}")
    df = read_lakehouse_table(spark, landing_lakehouse_id, table_name)

    original_count = df.count()
    df = df.dropDuplicates()
    dedup_count = df.count()

    df = df.dropna(how="all")
    clean_count = df.count()

    df = df.withColumn("_ingested_at", F.current_timestamp())

    write_lakehouse_table(df, bronze_lakehouse_id, table_name, mode="overwrite")
    print(f"[transform] {table_name}: {original_count} -> {dedup_count} (dedup) -> {clean_count} (dropna) written to bronze.")

print("[transform] Complete.")
