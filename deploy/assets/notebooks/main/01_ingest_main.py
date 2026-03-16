# Fabric Notebook
# 01_ingest_main.py - Ingest source tables into Landing lakehouse

# %run ../modules/config_module

params = get_notebook_params()

source_lakehouse_id = params["source_lakehouse_id"]
landing_lakehouse_id = params["landing_lakehouse_id"]
source_tables = parse_list_param(params["source_tables"])

if not source_lakehouse_id:
    raise ValueError("source_lakehouse_id is required.")
if not landing_lakehouse_id:
    raise ValueError("landing_lakehouse_id is required.")
if not source_tables:
    raise ValueError("source_tables list is empty.")

for table_name in source_tables:
    print(f"[ingest] Reading source table: {table_name}")
    df = read_lakehouse_table(spark, source_lakehouse_id, table_name)
    row_count = df.count()
    print(f"[ingest] {table_name}: {row_count} rows")
    write_lakehouse_table(df, landing_lakehouse_id, table_name, mode="overwrite")
    print(f"[ingest] Wrote {table_name} to landing lakehouse.")

print("[ingest] Complete.")
