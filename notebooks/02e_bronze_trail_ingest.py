# Databricks notebook source
from huggingface_hub import hf_hub_download
import os

os.environ["HF_TOKEN"] = "hf_token"  # <-- your actual token

# Download both TRAIL parquet files
for filename in ["data/gaia-00000-of-00001-33a2e72d362d688a.parquet", 
                  "data/swe_bench-00000-of-00001-91aa04220f7198b4.parquet"]:
    local = hf_hub_download(
        repo_id="PatronusAI/TRAIL",
        filename=filename,
        repo_type="dataset",
        token=os.environ["HF_TOKEN"],
        local_dir="/tmp/trail"
    )
    print(f"✅ Downloaded: {local}")

# COMMAND ----------

# Copy both files to DBFS
dbutils.fs.cp("file:/tmp/trail/data/gaia-00000-of-00001-33a2e72d362d688a.parquet", "dbfs:/tmp/trail_gaia.parquet")
dbutils.fs.cp("file:/tmp/trail/data/swe_bench-00000-of-00001-91aa04220f7198b4.parquet", "dbfs:/tmp/trail_swe.parquet")

# Read both
df_gaia = spark.read.parquet("dbfs:/tmp/trail_gaia.parquet")
df_swe = spark.read.parquet("dbfs:/tmp/trail_swe.parquet")

print(f"GAIA: {df_gaia.count()} rows")
df_gaia.printSchema()

print(f"\nSWE-Bench: {df_swe.count()} rows")
df_swe.printSchema()

# COMMAND ----------

from pyspark.sql import functions as F
import uuid

CATALOG = "bootcamp_students"
target_table = f"{CATALOG}.vadlamani_bronze.trail_raw"

# Union both TRAIL sources with a source label
df_gaia_tagged = df_gaia.withColumn("trail_source", F.lit("gaia"))
df_swe_tagged = df_swe.withColumn("trail_source", F.lit("swe_bench"))

# They might have different schemas, so read them separately
run_id = str(uuid.uuid4())

df_all = df_gaia_tagged.unionByName(df_swe_tagged, allowMissingColumns=True)

df_bronze = (
    df_all
    .withColumn("source", F.lit("trail"))
    .withColumn("ingest_ts", F.current_timestamp())
    .withColumn("run_id", F.lit(run_id))
)

# Idempotent write
try:
    existing = spark.sql(f"SELECT COUNT(*) as cnt FROM {target_table}").first()["cnt"]
    if existing > 0:
        print(f"⚠️ TRAIL already loaded ({existing} rows). Skipping.")
    else:
        df_bronze.write.mode("append").saveAsTable(target_table)
        print(f"✅ Created {target_table} with {df_bronze.count()} rows")
except Exception:
    df_bronze.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(target_table)
    print(f"✅ Created {target_table} with {df_bronze.count()} rows")

# Verify
spark.sql(f"SELECT COUNT(*) as total FROM {target_table}").show()