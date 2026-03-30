# Databricks notebook source
# MAGIC %pip install datasets huggingface_hub --upgrade
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Configuration
CATALOG = "bootcamp_students"
SCHEMA = "vadlamani_bronze"
TABLE = "toucan_raw"
SUBSET = "SFT"

# COMMAND ----------

import urllib.request
from pyspark.sql.types import StructType, StructField, StringType

url = "https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M/resolve/refs%2Fconvert%2Fparquet/SFT/train/0000.parquet"

# Step 1: Download to local disk
local_path = "/tmp/toucan_sft.parquet"
urllib.request.urlretrieve(url, local_path)
print("✅ Downloaded to local disk")

# Step 2: Copy to DBFS so Spark can see it
dbutils.fs.cp("file:/tmp/toucan_sft.parquet", "dbfs:/tmp/toucan_sft.parquet")
print("✅ Copied to DBFS")

# Step 3: Define expected schema — enforces column names and types
expected_schema = StructType([
    StructField("uuid", StringType(), False),
    StructField("subset_name", StringType(), True),
    StructField("question", StringType(), True),
    StructField("target_tools", StringType(), True),
    StructField("tools", StringType(), True),
    StructField("messages", StringType(), True),
])

# Step 4: Read with enforced schema — rejects data that doesn't match
df = spark.read.schema(expected_schema).parquet("dbfs:/tmp/toucan_sft.parquet")

# Step 5: Validate schema matches what we expect
actual_cols = set(df.columns)
expected_cols = set([f.name for f in expected_schema.fields])
assert actual_cols == expected_cols, f"Schema mismatch: {actual_cols.symmetric_difference(expected_cols)}"

print(f"✅ Schema enforced. {df.count()} rows loaded")
df.printSchema()

# COMMAND ----------

from pyspark.sql import functions as F
import uuid

run_id = str(uuid.uuid4())

df_bronze = (
    df
    .withColumn("source", F.lit("toucan_sft"))
    .withColumn("ingest_ts", F.current_timestamp())
    .withColumn("run_id", F.lit(run_id))
)

print(f"✅ Metadata columns added. Run ID: {run_id}")
df_bronze.printSchema()

# COMMAND ----------

target_table = f"{CATALOG}.{SCHEMA}.{TABLE}"

# Idempotent ingestion — check if this source was already loaded
try:
    existing_count = spark.sql(f"""
        SELECT COUNT(*) as cnt 
        FROM {target_table} 
        WHERE source = 'toucan_sft'
    """).first()["cnt"]
    
    if existing_count > 0:
        print(f"⚠️ Table already has {existing_count} rows from toucan_sft")
        print("Skipping to avoid duplicate ingestion. Drop table first to reload.")
    else:
        df_bronze.write.mode("append").saveAsTable(target_table)
        print(f"✅ Appended {df_bronze.count()} rows to {target_table}")

except Exception:
    # Table doesn't exist yet — create it
    (
        df_bronze
        .write
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(target_table)
    )
    print(f"✅ Created {target_table} with {df_bronze.count()} rows")

# COMMAND ----------

target_table = f"{CATALOG}.{SCHEMA}.{TABLE}"

row_count = spark.sql(f"SELECT COUNT(*) as row_count FROM {target_table}")
row_count.show()

spark.sql(f"""
    SELECT uuid, subset_name, LEFT(question, 80) as question_preview, source, ingest_ts 
    FROM {target_table} 
    LIMIT 5
""").show(truncate=False)