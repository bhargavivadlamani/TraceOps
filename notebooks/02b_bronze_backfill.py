# Databricks notebook source
import urllib.request
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
import uuid

CATALOG = "bootcamp_students"
SCHEMA = "vadlamani_bronze"
TABLE = "toucan_raw"
target_table = f"{CATALOG}.{SCHEMA}.{TABLE}"

expected_schema = StructType([
    StructField("uuid", StringType(), False),
    StructField("subset_name", StringType(), True),
    StructField("question", StringType(), True),
    StructField("target_tools", StringType(), True),
    StructField("tools", StringType(), True),
    StructField("messages", StringType(), True),
])

base_url = "https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M/resolve/refs%2Fconvert%2Fparquet/SFT/train"

for shard in ["0001", "0002", "0003"]:
    try:
        url = f"{base_url}/{shard}.parquet"
        local_path = f"/tmp/toucan_sft_{shard}.parquet"
        dbfs_path = f"dbfs:/tmp/toucan_sft_{shard}.parquet"
        
        # Download
        urllib.request.urlretrieve(url, local_path)
        dbutils.fs.cp(f"file:{local_path}", dbfs_path)
        
        # Read with schema enforcement
        df = spark.read.schema(expected_schema).parquet(dbfs_path)
        run_id = str(uuid.uuid4())
        
        # Check if this shard was already loaded
        source_label = f"toucan_sft_{shard}"
        existing = spark.sql(f"""
            SELECT COUNT(*) as cnt FROM {target_table} 
            WHERE source = '{source_label}'
        """).first()["cnt"]
        
        if existing > 0:
            print(f"⚠️ Shard {shard}: already loaded ({existing} rows). Skipping.")
            continue
        
        # Add metadata and append
        df_bronze = (
            df
            .withColumn("source", F.lit(source_label))
            .withColumn("ingest_ts", F.current_timestamp())
            .withColumn("run_id", F.lit(run_id))
        )
        
        df_bronze.write.mode("append").saveAsTable(target_table)
        print(f"✅ Shard {shard}: loaded {df.count()} rows")
        
    except Exception as e:
        print(f"⚠️ Shard {shard}: {e}")

# Final count
total = spark.sql(f"SELECT COUNT(*) as total FROM {target_table}").first()["total"]
print(f"\n✅ Total Bronze rows: {total}")