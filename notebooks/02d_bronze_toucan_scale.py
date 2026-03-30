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

base_url = "https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M/resolve/refs%2Fconvert%2Fparquet"

# Load Kimi-K2 (519K rows) and Qwen3 (552K rows)
subsets = ["Kimi-K2", "Qwen3"]

for subset in subsets:
    print(f"\n{'='*60}")
    print(f"Loading subset: {subset}")
    print(f"{'='*60}")
    
    for shard_num in range(20):  # Try up to 20 shards per subset
        shard = f"{shard_num:04d}"
        source_label = f"toucan_{subset.lower().replace('-','_')}_{shard}"
        
        try:
            # Check if already loaded
            existing = spark.sql(f"""
                SELECT COUNT(*) as cnt FROM {target_table}
                WHERE source = '{source_label}'
            """).first()["cnt"]
            
            if existing > 0:
                print(f"  ⚠️ Shard {shard}: already loaded ({existing} rows). Skipping.")
                continue
            
            # Download
            url = f"{base_url}/{subset}/train/{shard}.parquet"
            local_path = f"/tmp/toucan_{subset}_{shard}.parquet"
            dbfs_path = f"dbfs:/tmp/toucan_{subset}_{shard}.parquet"
            
            urllib.request.urlretrieve(url, local_path)
            dbutils.fs.cp(f"file:{local_path}", dbfs_path)
            
            # Read with schema enforcement
            df = spark.read.schema(expected_schema).parquet(dbfs_path)
            row_count = df.count()
            run_id = str(uuid.uuid4())
            
            # Add metadata and append
            df_bronze = (
                df
                .withColumn("source", F.lit(source_label))
                .withColumn("ingest_ts", F.current_timestamp())
                .withColumn("run_id", F.lit(run_id))
            )
            
            df_bronze.write.mode("append").saveAsTable(target_table)
            print(f"  ✅ Shard {shard}: loaded {row_count} rows")
            
            # Clean up DBFS to save space
            dbutils.fs.rm(dbfs_path)
            
        except Exception as e:
            if "404" in str(e):
                print(f"  ⚠️ Shard {shard}: no more shards. Moving to next subset.")
                break
            else:
                print(f"  ❌ Shard {shard}: {e}")

# Final count
total = spark.sql(f"SELECT COUNT(*) as total FROM {target_table}").first()["total"]
print(f"\n{'='*60}")
print(f"✅ TOTAL BRONZE ROWS: {total}")
print(f"{'='*60}")

# Show breakdown by source
spark.sql(f"""
    SELECT 
        CASE 
            WHEN source LIKE '%sft%' THEN 'SFT'
            WHEN source LIKE '%kimi%' THEN 'Kimi-K2'
            WHEN source LIKE '%qwen%' THEN 'Qwen3'
        END as subset,
        COUNT(*) as rows
    FROM {target_table}
    GROUP BY 1
    ORDER BY 1
""").show()

# COMMAND ----------

for subset in ["Kimi-K2", "Qwen3"]:
    print(f"\n{'='*60}")
    print(f"Continuing: {subset} (shards 0020-0049)")
    print(f"{'='*60}")
    
    for shard_num in range(20, 50):
        shard = f"{shard_num:04d}"
        source_label = f"toucan_{subset.lower().replace('-','_')}_{shard}"
        
        try:
            existing = spark.sql(f"""
                SELECT COUNT(*) as cnt FROM {target_table}
                WHERE source = '{source_label}'
            """).first()["cnt"]
            
            if existing > 0:
                print(f"  ⚠️ Shard {shard}: already loaded. Skipping.")
                continue
            
            url = f"{base_url}/{subset}/train/{shard}.parquet"
            local_path = f"/tmp/toucan_{subset}_{shard}.parquet"
            dbfs_path = f"dbfs:/tmp/toucan_{subset}_{shard}.parquet"
            
            urllib.request.urlretrieve(url, local_path)
            dbutils.fs.cp(f"file:{local_path}", dbfs_path)
            
            df = spark.read.schema(expected_schema).parquet(dbfs_path)
            row_count = df.count()
            run_id = str(uuid.uuid4())
            
            df_bronze = (
                df
                .withColumn("source", F.lit(source_label))
                .withColumn("ingest_ts", F.current_timestamp())
                .withColumn("run_id", F.lit(run_id))
            )
            
            df_bronze.write.mode("append").saveAsTable(target_table)
            print(f"  ✅ Shard {shard}: loaded {row_count} rows")
            
            dbutils.fs.rm(dbfs_path)
            
        except Exception as e:
            if "404" in str(e):
                print(f"  ⚠️ Shard {shard}: no more shards. Done with {subset}.")
                break
            else:
                print(f"  ❌ Shard {shard}: {e}")

total = spark.sql(f"SELECT COUNT(*) as total FROM {target_table}").first()["total"]
print(f"\n✅ TOTAL BRONZE ROWS: {total}")

spark.sql(f"""
    SELECT 
        CASE 
            WHEN source LIKE '%sft%' THEN 'SFT'
            WHEN source LIKE '%kimi%' THEN 'Kimi-K2'
            WHEN source LIKE '%qwen%' THEN 'Qwen3'
        END as subset,
        COUNT(*) as rows
    FROM {target_table}
    GROUP BY 1
    ORDER BY 1
""").show()