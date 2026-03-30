# Databricks notebook source
# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, StructType, StructField

CATALOG = "bootcamp_students"
BRONZE = f"{CATALOG}.vadlamani_bronze.trail_raw"
SILVER_TRAIL = f"{CATALOG}.vadlamani_silver.trail_spans"

df = spark.table(BRONZE)
print(f"✅ Read {df.count()} rows from Bronze TRAIL")

# COMMAND ----------

# Parse trace JSON to extract span count and trace_id
# Parse labels JSON to extract error details

trace_schema = StructType([
    StructField("trace_id", StringType()),
    StructField("spans", ArrayType(StructType([
        StructField("span_id", StringType()),
        StructField("parent_span_id", StringType()),
        StructField("span_name", StringType()),
        StructField("span_kind", StringType()),
        StructField("service_name", StringType())
    ])))
])

labels_schema = StructType([
    StructField("trace_id", StringType()),
    StructField("errors", ArrayType(StructType([
        StructField("category", StringType()),
        StructField("location", StringType()),
        StructField("evidence", StringType()),
        StructField("description", StringType())
    ])))
])

df_parsed = (
    df
    .withColumn("trace_parsed", F.from_json("trace", trace_schema))
    .withColumn("labels_parsed", F.from_json("labels", labels_schema))
    .withColumn("trace_id", F.col("trace_parsed.trace_id"))
    .withColumn("span_count", F.size("trace_parsed.spans"))
    .withColumn("error_count", F.size("labels_parsed.errors"))
    .withColumn("parse_status",
        F.when(F.col("trace_parsed").isNull() | F.col("labels_parsed").isNull(), "FAILED")
         .otherwise("OK"))
)

df_parsed.groupBy("parse_status").count().show()

# COMMAND ----------

# Explode errors to get one row per error with its category
df_errors = (
    df_parsed
    .filter(F.col("parse_status") == "OK")
    .select(
        "trace_id",
        "span_count",
        "error_count",
        F.col("trail_source"),
        F.explode_outer("labels_parsed.errors").alias("error")
    )
    .select(
        "trace_id",
        "span_count",
        "error_count",
        "trail_source",
        F.col("error.category").alias("error_category"),
        F.col("error.location").alias("error_span_id"),
        F.col("error.evidence").alias("error_evidence"),
        F.col("error.description").alias("error_description")
    )
    .withColumn("ingest_date", F.current_date())
)

print(f"✅ Exploded to {df_errors.count()} error rows from {df_parsed.filter(F.col('parse_status') == 'OK').count()} traces")
df_errors.show(5, truncate=50)

# COMMAND ----------

# Write Silver TRAIL table
(
    df_errors
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SILVER_TRAIL)
)

print(f"✅ Silver TRAIL written: {SILVER_TRAIL}")

# Summary stats
print("\n=== Error Categories ===")
df_errors.groupBy("error_category").agg(
    F.count("*").alias("error_count"),
    F.countDistinct("trace_id").alias("affected_traces")
).orderBy(F.desc("error_count")).show(truncate=False)

print("\n=== Errors by Source ===")
df_errors.groupBy("trail_source").agg(
    F.count("*").alias("errors"),
    F.countDistinct("trace_id").alias("traces")
).show()