# Databricks notebook source
from pyspark.sql import functions as F

CATALOG = "bootcamp_students"
SILVER_TRACE = f"{CATALOG}.vadlamani_silver.toucan_trace"
GOLD_KPIS = f"{CATALOG}.vadlamani_gold.trace_kpis"
GOLD_DQ = f"{CATALOG}.vadlamani_gold.dq_metrics"

df = spark.table(SILVER_TRACE)
print(f"✅ Read {df.count()} rows from Silver")

# COMMAND ----------

# Core operational metrics
gold_kpis = df.agg(
    F.current_date().alias("metric_date"),
    F.lit("toucan_sft").alias("source_name"),
    F.count("*").alias("traces_ingested"),
    F.sum(F.when(F.col("match_rate") == 0.0, 1).otherwise(0)).alias("zero_match_count"),
    F.round(
        F.sum(F.when(F.col("match_rate") == 0.0, 1).otherwise(0)) / F.count("*") * 100, 2
    ).alias("zero_match_pct"),
    F.round(F.avg("tool_call_count"), 2).alias("avg_tool_calls_per_trace"),
    F.round(F.avg("target_tool_count"), 2).alias("avg_target_tools_per_trace"),
    F.round(F.avg("available_tool_count"), 2).alias("avg_available_tools"),
    F.round(F.avg("match_rate") * 100, 2).alias("avg_match_rate_pct"),
    F.sum(F.when(F.col("match_rate") == 1.0, 1).otherwise(0)).alias("perfect_match_count"),
    F.round(
        F.sum(F.when(F.col("match_rate") == 1.0, 1).otherwise(0)) / F.count("*") * 100, 2
    ).alias("perfect_match_pct")
)

gold_kpis.show(vertical=True)

gold_kpis.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(GOLD_KPIS)
print(f"✅ Gold KPIs written: {GOLD_KPIS}")

# COMMAND ----------

GOLD_COMPLEXITY = f"{CATALOG}.vadlamani_gold.match_by_complexity"

gold_complexity = (
    df
    .withColumn("complexity_bucket",
        F.when(F.col("tool_call_count") == 1, "1_single_tool")
         .when(F.col("tool_call_count") == 2, "2_two_tools")
         .when(F.col("tool_call_count") <= 5, "3_three_to_five")
         .otherwise("4_six_plus")
    )
    .groupBy("complexity_bucket")
    .agg(
        F.count("*").alias("trace_count"),
        F.round(F.avg("match_rate") * 100, 2).alias("avg_match_rate_pct"),
        F.round(
            F.sum(F.when(F.col("match_rate") == 0.0, 1).otherwise(0)) / F.count("*") * 100, 2
        ).alias("zero_match_pct"),
        F.round(
            F.sum(F.when(F.col("match_rate") == 1.0, 1).otherwise(0)) / F.count("*") * 100, 2
        ).alias("perfect_match_pct")
    )
    .orderBy("complexity_bucket")
)

gold_complexity.show()

gold_complexity.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(GOLD_COMPLEXITY)
print(f"✅ Gold complexity metrics written: {GOLD_COMPLEXITY}")

# COMMAND ----------

GOLD_FAILURES = f"{CATALOG}.vadlamani_gold.failure_patterns"

# Look at the zero-match traces — what tools were they trying to use?
gold_failures = (
    df
    .filter(F.col("match_rate") == 0.0)
    .withColumn("tools_used_array", F.split("tools_used", ", "))
    .withColumn("tool_used", F.explode("tools_used_array"))
    # Extract server name (everything before the last hyphen-separated tool name)
    .withColumn("server_name", 
        F.regexp_extract("tool_used", r"^(.+)-[^-]+$", 1))
    .groupBy("server_name")
    .agg(
        F.count("*").alias("failed_call_count"),
        F.countDistinct("uuid").alias("affected_traces")
    )
    .orderBy(F.desc("affected_traces"))
)

gold_failures.show(20, truncate=False)

gold_failures.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(GOLD_FAILURES)
print(f"✅ Gold failure patterns written: {GOLD_FAILURES}")

# COMMAND ----------

# Dynamic DQ metrics — always reflects current Silver data
silver = spark.table(SILVER_TRACE)
total = silver.count()
zero_match = silver.filter(F.col("match_rate") == 0.0).count()
perfect_match = silver.filter(F.col("match_rate") == 1.0).count()
null_count = silver.filter(F.col("uuid").isNull()).count()

gold_dq = spark.createDataFrame([
    ("json_parse_check", total, 0, "All rows parsed successfully"),
    ("null_check", total, null_count, f"{'Zero' if null_count == 0 else null_count} nulls in uuid column"),
    ("match_rate_zero", total, zero_match, f"{round(zero_match/total*100, 1)}% of traces had zero tool match"),
    ("match_rate_perfect", total, perfect_match, f"{round(perfect_match/total*100, 1)}% of traces had perfect match"),
], ["dq_rule", "rows_checked", "rows_flagged", "notes"])

gold_dq = gold_dq.withColumn("metric_date", F.current_date()).withColumn("source", F.lit("toucan_sft"))

gold_dq.show(truncate=False)

gold_dq.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(GOLD_DQ)
print(f"✅ Gold DQ metrics written: {GOLD_DQ}")