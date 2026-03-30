# Databricks notebook source
# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, StructType, StructField

CATALOG = "bootcamp_students"
BRONZE = f"{CATALOG}.vadlamani_bronze.toucan_raw"
SILVER_TRACE = f"{CATALOG}.vadlamani_silver.toucan_trace"
QUARANTINE = f"{CATALOG}.vadlamani_silver.quarantine"

# Read Bronze
df_bronze = spark.table(BRONZE)
bronze_count = df_bronze.count()

# Incremental: only process rows not already in Silver
silver_exists = spark.catalog.tableExists(SILVER_TRACE)

if silver_exists:
    df_bronze_keyed = df_bronze.withColumn("model_name",
        F.when(F.col("source").like("%kimi%"), "Kimi-K2")
         .when(F.col("source").like("%qwen%"), "Qwen3")
         .otherwise("SFT"))
    existing_keys = spark.table(SILVER_TRACE).select("uuid", "model_name")
    df = df_bronze_keyed.join(existing_keys, on=["uuid", "model_name"], how="left_anti")
    new_count = df.count()
    print(f"✅ Bronze has {bronze_count} rows, Silver already has {existing_keys.count()} rows")
    print(f"✅ Processing {new_count} NEW rows only")
else:
    df = df_bronze
    new_count = bronze_count
    print(f"✅ Silver table doesn't exist yet. Processing all {new_count} rows")

# COMMAND ----------

# Skip if no new rows to process
if new_count == 0:
    print("✅ No new rows — Silver is up to date")
else:
    # Extended schema that captures function_call field for Kimi-K2/Qwen3
    message_schema = ArrayType(
        StructType([
            StructField("role", StringType()),
            StructField("content", StringType()),
            StructField("function_call", StructType([
                StructField("name", StringType()),
                StructField("arguments", StringType())
            ])),
            StructField("name", StringType())
        ])
    )

    df_transformed = (
        df
        .withColumn("messages_array", F.from_json("messages", message_schema))
        .withColumn("parse_status", 
            F.when(F.col("messages_array").isNull(), "FAILED").otherwise("OK"))
        # Derive model name from source column
        .withColumn("model_name",
            F.when(F.col("source").like("%kimi%"), "Kimi-K2")
             .when(F.col("source").like("%qwen%"), "Qwen3")
             .otherwise("SFT"))
        .withColumn("total_messages", F.size("messages_array"))
        
        # Count tool calls — SFT uses "tool_call" role, Kimi/Qwen use "assistant" with function_call
        .withColumn("sft_tool_calls", F.size(F.filter("messages_array", lambda x: x["role"] == "tool_call")))
        .withColumn("fc_tool_calls", F.size(F.filter("messages_array", lambda x: x["function_call"]["name"].isNotNull())))
        .withColumn("tool_call_count", F.greatest("sft_tool_calls", "fc_tool_calls"))
        
        # Count tool responses — SFT uses "tool_response", Kimi/Qwen use "function"
        .withColumn("sft_responses", F.size(F.filter("messages_array", lambda x: x["role"] == "tool_response")))
        .withColumn("fc_responses", F.size(F.filter("messages_array", lambda x: x["role"] == "function")))
        .withColumn("tool_response_count", F.greatest("sft_responses", "fc_responses"))
        
        # Extract tool names — handle both formats
        # SFT: tool name in content of tool_call messages
        .withColumn("sft_tool_calls_raw", 
            F.filter("messages_array", lambda x: x["role"] == "tool_call"))
        .withColumn("sft_tools_array",
            F.transform("sft_tool_calls_raw", 
                lambda x: F.regexp_extract(x["content"], r"'name':\s*'([^']+)'", 1)))
        # Kimi/Qwen: tool name in function_call.name field
        .withColumn("fc_tool_calls_raw",
            F.filter("messages_array", lambda x: x["function_call"]["name"].isNotNull()))
        .withColumn("fc_tools_array",
            F.transform("fc_tool_calls_raw", lambda x: x["function_call"]["name"]))
        # Merge both — one will be empty depending on format
        .withColumn("tools_used_array",
            F.when(F.size("fc_tools_array") > 0, F.col("fc_tools_array"))
             .otherwise(F.col("sft_tools_array")))
        .withColumn("tools_used", F.array_join("tools_used_array", ", "))
        
        # Parse target tools
        .withColumn("target_tools_array", F.split(F.trim("target_tools"), r",\s*"))
        .withColumn("target_tool_count", F.size("target_tools_array"))
        # Parse available tools
        .withColumn("tools_json", F.from_json("tools", ArrayType(
            StructType([
                StructField("type", StringType()),
                StructField("function", StructType([
                    StructField("name", StringType()),
                    StructField("description", StringType())
                ]))
            ])
        )))
        .withColumn("available_tool_count", 
            F.when(F.col("tools_json").isNull(), 0).otherwise(F.size("tools_json")))
        # FIXED MATCH: endsWith to handle MCP server prefix
        .withColumn("target_tools_matched",
            F.size(F.filter("target_tools_array", 
                lambda target: F.exists("tools_used_array", 
                    lambda used: used.endswith(target)
                )
            ))
        )
        .withColumn("match_rate",
            F.when(F.col("target_tool_count") == 0, F.lit(None))
             .otherwise(F.col("target_tools_matched") / F.col("target_tool_count")))
    )

    # Split good and bad records
    df_good = df_transformed.filter(F.col("parse_status") == "OK")
    df_bad = df_transformed.filter(F.col("parse_status") == "FAILED")

    print(f"✅ Good records: {df_good.count()}")
    print(f"⚠️ Quarantine records: {df_bad.count()}")

# COMMAND ----------

if new_count > 0:
    # Select final Silver columns
    silver_output = (
        df_good
        .select(
            "uuid", "subset_name", "model_name", "question",
            "total_messages", "tool_call_count", "tool_response_count",
            "tools_used", "target_tools",
            "target_tool_count", "available_tool_count",
            "target_tools_matched", "match_rate",
            "parse_status"
        )
        .withColumn("ingest_date", F.current_date())
    )

    if not silver_exists:
        (
            silver_output
            .write
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(SILVER_TRACE)
        )
        print(f"✅ Silver table created: {SILVER_TRACE}")
    else:
        silver_output.write.mode("append").saveAsTable(SILVER_TRACE)
        print(f"✅ Appended new rows to Silver")

    # Write quarantine if any bad records
    if df_bad.count() > 0:
        (
            df_bad
            .select(
                F.lit("toucan_raw").alias("source_table"),
                "uuid",
                "messages",
                F.lit("json_parse_failed").alias("dq_rule_failed"),
                "parse_status",
                F.current_timestamp().alias("quarantine_ts")
            )
            .write
            .mode("append")
            .option("overwriteSchema", "true")
            .saveAsTable(QUARANTINE)
        )
        print(f"✅ Quarantine records written: {QUARANTINE}")
    else:
        print("✅ No quarantine records")

# COMMAND ----------

# Verify final Silver state
result = spark.table(SILVER_TRACE)
total = result.count()
print(f"✅ Total Silver rows: {total}")

result.select(
    F.avg("tool_call_count").alias("avg_tool_calls"),
    F.avg("target_tool_count").alias("avg_target_tools"),
    F.round(F.avg("match_rate"), 4).alias("avg_match_rate"),
    F.avg("available_tool_count").alias("avg_available_tools")
).show()

# Breakdown by model
result.groupBy("model_name").agg(
    F.count("*").alias("traces"),
    F.round(F.avg("match_rate") * 100, 2).alias("avg_match_rate_pct"),
    F.round(F.avg("tool_call_count"), 2).alias("avg_tool_calls")
).orderBy("model_name").show()

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, StructType, StructField

CATALOG = "bootcamp_students"
BRONZE = f"{CATALOG}.vadlamani_bronze.toucan_raw"
SILVER_TRACE = f"{CATALOG}.vadlamani_silver.toucan_trace"
QUARANTINE = f"{CATALOG}.vadlamani_silver.quarantine"

# Read Bronze
df_bronze = spark.table(BRONZE)
bronze_count = df_bronze.count()

# Incremental: only process rows not already in Silver
silver_exists = spark.catalog.tableExists(SILVER_TRACE)

if silver_exists:
    existing_uuids = spark.table(SILVER_TRACE).select("uuid", "subset_name")
    df = df_bronze.join(existing_keys, on=["uuid", "subset_name"], how="left_anti")
    new_count = df.count()
    print(f"✅ Bronze has {bronze_count} rows, Silver already has {existing_uuids.count()} rows")
    print(f"✅ Processing {new_count} NEW rows only")
else:
    df = df_bronze
    new_count = bronze_count
    print(f"✅ Silver table doesn't exist yet. Processing all {new_count} rows")

# COMMAND ----------

# Skip if no new rows to process
if new_count == 0:
    print("✅ No new rows — Silver is up to date")
else:
    df_transformed = (
        df
        .withColumn("messages_array", F.from_json("messages", ArrayType(
            StructType([
                StructField("role", StringType()),
                StructField("content", StringType())
            ])
        )))
        .withColumn("total_messages", F.size("messages_array"))
        .withColumn("tool_call_count", F.size(F.filter("messages_array", lambda x: x["role"] == "tool_call")))
        .withColumn("tool_response_count", F.size(F.filter("messages_array", lambda x: x["role"] == "tool_response")))
        .withColumn("parse_status", 
            F.when(F.col("messages_array").isNull(), "FAILED").otherwise("OK"))
        # Extract full tool names
        .withColumn("tool_calls_raw", 
            F.filter("messages_array", lambda x: x["role"] == "tool_call"))
        .withColumn("tools_used_array",
            F.transform("tool_calls_raw", 
                lambda x: F.regexp_extract(x["content"], r"'name':\s*'([^']+)'", 1)))
        .withColumn("tools_used", F.array_join("tools_used_array", ", "))
        # Parse target tools
        .withColumn("target_tools_array", F.split(F.trim("target_tools"), r",\s*"))
        .withColumn("target_tool_count", F.size("target_tools_array"))
        # Parse available tools
        .withColumn("tools_json", F.from_json("tools", ArrayType(
            StructType([
                StructField("type", StringType()),
                StructField("function", StructType([
                    StructField("name", StringType()),
                    StructField("description", StringType())
                ]))
            ])
        )))
        .withColumn("available_tool_count", 
            F.when(F.col("tools_json").isNull(), 0).otherwise(F.size("tools_json")))
        # FIXED MATCH: endsWith to handle MCP server prefix
        .withColumn("target_tools_matched",
            F.size(F.filter("target_tools_array", 
                lambda target: F.exists("tools_used_array", 
                    lambda used: used.endswith(target)
                )
            ))
        )
        .withColumn("match_rate",
            F.when(F.col("target_tool_count") == 0, F.lit(None))
             .otherwise(F.col("target_tools_matched") / F.col("target_tool_count")))
    )

    # Split good and bad records
    df_good = df_transformed.filter(F.col("parse_status") == "OK")
    df_bad = df_transformed.filter(F.col("parse_status") == "FAILED")

    print(f"✅ Good records: {df_good.count()}")
    print(f"⚠️ Quarantine records: {df_bad.count()}")

# COMMAND ----------

if new_count > 0:
    # Select final Silver columns
    silver_output = (
        df_good
        .select(
            "uuid", "subset_name", "question",
            "total_messages", "tool_call_count", "tool_response_count",
            "tools_used", "target_tools",
            "target_tool_count", "available_tool_count",
            "target_tools_matched", "match_rate",
            "parse_status"
        )
        .withColumn("ingest_date", F.current_date())
    )

    if not silver_exists:
        # First run — create the table
        (
            silver_output
            .write
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(SILVER_TRACE)
        )
        print(f"✅ Silver table created: {SILVER_TRACE}")
    else:
        # Incremental — append new rows only
        silver_output.write.mode("append").saveAsTable(SILVER_TRACE)
        print(f"✅ Appended {silver_output.count()} new rows to Silver")

    # Write quarantine if any bad records
    if df_bad.count() > 0:
        (
            df_bad
            .select(
                F.lit("toucan_raw").alias("source_table"),
                "uuid",
                "messages",
                F.lit("json_parse_failed").alias("dq_rule_failed"),
                "parse_status",
                F.current_timestamp().alias("quarantine_ts")
            )
            .write
            .mode("append")
            .option("overwriteSchema", "true")
            .saveAsTable(QUARANTINE)
        )
        print(f"✅ Quarantine records written: {QUARANTINE}")
    else:
        print("✅ No quarantine records")

# COMMAND ----------

# Verify final Silver state
result = spark.table(SILVER_TRACE)
total = result.count()
print(f"✅ Total Silver rows: {total}")

result.select(
    F.avg("tool_call_count").alias("avg_tool_calls"),
    F.avg("target_tool_count").alias("avg_target_tools"),
    F.round(F.avg("match_rate"), 4).alias("avg_match_rate"),
    F.avg("available_tool_count").alias("avg_available_tools")
).show()