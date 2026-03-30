# Databricks notebook source
from pyspark.sql import functions as F

CATALOG = "bootcamp_students"

# ── Bronze checks ──
bronze = spark.table(f"{CATALOG}.vadlamani_bronze.toucan_raw")
bronze_count = bronze.count()
checks = []

# Check 1: No nulls in primary key
null_uuids = bronze.filter(F.col("uuid").isNull()).count()
checks.append(("bronze_null_uuid", null_uuids == 0, null_uuids))

# Check 2: No empty messages
empty_msgs = bronze.filter(F.length("messages") < 10).count()
checks.append(("bronze_empty_messages", empty_msgs == 0, empty_msgs))

# Check 3: Unique composite key (uuid + model_name)
distinct = bronze.withColumn("model_name",
    F.when(F.col("source").like("%kimi%"), "Kimi-K2")
     .when(F.col("source").like("%qwen%"), "Qwen3")
     .otherwise("SFT")
).select("uuid", "model_name").distinct().count()
checks.append(("bronze_composite_key_unique", bronze_count == distinct, bronze_count - distinct))

# Check 4: All sources accounted for
sources = bronze.select("source").distinct().collect()
source_list = [r["source"] for r in sources]
checks.append(("bronze_known_sources", all("toucan" in s for s in source_list), len(source_list)))

# ── Silver checks ──
silver = spark.table(f"{CATALOG}.vadlamani_silver.toucan_trace")
silver_count = silver.count()

# Check 5: Silver count matches Bronze
checks.append(("silver_count_matches_bronze", silver_count == bronze_count, 
               abs(silver_count - bronze_count)))

# Check 6: No nulls in match_rate
null_match = silver.filter(F.col("match_rate").isNull()).count()
checks.append(("silver_null_match_rate", null_match == 0, null_match))

# Check 7: match_rate between 0 and 1
bad_range = silver.filter((F.col("match_rate") < 0) | (F.col("match_rate") > 1)).count()
checks.append(("silver_match_rate_range", bad_range == 0, bad_range))

# Check 8: tool_call_count >= 0
negative_tools = silver.filter(F.col("tool_call_count") < 0).count()
checks.append(("silver_no_negative_tools", negative_tools == 0, negative_tools))

# Check 9: No duplicate composite keys in Silver
silver_distinct = silver.select("uuid", "model_name").distinct().count()
checks.append(("silver_composite_key_unique", silver_count == silver_distinct, silver_count - silver_distinct))

# ── Gold checks ──
gold_kpis = spark.table(f"{CATALOG}.vadlamani_gold.trace_kpis")

# Check 10: Gold KPIs exist and have data
kpi_count = gold_kpis.count()
checks.append(("gold_kpis_populated", kpi_count > 0, kpi_count))

# Check 11: traces_ingested matches Silver
kpi_row = gold_kpis.first()
checks.append(("gold_kpi_count_matches_silver", 
               kpi_row["traces_ingested"] == silver_count,
               abs(kpi_row["traces_ingested"] - silver_count)))

# ── Print Results ──
print("=" * 65)
print("  AGENTTRACEOPS — DATA QUALITY CHECK RESULTS")
print("=" * 65)
all_passed = True
pass_count = 0
fail_count = 0

for name, passed, violations in checks:
    status = "✅ PASS" if passed else "❌ FAIL"
    if passed:
        pass_count += 1
    else:
        fail_count += 1
        all_passed = False
    print(f"  {status}  {name:40s}  violations: {violations}")

print("=" * 65)
print(f"  Results: {pass_count} passed, {fail_count} failed out of {len(checks)} checks")
if all_passed:
    print("  ✅ ALL CHECKS PASSED — pipeline is healthy")
else:
    print("  ❌ SOME CHECKS FAILED — investigate before proceeding")
print("=" * 65)

# COMMAND ----------

spark.sql("""
    SELECT subset_name, COUNT(*) as cnt, COUNT(DISTINCT uuid) as unique_uuids
    FROM bootcamp_students.vadlamani_bronze.toucan_raw
    GROUP BY subset_name
""").show(truncate=False)

# COMMAND ----------

total = spark.sql("SELECT COUNT(*) as cnt FROM bootcamp_students.vadlamani_bronze.toucan_raw").first()["cnt"]
distinct = spark.sql("""
    SELECT COUNT(*) as cnt FROM (
        SELECT DISTINCT uuid, 
            CASE 
                WHEN source LIKE '%kimi%' THEN 'Kimi-K2'
                WHEN source LIKE '%qwen%' THEN 'Qwen3'
                ELSE 'SFT'
            END as model_name
        FROM bootcamp_students.vadlamani_bronze.toucan_raw
    )
""").first()["cnt"]

print(f"Total rows: {total}")
print(f"Distinct (uuid, model): {distinct}")
print(f"Duplicates: {total - distinct}")

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS bootcamp_students.vadlamani_silver.toucan_trace")
print("✅ Silver dropped")