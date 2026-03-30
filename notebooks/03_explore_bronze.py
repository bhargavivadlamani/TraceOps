# Databricks notebook source
# How big are these message blobs?
from pyspark.sql import functions as F

df = spark.table("bootcamp_students.vadlamani_bronze.toucan_raw")

df.select(
    F.length("messages").alias("msg_length"),
    F.length("tools").alias("tools_length"),
    F.length("target_tools").alias("target_length")
).summary("min", "mean", "max").show()

# COMMAND ----------

# Grab one row and look at the messages JSON
row = df.select("uuid", "question", "messages").first()

print("QUESTION:", row["question"][:200])
print("\n" + "="*80)
print("\nMESSAGES (first 2000 chars):")
print(row["messages"][:2000])

# COMMAND ----------

df.select("target_tools").show(10, truncate=False)

# COMMAND ----------

df.select(
    F.substring("tools", 1, 300).alias("tools_preview")
).show(5, truncate=False)

# COMMAND ----------

df.select(
    [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in df.columns]
).show(vertical=True)