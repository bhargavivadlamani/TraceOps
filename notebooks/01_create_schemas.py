# Databricks notebook source
# ── Notebook 01: Create Schemas ──
CATALOG = "bootcamp_students"

spark.sql(f"USE CATALOG {CATALOG}")

spark.sql("CREATE SCHEMA IF NOT EXISTS vadlamani_bronze")
spark.sql("CREATE SCHEMA IF NOT EXISTS vadlamani_silver")
spark.sql("CREATE SCHEMA IF NOT EXISTS vadlamani_gold")

print("✅ Schemas created:")
print(f"  - {CATALOG}.vadlamani_bronze")
print(f"  - {CATALOG}.vadlamani_silver")
print(f"  - {CATALOG}.vadlamani_gold")