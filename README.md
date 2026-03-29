# AgentTraceOps

A Databricks lakehouse pipeline that ingests 119K AI agent traces, parses tool-call behavior, and answers one question: **when agents have access to the right tools, do they actually use them?**

The answer: **less than half the time.**

## What I Found

I analyzed 119,287 agent execution traces from the Toucan-1.5M dataset. Each trace records a user question, the tools an agent had available, the tools it was expected to use, and the full conversation of what it actually did.

**Finding 1 — Agents either nail it or completely miss.** 47.85% of traces achieved a perfect tool match. 52.07% achieved zero match. Only 0.08% fell in between. There is almost no middle ground — agents don't partially fail, they fail entirely.

**Finding 2 — Single-tool tasks are nearly flawless. Multi-tool tasks break down.** When an agent only needs one tool, it picks the right one 99.9% of the time. The moment it needs two or more tools, accuracy drops to 25–33%. The failure mode isn't "picking bad tools" — it's coordinating across multiple tools.

**Finding 3 — Certain MCP servers are disproportionately associated with failures.** The dictionary server alone accounted for ~40K failed tool calls. Weather-related servers collectively represent the largest failure cluster. These aren't random — they point to specific tool schemas or descriptions that confuse agents.

**Finding 4 — First impressions lie.** My initial 39K-row sample showed a 78.5% match rate. After scaling to 119K rows, the real rate was 47.9%. The first shard was unrepresentatively clean.

## How the Pipeline Works

Raw parquet files from Hugging Face go through three layers before becoming dashboard-ready metrics.

**Bronze** stores the raw data exactly as it arrived — 119,287 rows across 3 parquet shards, each tagged with a source label, ingestion timestamp, and run ID. The ingestion is idempotent: running the notebook twice doesn't create duplicates because it checks if each shard's source label already exists before writing.

**Silver** parses the nested JSON in each trace. The `messages` column (a JSON array of user/assistant/tool_call/tool_response objects) gets broken apart to count tool calls, extract tool names, and compare them against the expected tools. A key challenge here: actual tool names include an MCP server prefix (`lyrical-mcp-find_rhymes`) while target tools use short names (`find_rhymes`). I used `endsWith` matching to bridge this gap. Silver processing is incremental — it tracks which UUIDs are already processed and only transforms new Bronze rows.

**Gold** aggregates Silver into four tables: headline KPIs, match rates bucketed by tool complexity, failure patterns grouped by MCP server, and dynamically computed data quality metrics.

The full pipeline is orchestrated as a Databricks Workflow with six tasks running in sequence:

```
create_schemas → bronze_ingest → bronze_backfill → silver_transform → gold_kpis → dq_checks
```

## What the Dashboard Shows
<img width="4000" height="3339" alt="dashboard-1" src="https://github.com/user-attachments/assets/7ce3a54a-7b1a-4141-aca3-1a4250515892" />


Four KPI tiles at the top: traces ingested (119.29K), average match rate (47.9%), zero match rate (52.07%), and perfect match rate (47.85%).

Below that, two charts. The first shows match rate by tool complexity — a steep cliff from single-tool (near 100%) to multi-tool (~25-33%). The second shows the top 10 MCP servers associated with failed traces, with dictionary and weather servers dominating.

At the bottom, a data quality summary table showing all checks passing across 119,287 rows.

## Engineering Decisions

**Why idempotent ingestion?** Each parquet shard gets a unique `source` label (e.g., `toucan_sft_0001`). Before writing, the notebook checks if that label already exists in Bronze. This means the pipeline is safe to rerun — it won't duplicate data.

**Why schema enforcement?** Bronze reads data with an explicitly defined `StructType` rather than letting Spark infer the schema. If a future shard has different columns, the pipeline fails immediately instead of silently loading mismatched data.

**Why incremental Silver?** Silver uses a `LEFT ANTI JOIN` on UUID to skip rows that are already processed. When new shards are added to Bronze, only the delta flows through Silver — no full reprocessing.

**Why quarantine?** Records that fail JSON parsing get routed to a separate quarantine table instead of being silently dropped. In this dataset zero rows were quarantined, but the mechanism exists for when messier data arrives.

**Why 11 DQ checks?** Automated validation runs across all three layers: null keys, empty messages, UUID uniqueness, match rate ranges, row count reconciliation between layers, and Gold KPI alignment. All 11 pass.

## Tables

```
bootcamp_students.vadlamani_bronze
  └── toucan_raw              119,287 rows    Raw traces + ingestion metadata

bootcamp_students.vadlamani_silver
  └── toucan_trace            119,287 rows    Parsed traces with match rates

bootcamp_students.vadlamani_gold
  ├── trace_kpis              1 row           Headline operational metrics
  ├── match_by_complexity     4 rows          Match rate by tool count bucket
  ├── failure_patterns        50+ rows        Failed calls grouped by MCP server
  └── dq_metrics              4 rows          Data quality check results
```

## Notebooks

```
AgentTraceOps/
  01_create_schemas.py              Creates bronze/silver/gold schemas (run once)
  02_bronze_toucan_ingest.py        Downloads shard 0000 → Bronze (idempotent)
  02b_bronze_backfill.py            Downloads shards 0001-0002 → Bronze (idempotent)
  03_explore_bronze.py              Data profiling and exploration
  04_silver_toucan.py               Parses traces, computes match rates (incremental)
  05_gold_kpis.py                   Aggregates into KPI, complexity, failure, DQ tables
  06_dq_checks.py                   11 automated checks across all layers
```

## Workarounds I Ran Into

**Hugging Face datasets library breaks on Databricks Runtime 18.1.** The `load_dataset()` function throws a `maxdepth` error due to a conflict with Databricks' patched `huggingface_hub`. Fix: bypass the library entirely and read parquet files directly from Hugging Face URLs.

**Spark can't read from local `/tmp/` on Databricks.** Downloaded files land on the driver's local disk, but Spark expects DBFS paths. Fix: `dbutils.fs.cp("file:/tmp/file.parquet", "dbfs:/tmp/file.parquet")` to copy to DBFS before reading.

**Tool name prefix mismatch tanks match rates to 0%.** Actual tool call names include the MCP server as a prefix (`lyrical-mcp-find_rhymes`) but target tools are just the short name (`find_rhymes`). `array_intersect` returns zero matches. Fix: use `endsWith()` matching — `"lyrical-mcp-find_rhymes".endswith("find_rhymes")` returns true.

## If This Were Production

Things I'd add with more time: Delta MERGE for true upsert logic instead of append-based incremental loading. Table partitioning by `subset_name` for query performance at scale. Delta Live Tables with built-in expectations for declarative DQ. Streaming ingestion via Auto Loader for real-time trace processing. SQL Alerts when zero-match rate exceeds a threshold. Integration with the TRAIL dataset for error categorization. Loading the remaining Toucan subsets (Kimi-K2, Qwen3, OSS) to scale to 1.6M+ rows.

## Stack

Databricks · Unity Catalog · Delta Lake · PySpark · Databricks Workflows · Databricks AI/BI Dashboard

## Data Source

[Toucan-1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M) by Agent-Ark — SFT subset (119,287 trajectories from 3 parquet shards). Apache 2.0 license.
