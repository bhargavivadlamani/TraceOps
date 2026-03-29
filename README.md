# TraceOps
# AgentTraceOps Lakehouse

**AI agent traces are messy logs. This pipeline turns them into operational intelligence.**

## The Finding

| Metric | Value |
|--------|-------|
| Traces analyzed | 119,287 |
| Avg tool match rate | 47.9% |
| Perfect match (100%) | 47.85% of traces |
| Zero match (0%) | 52.07% of traces |
| Top failure source | Dictionary server (~40K+ affected traces) |
| Data source | Toucan-1.5M SFT subset |

Agents either get every tool right or every tool wrong — almost nothing in between. Single-tool traces succeed 99.9% of the time, but multi-tool coordination drops accuracy to ~25%. Initial analysis on 39K traces showed a 78.5% match rate; scaling to 119K revealed the true rate is 47.9%.

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    AGENTTRACEOPS PIPELINE                          │
│                                                                    │
│  Hugging Face             Databricks              Dashboard        │
│  (Toucan-1.5M)                                                     │
│                                                                    │
│  SFT subset ──►  BRONZE              ──►  4-tile KPI panel         │
│  3 parquet        toucan_raw               Match Rate by Complexity│
│  shards           119,287 rows        ──►  Top 10 Failure Servers  │
│                        │                   DQ Summary Table        │
│                   SILVER                                           │
│                   toucan_trace                                     │
│                   119,287 rows                                     │
│                   + match_rate                                     │
│                   + tools_used                                     │
│                        │                                           │
│                   GOLD                                             │
│                   trace_kpis                                       │
│                   match_by_complexity                               │
│                   failure_patterns                                  │
│                   dq_metrics                                       │
└────────────────────────────────────────────────────────────────────┘
```

## Key Findings

**1. Tool selection is bimodal.** 47.85% of traces achieve perfect tool matches. 52.07% achieve zero matches. Only 0.08% fall in between. Agents don't partially fail — they either nail it or miss entirely.

**2. Multi-tool coordination is the weak point.** Single-tool traces: 99.9% match rate. Two-tool traces: ~25%. Three-to-five tools: ~33%. The failure mode isn't picking bad tools — it's coordinating across multiple tools.

**3. Weather and dictionary servers dominate failures.** The dictionary server alone accounts for ~40K+ failed tool calls. Weather-related servers (united-states-weather, weather-forecast-service, weather-api-server, etc.) collectively represent the largest failure cluster.

**4. Data quality varies across shards.** The first parquet shard showed 78.5% match rate. After loading all three shards, the rate dropped to 47.9% — a reminder that initial samples can be misleading.

## Tables

### Bronze Layer
| Table | Rows | Description |
|-------|------|-------------|
| `vadlamani_bronze.toucan_raw` | 119,287 | Raw Toucan SFT traces with ingestion metadata |

### Silver Layer
| Table | Rows | Description |
|-------|------|-------------|
| `vadlamani_silver.toucan_trace` | 119,287 | Parsed traces with tool counts, match rates, parse status |

### Gold Layer
| Table | Description |
|-------|-------------|
| `vadlamani_gold.trace_kpis` | Headline metrics: match rates, tool counts, ingestion stats |
| `vadlamani_gold.match_by_complexity` | Match rate broken down by tool call count buckets |
| `vadlamani_gold.failure_patterns` | Top MCP servers associated with zero-match traces |
| `vadlamani_gold.dq_metrics` | Data quality check results per pipeline run |

## Gold Table: Data Dictionary

**Table: `vadlamani_gold.trace_kpis`**

| Column | Type | Description |
|--------|------|-------------|
| metric_date | DATE | Date the metrics were computed |
| source_name | STRING | Data source identifier |
| traces_ingested | LONG | Total traces processed |
| zero_match_count | LONG | Traces where no target tools were matched |
| zero_match_pct | DOUBLE | Percentage of zero-match traces |
| avg_tool_calls_per_trace | DOUBLE | Average number of tool calls per trace |
| avg_target_tools_per_trace | DOUBLE | Average number of expected tools per trace |
| avg_available_tools | DOUBLE | Average tools available to the agent |
| avg_match_rate_pct | DOUBLE | Average tool match rate as percentage |
| perfect_match_count | LONG | Traces with 100% tool match |
| perfect_match_pct | DOUBLE | Percentage of perfect-match traces |

## Notebook Execution Order

```
AgentTraceOps/
  01_create_schemas.py          ← Creates bronze/silver/gold schemas (run once)
  02_bronze_toucan_ingest.py    ← Downloads shard 0000, writes to Bronze (idempotent)
  02b_bronze_backfill.py        ← Downloads shards 0001-0002, appends to Bronze (idempotent)
  03_explore_bronze.py          ← Data exploration and profiling
  04_silver_toucan.py           ← Parses traces, computes match rates (incremental)
  05_gold_kpis.py               ← Aggregates KPIs, complexity, failures, DQ metrics
  06_dq_checks.py               ← 11 automated DQ checks across all layers
```

Orchestrated via Databricks Workflow: `create_schemas → bronze_ingest → bronze_backfill → silver_transform → gold_kpis → dq_checks`

## Pipeline Design Principles

**Idempotent ingestion.** Bronze checks if a source has already been loaded before writing. Running the ingestion notebook multiple times never creates duplicates.

**Schema enforcement.** Bronze reads data with an explicitly defined schema rather than relying on Spark inference. If upstream data changes shape, the pipeline fails loudly instead of silently ingesting bad data.

**Incremental Silver processing.** Silver uses a `LEFT ANTI JOIN` against existing UUIDs to process only new Bronze rows. Adding a new parquet shard to Bronze and rerunning Silver processes only the delta.

**Quarantine handling.** Records that fail JSON parsing are routed to a quarantine table instead of being silently dropped. Zero rows were quarantined in this dataset, but the mechanism is in place.

**Automated DQ checks.** 11 checks validate data integrity across Bronze (null keys, empty messages, unique UUIDs), Silver (match rate ranges, no negative counts, row count reconciliation), and Gold (KPI population, count alignment).

## Known Workarounds

```python
# Hugging Face datasets library conflicts with Databricks runtime 18.1
# maxdepth error in HfFileSystem.find() — bypass by reading parquet directly
df = spark.read.parquet("dbfs:/tmp/toucan_sft.parquet")  # ✅
ds = load_dataset("Agent-Ark/Toucan-1.5M", ...)          # ❌ maxdepth error

# Local disk vs DBFS — Spark can't read from /tmp/ directly
dbutils.fs.cp("file:/tmp/file.parquet", "dbfs:/tmp/file.parquet")  # ✅
spark.read.parquet("/tmp/file.parquet")                             # ❌ PATH_NOT_FOUND

# Tool name prefix mismatch — actual calls include MCP server prefix
# tools_used: "lyrical-mcp-find_rhymes"  vs  target_tools: "find_rhymes"
# Fix: use endsWith() matching instead of exact array_intersect
used.endswith(target)  # ✅ matches "lyrical-mcp-find_rhymes" to "find_rhymes"
F.array_intersect()    # ❌ returns zero matches due to prefix mismatch
```

## Production Improvements

If this were a production system, the following would be added:

- **Delta MERGE** for upsert logic instead of append-based incremental loading
- **Delta table partitioning** by `subset_name` or `ingest_date` for query performance at scale
- **DLT (Delta Live Tables)** with built-in expectations for declarative DQ
- **Streaming ingestion** for real-time trace processing via Auto Loader
- **Alerting** via Databricks SQL Alerts when zero-match rate exceeds thresholds
- **TRAIL dataset integration** for error categorization and failure taxonomy enrichment
- **Additional Toucan subsets** (Kimi-K2, Qwen3, OSS) to scale to 1.6M+ rows

## Stack

Databricks · Delta Lake · PySpark · Unity Catalog · Databricks Workflows · Databricks AI/BI Dashboard
