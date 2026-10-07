# Binding the Limit

Would capping how fast cars can go save enough lives in Great Britain to be worth the extra
travel time? And would it beat simply holding every car to each road's posted limit?

This project answers that with the government's own data and appraisal values: every
police-reported injury collision since 1979, DfT traffic and speed statistics, and DfT's
official values for a prevented casualty and for an hour of travel time.

**Status:** Phase 3 of 9 complete. All 37.5 million STATS19 rows (1979–2025) are loaded into
PostgreSQL and modelled with dbt into a tested star schema. The pipeline reproduces DfT's
published 2025 headline figures (127,883 casualties, 1,538 deaths, 29,918 killed or seriously
injured), checked automatically on every build. The marts are exported to Parquet.

## Pipeline

```
DfT CSV files → PostgreSQL (raw) → dbt: staging → intermediate → star schema → Parquet
```

| Layer | What it holds |
|---|---|
| `raw` | The three STATS19 tables exactly as published, with primary and foreign keys |
| `staging` | Clear names, `-1` "unknown" codes turned into nulls (views) |
| `intermediate` | km/h speed limits, DfT road categories, keys, reported and adjusted severity (views) |
| `marts` | Star schema: `fct_casualty`, `fct_collision`, `dim_date`, `dim_road`, `dim_police_force`, `dim_casualty_profile`, plus `agg_casualties_year_road` (tables) |

29 data tests run on every build: key uniqueness, fact-to-dimension relationships, allowed
values, row conservation from raw to facts, the rules of DfT's severity adjustment, and
reconciliation against DfT's published totals.

## Run it

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and PostgreSQL 17 with a database
called `speed_gb`.

```bash
make setup      # create the Python environment
make ingest     # download the raw data (about 4.3 GB) and write ingest/manifest.json
make db         # build the raw database: schema, bulk load, keys, code lookups
make transform  # build the dbt layers and run all 29 tests
make export     # export the marts to Parquet in data/marts/
make docs       # browse the documentation and lineage graph
```

## Deliberately not used

Spark, Kafka, Airflow and Kubernetes. The source data is 4.3 GB of CSV: one PostgreSQL
database on a laptop handles it comfortably, and distributed tools would add cost and
complexity without benefit.