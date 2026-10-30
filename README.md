# Binding the Limit

Would capping how fast cars can go save enough lives in Great Britain to be worth the extra
travel time? And would it beat simply holding every car to each road's posted limit?

This project answers that with the government's own data and appraisal values: every
police-reported injury collision since 1979, DfT traffic and speed statistics, and DfT's
official values for a prevented casualty and for an hour of travel time.

**Status:** Phase 4 of 9 complete. All 37.5 million STATS19 rows (1979–2025) are loaded into
PostgreSQL and modelled with dbt into a tested star schema, joined to DfT motor-vehicle
traffic to give rates per billion vehicle-km. The pipeline reproduces DfT's published 2025
headline figures and all 15 of its 2025 casualty counts by road type, checked on every build.

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

49 data tests run on every build: key uniqueness, fact-to-dimension relationships, allowed
values, row conservation from raw to facts, the rules of DfT's severity adjustment, and
reconciliation against DfT's published totals.


## RQ1: where harm happens, per kilometre driven

![Deaths vs traffic by road environment, 2025](figures/rq1_share_deaths_vs_traffic.png)

- Rural roads carried 45% of 2025 traffic but 60% of deaths. Rural minor roads had the
  highest death rate: 4.4 per billion vehicle-km, against 0.8 on motorways.
- Serious injuries rank the other way: adjusted KSI rates are highest on urban roads
  (88–93 per billion vehicle-km) and lowest on rural A roads apart from motorways (37).
- These are descriptive rates, not causal estimates. Traffic is DfT's motor-vehicle estimate;
  DfT's own published rates also include pedal cycles, so they run 1–3% lower.

## Run it

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and PostgreSQL 17 with a database
called `speed_gb`.

```bash
make setup      # create the Python environment
make ingest     # download the raw data (about 4.3 GB) and write ingest/manifest.json
make db         # build the raw database: schema, bulk load, keys, code lookups
make transform  # build the dbt layers and run all 49 tests
make export     # export the marts to Parquet in data/marts/
make docs       # browse the documentation and lineage graph
make rq1        # RQ1 rates and charts, written to figures/
```

## Deliberately not used

Spark, Kafka, Airflow and Kubernetes. The source data is 4.3 GB of CSV: one PostgreSQL
database on a laptop handles it comfortably, and distributed tools would add cost and
complexity without benefit.