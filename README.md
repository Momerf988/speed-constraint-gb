# Binding the Limit

Would capping how fast cars can go save enough lives in Great Britain to be worth the extra
travel time? And would it beat simply holding every car to each road's posted limit?

This project answers that with the government's own data and appraisal values: every
police-reported injury collision since 1979, DfT traffic and speed statistics, and DfT's
official values for a prevented casualty and for an hour of travel time.

**Status:** Phase 6 of 9 complete. All 37.5 million STATS19 rows (1979–2025) are loaded into
PostgreSQL and modelled with dbt into a tested star schema, joined to DfT traffic and
free-flow speed statistics. A casualty model in Python estimates the deaths and injuries each
policy would prevent, with Monte Carlo ranges and an explicit label wherever a scenario goes
beyond the evidence. Next: the travel time each policy costs, and whether it pays (Phase 7).

## Pipeline

```
DfT files → PostgreSQL (raw) → dbt: staging → intermediate → star schema → Parquet → Python model
```

| Layer | What it holds |
|---|---|
| `raw` | The three STATS19 tables exactly as published, with primary and foreign keys; DfT traffic (TRA0202) and free-flow speeds (SPE0102) |
| `staging` | Clear names, `-1` "unknown" codes turned into nulls (views) |
| `intermediate` | km/h speed limits, DfT road categories, keys, reported and adjusted severity (views) |
| `marts` | Star schema: `fct_casualty`, `fct_collision`, `dim_date`, `dim_road`, `dim_police_force`, `dim_casualty_profile`, plus aggregates for rates, speed bands and the casualty model's baseline (tables) |
| `model/` | Python: speed distributions, policies, speed-risk curves and the Monte Carlo, with every parameter traced to its source |

54 data tests run on every build: key uniqueness, fact-to-dimension relationships, allowed
values, row conservation from raw to facts and into the model's baseline, the rules of DfT's
severity adjustment, and reconciliation against DfT's published casualty and speed figures.
41 pytest tests cover the model.

## RQ1: where harm happens, per kilometre driven

![Deaths vs traffic by road environment, 2025](figures/rq1_share_deaths_vs_traffic.png)

- Rural roads carried 45% of 2025 traffic but 60% of deaths. Rural minor roads had the
  highest death rate: 4.4 per billion vehicle-km, against 0.8 on motorways.
- Serious injuries rank the other way: adjusted KSI rates are highest on urban roads
  (88–93 per billion vehicle-km) and lowest on rural A roads apart from motorways (37).
- These are descriptive rates, not causal estimates. Traffic is DfT's motor-vehicle estimate;
  DfT's own published rates also include pedal cycles, so they run 1–3% lower.

## RQ2 and RQ4: the deaths each policy would prevent

![Deaths prevented beyond business-as-usual ISA](figures/rq2_rq4_deaths_prevented.png)

Each policy is measured against business as usual: the overridable speed limiter (ISA) that
GB has decided to mandate, assumed fitted to every car, with drivers overriding it or
switching it off part of the time. Figures are deaths a year at 2022–2024 casualty levels:
a central estimate, with the 5th–95th percentile of 10,000 Monte Carlo draws.

- **Holding every car to the posted limit (RQ4)** would prevent at least 125 more deaths a
  year (62–214) on the four road types with DfT speed data, which have 71% of GB deaths. Most
  of it is on 48 km/h roads: 75 (33–137).
- **A flat cap at 80 km/h (RQ2)** would prevent 81 more deaths a year (58–103) on 97 km/h
  single carriageways, four times what posted-limit compliance achieves there (21): only 9%
  of cars exceed the limit on those roads, but over half drive faster than 80 km/h.
- **Lower caps, and any cap on motorways, lie beyond the evidence.** They cut mean speeds by
  15–62 km/h, while the speed-risk models were fitted on changes of at most 14 km/h. For these
  the model gives only the range across models (for example 100–356 deaths a year for a
  60 km/h cap on 97 km/h roads), not a point estimate.
- **The risk curve matters most.** Brenac et al.'s flatter curve cuts the compliance estimate
  from 125 to 27. Measuring speed differences from today's mean instead of the new one, or
  changing the speed distribution's shape, moves it by less than 15%.
- **Checked against published work.** If every casualty in a car or van collision can respond,
  full compliance leaves 66% of injuries on 48 km/h roads, against 60% in ISA-UK (Tate and
  Carsten 2008). It prevents 24% of deaths on the modelled roads, against 25% estimated for
  Sweden (Vadeby 2023).

Limits: DfT surveys speeds where traffic flows freely, which overstates speeding; the risk
curves come from South Australia; the results assume the whole fleet is fitted, with no
rollout path; and 29% of deaths are on road types without DfT speed data. "At least" marks
results where the urban curve, fitted only on cars faster than the traffic mean, cuts the
estimate short.

## Run it

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and PostgreSQL 17 with a database
called `speed_gb`.

```bash
make setup      # create the Python environment
make ingest     # download the raw data (about 4.3 GB) and write ingest/manifest.json
make db         # build the raw database: schema, bulk load, keys, code lookups, traffic, speeds
make transform  # build the dbt layers and run all 54 data tests
make export     # export the marts to Parquet in data/marts/
make docs       # browse the documentation and lineage graph
make rq1        # RQ1 rates and charts, written to figures/
make test       # run the 41 model tests
make speeds     # speed distributions from DfT's bands, checked against DfT's own figures
make rq2        # RQ2 and RQ4: casualties prevented by each policy, with Monte Carlo ranges
```

## Deliberately not used

Spark, Kafka, Airflow and Kubernetes. The source data is 4.3 GB of CSV: one PostgreSQL
database on a laptop handles it comfortably, and distributed tools would add cost and
complexity without benefit.
