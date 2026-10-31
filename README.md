# Binding the Limit

[![CI](https://github.com/Momerf988/speed-constraint-gb/actions/workflows/ci.yml/badge.svg)](https://github.com/Momerf988/speed-constraint-gb/actions/workflows/ci.yml)

Would capping how fast cars can go save enough lives in Great Britain to be worth the extra
travel time? And would it beat simply holding every car to each road's posted limit?

This project answers that with the government's own data and appraisal values: every
police-reported injury collision since 1979, DfT traffic and speed statistics, and DfT's
official values for a prevented casualty and for an hour of travel time.

**Status:** Phases 1–8 complete; Phase 9 under way (CI/CD done, Azure deployment next). All 37.5 million STATS19 rows (1979–2025) are loaded into
PostgreSQL and modelled with dbt into a tested star schema, joined to DfT traffic and
free-flow speed statistics. A casualty model in Python estimates the deaths and injuries each
policy would prevent, with Monte Carlo ranges and an explicit label wherever a scenario goes
beyond the evidence, and an appraisal prices them against the travel time each policy costs,
using DfT's TAG values. A causal study of Wales's 2023 default 32 km/h limit and a
machine-learning severity model check the picture against what actually happened on the roads.

## Pipeline

```
DfT files → PostgreSQL (raw) → dbt: staging → intermediate → star schema → Parquet → Python model
```

| Layer | What it holds |
|---|---|
| `raw` | The three STATS19 tables exactly as published, with primary and foreign keys; DfT traffic (TRA0202, TRA0204) and free-flow speeds (SPE0102) |
| `staging` | Clear names, `-1` "unknown" codes turned into nulls (views) |
| `intermediate` | km/h speed limits, DfT road categories, keys, reported and adjusted severity (views) |
| `marts` | Star schema: `fct_casualty`, `fct_collision`, `dim_date`, `dim_road`, `dim_police_force`, `dim_casualty_profile`, plus aggregates for rates, speed bands, the casualty baseline, traffic per road type and collisions per casualty (tables) |
| `model/` | Python: speed distributions, policies, speed-risk curves, Monte Carlo and appraisal, with every parameter and £ value traced to its source cell; difference-in-differences and the severity model |

80 data tests run on every build: key uniqueness, fact-to-dimension relationships, allowed
values, row conservation from raw to facts and into the model's inputs, the rules of DfT's
severity adjustment, and reconciliation against DfT's published casualty, traffic and speed
figures. 71 pytest tests cover the models, the appraisal and the causal and machine-learning code.

## Continuous integration and delivery

Every push runs three jobs on GitHub Actions (`.github/workflows/ci.yml`):

- **Python tests:** all 71; the one that re-reads DfT's raw appraisal spreadsheet is skipped,
  because CI does not hold it.
- **The whole pipeline on a sample:** raw load, keys, code lookups, traffic, speeds, every dbt
  model and 78 of the 80 data tests, then the Parquet export, in a throwaway Postgres. The
  sample (`ci/sample/`, 2.7 MB, made by `ci/make_sample.py`) is 1 in 250 collisions from 2015
  onwards, with all their vehicles and casualties, copied byte for byte. The two tests that
  compare the build with DfT's national totals need every row; they are tagged `full_data` and
  run only on the full build.
- **A Docker image of the pipeline**, checked to start; pushes to `main` publish it as
  `ghcr.io/momerf988/speed-constraint-gb`. It runs against any Postgres given through the
  `PG*` and `DATABASE_URL` environment variables, which is what the planned Azure deployment
  will use.

Building from scratch for CI exposed one real problem: DfT's 2025 data guide gives casualty
type 23 two labels. `db/lookups.py` keeps the one that matches vehicle type 23 (electric
motorcycle), records why, and still stops on any other conflict.

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

## RQ3: does each policy pay for itself?

![Benefit-cost ratio of caps below the limit](figures/rq3_benefit_cost_ratio.png)

Casualties prevented are valued with TAG data book v2.03 (£2.65m per death, £295,069 per
serious and £22,668 per slight injury, plus police, insurance and damage costs per
collision), and travel time at TAG's £15.65 per car-hour and £28.76 per van-hour, all in 2023
prices. Following DfT practice, time a driver gained by breaking the limit is not counted.
Figures are £ a year beyond business-as-usual ISA, with the 5th–95th percentile in brackets.

- **Holding cars to the posted limit costs no lawful time**, so it pays wherever it prevents
  casualties: at least £1.23bn a year on the four road types, £756m (£329m–£1.4bn) of it on
  48 km/h roads. Even if time gained by speeding were counted, it would still pay on 48 and
  97 km/h roads; on 32 km/h roads and motorways it would pay only if under 22% and 17% of
  traffic flowed freely.
- **An 80 km/h cap on 97 km/h single carriageways does not pay on central assumptions:**
  £408m of casualties prevented against £718m of lawful time, a benefit-cost ratio of 0.57
  (0.37–0.84). It pays only if under 40% of the traffic on those roads flows freely, or if
  the Exponential model is right, or if injuries never reported to the police are counted.
- **Lower caps, and any cap on motorways, fail even on the most favourable model.** A 60 km/h
  cap on 97 km/h roads would cost £4.4bn of lawful time a year for £0.5–2.1bn of casualties
  prevented; motorway caps return at most 23p per £1 of time lost.
- **Against the 2035 target** of 65% fewer KSI than 2022–2024 (18,801 fewer a year), the
  mandated ISA in every car would deliver at least 14%, and posted-limit compliance at least a
  further 13% (7–24%).

Limits: no published figure gives the share of traffic that flows freely (70% is assumed on
97 km/h roads and motorways, 40% on urban roads), so the break-even share is reported; DfT
publishes traffic by road class, not speed limit, so it is split by casualty share, which
overstates traffic on single carriageways; vans are assumed to drive like cars; fuel,
emissions and vehicle operating costs are left out.

## Wales: did the 32 km/h default cut casualties?

![Wales against England, quarter by quarter](figures/wales_event_study.png)

Wales made 32 km/h (20 mph) the default on its restricted roads on 17 September 2023. A
difference-in-differences design compares monthly casualties on Welsh 32 and 48 km/h roads with
the same roads in England, police force by police force, from January 2022 to December 2025
(Poisson regression with force and month fixed effects).

- Casualties on these Welsh roads fell 17.5% more than England's trend implies: a rate ratio of
  0.825 (95% interval 0.687–0.991). Random sets of four English forces showed a change as large
  in only 3.8% of 2,000 draws. Adjusted KSI fell 14% more (0.858, 0.747–0.985).
- Welsh roads the policy did not touch, at 64 km/h and above, show no change (1.065,
  0.971–1.169), so the fall is not a Wales-wide shift.
- The comparison starts in 2022 because the event study shows Wales was already falling
  relative to England between 2019 and 2022. Counting 2017–2019 as "before" would credit the
  policy with that older fall, and give 0.748.
- Under the Exponential speed-risk model, a 17.5% fall needs mean speeds about 3.2 km/h lower
  across these roads. Transport for Wales reports larger falls, about 6.4 km/h, at its 38
  monitoring sites (secondary summaries); this road group also includes 48 km/h roads that
  kept their limit.

Limits: with four Welsh forces, the intervals are indicative; England's own move to 20 mph (a
quarter of these English casualties are now on 32 km/h roads) biases the estimate towards
zero; there is no traffic count by force and month.

## What makes a casualty killed or seriously injured?

![Which fields drive severity, and the speed limit's part](figures/severity_drivers.png)

An XGBoost model learns each casualty's adjusted probability of being killed or seriously
injured from 27 fields recorded at the scene (2021–2025, 652,821 casualties). Fields that follow
from the outcome or reflect police recording (number of casualties, police force,
contributory factors) are left out, and every check keeps each collision's casualties together.
Its settings were tuned by a random search with early stopping, on 2021–2024 only.

- Out of fold, it reaches an average precision of 0.457 (± 0.003 across folds), against 0.218
  with no skill and 0.426 for logistic regression on the same fields, and beat logistic
  regression in all five folds. Its probabilities are calibrated to within 0.5 percentage points.
- On 2025, a year it never saw, it keeps its edge: average precision 0.462 (95% interval
  0.455–0.467) against 0.434 for logistic regression, though ROC AUC slips from 0.745 to 0.731.
  Tuning added only 0.003 over the default settings: the gain comes from the type of model.
- It beats no skill in every group, by 1.5 to 2.2 times, and is weakest for cyclists (ROC AUC
  0.65).
- Who was hurt matters most (pedestrian, rider or occupant), then what the vehicle was doing,
  where it was hit and the casualty's age; the speed limit comes fifth.
- Holding the other fields, the odds of KSI rise steadily with the limit: 0.93 at 32 km/h, 1.20
  at 64, 1.35 at 80, 1.47 at 97 and 1.43 at 113 km/h, against 48 km/h roads. Raw figures hide
  this on motorways (0.95), where crashes involve different people and impacts.

Scope: this shows which crashes end up more severe. It does not estimate what a cap would do,
because a cap also changes which crashes happen at all. SHAP values explain the model, not the
world: related fields, such as casualty type and vehicle type, share credit between them.

## Run it

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and PostgreSQL 17 with a database
called `speed_gb`.

```bash
make setup      # create the Python environment
make ingest     # download the raw data (about 4.3 GB) and write ingest/manifest.json
make db         # build the raw database: schema, bulk load, keys, code lookups, traffic, speeds
make transform  # build the dbt layers and run all 80 data tests
make export     # export the marts to Parquet in data/marts/
make docs       # browse the documentation and lineage graph
make rq1        # RQ1 rates and charts, written to figures/
make values     # extract DfT's appraisal values into model/parameters/appraisal_values.csv
make test       # run the 71 model tests
make speeds     # speed distributions from DfT's bands, checked against DfT's own figures
make rq2        # RQ2 and RQ4: casualties prevented by each policy, with Monte Carlo ranges
make rq3        # RQ3: benefit, time cost and benefit-cost ratio of each policy
make wales      # Wales 32 km/h default: difference-in-differences against England
make tune       # tune the severity model on 2021-2024 (about 10 minutes)
make severity   # severity model: XGBoost with SHAP, cross-validated by collision
make holdout    # severity model tested on 2025, a year it never saw, overall and by group
make sample     # remake the small STATS19 sample in ci/sample/ that CI runs on
```

Or run the pipeline from its Docker image against any Postgres:

```bash
docker run --rm -e PGHOST=host.docker.internal -e PGUSER="$USER" -e PGDATABASE=speed_gb \
    -e DATABASE_URL="host=host.docker.internal user=$USER dbname=speed_gb" \
    ghcr.io/momerf988/speed-constraint-gb
```

## Data and licence

STATS19, the traffic and speed tables, RAS4001 and the TAG data book are published by the
Department for Transport. Contains public sector information licensed under the
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
The CI sample in `ci/sample/` is a subset of those files, unchanged.

## Deliberately not used

Spark, Kafka, Airflow and Kubernetes. The source data is 4.3 GB of CSV: one PostgreSQL
database on a laptop handles it comfortably, and distributed tools would add cost and
complexity without benefit.
