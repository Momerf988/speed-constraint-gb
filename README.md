# Binding the Limit

[![CI](https://github.com/Momerf988/speed-constraint-gb/actions/workflows/ci.yml/badge.svg)](https://github.com/Momerf988/speed-constraint-gb/actions/workflows/ci.yml)

Would capping how fast cars can go save enough lives in Great Britain to be worth the extra
travel time? And would it beat simply holding every car to each road's posted limit?

This project answers that with the government's own data and appraisal values: every
police-reported injury collision since 1979, DfT traffic and speed statistics, and DfT's
official values for a prevented casualty and for an hour of travel time.

**Status:** Phases 1–8 complete; Phase 9 under way (CI/CD done, Azure deployment next). The
casualty, appraisal and Wales results were revised in November 2026 after an examiner-style review:
vans now have their own speeds and limits, the baseline represents the ISA being mandated more
faithfully, fuel and carbon are counted, and every headline is reported with the range across models. All 37.5 million STATS19 rows (1979–2025) are loaded into
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
| `raw` | The three STATS19 tables exactly as published, with primary and foreign keys; DfT traffic (TRA0202, TRA0204) and free-flow speeds of cars and vans (SPE0102) |
| `staging` | Clear names, `-1` "unknown" codes turned into nulls (views) |
| `intermediate` | km/h speed limits, DfT road categories, keys, reported and adjusted severity (views) |
| `marts` | Star schema: `fct_casualty`, `fct_collision`, `dim_date`, `dim_road`, `dim_police_force`, `dim_casualty_profile`, plus aggregates for rates, car and van speed bands, the casualty baseline, traffic per road type, collisions per casualty and a monthly police-force panel (tables) |
| `model/` | Python: speed distributions, policies, speed-risk curves, Monte Carlo, structural range and appraisal (time, casualties, fuel, carbon), with every parameter and £ value traced to its source cell; difference-in-differences, synthetic control, triple difference and permutation tests; the severity model |

91 data tests run on every build: key uniqueness, fact-to-dimension relationships, allowed
values, row conservation from raw to facts and into the model's inputs, the rules of DfT's
severity adjustment, and reconciliation against DfT's published casualty, traffic and speed
figures. 114 pytest tests cover the models, the appraisal and the causal and machine-learning code.

## Continuous integration and delivery

Every push runs three jobs on GitHub Actions (`.github/workflows/ci.yml`):

- **Python tests:** all 114; the one that re-reads DfT's raw appraisal spreadsheet is skipped,
  because CI does not hold it.
- **The whole pipeline on a sample:** raw load, keys, code lookups, traffic, speeds, every dbt
  model and 89 of the 91 data tests, then the Parquet export, in a throwaway Postgres. The
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

Each policy is measured against business as usual: the ISA that GB has decided to mandate,
fitted to every car and van. Following the regulation and TRL's 2024 appraisal for DfT, half the
fleet has a speed control function and half a warning only. Drivers switch it off for some
journeys and override it, and they override in order to speed. Figures are deaths a year at
2022–2024 casualty levels, given as central [5th–95th percentile of 10,000 Monte Carlo draws]
{lowest–highest across the models}.

- **Holding every vehicle to its legal limit (RQ4)** would prevent 213 more deaths a year
  [131–285] {57–259} on the four road types with DfT speed data, which have 71% of GB deaths.
  60% of that is on urban roads, 104 [50–163] on 48 km/h roads alone. The mandated ISA itself
  prevents 69 [28–143] {21–124}.
- **Vans have their own limit.** Most vans are limited to 80 km/h on single carriageways, and 53%
  of them drive faster than that. DfT publishes no van speed bands for those roads, because its
  counters cannot weigh a van, so the model moves the car distribution to the published van mean.
- **A cap at 80 km/h on 97 km/h single carriageways (RQ2)** would prevent 152 more deaths a year
  [105–181] {60–222}, twice what compliance achieves there, because 56% of cars there drive faster
  than 80 km/h. That is close to Kloeden's estimate for an 80 km/h limit on rural South Australian
  roads.
- **Lower caps, and any cap on motorways, lie beyond the evidence.** They cut mean speeds by
  15–62 km/h, while the speed-risk models were fitted on changes of at most 14 km/h and initial
  speeds up to 95 km/h. Only the range across models is given for them (for example 179–367
  deaths a year for a 60 km/h cap on 97 km/h roads).
- **Model choice matters more than parameter values.** Brenac et al.'s flatter individual risk
  curve cuts the compliance estimate from 213 to 57, far below the Monte Carlo's 5th percentile.
  `make structural` gives every headline its central value, Monte Carlo range and structural
  range, and names the model behind each extreme.
- **Checked against published work, honestly.** With every car or van collision allowed to respond
  (f = 1), compliance prevents 36% of injuries on 48 km/h roads, against ISA-UK's 40%; at the
  central f it is 20%. At the central f, deaths prevented match Sweden's 25% (24%). No single
  value of f matches both, so validation tests plausibility rather than picking a model.

Limits: DfT surveys speeds where traffic flows freely, which overstates speeding; the risk curves,
and the shares of crashes involving a free-travelling vehicle (56% urban, 80% rural), come from
South Australia; the results assume the whole fleet is fitted; and 29% of deaths are on road types
without DfT speed data.

## RQ3: does each policy pay for itself?

![Ratio of benefit to time cost for caps below the limit](figures/rq3_benefit_cost_ratio.png)

Casualties prevented are valued with TAG data book v2.03 (£2.65m per death, £295,069 per serious
and £22,668 per slight injury, plus police, insurance and damage costs per collision). Fuel follows
TAG's speed–consumption curves by vehicle and fuel, at resource cost, and carbon TAG's central
value; travel time is valued at £15.65 per car-hour and £28.76 per van-hour, plus TAG's vehicle
capital term, all in 2023 prices. Following DfT practice, time a driver gained by breaking the
limit is not counted, each vehicle against its own limit. "Ratio" is annual benefit over annual
time cost, with no present values or hardware cost.

- **Holding vehicles to their limits costs no lawful time**, so it pays wherever it prevents
  casualties: £2.13bn a year [£1.33–3.16bn] {£0.48–2.51bn} on the four road types.
- **An 80 km/h cap on 97 km/h single carriageways pays:** £784m of casualties, £58m of fuel and
  £62m of carbon, against £489m of lawful car time (a van above 80 km/h was already breaking its
  limit). The ratio is 1.83 [1.27–2.36] {0.85–2.63}. It fails only under Brenac's flatter curve,
  and pays even if all traffic on those roads flows freely.
- **A flat limiter on every road does not pay.** At 80 km/h on all four road types the ratio is
  0.58 [0.49–0.65] {0.45–0.68}, because motorway time dominates. At 97 km/h it is about break-even,
  1.06 [0.86–1.23] {0.81–1.20}, and is worth less than compliance.
- **Four caps fail whatever their safety effect.** Even if they prevented every casualty on the
  road type, a 48 km/h cap on 97 km/h roads and motorway caps at 48, 60 and 80 km/h could not
  repay the time they cost, with or without unreported injuries.
- **Against the 2035 target** of 65% fewer KSI than 2022–2024 (18,801 fewer a year), the mandated
  ISA in every vehicle would deliver 6% [3–14%] {2–12%} and compliance a further 23% [14–34%]
  {5–27%}. These are full-fleet steady states at 2022–2024 levels, not forecasts.

Limits: no published figure gives the share of traffic that flows freely (70% is assumed on
97 km/h roads and motorways, 40% on urban roads), so the break-even share is reported; DfT
publishes traffic by road class, not speed limit, so it is split by casualty share and tested 30%
either way; TAG's fuel curves are for average link speeds.

## Wales: did the 32 km/h default cut casualties?

![Wales against England, quarter by quarter](figures/wales_event_study.png)

Wales made 32 km/h (20 mph) the default on its restricted roads on 17 September 2023. A
difference-in-differences design compares monthly casualties on Welsh 32 and 48 km/h roads with
the same roads in England, police force by police force, from January 2022 to December 2025
(Poisson regression with force and month fixed effects). It replicates and extends Xu (2026), who
found 19.4% fewer pedestrian and cyclist casualties with a similar design.

- Casualties on these Welsh roads fell 17.5% more than England's change implies: a rate ratio of
  0.825 (95% interval 0.687–0.991). Adjusted KSI fell 14.2% more (0.858, 0.747–0.985).
- **The before window was chosen after seeing the pre-trends**, so other readings are reported
  beside it. If Wales's 2017–2023 decline relative to England had carried on, the ratio is 0.849
  (0.670–1.075); counting 2017–2019 as "before" gives 0.748.
- **How unusual is it, with four Welsh forces?** 3.8% of all 82,251 sets of four English forces
  changed as much. Two English forces have incomplete records by DfT's own account; without them,
  none did (p < 0.001). A synthetic control puts Wales 28% below its English counterpart, but 5 of
  39 English forces depart as much from theirs (p = 0.15). Leaving out North Wales gives 0.908.
- **Displacement or a Wales-wide shift?** Other Welsh roads rose relative to England (1.065,
  0.971–1.169). Across all Welsh roads together, the net change is 0.944 (0.812–1.096); against
  other Welsh roads, the affected roads fell 22.6% (triple difference).
- Under the Exponential speed–risk model, the fall corresponds to 2.9–5.6 km/h lower mean speeds,
  depending on the coefficient: an illustration, not a test.

So the direction holds under every reading, but the significance depends on the test: the evidence
is consistent with a fall of about a sixth, not proof of one.

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
- With the other recorded fields held as they are, the odds of KSI rise with the limit: setting
  every casualty's limit in turn gives 0.92 at 32 km/h, 1.21 at 64, 1.36 at 80, 1.47 at 97 and
  1.51 at 113 km/h, against 48 km/h roads, and the logistic regression's adjusted odds ratio at
  97 km/h is 1.50. Raw figures hide this on motorways (0.95), where crashes involve different
  people and impacts.

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
make transform  # build the dbt layers and run all 91 data tests
make export     # export the marts to Parquet in data/marts/
make docs       # browse the documentation and lineage graph
make rq1        # RQ1 rates and charts, written to figures/
make values     # extract DfT's appraisal values into model/parameters/appraisal_values.csv
make test       # run the 114 model tests
make speeds     # speed distributions from DfT's bands, checked against DfT's own figures
make rq2        # RQ2 and RQ4: casualties prevented by each policy, with Monte Carlo ranges
make rq3        # RQ3: casualties, fuel, carbon and time cost of each policy, with the bound
make structural # every headline: central, Monte Carlo range and range across models
make wales      # Wales 32 km/h default: difference-in-differences, synthetic control, permutation tests
make tune       # tune the severity model on 2021-2024 (about 10 minutes)
make severity   # severity model: XGBoost with SHAP and limit contrasts, cross-validated by collision
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
