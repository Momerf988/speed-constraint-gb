.PHONY: setup ingest db transform docs export rq1 test speeds rq2 values rq3 structural wales tune severity holdout sample

# Create the Python environment from pyproject.toml.
setup:
	uv sync

# Phase 1: download raw source files and record them in ingest/manifest.json.
ingest:
	uv run python ingest/download.py

# Phase 2, 4, 6 and 7: rebuild the raw database from the downloaded files.
# Drops and recreates the raw schema, loads the CSVs, adds keys and code lookups,
# then loads DfT traffic (TRA0202), free-flow speeds (SPE0102) and traffic by vehicle type (TRA0204).
db:
	psql -d speed_gb -v ON_ERROR_STOP=1 -f db/schema.sql
	uv run python db/load.py
	psql -d speed_gb -v ON_ERROR_STOP=1 -f db/constraints.sql
	uv run python db/lookups.py
	uv run python db/load_traffic.py
	uv run python db/load_speed.py
	uv run python db/load_traffic_vehicle.py

# Phase 3: build the staging, intermediate and mart layers with dbt, and run every test.
transform:
	cd dbt && uv run dbt build

# Phase 3: browse the dbt documentation and lineage graph in a web browser.
docs:
	cd dbt && uv run dbt docs generate && uv run dbt docs serve

# Phase 3: export the marts to Parquet files in data/marts/.
export:
	uv run python export/export_parquet.py

# Phase 4: RQ1 rates and charts, written to figures/.
rq1:
	uv run python analysis/rq1_rates.py

# Phase 6: run the Python model tests.
test:
	uv run pytest -q

# Phase 6: speed distributions from DfT's bands, written to results/ and figures/.
speeds:
	uv run python -m analysis.fit_speeds

# Phase 6: RQ2 and RQ4, deaths and injuries prevented by each policy, written to results/ and figures/.
rq2:
	uv run python -m analysis.rq2_rq4

# Phase 7: extract DfT's appraisal values (TAG data book, RAS4001) into model/parameters/appraisal_values.csv.
values:
	uv run python ingest/extract_appraisal_values.py

# Phase 7: RQ3 cost-benefit appraisal, written to results/ and figures/.
rq3:
	uv run python -m analysis.rq3

# Paper revision: every headline with its central value, Monte Carlo range and structural range. Run after rq2 and rq3.
structural:
	uv run python -m analysis.structural

# Phase 8: Wales 32 km/h default, difference-in-differences against England, written to results/ and figures/.
wales:
	uv run python -m analysis.wales

# Phase 8: tune the severity model on 2021-2024 (about 10 minutes), written to model/parameters/ and results/.
tune:
	uv run python -m analysis.severity_tuning

# Phase 8: severity model (XGBoost with the tuned settings, SHAP), written to results/ and figures/.
severity:
	uv run python -m analysis.severity

# Phase 8: test the tuned severity model on 2025, a year it never saw, written to results/ and figures/.
holdout:
	uv run python -m analysis.severity_holdout

# Phase 9: remake the small STATS19 sample in ci/sample/ that CI builds the pipeline on.
sample:
	uv run python ci/make_sample.py
