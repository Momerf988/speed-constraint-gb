.PHONY: setup ingest db transform docs export rq1

# Create the Python environment from pyproject.toml.
setup:
	uv sync

# Phase 1: download raw source files and record them in ingest/manifest.json.
ingest:
	uv run python ingest/download.py

# Phase 2 and 4: rebuild the raw database from the downloaded files.
# Drops and recreates the raw schema, loads the CSVs, adds keys and code lookups,
# then loads DfT traffic (TRA0202) and free-flow speeds (SPE0102).
db:
	psql -d speed_gb -v ON_ERROR_STOP=1 -f db/schema.sql
	uv run python db/load.py
	psql -d speed_gb -v ON_ERROR_STOP=1 -f db/constraints.sql
	uv run python db/lookups.py
	uv run python db/load_traffic.py
	uv run python db/load_speed.py
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