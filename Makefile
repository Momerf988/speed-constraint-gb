.PHONY: setup ingest db transform docs export

# Create the Python environment from pyproject.toml.
setup:
	uv sync

# Phase 1: download raw source files and record them in ingest/manifest.json.
ingest:
	uv run python ingest/download.py

# Phase 2: rebuild the raw database from the downloaded files.
# Drops and recreates the raw schema, loads the CSVs, then adds keys and code lookups.
db:
	psql -d speed_gb -v ON_ERROR_STOP=1 -f db/schema.sql
	uv run python db/load.py
	psql -d speed_gb -v ON_ERROR_STOP=1 -f db/constraints.sql
	uv run python db/lookups.py

# Phase 3: build the staging, intermediate and mart layers with dbt, and run every test.
transform:
	cd dbt && uv run dbt build

# Phase 3: browse the dbt documentation and lineage graph in a web browser.
docs:
	cd dbt && uv run dbt docs generate && uv run dbt docs serve

# Phase 3: export the marts to Parquet files in data/marts/.
export:
	uv run python export/export_parquet.py

db:
	psql -d speed_gb -v ON_ERROR_STOP=1 -f db/schema.sql
	uv run python db/load.py
	psql -d speed_gb -v ON_ERROR_STOP=1 -f db/constraints.sql
	uv run python db/lookups.py
	uv run python db/load_traffic.py