.PHONY: setup ingest db

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
