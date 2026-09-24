.PHONY: setup ingest

# Create the Python environment from pyproject.toml.
setup:
	uv sync

# Phase 1: download raw source files and record them in ingest/manifest.json.
ingest:
	uv run python ingest/download.py
