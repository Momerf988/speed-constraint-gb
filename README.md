# Binding the Limit

Would capping how fast cars can go save enough lives in Great Britain to be worth the extra
travel time? And would it beat simply holding every car to each road's posted limit?

This project answers that with the government's own data and appraisal values: every
police-reported injury collision since 1979, DfT traffic and speed statistics, and DfT's
official values for a prevented casualty and for an hour of travel time.

**Status:** Phase 2 of 9 complete. All 37.5 million STATS19 rows (1979–2025) are loaded into
PostgreSQL with primary keys, foreign keys and code lookups.

## Run it

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and PostgreSQL 17 with a database
called `speed_gb`.

```bash
make setup    # create the Python environment
make ingest   # download the raw data (about 4.3 GB) and write ingest/manifest.json
make db       # build the raw database: schema, bulk load, keys, code lookups
```
