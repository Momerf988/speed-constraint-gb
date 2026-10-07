"""Export the dbt marts from Postgres to Parquet files in data/marts/.

Parquet stores each column together and compresses it, so the files are far smaller
than the database tables and fast to read from Python, DuckDB and cloud storage.
Everything downstream of Phase 3 (models, dashboard, Azure) reads these files rather
than the database.

DuckDB does the work: it attaches the Postgres database, reads each table and writes
it straight to Parquet.

Usage:
    python export/export_parquet.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "marts"
DSN = os.environ.get("DATABASE_URL", "dbname=speed_gb")

TABLES = [
    "dim_date",
    "dim_road",
    "dim_police_force",
    "dim_casualty_profile",
    "fct_collision",
    "fct_casualty",
    "agg_casualties_year_road",
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("install postgres; load postgres;")
    con.execute(f"attach '{DSN}' as pg (type postgres, read_only)")

    print(f"{'table':28s} {'rows':>12s} {'postgres':>10s} {'parquet':>10s}")
    for table in TABLES:
        start = time.time()
        path = OUT / f"{table}.parquet"
        part = path.with_name(path.name + ".part")  # never leave a half-written file behind

        con.execute(f"copy (select * from pg.marts.{table}) to '{part}' (format parquet, compression zstd)")
        part.rename(path)

        rows = con.execute(f"select count(*) from read_parquet('{path}')").fetchone()[0]
        pg_bytes = con.execute(
            f"select * from postgres_query('pg', 'select pg_total_relation_size(''marts.{table}'')')"
        ).fetchone()[0]
        print(f"{table:28s} {rows:>12,} {pg_bytes / 1e6:>8.1f}MB {path.stat().st_size / 1e6:>8.1f}MB"
              f"  ({time.time() - start:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())