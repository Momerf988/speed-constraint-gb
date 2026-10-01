"""Load the raw STATS19 CSV files into the raw schema in Postgres.

All three tables are emptied and reloaded from their CSV files with COPY, so the
script can be rerun safely. Values are loaded exactly as published; cleaning
happens later, in dbt.

Usage:
    python db/load.py                # load all three files in full
    python db/load.py --sample 1000  # load only the first 1000 rows of each file
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from itertools import islice
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DSN = os.environ.get("DATABASE_URL", "dbname=speed_gb")

# Parents before children, so foreign keys (once added) are satisfied as rows arrive.
TABLES = ["collision", "vehicle", "casualty"]

CHUNK = 1 << 20  # 1 MiB
REPORT_EVERY = 250 * CHUNK


def csv_path(table: str) -> Path:
    return RAW / f"dft-road-casualty-statistics-{table}-1979-latest-published-year.csv"


def load_table(cur: psycopg.Cursor, table: str, sample: int | None) -> None:
    """Stream one CSV into raw.<table> and check the row count against the file."""
    path = csv_path(table)
    start = time.time()
    lines = 0

    # HEADER MATCH makes Postgres refuse the file unless its header names match the
    # table's columns exactly, in order. That catches a misaligned schema before
    # any wrong data goes in.
    with cur.copy(f"copy raw.{table} from stdin with (format csv, header match)") as copy, open(path, "rb") as f:
        if sample:
            for line in islice(f, sample + 1):  # +1 for the header line
                copy.write(line)
                lines += 1
        else:
            size = path.stat().st_size
            done = 0
            last = b""
            while chunk := f.read(CHUNK):
                copy.write(chunk)
                lines += chunk.count(b"\n")
                done += len(chunk)
                last = chunk[-1:]
                if done % REPORT_EVERY < CHUNK:
                    print(f"    {done / 1e6:,.0f} of {size / 1e6:,.0f} MB", flush=True)
            if last and last != b"\n":
                lines += 1  # final line has no trailing newline

    cur.execute(f"select count(*) from raw.{table}")
    loaded = cur.fetchone()[0]
    expected = lines - 1  # minus the header
    status = "matches file" if loaded == expected else f"MISMATCH, file has {expected:,} data lines"
    print(f"done  raw.{table}: {loaded:,} rows in {time.time() - start:.0f}s ({status})", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", type=int, help="load only the first N rows of each file")
    args = parser.parse_args()

    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        # The CSV dates are day/month/year (e.g. 05/05/2013). Postgres assumes
        # month/day unless told otherwise.
        cur.execute("set datestyle = 'ISO, DMY'")

        # Empty all three together: Postgres refuses to truncate a table on its own
        # once another table's foreign key points at it.
        cur.execute("truncate " + ", ".join(f"raw.{t}" for t in TABLES))

        for table in TABLES:
            print(f"load  raw.{table} from {csv_path(table).name}", flush=True)
            load_table(cur, table, args.sample)

        # Refresh the statistics the query planner uses to choose between plans.
        for table in TABLES:
            cur.execute(f"analyze raw.{table}")
    # Leaving the `with` block commits: either every table loads, or none does.
    return 0


if __name__ == "__main__":
    sys.exit(main())
