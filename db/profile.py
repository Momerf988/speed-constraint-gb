"""Profile a raw STATS19 file: type, min, max, distinct values and nulls per column.

Usage:
    python db/profile.py collision    # or vehicle, or casualty
"""

import sys

import duckdb

TABLES = ["collision", "vehicle", "casualty"]

if len(sys.argv) != 2 or sys.argv[1] not in TABLES:
    sys.exit(f"usage: python db/profile.py {{{','.join(TABLES)}}}")

path = f"data/raw/dft-road-casualty-statistics-{sys.argv[1]}-1979-latest-published-year.csv"
duckdb.sql(f"summarize select * from read_csv('{path}', sample_size=-1)").show(max_rows=100, max_width=250)
