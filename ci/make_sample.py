"""Make the small STATS19 sample that continuous integration runs the pipeline on.

GitHub's servers cannot hold the 4.3 GB of full CSV files, so CI builds the database and every
dbt layer from a sample instead: about 1 in 250 collisions from 2015 onwards, with every vehicle
and casualty in them. Collisions are picked by a hash of their index, so the same ones are
picked every time, and lines are copied byte for byte, so the sample has exactly the published
format. The small DfT spreadsheets the loaders need are copied whole.

Tests that compare the build with DfT's national published totals cannot pass on a sample; they
are tagged full_data in dbt and left out of the CI run.

Writes ci/sample/.

Usage:
    python ci/make_sample.py
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "ci" / "sample"

FIRST_YEAR = 2015
ONE_IN = 250
TABLES = ["collision", "vehicle", "casualty"]     # parents first: vehicles and casualties follow their collisions
SPREADSHEETS = [
    "dft-road-casualty-statistics-road-safety-open-dataset-data-guide-2025.xlsx",
    "tra0202-km-by-road-class.ods",
    "spe0102.ods",
    "tra0204-km-by-vehicle-and-road-type.ods",
]


def csv_name(table: str) -> str:
    return f"dft-road-casualty-statistics-{table}-1979-latest-published-year.csv"


def chosen(collision_index: bytes) -> bool:
    """The same 1 in ONE_IN collisions every time, spread evenly across police forces and years."""
    return int(hashlib.md5(collision_index).hexdigest()[:8], 16) % ONE_IN == 0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    keep: set[bytes] = set()
    for table in TABLES:
        kept = 0
        with open(RAW / csv_name(table), "rb") as source, open(OUT / csv_name(table), "wb") as sample:
            sample.write(source.readline())                          # the header
            for line in source:
                index, year = line.split(b",", 2)[:2]
                if table == "collision":
                    if int(year) >= FIRST_YEAR and chosen(index):
                        keep.add(index)
                    else:
                        continue
                elif index not in keep:
                    continue
                sample.write(line)
                kept += 1
        print(f"{csv_name(table)}: {kept:,} rows, {(OUT / csv_name(table)).stat().st_size / 1e6:.1f} MB")
    for name in SPREADSHEETS:
        shutil.copyfile(RAW / name, OUT / name)
        print(f"{name}: copied whole")
    return 0


if __name__ == "__main__":
    sys.exit(main())
