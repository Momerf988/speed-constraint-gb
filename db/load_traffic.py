"""Load DfT table TRA0202 (traffic in billion vehicle-km by road class) into Postgres.

TRA0202 is a spreadsheet (ODS), not a CSV: the table sits below title rows, its text
cells carry stray whitespace, and it has a notes column. This script finds the table,
checks the headers and units are exactly as expected, and loads it unchanged into
raw.traffic_tra0202, one row per year. Reshaping into road categories happens in dbt.

Usage:
    python db/load_traffic.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import psycopg

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "raw" / "tra0202-km-by-road-class.ods"
SHEET = "TRA0202"  # 1993 onwards; TRA0202_historic uses an older urban/rural definition
DSN = os.environ.get("DATABASE_URL", "dbname=speed_gb")

# Exact header text in the spreadsheet -> column name in raw.traffic_tra0202.
# If DfT changes the layout, the load stops instead of putting numbers in the wrong column.
COLUMNS = {
    "Year": "year",
    "Notes": "notes",
    "Units": "units",
    "Major Roads: Motorway [note 1]": "motorway",
    "Major Roads: Rural 'A' Roads [note 2]": "rural_a_roads",
    "Major Roads: Urban 'A' Roads [note 2]": "urban_a_roads",
    "Major Roads: All 'A' Roads": "all_a_roads",
    "All Major Roads": "all_major_roads",
    "Minor Roads: Rural [note 2]": "rural_minor_roads",
    "Minor Roads: Urban [note 2]": "urban_minor_roads",
    "All Minor Roads": "all_minor_roads",
    "All Roads": "all_roads",
}
NUMERIC = list(COLUMNS.values())[3:]

DDL = """
drop table if exists raw.traffic_tra0202;
create table raw.traffic_tra0202 (
    year               smallint primary key,
    notes              text,
    units              text not null,
    motorway           numeric(6, 1) not null,
    rural_a_roads      numeric(6, 1) not null,
    urban_a_roads      numeric(6, 1) not null,
    all_a_roads        numeric(6, 1) not null,
    all_major_roads    numeric(6, 1) not null,
    rural_minor_roads  numeric(6, 1) not null,
    urban_minor_roads  numeric(6, 1) not null,
    all_minor_roads    numeric(6, 1) not null,
    all_roads          numeric(6, 1) not null
);
"""


def clean(value):
    """Spreadsheet text cells carry layout whitespace ('   Year   '); collapse it."""
    if isinstance(value, str):
        return " ".join(value.split())
    return value


def read_tra0202(path: Path) -> list[tuple]:
    sheet = pd.read_excel(path, sheet_name=SHEET, header=None, engine="odf")
    sheet = sheet.map(clean)

    # Find the header row rather than assuming its position: title rows above it can change.
    header_row = sheet.index[sheet[0] == "Year"][0]
    headers = sheet.iloc[header_row].dropna().tolist()
    if headers != list(COLUMNS):
        raise ValueError(f"TRA0202 headers have changed:\n  expected {list(COLUMNS)}\n  found    {headers}")

    table = sheet.iloc[header_row + 1:, :len(COLUMNS)].dropna(how="all")
    table.columns = list(COLUMNS.values())

    units = table["units"].unique().tolist()
    if units != ["Billion vehicle kilometres"]:
        raise ValueError(f"unexpected units: {units}")

    return [
        (int(row["year"]), row["notes"], row["units"], *(float(row[c]) for c in NUMERIC))
        for _, row in table.iterrows()
    ]


def main() -> int:
    rows = read_tra0202(SOURCE)
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(DDL)
        with cur.copy(f"copy raw.traffic_tra0202 ({', '.join(COLUMNS.values())}) from stdin") as copy:
            for row in rows:
                copy.write_row(row)
    # Leaving the `with` block commits: the table is replaced completely, or not at all.
    print(f"raw.traffic_tra0202: {len(rows)} rows, years {rows[0][0]}-{rows[-1][0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())