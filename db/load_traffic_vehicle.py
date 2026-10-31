"""Load DfT table TRA0204 (traffic in billion vehicle-km by vehicle type and road class) into Postgres.

TRA0204 is a spreadsheet (ODS) with one row per road class and vehicle type, and one column
per year. Its text cells carry layout whitespace, and its year headers are a mix of numbers
(2023) and text with note markers ("2020 [note 3]"). This script finds the table, checks the
headers and units are exactly as expected, and loads every published value unchanged into
raw.traffic_tra0204, one row per road class, vehicle type and year. DfT's own total rows
("All Motor Vehicles", "All Roads") are kept, so dbt can test the parts against them.

Usage:
    python db/load_traffic_vehicle.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd
import psycopg

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "raw" / "tra0204-km-by-vehicle-and-road-type.ods"
SHEET = "TRA0204"
DSN = os.environ.get("DATABASE_URL", "dbname=speed_gb")

# Exact text of the descriptive headers -> column name in raw.traffic_tra0204.
# Every column after these is a year. If DfT changes the layout, the load stops.
COLUMNS = {
    "Road Type": "road_type",
    "Road Management": "road_management",
    "Rural Urban Classification": "rural_urban",
    "Vehicle": "vehicle",
    "Notes": "notes",
    "Units": "units",
}
YEAR_HEADER = re.compile(r"(\d{4})(?:\.0)?(?: (\[note \d+\]))?")

DDL = """
drop table if exists raw.traffic_tra0204;
create table raw.traffic_tra0204 (
    road_type           text not null,
    road_management     text not null,
    rural_urban         text not null,
    vehicle             text not null,
    notes               text,
    year                smallint not null,
    year_note           text,
    billion_vehicle_km  numeric(6, 1) not null,
    primary key (road_type, road_management, rural_urban, vehicle, year)
);
"""


def clean(value):
    """Spreadsheet text cells carry layout whitespace ('   Motorway   '); collapse it."""
    if isinstance(value, str):
        return " ".join(value.split())
    return value


def year_of(header) -> tuple[int, str | None]:
    """2023.0 -> (2023, None); '2020 [note 3]' -> (2020, '[note 3]')."""
    match = YEAR_HEADER.fullmatch(str(header))
    if not match:
        raise ValueError(f"TRA0204: unexpected year header {header!r}")
    return int(match.group(1)), match.group(2)


def read_tra0204(path: Path) -> list[tuple]:
    sheet = pd.read_excel(path, sheet_name=SHEET, header=None, engine="odf")
    sheet = sheet.map(clean)

    # Find the header row rather than assuming its position: title rows above it can change.
    header_row = sheet.index[sheet[0] == "Road Type"][0]
    headers = sheet.iloc[header_row].dropna().tolist()
    if headers[:len(COLUMNS)] != list(COLUMNS):
        raise ValueError(f"TRA0204 headers have changed:\n  expected {list(COLUMNS)}\n  found    {headers[:len(COLUMNS)]}")
    years = [year_of(h) for h in headers[len(COLUMNS):]]

    table = sheet.iloc[header_row + 1:, :len(headers)].dropna(how="all")
    units = table[len(COLUMNS) - 1].unique().tolist()
    if units != ["Billion vehicle kilometres"]:
        raise ValueError(f"unexpected units: {units}")

    rows = []
    for _, row in table.iterrows():
        labels = row.iloc[:len(COLUMNS)].tolist()
        road_type, management, rural_urban, vehicle, notes = labels[:5]
        for (year, year_note), value in zip(years, row.iloc[len(COLUMNS):]):
            rows.append((road_type, management, rural_urban, vehicle, notes, year, year_note, float(value)))
    return rows


def main() -> int:
    rows = read_tra0204(SOURCE)
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(DDL)
        columns = "road_type, road_management, rural_urban, vehicle, notes, year, year_note, billion_vehicle_km"
        with cur.copy(f"copy raw.traffic_tra0204 ({columns}) from stdin") as copy:
            for row in rows:
                copy.write_row(row)
    # Leaving the `with` block commits: the table is replaced completely, or not at all.
    years = sorted({row[5] for row in rows})
    print(f"raw.traffic_tra0204: {len(rows)} rows, {len(rows) // len(years)} road class and vehicle rows, "
          f"years {years[0]}-{years[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
