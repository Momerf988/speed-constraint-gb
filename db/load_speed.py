"""Load DfT table SPE0102 (free-flow vehicle speeds by road type and vehicle type) into Postgres.

SPE0102 is a spreadsheet with one sheet per year (2018 onwards). Each sheet stacks four road
types, and each road type is a block of rows: 5 mph speed bands, the share exceeding the
limit, the average speed and the number of vehicles observed. Values are kept as text,
exactly as published, because some cells hold markers rather than numbers: "[low]" means a
share too small to print but not zero. Parsing happens in dbt.

Usage:
    python db/load_speed.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd
import psycopg

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "raw" / "spe0102.ods"
DSN = os.environ.get("DATABASE_URL", "dbname=speed_gb")
YEAR_SHEET = re.compile(r"SPE0102_\((\d{4})\)")

# Exact header text in each yearly sheet -> column name in raw.speed_spe0102.
# "Artciulated" is DfT's own spelling in the file: the check must match the file as
# published, typo included. DfT's "Units" column actually describes each row's measure.
COLUMNS = {
    "Road type": "road_type",
    "Notes": "notes",
    "Units": "measure",
    "Cars [note 4]": "cars",
    "Light Commercial Vehicles [notes 5, 6]": "light_commercial_vehicles",
    "Short Buses [note 7]": "short_buses",
    "Long Buses [note 7]": "long_buses",
    "Rigid Heavy Goods Vehicles 2 axles [notes 1, 2]": "rigid_hgv_2_axles",
    "Rigid Heavy Goods Vehicles 3 or more axles [notes 1, 2]": "rigid_hgv_3_plus_axles",
    "All Rigid Heavy Goods Vehicles [notes 1, 2]": "all_rigid_hgv",
    "Articulated Heavy Goods Vehicles 3 or 4 axles [notes 1, 2]": "artic_hgv_3_4_axles",
    "Articulated Heavy Goods Vehicles 5 or more axles [notes 1, 2]": "artic_hgv_5_plus_axles",
    "All Artciulated Heavy Goods Vehicles [notes 1, 2]": "all_artic_hgv",
    "Motorcycles [note 8]": "motorcycles",
}

DDL = """
drop table if exists raw.speed_spe0102;
create table raw.speed_spe0102 (
    year                       smallint not null,
    row_order                  smallint not null,
    road_type                  text not null,
    notes                      text,
    measure                    text not null,
    cars                       text,
    light_commercial_vehicles  text,
    short_buses                text,
    long_buses                 text,
    rigid_hgv_2_axles          text,
    rigid_hgv_3_plus_axles     text,
    all_rigid_hgv              text,
    artic_hgv_3_4_axles        text,
    artic_hgv_5_plus_axles     text,
    all_artic_hgv              text,
    motorcycles                text,
    primary key (year, road_type, measure)
);
"""


def clean(value):
    """Spreadsheet text cells carry layout whitespace; collapse it."""
    if isinstance(value, str):
        return " ".join(value.split())
    return value


def as_text(value) -> str | None:
    """Numbers become their full-precision text; markers such as "[low]" stay as they are."""
    if pd.isna(value):
        return None
    if isinstance(value, float):
        return repr(value)
    return str(value)


def read_spe0102(path: Path) -> list[tuple]:
    book = pd.read_excel(path, sheet_name=None, header=None, engine="odf")
    years = {int(m.group(1)): name for name in book if (m := YEAR_SHEET.fullmatch(name))}
    if not years:
        raise ValueError(f"no yearly sheets found; the sheets are {list(book)}")

    rows = []
    for year, name in sorted(years.items()):
        sheet = book[name].map(clean)
        header_row = sheet.index[sheet[0] == "Road type"][0]
        headers = sheet.iloc[header_row].dropna().tolist()
        if headers != list(COLUMNS):
            raise ValueError(f"{name}: headers have changed:\n  expected {list(COLUMNS)}\n  found    {headers}")
        table = sheet.iloc[header_row + 1:, :len(COLUMNS)].dropna(how="all")
        for order, (_, row) in enumerate(table.iterrows(), start=1):
            rows.append((year, order, *(as_text(value) for value in row.tolist())))
    return rows


def main() -> int:
    rows = read_spe0102(SOURCE)
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(DDL)
        columns = "year, row_order, " + ", ".join(COLUMNS.values())
        with cur.copy(f"copy raw.speed_spe0102 ({columns}) from stdin") as copy:
            for row in rows:
                copy.write_row(row)
    years = sorted({row[0] for row in rows})
    print(f"raw.speed_spe0102: {len(rows)} rows, years {years[0]}-{years[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())