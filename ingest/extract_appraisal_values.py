"""Extract the appraisal values the cost-benefit model needs, from DfT's own files.

Reads exact cells of the TAG data book (a macro workbook of 122 sheets, where each table sits
at a fixed position among notes) and two tables of RAS4001, and writes them, one value per
row with its source cell, to model/parameters/appraisal_values.csv. The model reads only that
file, so it runs without the raw spreadsheets, and every number can be traced to a cell.

Before reading each value, the script checks the labels beside it (row name, column header,
price year). If DfT moves a table in a future version, it stops instead of reading the wrong
cell. Cells are read as values saved in the file: the workbook's macros are never run.

TAG values are in 2023 prices and 2023 values, market prices. RAS4001 is used for two
sensitivity tests only: its older casualty values, and DfT's estimate of injuries never
reported to the police.

Usage:
    python ingest/extract_appraisal_values.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TAG = ROOT / "data" / "raw" / "tag-data-book-v2-03.xlsm"
RAS4001 = ROOT / "data" / "raw" / "ras4001.ods"
OUT = ROOT / "model" / "parameters" / "appraisal_values.csv"

TAG_VERSION = "May 2026 v2.03"
PRICE_YEAR = 2023

# (value_id, sheet, row label cell, expected label, value cell, column header cell, expected header, unit, scale)
# "scale" turns TAG's percentages (0-100) into shares (0-1).
TAG_CELLS = [
    ("casualty_fatal", "A4.1.1", "A27", "Fatal", "G27", "G25", "Total", "£ per casualty", 1),
    ("casualty_serious", "A4.1.1", "A28", "Serious", "G28", "G25", "Total", "£ per casualty", 1),
    ("casualty_slight", "A4.1.1", "A29", "Slight", "G29", "G25", "Total", "£ per casualty", 1),
    ("collision_police_fatal", "A4.1.3", "A27", "Fatal", "G27", "G25", "Police cost", "£ per collision", 1),
    ("collision_police_serious", "A4.1.3", "A28", "Serious", "G28", "G25", "Police cost", "£ per collision", 1),
    ("collision_police_slight", "A4.1.3", "A29", "Slight", "G29", "G25", "Police cost", "£ per collision", 1),
    ("collision_damage_fatal", "A4.1.3", "A27", "Fatal", "H27", "H25", "Damage to", "£ per collision", 1),
    ("collision_damage_serious", "A4.1.3", "A28", "Serious", "H28", "H25", "Damage to", "£ per collision", 1),
    ("collision_damage_slight", "A4.1.3", "A29", "Slight", "H29", "H25", "Damage to", "£ per collision", 1),
    ("collision_insurance_fatal", "A4.1.3", "A27", "Fatal", "I27", "I25", "Insurance", "£ per collision", 1),
    ("collision_insurance_serious", "A4.1.3", "A28", "Serious", "I28", "I25", "Insurance", "£ per collision", 1),
    ("collision_insurance_slight", "A4.1.3", "A29", "Slight", "I29", "I25", "Insurance", "£ per collision", 1),
    ("vot_car_work", "A1.3.1", "A28", "Car driver", "F28", "F26", "Market", "£ per person-hour", 1),
    ("vot_van_freight", "A1.3.1", "A30", "LGV Freight (driver or passenger)", "F30", "F26", "Market", "£ per person-hour", 1),
    ("vot_van_services", "A1.3.1", "A31", "LGV Services (driver or passenger)", "F31", "F26", "Market", "£ per person-hour", 1),
    ("vot_commuting", "A1.3.1", "A47", "Commuting", "F47", "F45", "Market", "£ per person-hour", 1),
    ("vot_other", "A1.3.1", "A48", "Other", "F48", "F45", "Market", "£ per person-hour", 1),
    ("car_occupancy_work", "A1.3.3", "A27", "Work", "J27", "J25", "All Week Average", "people per vehicle-km", 1),
    ("car_occupancy_commuting", "A1.3.3", "A28", "Commuting", "J28", "J25", "All Week Average", "people per vehicle-km", 1),
    ("car_occupancy_other", "A1.3.3", "A29", "Other", "J29", "J25", "All Week Average", "people per vehicle-km", 1),
    ("van_occupancy_work", "A1.3.3", "B40", "Work", "J40", "J38", "Average", "people per vehicle-km", 1),
    ("van_occupancy_nonwork", "A1.3.3", "B41", "Non Work", "J41", "J38", "Average", "people per vehicle-km", 1),
    ("car_km_share_work", "A1.3.4", "B27", "Work", "J27", "J24", "All Week", "share of vehicle-km", 0.01),
    ("car_km_share_commuting", "A1.3.4", "B28", "Commuting", "J28", "J24", "All Week", "share of vehicle-km", 0.01),
    ("car_km_share_other", "A1.3.4", "B29", "Other", "J29", "J24", "All Week", "share of vehicle-km", 0.01),
    ("van_km_share_freight", "A1.3.4", "B30", "Work (freight)", "J30", "J24", "All Week", "share of vehicle-km", 0.01),
    ("van_km_share_services", "A1.3.4", "B31", "Work (services)", "J31", "J24", "All Week", "share of vehicle-km", 0.01),
    ("van_km_share_commuting", "A1.3.4", "B32", "Non-work (commute)", "J32", "J24", "All Week", "share of vehicle-km", 0.01),
    ("van_km_share_other", "A1.3.4", "B33", "Non-work (other)", "J33", "J24", "All Week", "share of vehicle-km", 0.01),
]
PRICE_YEAR_SHEETS = ("A4.1.1", "A4.1.3", "A1.3.1")   # sheets that state their price and value year in B14:B15


def clean(value):
    """TAG labels carry stray spaces ('Work ', 'Other '); collapse them."""
    return " ".join(value.split()) if isinstance(value, str) else value


def tag_values(path: Path) -> list[dict]:
    book = openpyxl.load_workbook(path, read_only=True, data_only=True, keep_vba=False)
    for sheet in ("A4.1.1", "A1.3.1"):
        version = clean(book[sheet]["A7"].value)
        if version != TAG_VERSION:
            raise ValueError(f"{sheet}: expected TAG data book {TAG_VERSION!r}, found {version!r}")
    for sheet in PRICE_YEAR_SHEETS:
        years = (book[sheet]["B14"].value, book[sheet]["B15"].value)
        if tuple(int(y) for y in years) != (PRICE_YEAR, PRICE_YEAR):
            raise ValueError(f"{sheet}: expected {PRICE_YEAR} prices and values, found {years}")
    # Occupancies are from 2010 (cars) and 2000 (vans); TAG holds them constant, which is
    # only true while its table of annual changes is all zero.
    changes = [book["A1.3.3"][f"{col}{row}"].value for row in (50, 51) for col in "DEFGHIJ"]
    if any(float(c) != 0 for c in changes):
        raise ValueError(f"A1.3.3: occupancy is no longer constant over time: {changes}")

    rows = []
    for value_id, sheet, label_cell, label, value_cell, header_cell, header, unit, scale in TAG_CELLS:
        ws = book[sheet]
        found_label, found_header = clean(ws[label_cell].value), clean(ws[header_cell].value)
        if found_label != label or found_header != header:
            raise ValueError(f"{sheet}: expected {label!r} in {label_cell} and {header!r} in {header_cell}, "
                             f"found {found_label!r} and {found_header!r}")
        rows.append({
            "value_id": value_id,
            "value": float(ws[value_cell].value) * scale,
            "unit": unit,
            "source": f"TAG data book {TAG_VERSION}",
            "location": f"{sheet}!{value_cell}",
            "price_year": PRICE_YEAR if unit.startswith("£") else "",
        })
    return rows


def ras4001_values(path: Path) -> list[dict]:
    average = pd.read_excel(path, sheet_name="Average_value", header=4, engine="odf")
    average["Cost per casualty (£)"] = pd.to_numeric(average["Cost per casualty (£)"], errors="coerce")  # "-" -> NaN
    rows = []
    for severity in ("Fatal", "Serious", "Slight"):
        match = average[(average["Price year"] == PRICE_YEAR) & (average["Severity"] == severity)]
        if len(match) != 1:
            raise ValueError(f"RAS4001 Average_value: expected one {PRICE_YEAR} {severity} row, found {len(match)}")
        rows.append({
            "value_id": f"previous_casualty_{severity.lower()}",
            "value": float(match["Cost per casualty (£)"].iloc[0]),
            "unit": "£ per casualty",
            "source": "RAS4001 (2026), based on the September 2024 TAG data book",
            "location": f"Average_value: {PRICE_YEAR}, {severity}",
            "price_year": PRICE_YEAR,
        })

    total = pd.read_excel(path, sheet_name="Total_value", header=3, engine="odf")
    latest = total["Collision data year"].max()
    year = total[total["Collision data year"] == latest].set_index("Severity [note 4]")["Total (£ million)"]
    unreported = year["Non-fatal injury collisions not reported to the police"]
    reported = year["Serious"] + year["Slight"]
    rows.append({
        "value_id": "unreported_to_reported_nonfatal",
        "value": float(unreported / reported),
        "unit": "ratio",
        "source": "RAS4001 (2026)",
        "location": f"Total_value: {latest}, unreported non-fatal / (serious + slight)",
        "price_year": "",
    })
    return rows


def main() -> int:
    rows = tag_values(TAG) + ras4001_values(RAS4001)
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{OUT.relative_to(ROOT)}: {len(rows)} values")
    for row in rows:
        print(f"  {row['value_id']:34s} {row['value']:>14,.4f}  {row['unit']:22s} {row['location']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
