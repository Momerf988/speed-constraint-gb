"""Extract the appraisal values the cost-benefit model needs, from DfT's own files.

Reads exact cells of the TAG data book (a macro workbook of 122 sheets, where each table sits
at a fixed position among notes) and two tables of RAS4001, and writes them, one value per
row with its source cell, to model/parameters/appraisal_values.csv. The model reads only that
file, so it runs without the raw spreadsheets, and every number can be traced to a cell.

Before reading each value, the script checks the labels beside it (row name, column header,
price year). If DfT moves a table in a future version, it stops instead of reading the wrong
cell. Cells are read as values saved in the file: the workbook's macros are never run.

TAG values are in 2023 prices and 2023 values, market prices. Fuel and carbon are read for 2023,
the central year, and for 2035, a sensitivity test with far more electric vehicles, still in 2023
prices. RAS4001 is used for two sensitivity tests only: its older casualty values, and DfT's
estimate of injuries never reported to the police.

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

# Fuel and carbon, for 2023 (central) and 2035 (a sensitivity test): row of that year in each table.
FUEL_YEAR_ROWS = {2023: {"A1.3.11": 39, "A1.3.9": 45, "A1.3.7": 41, "A3.3": 40, "A3.4": 39},
                  2035: {"A1.3.11": 51, "A1.3.9": 57, "A1.3.7": 53, "A3.3": 52, "A3.4": 51}}
FUEL_CURVE_COLUMNS = {"petrol_car": "DEFG", "diesel_car": "HIJK", "petrol_van": "PQRS", "diesel_van": "TUVW"}
FUEL_SHARE_COLUMNS = {"petrol_car": ("D", "Petrol"), "diesel_car": ("E", "Diesel"), "electric_car": ("F", "Electric"),
                      "petrol_van": ("G", "Petrol"), "diesel_van": ("H", "Diesel"), "electric_van": ("I", "Electric")}
# Group headings above the columns those rows are read from: (sheet, cell, expected).
TAG_GROUP_LABELS = [
    ("A1.3.11", "D24", "Petrol Car"), ("A1.3.11", "H24", "Diesel Car"), ("A1.3.11", "P24", "Petrol LGV"),
    ("A1.3.11", "T24", "Diesel LGV"), ("A1.3.9", "D24", "Cars"), ("A1.3.9", "G24", "LGV"),
    ("A1.3.7", "D24", "Resource Cost"), ("A1.3.7", "I24", "Duty"), ("A1.3.7", "M24", "VAT rate"),
    ("A3.3", "D26", "kg CO2e/l"), ("A3.4", "G24", "Market price unit of account"),
    ("A1.3.14", "A27", "Car"), ("A1.3.14", "A33", "LGV"),
]
# Cells that do not change by year: speed ranges of the fuel curves, and vehicle capital saving.
TAG_FIXED_CELLS = [
    ("fuel_min_speed_petrol_car", "A1.3.8", "A27", "Petrol Car", "H27", "H25", "Min speed", "km/h", 1),
    ("fuel_max_speed_petrol_car", "A1.3.8", "A27", "Petrol Car", "I27", "I25", "Max speed", "km/h", 1),
    ("fuel_min_speed_diesel_car", "A1.3.8", "A28", "Diesel Car", "H28", "H25", "Min speed", "km/h", 1),
    ("fuel_max_speed_diesel_car", "A1.3.8", "A28", "Diesel Car", "I28", "I25", "Max speed", "km/h", 1),
    ("fuel_min_speed_petrol_van", "A1.3.8", "A29", "Petrol LGV", "H29", "H25", "Min speed", "km/h", 1),
    ("fuel_max_speed_petrol_van", "A1.3.8", "A29", "Petrol LGV", "I29", "I25", "Max speed", "km/h", 1),
    ("fuel_min_speed_diesel_van", "A1.3.8", "A30", "Diesel LGV", "H30", "H25", "Min speed", "km/h", 1),
    ("fuel_max_speed_diesel_van", "A1.3.8", "A30", "Diesel LGV", "I30", "I25", "Max speed", "km/h", 1),
    ("capital_saving_car_work", "A1.3.14", "B27", "Work Petrol", "E27", "E26", "b1 p / hr", "£ per vehicle-hour", 0.01),
    ("capital_saving_van_services", "A1.3.14", "B35", "Work (services)", "E35", "E26", "b1 p / hr", "£ per vehicle-hour", 0.01),
]


def fuel_cells() -> list[tuple]:
    """The year-by-year fuel and carbon cells, in the same form as TAG_CELLS."""
    cells = []
    for year, rows in FUEL_YEAR_ROWS.items():
        curve, share, price, co2, carbon = (rows[s] for s in ("A1.3.11", "A1.3.9", "A1.3.7", "A3.3", "A3.4"))
        for fuel, columns in FUEL_CURVE_COLUMNS.items():
            for parameter, column in zip("abcd", columns):
                cells.append((f"fuel_{parameter}_{fuel}_{year}", "A1.3.11", f"B{curve}", year, f"{column}{curve}",
                              f"{column}25", parameter, "litres per km parameter", 1))
        for fuel, (column, header) in FUEL_SHARE_COLUMNS.items():
            cells.append((f"km_share_{fuel}_{year}", "A1.3.9", f"B{share}", year, f"{column}{share}",
                          f"{column}25", header, "share of vehicle-km", 1))
        for name, column, header, unit, scale in (
                ("fuel_resource_petrol", "D", "Petrol", "£ per litre", 0.01), ("fuel_resource_diesel", "E", "Diesel", "£ per litre", 0.01),
                ("fuel_duty_petrol", "I", "Petrol", "£ per litre", 0.01), ("fuel_duty_diesel", "J", "Diesel", "£ per litre", 0.01),
                ("fuel_vat_petrol", "M", "Petrol", "share", 0.01), ("fuel_vat_diesel", "N", "Diesel", "share", 0.01)):
            cells.append((f"{name}_{year}", "A1.3.7", f"B{price}", year, f"{column}{price}", f"{column}25", header, unit, scale))
        for name, column, header in (("co2_per_litre_petrol", "D", "Petrol"), ("co2_per_litre_diesel", "E", "Diesel")):
            cells.append((f"{name}_{year}", "A3.3", f"B{co2}", year, f"{column}{co2}", f"{column}24", header, "kg CO2e per litre", 1))
        for name, column, header in (("carbon_low", "G", "Low"), ("carbon_central", "H", "Central"), ("carbon_high", "I", "High")):
            cells.append((f"{name}_{year}", "A3.4", f"B{carbon}", year, f"{column}{carbon}", f"{column}25", header,
                          "£ per tonne CO2e", 1))
    return cells


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

    for sheet in ("A1.3.7", "A1.3.14", "A3.4"):
        if int(book[sheet]["B14"].value) != PRICE_YEAR:
            raise ValueError(f"{sheet}: expected {PRICE_YEAR} prices, found {book[sheet]['B14'].value}")
    for sheet, cell, expected in TAG_GROUP_LABELS:
        if clean(book[sheet][cell].value) != expected:
            raise ValueError(f"{sheet}: expected {expected!r} in {cell}, found {book[sheet][cell].value!r}")

    rows = []
    for value_id, sheet, label_cell, label, value_cell, header_cell, header, unit, scale in (
            TAG_CELLS + TAG_FIXED_CELLS + fuel_cells()):
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
