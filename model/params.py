"""Read the speed-safety parameters file and refuse to hand the model anything unverified.

Every number the model uses comes from model/parameters/speed_safety_parameters.csv, where
each row records its source, where in the source it appears, and its status:
    primary     read in the original source
    secondary   quoted by another source, not yet seen in the original
    to_extract  identified but not yet filled in
The model asks for parameters by id through require(). It gets them only if every one is
"primary"; otherwise it stops with an error naming the rows. Rows the model never asks for
(such as the Kloeden 1997 reference table) can stay secondary without blocking anything.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

PARAMETERS_FILE = Path(__file__).resolve().parent / "parameters" / "speed_safety_parameters.csv"
STATUSES = {"primary", "secondary", "to_extract"}


@dataclass(frozen=True)
class Parameter:
    parameter_id: str
    model: str
    outcome: str
    values: tuple[float, ...]   # one number, or a curve's coefficients in the order the file gives them
    ci_low: float | None
    ci_high: float | None
    unit: str
    valid_range: str
    source: str
    location_in_source: str
    status: str

    @property
    def value(self) -> float:
        """The single value of a one-number parameter."""
        if len(self.values) != 1:
            raise ValueError(f"{self.parameter_id} has {len(self.values)} values; use .values")
        return self.values[0]


def _number(text: str) -> float | None:
    return float(text) if text.strip() else None


def load_parameters(path: Path = PARAMETERS_FILE) -> dict[str, Parameter]:
    """All rows of the parameters file, keyed by parameter_id, with basic checks."""
    parameters: dict[str, Parameter] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pid = row["parameter_id"]
            if pid in parameters:
                raise ValueError(f"duplicate parameter_id: {pid}")
            if row["status"] not in STATUSES:
                raise ValueError(f"{pid}: unknown status {row['status']!r}")
            text = row["value"].strip()
            if not text and row["status"] != "to_extract":
                raise ValueError(f"{pid}: no value, but status is {row['status']!r}")
            values = tuple(float(part) for part in text.split(";")) if text else ()
            parameters[pid] = Parameter(
                parameter_id=pid,
                model=row["model"],
                outcome=row["outcome"],
                values=values,
                ci_low=_number(row["ci_low"]),
                ci_high=_number(row["ci_high"]),
                unit=row["unit"],
                valid_range=row["valid_range"],
                source=row["source"],
                location_in_source=row["location_in_source"],
                status=row["status"],
            )
    return parameters


def require(*parameter_ids: str, parameters: dict[str, Parameter] | None = None) -> dict[str, Parameter]:
    """The requested parameters, but only if every one exists and is "primary"."""
    parameters = load_parameters() if parameters is None else parameters
    missing = [pid for pid in parameter_ids if pid not in parameters]
    unverified = [pid for pid in parameter_ids if pid in parameters and parameters[pid].status != "primary"]
    if missing or unverified:
        raise ValueError(
            "the model refused its parameters:\n"
            + (f"  not in the file: {missing}\n" if missing else "")
            + (f"  not verified in the original source: {unverified}\n" if unverified else "")
        )
    return {pid: parameters[pid] for pid in parameter_ids}