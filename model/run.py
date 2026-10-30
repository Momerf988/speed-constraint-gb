"""One full model run: every road type under every policy, for one set of assumptions.

Returns one row per road type and policy: mean speeds before and after, the casualties
prevented each year under each model (against today, and against business-as-usual ISA), and
how far the scenario lies from the evidence. analysis/rq2_rq4.py calls it once for the
central result, once per sensitivity test, and thousands of times for the Monte Carlo.

The preferred estimate is Model B where the road type has an individual risk curve, and the
Power Model on motorways, which have none (as ISA-UK did).

Evidence label of each row:
    within        mean-speed change inside Model A's range, and Model B a point estimate
    lower bound   inside Model A's range, but Model B cut short for over 10% of cars
    beyond        mean-speed change outside -14 to +7 km/h, so every model extrapolates
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from model.casualties import MAX_SHARE_CLAMPED, MODEL_A_RANGE_KMH, Assumptions, model_a, model_b
from model.scenarios import (ISA_SWITCHED_OFF_SHARE, POLICIES, ROAD_TYPES, apply_policy, isa_working_share,
                             original_speeds)
from model.speeds import SpeedDistribution, band_intervals, empirical_distribution, fit, fitted_distribution

BASELINE_YEARS = (2022, 2024)   # the Road Safety Strategy's own baseline
SPEED_YEAR = 2024               # latest DfT speed distributions
EVIDENCE_ORDER = ("within", "lower bound", "beyond")
PREVENTED_COLUMNS = (
    "killed_prevented", "killed_prevented_b", "killed_prevented_power", "killed_prevented_exp",
    "ksi_prevented_b",
    "casualties_prevented", "casualties_prevented_b", "casualties_prevented_power", "casualties_prevented_exp",
)


def load_inputs(marts_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Speed bands for SPEED_YEAR, and annual baseline casualties with car/van shares per road type."""
    bands = duckdb.sql(f"""
        select * from read_parquet('{marts_dir}/agg_speed_bands_cars.parquet') where year = {SPEED_YEAR}
    """).df()
    years = BASELINE_YEARS[1] - BASELINE_YEARS[0] + 1
    baseline = duckdb.sql(f"""
        select segment,
               sum(killed) / {years}                                 as killed,
               sum(ksi_adjusted) / {years}                           as ksi,
               sum(casualties) / {years}                             as casualties,
               sum(killed_car_van) / sum(killed)                     as k_fatal,
               sum(ksi_adjusted_car_van) / sum(ksi_adjusted)         as k_ksi,
               sum(casualties_car_van)::double / sum(casualties)     as k_casualties
        from read_parquet('{marts_dir}/agg_casualties_segment_year.parquet')
        where year between {BASELINE_YEARS[0]} and {BASELINE_YEARS[1]}
        group by 1
    """).df().set_index("segment")
    return bands, baseline


def speed_distributions(bands: pd.DataFrame, shape: str = "empirical") -> dict[str, SpeedDistribution]:
    """Today's speed distribution on each modelled road type: empirical, normal or lognormal."""
    distributions = {}
    for road in ROAD_TYPES:
        intervals, _ = band_intervals(bands[bands["road_type"] == road.dft_road_type])
        distributions[road.segment] = (
            empirical_distribution(intervals, road.segment) if shape == "empirical"
            else fitted_distribution(shape, *fit(intervals, shape), road.segment)
        )
    return distributions


def evidence_label(model_a_in_range: bool, model_b_lower_bound: bool) -> str:
    if not model_a_in_range:
        return "beyond"
    return "lower bound" if model_b_lower_bound else "within"


def worse(*labels: str) -> str:
    """The weakest evidence label among several, e.g. for a total over road types."""
    return max(labels, key=EVIDENCE_ORDER.index)


def run(distributions: dict[str, SpeedDistribution], baseline: pd.DataFrame, assumptions: Assumptions,
        switched_off: float = ISA_SWITCHED_OFF_SHARE, overrides: dict[str, float] | None = None) -> pd.DataFrame:
    rows = []
    for road in ROAD_TYPES:
        today = distributions[road.segment]
        base = baseline.loc[road.segment]
        slight = base["casualties"] - base["ksi"]
        override = road.override_share if overrides is None else overrides[road.segment]
        working = isa_working_share(road, switched_off, override)
        results = {}
        for policy in POLICIES:
            after = apply_policy(policy, today, road, working)
            a = model_a(today, after, base["k_fatal"], base["k_casualties"], assumptions)
            b = model_b(today, after, original_speeds(policy, today), road, base["k_ksi"], assumptions)
            in_range = MODEL_A_RANGE_KMH[0] <= a["mean_change_kmh"] <= MODEL_A_RANGE_KMH[1]
            r = {
                "segment": road.segment,
                "policy": policy,
                "mean_before_kmh": today.mean(),
                "mean_after_kmh": after.mean(),
                "mean_change_kmh": a["mean_change_kmh"],
                "pct_over_limit_after": 100 * after.share_at_or_above(road.limit_kmh + 0.05),
                "share_clamped": np.nan if b is None else b["share_clamped"],
                "evidence": evidence_label(in_range, b is not None and b["share_clamped"] > MAX_SHARE_CLAMPED),
                "preferred_model": "power" if b is None else "b",
                "rho": np.nan if b is None else b["rho"],
                "baseline_killed": base["killed"],
                "baseline_ksi": base["ksi"],
                "baseline_casualties": base["casualties"],
                "killed_prevented_b": np.nan if b is None else base["killed"] * (1 - b["fatal"]),
                "killed_prevented_power": base["killed"] * (1 - a["power_fatal"]),
                "killed_prevented_exp": base["killed"] * (1 - a["exponential_fatal"]),
                "ksi_prevented_b": np.nan if b is None else base["ksi"] * (1 - b["ksi"]),
                "casualties_prevented_b": (np.nan if b is None
                                           else base["ksi"] * (1 - b["ksi"]) + slight * (1 - b["slight"])),
                "casualties_prevented_power": base["casualties"] * (1 - a["power_casualties"]),
                "casualties_prevented_exp": base["casualties"] * (1 - a["exponential_casualties"]),
            }
            r["killed_prevented"] = r[f"killed_prevented_{r['preferred_model']}"]
            r["casualties_prevented"] = r[f"casualties_prevented_{r['preferred_model']}"]
            results[policy] = r
        bau = results["bau_isa"]
        for r in results.values():
            for column in PREVENTED_COLUMNS:
                r[f"{column}_vs_bau"] = r[column] - bau[column]
            r["evidence_vs_bau"] = worse(r["evidence"], bau["evidence"])
            rows.append(r)
    return pd.DataFrame(rows)
