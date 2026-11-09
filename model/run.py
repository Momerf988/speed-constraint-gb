"""One full model run: every road type under every policy, for one set of assumptions.

Returns one row per road type and policy: mean speeds before and after, the casualties
prevented each year under each model (against today, and against business-as-usual ISA), and
how far the scenario lies from the evidence. analysis/rq2_rq4.py calls it once for the
central result, once per sensitivity test, and thousands of times for the Monte Carlo.

The preferred estimate is Model B where the road type has an individual risk curve, and the
Power Model on motorways, which have none (as ISA-UK did).

Evidence label of each row:
    within        Model A's fitted range holds, and Model B is a point estimate
    lower bound   Model A's fitted range holds, but Model B is cut short for over 10% of vehicles
    beyond        mean-speed change outside -14 to +7 km/h, or a starting mean above 95 km/h: every
                  model extrapolates. Elvik et al. (2019) base their preferred values on initial speeds
                  up to 95 km/h, so every motorway result is beyond the evidence, whatever its size.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from model.appraisal import FUEL_YEAR, running_per_km
from model.casualties import (MAX_SHARE_CLAMPED, MODEL_A_MAX_INITIAL_KMH, MODEL_A_RANGE_KMH, Assumptions, CarVanShares,
                              model_a, model_b)
from model.scenarios import POLICIES, ROAD_TYPES, IsaUse, RoadType, Traffic, fleet_after, fleet_before
from model.speeds import SpeedDistribution, band_intervals, empirical_distribution, fit, fitted_distribution

BASELINE_YEARS = (2022, 2024)   # the Road Safety Strategy's own baseline
SPEED_YEAR = 2024               # latest DfT speed distributions
EVIDENCE_ORDER = ("within", "lower bound", "beyond")
PREVENTED_COLUMNS = (
    "killed_prevented", "killed_prevented_b", "killed_prevented_power", "killed_prevented_exp",
    "ksi_prevented_b",
    "casualties_prevented", "casualties_prevented_b", "casualties_prevented_power", "casualties_prevented_exp",
)
SAVED_COLUMNS = tuple(f"{vehicle}_{item}_saved_per_km" for vehicle in ("car", "van") for item in ("fuel", "tax", "co2"))


@dataclass(frozen=True)
class SpeedInputs:
    """What DfT publishes about speeds for SPEED_YEAR, plus vans' share of car and van vehicle-km."""
    car_bands: pd.DataFrame
    van_bands: pd.DataFrame
    van_summary: pd.DataFrame
    van_shares: dict[str, float]


def load_inputs(marts_dir) -> tuple[SpeedInputs, pd.DataFrame]:
    """Speed inputs, and annual baseline casualties with car/van shares per road type."""
    def table(name):
        return duckdb.sql(f"select * from read_parquet('{marts_dir}/{name}.parquet') where year = {SPEED_YEAR}").df()

    shares = duckdb.sql(f"""
        select segment, sum(van_billion_vehicle_km) / sum(car_billion_vehicle_km + van_billion_vehicle_km) as van_share
        from read_parquet('{marts_dir}/agg_traffic_segment.parquet')
        where year between {BASELINE_YEARS[0]} and {BASELINE_YEARS[1]}
        group by 1
    """).df()
    speeds = SpeedInputs(table("agg_speed_bands_cars"), table("agg_speed_bands_vans"), table("agg_speed_summary_vans"),
                         dict(zip(shares["segment"], shares["van_share"])))
    years = BASELINE_YEARS[1] - BASELINE_YEARS[0] + 1
    baseline = duckdb.sql(f"""
        select segment,
               sum(killed) / {years}                                 as killed,
               sum(ksi_adjusted) / {years}                           as ksi,
               sum(casualties) / {years}                             as casualties,
               sum(killed_car_van) / sum(killed)                     as k_fatal,
               sum(ksi_adjusted_car_van) / sum(ksi_adjusted)         as k_ksi,
               sum(casualties_car_van)::double / sum(casualties)     as k_casualties,
               (sum(casualties_car_van) - sum(ksi_adjusted_car_van))
                   / (sum(casualties) - sum(ksi_adjusted))           as k_slight
        from read_parquet('{marts_dir}/agg_casualties_segment_year.parquet')
        where year between {BASELINE_YEARS[0]} and {BASELINE_YEARS[1]}
        group by 1
    """).df().set_index("segment")
    return speeds, baseline


def speed_distribution(bands: pd.DataFrame, shape: str, label: str) -> SpeedDistribution:
    """One vehicle type on one road type: empirical, normal or lognormal."""
    intervals, _ = band_intervals(bands)
    return (empirical_distribution(intervals, label) if shape == "empirical"
            else fitted_distribution(shape, *fit(intervals, shape), label))


def speed_distributions(speeds: SpeedInputs, shape: str = "empirical") -> dict[str, Traffic]:
    """Today's cars and vans on each modelled road type.

    DfT publishes no van bands on national-limit single carriageways, because its counters cannot
    tell which limit a van has (SPE0102 note 6), only their average speed. There the vans take the
    car distribution, moved to the van average.
    """
    traffic = {}
    for road in ROAD_TYPES:
        cars = speed_distribution(speeds.car_bands[speeds.car_bands["road_type"] == road.dft_road_type], shape,
                                  f"{road.segment}, cars")
        van_bands = speeds.van_bands[speeds.van_bands["road_type"] == road.dft_road_type]
        if len(van_bands):
            vans = speed_distribution(van_bands, shape, f"{road.segment}, vans")
        else:
            summary = speeds.van_summary[speeds.van_summary["road_type"] == road.dft_road_type].iloc[0]
            vans = cars.shifted(summary["van_avg_free_flow_kmh"] - summary["car_avg_free_flow_kmh"], f"{road.segment}, vans")
        traffic[road.segment] = Traffic(cars, vans, speeds.van_shares[road.segment])
    return traffic


def evidence_label(model_a_in_range: bool, model_b_lower_bound: bool) -> str:
    if not model_a_in_range:
        return "beyond"
    return "lower bound" if model_b_lower_bound else "within"


def worse(*labels: str) -> str:
    """The weakest evidence label among several, e.g. for a total over road types."""
    return max(labels, key=EVIDENCE_ORDER.index)


def run(distributions: dict[str, Traffic], baseline: pd.DataFrame, assumptions: Assumptions,
        isa: IsaUse | None = None, roads: tuple[RoadType, ...] = ROAD_TYPES, fuel_year: int = FUEL_YEAR) -> pd.DataFrame:
    isa = IsaUse.central() if isa is None else isa
    rows = []
    for road in roads:
        traffic = distributions[road.segment]
        today = traffic.mixed()
        running_today = {"car": running_per_km(traffic.cars, "car", fuel_year),
                         "van": running_per_km(traffic.vans, "van", fuel_year)}
        base = baseline.loc[road.segment]
        slight = base["casualties"] - base["ksi"]
        k = CarVanShares(fatal=base["k_fatal"], ksi=base["k_ksi"], slight=base["k_slight"])
        results = {}
        for policy in POLICIES:
            cars, vans, after = fleet_after(policy, traffic, road, isa)
            a = model_a(today, after, base["k_fatal"], base["k_casualties"], assumptions)
            b = model_b(today, after, fleet_before(policy, traffic), road, k, assumptions)
            in_range = (MODEL_A_RANGE_KMH[0] <= a["mean_change_kmh"] <= MODEL_A_RANGE_KMH[1]
                        and today.mean() <= MODEL_A_MAX_INITIAL_KMH)
            r = {
                "segment": road.segment,
                "policy": policy,
                "mean_before_kmh": today.mean(),
                "mean_after_kmh": after.mean(),
                "mean_change_kmh": a["mean_change_kmh"],
                "car_mean_after_kmh": cars.mean(),
                "van_mean_after_kmh": vans.mean(),
                "pct_cars_over_limit_after": 100 * cars.share_at_or_above(road.limit_kmh + 0.05),
                "pct_vans_over_limit_after": 100 * vans.share_at_or_above(road.van_limit_kmh + 0.05),
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
            for vehicle, dist in (("car", cars), ("van", vans)):
                running = running_per_km(dist, vehicle, fuel_year)
                for item, value in running.items():
                    r[f"{vehicle}_{item}_saved_per_km"] = running_today[vehicle][item] - value
                r[f"{vehicle}_co2_per_km"] = running["co2"]
            r["killed_prevented"] = r[f"killed_prevented_{r['preferred_model']}"]
            r["casualties_prevented"] = r[f"casualties_prevented_{r['preferred_model']}"]
            results[policy] = r
        bau = results["bau_isa"]
        for r in results.values():
            for column in PREVENTED_COLUMNS + SAVED_COLUMNS:
                r[f"{column}_vs_bau"] = r[column] - bau[column]
            r["evidence_vs_bau"] = worse(r["evidence"], bau["evidence"])
            rows.append(r)
    return pd.DataFrame(rows)
