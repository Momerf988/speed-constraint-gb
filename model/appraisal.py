"""Put £ on each policy: the casualties it prevents against the travel time it costs.

Benefit: deaths, serious and slight injuries prevented (from the casualty model), each valued
at TAG's average value of prevention per casualty, plus the police, insurance and property
damage costs that TAG values per collision, using STATS19's collisions per casualty. Then the
fuel and carbon saved by the free-flowing share of traffic.

Fuel follows TAG unit A1.3: litres per km = a/v + b + c*v + d*v^2, by vehicle and fuel, held at
the speed limits of each curve, and averaged over the speed distribution before and after a
policy. An approximation: the curves are for average link speeds, and they are applied here to
spot speeds. TAG treats an electric vehicle's energy use as independent of speed, so electric
vehicles save nothing here. Drivers save fuel at its pump price, and the government loses the
duty and VAT on it. Those cancel, leaving the resource cost, which is what counts as the benefit;
the fall in tax is reported beside it. Carbon is valued at TAG's central market-price value.

Cost: the extra hours car and van occupants spend on the road, valued at TAG's market-price
value of time per vehicle-hour. A vehicle loses 1/new speed - 1/old speed hours per km it drives.
Only free-flowing traffic is slowed: in congestion, speeds are already below any cap. Following
DfT practice (Carsten and Tate 2005; ISA-UK), time a driver gained by breaking the limit is not
counted, so only lawful time lost is a cost, each vehicle against its own limit: a van above
80 km/h on a single carriageway was already breaking its limit. A sensitivity test counts all of it.
TAG also charges working cars and service vans for time on the road through the vehicle capital
term of its non-fuel operating cost, C = a1 + b1/v. Only b1/v moves with speed, and it is charged
on the same hours. The rest of a vehicle's non-fuel cost depends on distance, not speed.

Everything is per year, at 2022-2024 casualty and traffic levels, with every car fitted, in
2023 prices and values (TAG data book v2.03, market prices). Because benefits and time costs
both grow with the share of cars fitted, the ratio of the two does not depend on how quickly
the fleet is fitted.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from model.scenarios import RoadType, apply_policy, original_speeds
from model.speeds import SpeedDistribution

VALUES_FILE = Path(__file__).resolve().parent / "parameters" / "appraisal_values.csv"
SEVERITIES = ("fatal", "serious", "slight")
SAME_SPEED_KMH = 0.5   # a km/h cap this close below an mph limit is that limit in practice (80 and 50 mph = 80.47)
FUEL_YEAR = 2023       # fleet, fuel curves, fuel prices and carbon value of the central appraisal

# Share of car and van km driven in free flow, where drivers choose their own speed. No
# published source gives it, so these are stated judgements, drawn from wide ranges in the
# Monte Carlo; the break-even share is reported so a reader can apply their own.
FREE_FLOW_SHARE = {
    "32 km/h roads": 0.4,
    "48 km/h roads": 0.4,
    "97 km/h single carriageways": 0.7,
    "113 km/h motorways": 0.7,
}
FREE_FLOW_RANGES = {
    "32 km/h roads": (0.2, 0.6),
    "48 km/h roads": (0.2, 0.6),
    "97 km/h single carriageways": (0.5, 0.9),
    "113 km/h motorways": (0.5, 0.9),
}


@dataclass(frozen=True)
class Values:
    """The £ values one appraisal run uses."""
    per_casualty: dict[str, float]      # £ per casualty, by severity
    per_collision: dict[str, float]     # £ per collision: police, property damage, insurance
    car_hour: float                     # £ per car-hour, all occupants
    van_hour: float                     # £ per van-hour, all occupants
    unreported_ratio: float             # non-fatal injuries never reported to police, per reported one
    car_capital_hour: float = 0.0       # £ per car-hour of vehicle capital: TAG's b1, working cars only
    van_capital_hour: float = 0.0       # £ per van-hour of vehicle capital: TAG's b1, service vans only
    carbon_tonne: float = 0.0           # £ per tonne of CO2e, central market-price value


def read_values(path: Path = VALUES_FILE) -> dict[str, float]:
    with open(path, newline="") as f:
        return {row["value_id"]: float(row["value"]) for row in csv.DictReader(f)}


def vehicle_hour_values(v: dict[str, float]) -> tuple[float, float]:
    """£ per vehicle-hour for cars and vans: share of km by purpose x people per vehicle x £ per person-hour."""
    car = (v["car_km_share_work"] * v["car_occupancy_work"] * v["vot_car_work"]
           + v["car_km_share_commuting"] * v["car_occupancy_commuting"] * v["vot_commuting"]
           + v["car_km_share_other"] * v["car_occupancy_other"] * v["vot_other"])
    van = (v["van_km_share_freight"] * v["van_occupancy_work"] * v["vot_van_freight"]
           + v["van_km_share_services"] * v["van_occupancy_work"] * v["vot_van_services"]
           + v["van_km_share_commuting"] * v["van_occupancy_nonwork"] * v["vot_commuting"]
           + v["van_km_share_other"] * v["van_occupancy_nonwork"] * v["vot_other"])
    return car, van


def load_values(casualty_values: str = "tag", path: Path = VALUES_FILE) -> Values:
    """casualty_values: "tag" (TAG v2.03, central) or "previous" (RAS4001's, from the 2024 TAG)."""
    v = read_values(path)
    prefix = {"tag": "casualty", "previous": "previous_casualty"}[casualty_values]
    car, van = vehicle_hour_values(v)
    return Values(
        per_casualty={s: v[f"{prefix}_{s}"] for s in SEVERITIES},
        per_collision={s: v[f"collision_police_{s}"] + v[f"collision_damage_{s}"] + v[f"collision_insurance_{s}"]
                       for s in SEVERITIES},
        car_hour=car,
        van_hour=van,
        unreported_ratio=v["unreported_to_reported_nonfatal"],
        car_capital_hour=v["car_km_share_work"] * v["capital_saving_car_work"],
        van_capital_hour=v["van_km_share_services"] * v["capital_saving_van_services"],
        carbon_tonne=v[f"carbon_central_{FUEL_YEAR}"],
    )


@dataclass(frozen=True)
class Fuel:
    """One fuel in one vehicle type: its share of the vehicle type's km, its curve, and what a litre is worth."""
    km_share: float
    a: float
    b: float
    c: float
    d: float
    min_speed: float
    max_speed: float
    resource: float         # £ per litre
    tax: float              # £ per litre: duty, plus VAT on the km not driven for work
    co2_kg: float           # kg CO2e per litre

    def litres_per_km(self, speeds_kmh: np.ndarray) -> np.ndarray:
        v = np.clip(speeds_kmh, self.min_speed, self.max_speed)
        return self.a / v + self.b + self.c * v + self.d * v * v


@cache
def load_fuels(year: int = FUEL_YEAR) -> dict[str, tuple[Fuel, ...]]:
    """The petrol and diesel shares of cars and of vans, for one year. Electric km save no energy, so they are left out."""
    v = read_values()
    non_work = {"car": v["car_km_share_commuting"] + v["car_km_share_other"],
                "van": v["van_km_share_commuting"] + v["van_km_share_other"]}

    def fuel(vehicle: str, kind: str) -> Fuel:
        name = f"{kind}_{vehicle}"
        resource, duty = v[f"fuel_resource_{kind}_{year}"], v[f"fuel_duty_{kind}_{year}"]
        return Fuel(km_share=v[f"km_share_{name}_{year}"],
                    a=v[f"fuel_a_{name}_{year}"], b=v[f"fuel_b_{name}_{year}"],
                    c=v[f"fuel_c_{name}_{year}"], d=v[f"fuel_d_{name}_{year}"],
                    min_speed=v[f"fuel_min_speed_{name}"], max_speed=v[f"fuel_max_speed_{name}"],
                    resource=resource,
                    tax=duty + non_work[vehicle] * v[f"fuel_vat_{kind}_{year}"] * (resource + duty),
                    co2_kg=v[f"co2_per_litre_{kind}_{year}"])

    return {vehicle: (fuel(vehicle, "petrol"), fuel(vehicle, "diesel")) for vehicle in ("car", "van")}


def running_per_km(dist: SpeedDistribution, vehicle: str, year: int = FUEL_YEAR) -> dict[str, float]:
    """Average fuel resource cost (£), tax on that fuel (£) and CO2e (kg) per vehicle-km over a speed distribution."""
    totals = {"fuel": 0.0, "tax": 0.0, "co2": 0.0}
    for fuel in load_fuels(year)[vehicle]:
        litres = fuel.km_share * dist.expectation(fuel.litres_per_km)
        totals["fuel"] += litres * fuel.resource
        totals["tax"] += litres * fuel.tax
        totals["co2"] += litres * fuel.co2_kg
    return totals


def casualty_benefit(killed, ksi, casualties, values: Values, collisions_per_casualty: dict[str, float],
                     include_unreported: bool = False):
    """£ of casualties prevented. Accepts numbers or NumPy arrays (one per Monte Carlo draw)."""
    serious, slight = ksi - killed, casualties - ksi
    unit = {s: values.per_casualty[s] + collisions_per_casualty[s] * values.per_collision[s] for s in SEVERITIES}
    nonfatal = serious * unit["serious"] + slight * unit["slight"]
    if include_unreported:
        nonfatal = nonfatal * (1 + values.unreported_ratio)
    return killed * unit["fatal"] + nonfatal


def ksi_from_casualties(casualties_prevented, baseline_ksi: float, baseline_casualties: float):
    """KSI prevented where a model gives only all casualties (Model A, and every model on motorways).

    Assumes KSI fall at the same rate as all casualties. KSI fall faster than slight injuries
    when speeds drop, so this understates the benefit: the cautious direction.
    """
    return casualties_prevented * baseline_ksi / baseline_casualties


def hours_lost_per_km(policy: str, today: SpeedDistribution, road: RoadType, isa_working: float | np.ndarray,
                      count_speeding_time: bool = False, vehicle: str = "car",
                      limit_read: float | None = None) -> float:
    """Average extra hours per vehicle-km under a policy, compared with today's speeds.

    Lawful time only, by default: a car slowed from 110 to 80 km/h on a 97 km/h road loses
    1/80 - 1/97 hours per km; the 1/97 - 1/110 it gained by speeding is not counted. A van
    slowed from 90 to 80 km/h there loses none, because its limit is 80 km/h.
    """
    after = apply_policy(policy, today, road, isa_working, vehicle, limit_read)
    before = original_speeds(policy, today)
    if count_speeding_time:
        extra = 1 / after.speeds_kmh - 1 / before
    else:
        legal = road.legal_limit(vehicle)
        extra = np.maximum(0.0, 1 / after.speeds_kmh - 1 / np.minimum(before, legal))
        extra = np.where(after.speeds_kmh >= legal - SAME_SPEED_KMH, 0.0, extra)
    return float(np.sum(after.probabilities * extra))


def time_cost(car_hours_per_km, van_hours_per_km, car_billion_km: float, van_billion_km: float, values: Values,
              free_flow_share):
    """£ a year of the time lost by the free-flowing share of car and van traffic."""
    return (car_hours_per_km * car_billion_km * values.car_hour
            + van_hours_per_km * van_billion_km * values.van_hour) * 1e9 * free_flow_share


def capital_cost(car_hours_per_km, van_hours_per_km, car_billion_km: float, van_billion_km: float, values: Values,
                 free_flow_share):
    """£ a year of vehicle capital tied up by those hours: the speed-dependent part of TAG's non-fuel operating cost."""
    return (car_hours_per_km * car_billion_km * values.car_capital_hour
            + van_hours_per_km * van_billion_km * values.van_capital_hour) * 1e9 * free_flow_share


def break_even_free_flow(casualty_value, running_saving_all_free, cost_all_free):
    """The free-flowing share of traffic at which cost equals benefit (inf if it never does).

    Time cost and running savings grow with the share of traffic in free flow; the casualty
    benefit does not, because the risk curves already say which casualties respond. So
    casualty_value + share * running_saving_all_free = share * cost_all_free.
    """
    net_cost = np.asarray(cost_all_free, dtype=float) - np.asarray(running_saving_all_free, dtype=float)
    return np.divide(casualty_value, net_cost, out=np.full_like(net_cost, np.inf), where=net_cost > 0)


def cap_beyond_compliance(road: RoadType, cap_kmh: float) -> bool:
    """Whether a cap adds anything to posted-limit compliance on this road type.

    Compliance holds each vehicle to its own limit, at no lawful time cost. A cap added to it holds
    each vehicle to the lower of the two. If the cap is at or above every vehicle's limit, that is
    compliance itself, so the cap adds nothing (False). If it is at or below every vehicle's limit, it
    is the cap alone, so what it adds is the cap's result minus compliance's (True). Within
    SAME_SPEED_KMH counts as at the limit. A cap between a van's and a car's limit fits neither rule
    and is refused; none of the modelled caps falls there.
    """
    limits = (road.limit_kmh, road.van_limit_kmh)
    if cap_kmh >= max(limits) - SAME_SPEED_KMH:
        return False
    if cap_kmh <= min(limits) + SAME_SPEED_KMH:
        return True
    raise ValueError(f"a {cap_kmh} km/h cap lies between the van and car limits on {road.segment}")
