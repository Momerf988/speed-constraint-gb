"""Turn a change in car speeds into a change in casualties, with two kinds of model.

Model B, individual (the main model): each car's relative risk of a casualty crash comes from
a Kloeden speed-difference curve, where the difference is the car's speed minus the mean
speed of traffic. Averaging over the speed distribution before and after a policy gives the
change in crash involvement, rho. Only some casualties can respond:
    k   the share in collisions involving a car, taxi or van, by severity and road type (STATS19)
    f   the share of casualty crashes with at least one free-travelling vehicle of any type:
        0.56 in Adelaide's 60 km/h zones (CR 207, Section 3.7), an assumption of its authors, and
        134 of 167 rural crashes (CR 204, Section 3.9). k and f restrict different things, so
        neither counts the other: f covers trucks and motorcycles too, and k then keeps only the
        collisions a car or van limiter can reach. Multiplying them treats the two as independent.
so KSI change by q_ksi = 1 - k_ksi * f * (1 - rho). Kloeden's crashes were ambulance cases, which
ISA-UK treated as KSI. The casualties that respond do so more steeply for deaths and less
steeply for slight injuries, by the ratio of Elvik's (2009) road-user exponents:
    q_fatal  = 1 - k_fatal  * f * (1 - rho ** (e_fatal  / e_serious))
    q_slight = 1 - k_slight * f * (1 - rho ** (e_slight / e_serious))

What the difference is measured from after a policy is an assumption, and there are three:
    new        the mean after the policy (central). ISA-UK did this on rural roads, and called it
               conservative because it gives smaller reductions against today. It means a uniform
               slowdown earns no credit: only a change in the shape of the distribution counts.
    old        today's mean, which Kloeden et al. (2002) kept fixed as an absolute reference point
    absolute   on urban roads, the absolute curve: speed minus 60 km/h, held at 45 and 80 km/h, as
               ISA-UK did on 48 and 64 km/h roads. Rural roads keep the new mean, as ISA-UK did.
None of these is conservative for every comparison: which gives the larger increment beyond
business-as-usual ISA depends on how much that baseline has already slowed traffic.

Each curve is used only on the speed differences it was fitted on; beyond them the risk is
held at the edge value. That never credits a benefit the evidence does not show, but it cuts
short the credit for cars slowed past an edge. When more than 10% of cars are cut short, the
Model B result is labelled a lower bound.

Model A, aggregate (the cross-check, and the only model on motorways): the Power and
Exponential models applied to the change in mean car speed, for deaths and all casualties.
They were fitted on mean-speed changes of -14 to +7 km/h; beyond that they extrapolate.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from model.params import require
from model.risk import exponential_model, power_model, quadratic_relative_risk
from model.scenarios import RoadType
from model.speeds import SpeedDistribution

MODEL_A_RANGE_KMH = (-14.0, 7.0)    # Elvik et al. (2019): mean-speed changes in the source studies
MODEL_A_MAX_INITIAL_KMH = 95.0      # Elvik et al. (2019): preferred values rest on initial speeds up to 95 km/h
MAX_SHARE_CLAMPED = 0.10            # above this, Model B is a lower bound, not a point estimate
ABSOLUTE_CURVE = "kloeden2002_urban_absolute"
ABSOLUTE_REFERENCE_KMH = 60.0       # the curve is written as speed minus 60 km/h
ABSOLUTE_DV_RANGE_KMH = (-15.0, 20.0)   # ISA-UK held it at 45 and 80 km/h, where its intervals grow very wide

SEVERITY_EXPONENTS = {               # Elvik (2009) Table S.1, road-user rows
    "urban": ("elvik2009_urban_fatalities", "elvik2009_urban_seriously_injured", "elvik2009_urban_slightly_injured"),
    "rural": ("elvik2009_rural_fatalities", "elvik2009_rural_seriously_injured", "elvik2009_rural_slightly_injured"),
}


@dataclass(frozen=True)
class CarVanShares:
    """k: the share of casualties, by severity, in collisions involving a car, taxi or van."""
    fatal: float
    ksi: float
    slight: float


@dataclass(frozen=True)
class Assumptions:
    """Everything a single model run depends on. The Monte Carlo varies these."""
    reference_mean: str              # "new", "old" or "absolute": see the module docstring
    free_travel_share: float         # f on urban roads
    free_travel_share_rural: float   # f on rural roads
    power_exponent_fatal: float
    power_exponent_injury: float
    exponential_coef_fatal: float
    exponential_coef_injury: float
    urban_curve: str
    rural_curve: str
    urban_fatal_ratio: float         # e_fatal / e_serious on urban roads
    rural_fatal_ratio: float         # e_fatal / e_serious on rural roads

    @classmethod
    def central(cls) -> "Assumptions":
        """The central assumptions, read from the parameters file."""
        p = require("power_exponent_fatalities", "power_exponent_injury_crashes", "exponential_coef_fatalities",
                    "exponential_coef_injury_crashes", "kloeden2002_free_speed_share_urban60",
                    "kloeden2001_free_speed_share_rural")
        return cls(
            reference_mean="new",
            free_travel_share=p["kloeden2002_free_speed_share_urban60"].value,
            free_travel_share_rural=p["kloeden2001_free_speed_share_rural"].value,
            power_exponent_fatal=p["power_exponent_fatalities"].value,
            power_exponent_injury=p["power_exponent_injury_crashes"].value,
            exponential_coef_fatal=p["exponential_coef_fatalities"].value,
            exponential_coef_injury=p["exponential_coef_injury_crashes"].value,
            urban_curve="kloeden2002_urban_difference",
            rural_curve="kloeden2001_rural_difference",
            urban_fatal_ratio=severity_powers("urban")[0],
            rural_fatal_ratio=severity_powers("rural")[0],
        )


@cache
def curve_coefficients(curve_id: str) -> tuple[float, float]:
    """(a, b) of an individual curve exp(a*dv + b*dv^2); Brenac's exponential has b = 0."""
    values = require(curve_id)[curve_id].values
    return (values[0], 0.0) if len(values) == 1 else (values[0], values[1])


@cache
def severity_powers(environment: str) -> tuple[float, float]:
    """(e_fatal / e_serious, e_slight / e_serious) from Elvik (2009), for urban or rural roads."""
    ids = SEVERITY_EXPONENTS[environment]
    parameters = require(*ids)
    fatal, serious, slight = (parameters[pid].value for pid in ids)
    return fatal / serious, slight / serious


def crash_involvement(dist: SpeedDistribution, reference_kmh: float, a: float, b: float,
                      dv_range: tuple[float, float]) -> float:
    """Average relative crash risk over all cars, holding risk at the curve's edges."""
    dv = np.clip(dist.speeds_kmh - reference_kmh, *dv_range)
    return float(np.sum(quadratic_relative_risk(dv, a, b) * dist.probabilities))


def share_clamped(after: SpeedDistribution, before_speeds: np.ndarray, reference_before: float,
                  reference_after: float, dv_range: tuple[float, float]) -> float:
    """Share of all cars the policy slows whose speed difference, before or after, is past an edge."""
    low, high = dv_range
    dv_before = before_speeds - reference_before
    dv_after = after.speeds_kmh - reference_after
    slowed = after.speeds_kmh < before_speeds - 1e-9
    outside = (dv_before < low) | (dv_before > high) | (dv_after < low) | (dv_after > high)
    return float(after.probabilities[slowed & outside].sum())


def model_b(today: SpeedDistribution, after: SpeedDistribution, before_speeds: np.ndarray, road: RoadType,
            k: CarVanShares, assumptions: Assumptions) -> dict[str, float] | None:
    """KSI, fatal and slight ratios (after / today) under the individual model, or None without a curve."""
    if road.curve_id is None:
        return None
    a, b = curve_coefficients(assumptions.urban_curve if road.environment == "urban" else assumptions.rural_curve)
    reference_before, dv_range = today.mean(), road.dv_range_kmh
    reference_after = reference_before if assumptions.reference_mean == "old" else after.mean()
    if assumptions.reference_mean == "absolute" and road.environment == "urban":
        a, b = curve_coefficients(ABSOLUTE_CURVE)
        reference_before = reference_after = ABSOLUTE_REFERENCE_KMH
        dv_range = ABSOLUTE_DV_RANGE_KMH
    rho = (crash_involvement(after, reference_after, a, b, dv_range)
           / crash_involvement(today, reference_before, a, b, dv_range))
    f = assumptions.free_travel_share if road.environment == "urban" else assumptions.free_travel_share_rural
    fatal_power = assumptions.urban_fatal_ratio if road.environment == "urban" else assumptions.rural_fatal_ratio
    _, slight_power = severity_powers(road.environment)
    return {
        "rho": rho,
        "ksi": 1 - k.ksi * f * (1 - rho),
        "fatal": 1 - k.fatal * f * (1 - rho ** fatal_power),
        "slight": 1 - k.slight * f * (1 - rho ** slight_power),
        "share_clamped": share_clamped(after, before_speeds, reference_before, reference_after, dv_range),
    }


def model_a(today: SpeedDistribution, after: SpeedDistribution, k_fatal: float, k_casualties: float,
            assumptions: Assumptions) -> dict[str, float]:
    """Fatal and all-casualty ratios (after / today) under the Power and Exponential models."""
    before, now = today.mean(), after.mean()
    change = now - before

    def covered(ratio: float, k: float) -> float:
        return 1 - k * (1 - float(ratio))

    return {
        "mean_change_kmh": change,
        "power_fatal": covered(power_model(now, before, assumptions.power_exponent_fatal), k_fatal),
        "power_casualties": covered(power_model(now, before, assumptions.power_exponent_injury), k_casualties),
        "exponential_fatal": covered(exponential_model(change, assumptions.exponential_coef_fatal), k_fatal),
        "exponential_casualties": covered(exponential_model(change, assumptions.exponential_coef_injury), k_casualties),
    }
