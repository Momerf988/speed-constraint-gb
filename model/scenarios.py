"""The four modelled road types and the policies applied to their speed distributions.

Road types: the four that DfT publishes free-flow speed distributions for (SPE0102). Each
one has its posted limit, its STATS19 segment, the individual risk curve that applies to it
and the environment that picks its Elvik (2009) severity exponents.

Policies, each a change to the distribution of car speeds:
    today        speeds as DfT observed them
    bau_isa      business as usual: the overridable ISA that GB has decided to mandate, fitted
                 to every car. A share u of driving has ISA working (speeds held to the limit);
                 the rest is unchanged, because the driver overrides it or has switched it off.
                 u = (1 - share of driving with ISA switched off) * (1 - share of time overriding)
    compliance   every car held to the posted limit, with no override (mandatory ISA)
    cap_<N>      a flat vehicle limiter at N km/h on every road, on top of business-as-usual ISA
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.speeds import MPH_TO_KMH, SpeedDistribution


@dataclass(frozen=True)
class RoadType:
    segment: str            # name in agg_casualties_segment_year
    dft_road_type: str      # name in agg_speed_bands_cars
    limit_kmh: float        # posted limit, exactly (e.g. 30 mph = 48.28 km/h)
    curve_id: str | None    # individual risk curve in the parameters file; None = no curve (motorways)
    dv_range_kmh: tuple[float, float] | None   # range of speed differences the curve was fitted on
    environment: str        # "urban" or "rural", for the Elvik (2009) severity exponents
    override_share: float   # share of driving time with overridable ISA overridden (ISA-UK trials)


ROAD_TYPES = (
    RoadType("32 km/h roads", "20 mph speed limit roads", 20 * MPH_TO_KMH,
             "kloeden2002_urban_difference", (0.0, 20.0), "urban", 0.13),
    RoadType("48 km/h roads", "30 mph speed limit roads", 30 * MPH_TO_KMH,
             "kloeden2002_urban_difference", (0.0, 20.0), "urban", 0.05),
    RoadType("97 km/h single carriageways", "National Speed Limit Single Carriageways", 60 * MPH_TO_KMH,
             "kloeden2001_rural_difference", (-10.0, 30.0), "rural", 0.05),
    RoadType("113 km/h motorways", "Motorways", 70 * MPH_TO_KMH,
             None, None, "rural", 0.16),
)

CAPS_KMH = (48, 60, 80, 97)
POLICIES = ("today", "bau_isa", "compliance") + tuple(f"cap_{cap}" for cap in CAPS_KMH)
ISA_SWITCHED_OFF_SHARE = 0.30   # central assumption; Day et al. (2023): 3 of 20 drivers never use it, 11 sometimes


def mixture(parts: list[tuple[float, SpeedDistribution]], label: str) -> SpeedDistribution:
    """A population made of several groups of drivers, each with its own speed distribution."""
    speeds = np.concatenate([dist.speeds_kmh for _, dist in parts])
    probabilities = np.concatenate([weight * dist.probabilities for weight, dist in parts])
    return SpeedDistribution(speeds, probabilities / probabilities.sum(), label)


def isa_working_share(road: RoadType, switched_off: float = ISA_SWITCHED_OFF_SHARE,
                      override: float | None = None) -> float:
    """The share of driving in which overridable ISA actually holds the car to the limit."""
    return (1 - switched_off) * (1 - (road.override_share if override is None else override))


def original_speeds(policy: str, today: SpeedDistribution) -> np.ndarray:
    """Each car's speed before the policy, in the same order as apply_policy's result.

    Lets the model tell which cars a policy actually changed: capping keeps the order of the
    grid, and a mixture lists its groups one after the other.
    """
    if policy in ("today", "compliance"):
        return today.speeds_kmh
    return np.concatenate([today.speeds_kmh, today.speeds_kmh])


def apply_policy(policy: str, today: SpeedDistribution, road: RoadType, isa_working: float) -> SpeedDistribution:
    """The speed distribution on one road type under one policy."""
    if policy == "today":
        return today
    if policy == "compliance":
        return today.capped(road.limit_kmh, "compliance")
    if policy == "bau_isa":
        return mixture([(isa_working, today.capped(road.limit_kmh)), (1 - isa_working, today)], "bau_isa")
    if policy.startswith("cap_"):
        cap = float(policy.removeprefix("cap_"))
        return mixture([(isa_working, today.capped(min(cap, road.limit_kmh))),
                        (1 - isa_working, today.capped(cap))], policy)
    raise ValueError(f"unknown policy {policy!r}")
