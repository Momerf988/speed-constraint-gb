"""The four modelled road types and the policies applied to their speed distributions.

Road types: the four that DfT publishes free-flow speed distributions for (SPE0102). Each
one has its posted limit, its STATS19 segment, the individual risk curve that applies to it
and the environment that picks its Elvik (2009) severity exponents.

Policies, each a change to the distribution of car speeds:
    today        speeds as DfT observed them
    bau_isa      business as usual: the ISA that GB has decided to mandate, fitted to every car.
                 Above the limit, a share of driving is held to the limit (held_to_limit); the
                 rest is unchanged, because the driver has switched ISA off or overrides it, or
                 ignores a warning.
    compliance   every car held to the posted limit, with no override (mandatory ISA)
    cap_<N>      a flat vehicle limiter at N km/h on every road, on top of business-as-usual ISA

How GB's ISA is used (IsaUse):
    Delegated Regulation (EU) 2021/1958 lets a maker fit a warning or a speed control function.
    Both can be switched off for a journey, and speed control can be overridden. So:
        held(v) = effect * (1 - s) * (1 - o_v)
        effect  = w + (1 - w) * kappa
    s      share of driving with ISA switched off
    o_v    share of driving at speed v that is overridden. Drivers override in order to speed, so
           all override time falls above the limit: o_v = min(1, o / share of driving above the
           limit), where o is the share of driving time overridden in the ISA-UK trials
    w      share of the fleet with speed control; the rest has a warning only
    kappa  what a warning achieves, relative to speed control used the same way
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.params import require
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
ISA_SWITCHED_OFF_SHARE = 0.30   # judgement, informed by Day et al. (2023): of 20 drivers, 3 never use it, 11 sometimes


@dataclass(frozen=True)
class IsaUse:
    """How the mandated ISA is used. The Monte Carlo varies these."""
    switched_off: float             # s, a judgement
    speed_control_share: float      # w, from TRL (2024)
    warning_effect: float           # kappa, from TRL (2024)
    overrides: dict[str, float] | None = None   # o by segment; None = the ISA-UK trial shares in ROAD_TYPES
    override_targets_speeding: bool = True      # False = override falls at random, on every speed alike

    @classmethod
    def central(cls) -> "IsaUse":
        p = require("trl2024_speed_control_share", "trl2024_advisory_over_voluntary")
        return cls(switched_off=ISA_SWITCHED_OFF_SHARE,
                   speed_control_share=p["trl2024_speed_control_share"].value,
                   warning_effect=p["trl2024_advisory_over_voluntary"].value)

    def override_on(self, road: RoadType) -> float:
        return road.override_share if self.overrides is None else self.overrides[road.segment]

    @property
    def effect(self) -> float:
        """What the fleet's mix of systems achieves, relative to speed control in every car."""
        return self.speed_control_share + (1 - self.speed_control_share) * self.warning_effect


def mixture(parts: list[tuple[float | np.ndarray, SpeedDistribution]], label: str) -> SpeedDistribution:
    """A population made of several groups of drivers, each with its own speed distribution.

    A weight is one number, or one number per grid speed for groups whose share depends on speed.
    """
    speeds = np.concatenate([dist.speeds_kmh for _, dist in parts])
    probabilities = np.concatenate([weight * dist.probabilities for weight, dist in parts])
    return SpeedDistribution(speeds, probabilities / probabilities.sum(), label)


def isa_working_share(road: RoadType, switched_off: float = ISA_SWITCHED_OFF_SHARE,
                      override: float | None = None) -> float:
    """The share of driving with ISA working, if override falls at random on every speed."""
    return (1 - switched_off) * (1 - (road.override_share if override is None else override))


def held_to_limit(today: SpeedDistribution, road: RoadType, isa: IsaUse) -> np.ndarray | float:
    """The share of driving that ISA holds to the limit: one share per grid speed, or one number if override is random."""
    override = isa.override_on(road)
    if not isa.override_targets_speeding:
        return isa.effect * isa_working_share(road, isa.switched_off, override)
    above = today.speeds_kmh > road.limit_kmh
    share_above = today.probabilities[above].sum()
    overridden = np.where(above, min(1.0, override / share_above) if share_above > 0 else 0.0, 0.0)
    return isa.effect * (1 - isa.switched_off) * (1 - overridden)


def original_speeds(policy: str, today: SpeedDistribution) -> np.ndarray:
    """Each car's speed before the policy, in the same order as apply_policy's result.

    Lets the model tell which cars a policy actually changed: capping keeps the order of the
    grid, and a mixture lists its groups one after the other.
    """
    if policy in ("today", "compliance"):
        return today.speeds_kmh
    return np.concatenate([today.speeds_kmh, today.speeds_kmh])


def apply_policy(policy: str, today: SpeedDistribution, road: RoadType,
                 isa_working: float | np.ndarray) -> SpeedDistribution:
    """The speed distribution on one road type under one policy.

    isa_working is the share of driving ISA holds to the limit, as held_to_limit gives it.
    """
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
