"""The four modelled road types and the policies applied to their speed distributions.

Road types: the four that DfT publishes free-flow speed distributions for (SPE0102). Each
one has its posted limit, its STATS19 segment, the individual risk curve that applies to it
and the environment that picks its Elvik (2009) severity exponents.

Traffic on each road type is cars and vans, each with its own speeds and its own legal limit.
Most vans are goods vehicles up to 7.5 t: 80 km/h on single carriageways, against 97 km/h for
cars and car-derived vans (GOV.UK, "Speed limits"); on the other three road types modelled here the
limits are the same (they differ on dual carriageways, which DfT publishes no speeds for).

Policies, each a change to the distribution of speeds:
    today        speeds as DfT observed them
    bau_isa      business as usual: the ISA that GB has decided to mandate, fitted to every car
                 and van. Above the limit it reads, a share of driving is held to that limit
                 (held_to_limit); the rest is unchanged, because the driver has switched ISA off
                 or overrides it, or ignores a warning. ISA reads the posted limit, so it holds a
                 van to 97 km/h on a single carriageway. An assumption: a system that knows the
                 vehicle's category could show the van's own limit, which is tested as a sensitivity.
    compliance   every vehicle held to its own legal limit, with no override (mandatory ISA)
    cap_<N>      a flat limiter at N km/h on every car and van, on top of business-as-usual ISA

How GB's ISA is used (IsaUse):
    Delegated Regulation (EU) 2021/1958 lets a maker fit a warning or a speed control function.
    Both can be switched off for a journey, and speed control can be overridden. So:
        held(v) = effect * (1 - s) * (1 - o_v)
        effect  = w + (1 - w) * kappa
    s      share of driving with ISA switched off
    o_v    share of driving at speed v that is overridden. Drivers override in order to speed, so
           all override falls above the limit: o_v = min(1, o / share of driving above the limit),
           where o is the share of distance driven with ISA overridden in the ISA-UK trials, by
           speed limit (Lai and Carsten 2012, Table 5 and Fig. 8). Centrally the share above the
           limit is today's, from the DfT surveys; a structural variant takes the trial's own share
           instead (TRIAL_SHARE_OVER_LIMIT), which is lower, because the survey sites are chosen for
           free flow and the trials included congested driving
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
    override_share: float   # share of distance driven with ISA overridden (ISA-UK trials, Lai and Carsten 2012, Table 5)
    van_limit_kmh: float    # legal limit of goods vans up to 7.5 t

    def legal_limit(self, vehicle: str) -> float:
        return self.van_limit_kmh if vehicle == "van" else self.limit_kmh


ROAD_TYPES = (
    RoadType("32 km/h roads", "20 mph speed limit roads", 20 * MPH_TO_KMH,
             "kloeden2002_urban_difference", (0.0, 20.0), "urban", 0.133, 20 * MPH_TO_KMH),
    RoadType("48 km/h roads", "30 mph speed limit roads", 30 * MPH_TO_KMH,
             "kloeden2002_urban_difference", (0.0, 20.0), "urban", 0.081, 30 * MPH_TO_KMH),
    RoadType("97 km/h single carriageways", "National Speed Limit Single Carriageways", 60 * MPH_TO_KMH,
             "kloeden2001_rural_difference", (-10.0, 30.0), "rural", 0.013, 50 * MPH_TO_KMH),
    RoadType("113 km/h motorways", "Motorways", 70 * MPH_TO_KMH,
             None, None, "rural", 0.163, 70 * MPH_TO_KMH),
)

CAPS_KMH = (48, 60, 80, 97)
POLICIES = ("today", "bau_isa", "compliance") + tuple(f"cap_{cap}" for cap in CAPS_KMH)
ISA_SWITCHED_OFF_SHARE = 0.30   # judgement, informed by Day et al. (2023): of 20 drivers, 3 never use it, 11 sometimes

# Share of distance the ISA-UK drivers drove over the limit with ISA off (Lai and Carsten 2012, Fig. 6, Phase 1),
# by speed limit. Read from the bar chart, to about ±1 point: the chart's axis was calibrated on Phase 2, whose
# values Table 5 prints exactly (its "speeding observed in this trial" column is the ISA-active phase).
# Only the structural variant "override at the trial's own rate" uses these.
TRIAL_SHARE_OVER_LIMIT = {
    "32 km/h roads": 0.43,
    "48 km/h roads": 0.40,
    "97 km/h single carriageways": 0.07,
    "113 km/h motorways": 0.31,
}


@dataclass(frozen=True)
class IsaUse:
    """How the mandated ISA is used. The Monte Carlo varies these."""
    switched_off: float             # s, a judgement
    speed_control_share: float      # w, from TRL (2024)
    warning_effect: float           # kappa, from TRL (2024)
    overrides: dict[str, float] | None = None   # o by segment; None = the ISA-UK trial shares in ROAD_TYPES
    override_targets_speeding: bool = True      # False = override falls at random, on every speed alike
    vans_read_own_limit: bool = False           # True = ISA on a van shows the van's limit, not the posted one
    override_base: dict[str, float] | None = None   # what o is divided by, by segment; None = today's share above the limit

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

    def limit_read(self, road: RoadType, vehicle: str) -> float:
        """The limit ISA holds this vehicle to."""
        return road.legal_limit(vehicle) if self.vans_read_own_limit else road.limit_kmh


@dataclass(frozen=True)
class Traffic:
    """The vehicles a limiter reaches on one road type, by share of car and van vehicle-km."""
    cars: SpeedDistribution
    vans: SpeedDistribution
    van_share: float

    def mixed(self) -> SpeedDistribution:
        return mixture([(1 - self.van_share, self.cars), (self.van_share, self.vans)], "cars and vans")


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


def held_to_limit(today: SpeedDistribution, road: RoadType, isa: IsaUse, vehicle: str = "car") -> np.ndarray | float:
    """The share of driving that ISA holds to the limit: one share per grid speed, or one number if override is random."""
    override = isa.override_on(road)
    if not isa.override_targets_speeding:
        return isa.effect * isa_working_share(road, isa.switched_off, override)
    above = today.speeds_kmh > isa.limit_read(road, vehicle)
    share_above = today.probabilities[above].sum() if isa.override_base is None else isa.override_base[road.segment]
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


def apply_policy(policy: str, today: SpeedDistribution, road: RoadType, isa_working: float | np.ndarray,
                 vehicle: str = "car", limit_read: float | None = None) -> SpeedDistribution:
    """The speed distribution of one vehicle type on one road type under one policy.

    isa_working is the share of driving ISA holds to the limit, as held_to_limit gives it, and
    limit_read the limit ISA holds it to (the posted limit unless given).
    """
    limit_read = road.limit_kmh if limit_read is None else limit_read
    if policy == "today":
        return today
    if policy == "compliance":
        return today.capped(road.legal_limit(vehicle), "compliance")
    if policy == "bau_isa":
        return mixture([(isa_working, today.capped(limit_read)), (1 - isa_working, today)], "bau_isa")
    if policy.startswith("cap_"):
        cap = float(policy.removeprefix("cap_"))
        return mixture([(isa_working, today.capped(min(cap, limit_read))),
                        (1 - isa_working, today.capped(cap))], policy)
    raise ValueError(f"unknown policy {policy!r}")


def fleet_after(policy: str, traffic: Traffic, road: RoadType, isa: IsaUse) -> tuple[SpeedDistribution, ...]:
    """Cars, vans and both together after a policy, plus today's mix for the speeds it started from."""
    cars = apply_policy(policy, traffic.cars, road, held_to_limit(traffic.cars, road, isa, "car"),
                        "car", isa.limit_read(road, "car"))
    vans = apply_policy(policy, traffic.vans, road, held_to_limit(traffic.vans, road, isa, "van"),
                        "van", isa.limit_read(road, "van"))
    return cars, vans, mixture([(1 - traffic.van_share, cars), (traffic.van_share, vans)], policy)


def fleet_before(policy: str, traffic: Traffic) -> np.ndarray:
    """Each vehicle's speed before the policy, in the same order as fleet_after's mixture."""
    return np.concatenate([original_speeds(policy, traffic.cars), original_speeds(policy, traffic.vans)])
