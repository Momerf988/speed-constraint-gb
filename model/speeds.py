"""Turn DfT's 5 mph speed bands (SPE0102) into continuous speed distributions, in km/h.

The risk curves need a speed for every driver, and a cap moves only part of the
distribution, so the model needs continuous speeds rather than bands. Three versions:
    empirical   spreads each band's share evenly across the band, so it reproduces DfT's
                bands exactly; the central input
    normal      a fitted symmetric shape (sensitivity test)
    lognormal   a fitted skewed shape with a longer fast tail (sensitivity test)
The fitted shapes use maximum likelihood for grouped data: the parameters that make the
observed band shares most probable. On 2024 data they miss DfT's fast-tail shares by up to
a factor of two, which is why the empirical version is central.

"[low]" bands (too small to print, but not zero) are not dropped. Each run of them is merged
into one interval, and its share is inferred: the fast-end run gets the gap between DfT's
published "% exceeding" and the printed bands above the limit; the slow-end run gets the rest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats

FAMILIES = ("normal", "lognormal")
MPH_TO_KMH = 1.609344


@dataclass(frozen=True)
class Interval:
    lower_kmh: float
    upper_kmh: float   # np.inf for the open-ended top band
    share: float       # proportion of vehicles, 0 to 1


def band_intervals(bands: pd.DataFrame) -> tuple[list[Interval], dict[str, float]]:
    """One road type and year of bands -> intervals whose shares sum to 1, plus the inferred low masses (%)."""
    bands = bands.sort_values("band_lower_mph")
    limit_mph = bands["speed_limit_mph"].iloc[0]
    known = bands[~bands["is_low"]]
    missing = 100 - known["share_pct"].sum()
    at_or_above = known.loc[known["band_lower_mph"] >= limit_mph, "share_pct"].sum()
    top_mass = float(np.clip(bands["pct_exceeding"].iloc[0] - at_or_above, 0, max(missing, 0)))
    bottom_mass = max(missing - top_mass, 0.0)

    intervals = [
        Interval(r.band_lower_kmh, np.inf if pd.isna(r.band_upper_kmh) else r.band_upper_kmh, r.share_pct)
        for r in known.itertuples()
    ]
    low = bands[bands["is_low"]]
    for run, mass in ((low[low["band_lower_mph"] < limit_mph], bottom_mass),
                      (low[low["band_lower_mph"] >= limit_mph], top_mass)):
        if len(run) and mass > 0:
            upper = run["band_upper_kmh"].iloc[-1]
            intervals.append(Interval(run["band_lower_kmh"].iloc[0], np.inf if pd.isna(upper) else upper, mass))

    total = sum(i.share for i in intervals)
    intervals = sorted((Interval(i.lower_kmh, i.upper_kmh, i.share / total) for i in intervals), key=lambda i: i.lower_kmh)
    return intervals, {"top_low_pct": top_mass, "bottom_low_pct": bottom_mass}


def distribution(family: str, location: float, scale: float):
    """A frozen SciPy distribution in km/h. For lognormal, location and scale are on the log scale."""
    if family == "normal":
        return stats.norm(loc=location, scale=scale)
    if family == "lognormal":
        return stats.lognorm(s=scale, scale=np.exp(location))
    raise ValueError(f"unknown family {family!r}")


def fit(intervals: list[Interval], family: str) -> tuple[float, float]:
    """Maximum-likelihood (location, scale) for grouped data."""
    lower = np.array([i.lower_kmh for i in intervals])
    upper = np.array([i.upper_kmh for i in intervals])
    weight = np.array([i.share for i in intervals])
    finite_upper = np.where(np.isinf(upper), lower + (lower - np.roll(lower, 1)).max(), upper)
    mid = (lower + finite_upper) / 2
    mean, sd = np.average(mid, weights=weight), np.sqrt(np.average((mid - np.average(mid, weights=weight)) ** 2, weights=weight))
    start = (mean, np.log(sd)) if family == "normal" else (np.log(mean), np.log(sd / mean))

    def negative_log_likelihood(params):
        dist = distribution(family, params[0], np.exp(params[1]))
        probability = dist.cdf(upper) - dist.cdf(lower)
        return -np.sum(weight * np.log(np.maximum(probability, 1e-300)))

    result = optimize.minimize(negative_log_likelihood, start, method="Nelder-Mead",
                               options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 5000})
    if not result.success:
        raise RuntimeError(f"{family} fit failed: {result.message}")
    return float(result.x[0]), float(np.exp(result.x[1]))


GRID_STEP_KMH = 0.1
GRID_MAX_KMH = 250.0


@dataclass(frozen=True)
class SpeedDistribution:
    """Speeds on a fine grid (km/h) with the share of vehicles at each one.

    One representation for every shape, so a policy and a risk curve treat them all alike:
    a cap moves every speed above it down to the cap; a risk curve is averaged over the grid.
    """

    speeds_kmh: np.ndarray
    probabilities: np.ndarray
    label: str

    def mean(self) -> float:
        return float(np.sum(self.speeds_kmh * self.probabilities))

    def share_at_or_above(self, kmh: float) -> float:
        return float(self.probabilities[self.speeds_kmh >= kmh].sum())

    def expectation(self, function) -> float:
        """The average of function(speed) over all vehicles."""
        return float(np.sum(function(self.speeds_kmh) * self.probabilities))

    def capped(self, cap_kmh: float, label: str | None = None) -> "SpeedDistribution":
        """Every vehicle faster than the cap now drives at the cap; nobody else changes."""
        return SpeedDistribution(np.minimum(self.speeds_kmh, cap_kmh), self.probabilities,
                                 label or f"{self.label}, capped at {cap_kmh:g} km/h")


def _grid() -> np.ndarray:
    return np.arange(GRID_STEP_KMH / 2, GRID_MAX_KMH, GRID_STEP_KMH)


def empirical_distribution(intervals: list[Interval], label: str) -> SpeedDistribution:
    """Reproduces DfT's bands exactly: each band's share is spread evenly across the band.

    The open-ended top interval has no upper edge, so its share decays exponentially, at the
    rate the last two printed bands fall (e.g. on 97 km/h roads each 5 mph band holds about a
    third of the one before). Nothing above 250 km/h.
    """
    speeds = _grid()
    probabilities = np.zeros_like(speeds)
    finite = [i for i in intervals if np.isfinite(i.upper_kmh)]
    last_two = finite[-2:]
    width = last_two[1].upper_kmh - last_two[1].lower_kmh
    ratio = last_two[1].share / last_two[0].share
    decay_per_kmh = -np.log(ratio) / width if 0 < ratio < 1 else 1 / width
    for interval in intervals:
        if np.isfinite(interval.upper_kmh):
            inside = (speeds >= interval.lower_kmh) & (speeds < interval.upper_kmh)
            weights = inside.astype(float)
        else:
            inside = speeds >= interval.lower_kmh
            weights = np.where(inside, np.exp(-decay_per_kmh * (speeds - interval.lower_kmh)), 0.0)
        probabilities += interval.share * weights / weights.sum()
    return SpeedDistribution(speeds, probabilities / probabilities.sum(), label)


def fitted_distribution(family: str, location: float, scale: float, label: str) -> SpeedDistribution:
    """A fitted normal or lognormal shape, put on the same grid."""
    speeds = _grid()
    dist = distribution(family, location, scale)
    edges = np.append(speeds - GRID_STEP_KMH / 2, GRID_MAX_KMH)
    probabilities = np.diff(dist.cdf(edges))
    return SpeedDistribution(speeds, probabilities / probabilities.sum(), label)