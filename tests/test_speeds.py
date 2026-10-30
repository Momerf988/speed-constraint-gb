"""The speed-distribution code must recover answers we already know.

A fitting method is tested on data made from a known distribution: if it cannot find the
parameters it was built from, it cannot be trusted on DfT's data either.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from model.speeds import Interval, band_intervals, empirical_distribution, fit

EDGES = np.arange(40.0, 130.0, 8.0)  # bands 8 km/h wide, like 5 mph


def bands_from(dist) -> list[Interval]:
    """Intervals from a known distribution, with open-ended bottom and top bands."""
    lower = np.concatenate(([0.0], EDGES))
    upper = np.concatenate((EDGES, [np.inf]))
    return [Interval(lo, hi, float(dist.cdf(hi) - dist.cdf(lo))) for lo, hi in zip(lower, upper)]


def test_normal_fit_recovers_known_parameters():
    location, scale = fit(bands_from(stats.norm(loc=82, scale=11)), "normal")
    assert location == pytest.approx(82, abs=0.05)
    assert scale == pytest.approx(11, abs=0.05)


def test_lognormal_fit_recovers_known_parameters():
    location, scale = fit(bands_from(stats.lognorm(s=0.14, scale=np.exp(4.4))), "lognormal")
    assert location == pytest.approx(4.4, abs=0.005)
    assert scale == pytest.approx(0.14, abs=0.005)


def test_empirical_reproduces_every_band_exactly():
    intervals = bands_from(stats.norm(loc=82, scale=11))
    dist = empirical_distribution(intervals, "test")
    for interval in intervals:
        inside = (dist.speeds_kmh >= interval.lower_kmh) & (dist.speeds_kmh < interval.upper_kmh)
        assert dist.probabilities[inside].sum() == pytest.approx(interval.share, abs=1e-9)


def test_cap_moves_only_the_fast_vehicles():
    dist = empirical_distribution(bands_from(stats.norm(loc=82, scale=11)), "test")
    capped = dist.capped(80.0)
    assert capped.share_at_or_above(80.0 + 1e-9) == 0          # nobody faster than the cap
    slow = dist.speeds_kmh < 80.0
    assert np.array_equal(capped.speeds_kmh[slow], dist.speeds_kmh[slow])   # slower drivers untouched
    assert capped.probabilities.sum() == pytest.approx(1.0)
    assert capped.mean() < dist.mean()


def test_low_bands_get_the_mass_dft_published():
    # Printed bands plus "[low]" bands at both ends. DfT says 12% exceed the 50-unit limit,
    # but only 10% is printed above it, so the top "[low]" run must hold 2% and the bottom the rest.
    bands = pd.DataFrame({
        "speed_limit_mph": 50, "pct_exceeding": 12.0,
        "band_lower_mph": [0, 10, 20, 30, 40, 50, 60],
        "band_lower_kmh": [0, 10, 20, 30, 40, 50, 60],
        "band_upper_kmh": [10, 20, 30, 40, 50, 60, np.nan],
        "share_pct": [np.nan, 30.0, 30.0, 25.0, 2.0, 10.0, np.nan],
        "is_low": [True, False, False, False, False, False, True],
    })
    intervals, low = band_intervals(bands)
    assert low["top_low_pct"] == pytest.approx(2.0)
    assert low["bottom_low_pct"] == pytest.approx(1.0)
    assert sum(i.share for i in intervals) == pytest.approx(1.0)