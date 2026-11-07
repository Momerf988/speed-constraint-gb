"""Monte Carlo: how far the casualty results move across the values the sources allow.

Each draw picks every uncertain input at random, runs the whole model once and keeps the
results. The 5th to 95th percentile across draws is the range reported. Draws are uniform
between two bounds, because the sources give bounds rather than distributions.

What is drawn, and where each bound comes from (all in the parameters file or its notes):
    Power exponent, fatal          4.3 to 6.7      Elvik et al. (2019): fixed effects after trim-and-fill;
                                                   highest Table 5 fit
    Power exponent, injury         2.1 to 4.2      Elvik (2013); highest Elvik et al. (2019) Table 5 fit
    Exponential coef., fatal       0.069 to 0.086  Elvik (2013) fatal accidents; highest Table 5 fit
    Exponential coef., injury      0.034 to 0.067  Elvik (2013) injury crashes; highest Table 5 fit
    Free-travel share f, urban     0.40 to 0.70    Adelaide's 0.56, widened (a judgement): no GB figure exists
    Free-travel share f, rural     0.60 to 0.90    CR 204's 0.80, widened (a judgement): GB rural roads are busier
    ISA switched off               0.10 to 0.50    a judgement, informed by Day et al. (2023): 3 of 20 never
                                                   use it, 11 sometimes
    Fleet share with speed control 0 to 1          TRL (2024) assumes a half; the makers' choice is not published
    Warning effect, kappa          0.20 to 0.36    TRL (2024) Table 34: advisory / voluntary, by severity
    ISA overridden, by road type   +/- 5 points    around the ISA-UK trial shares (Lai and Carsten 2012)
    Urban curve                    one of four     Kloeden (2002) refits for case-speed errors of 0-7.5 km/h

Not drawn:
    Elvik's (2009) severity exponents stay at their best estimates. The urban fatality interval
    runs from -0.5 to 6.5, so drawing from it would let a slower speed raise deaths.
    Which mean speed differences are measured from, the shape of the speed distribution and
    Brenac's flatter curve are choices between models, not values within one. They are tested
    one at a time (analysis/rq2_rq4.py), so it stays visible which choice moves the answer.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from model.casualties import Assumptions
from model.run import run
from model.scenarios import IsaUse
from model.speeds import SpeedDistribution

N_DRAWS = 10_000
SEED = 20261030

ASSUMPTION_RANGES = {              # names match the fields of Assumptions
    "power_exponent_fatal": (4.3, 6.7),
    "power_exponent_injury": (2.1, 4.2),
    "exponential_coef_fatal": (0.069, 0.086),
    "exponential_coef_injury": (0.034, 0.067),
    "free_travel_share": (0.40, 0.70),
    "free_travel_share_rural": (0.60, 0.90),
}
SWITCHED_OFF_RANGE = (0.10, 0.50)
SPEED_CONTROL_RANGE = (0.0, 1.0)
WARNING_EFFECT_RANGE = (0.20, 0.36)
OVERRIDE_RANGES = {
    "32 km/h roads": (0.08, 0.18),
    "48 km/h roads": (0.02, 0.08),
    "97 km/h single carriageways": (0.02, 0.08),
    "113 km/h motorways": (0.11, 0.21),
}
URBAN_CURVES = (
    "kloeden2002_urban_difference_se0",
    "kloeden2002_urban_difference_se2_5",
    "kloeden2002_urban_difference",          # case-speed error 5 km/h: the authors' central fit
    "kloeden2002_urban_difference_se7_5",
)


def draw(n: int = N_DRAWS, seed: int = SEED) -> pd.DataFrame:
    """n sets of uncertain inputs, one row each. The same seed always gives the same draws."""
    rng = np.random.default_rng(seed)
    draws = pd.DataFrame({name: rng.uniform(low, high, n) for name, (low, high) in ASSUMPTION_RANGES.items()})
    draws["switched_off"] = rng.uniform(*SWITCHED_OFF_RANGE, n)
    for segment, (low, high) in OVERRIDE_RANGES.items():
        draws[f"override: {segment}"] = rng.uniform(low, high, n)
    draws["urban_curve"] = rng.choice(URBAN_CURVES, n)
    draws["speed_control_share"] = rng.uniform(*SPEED_CONTROL_RANGE, n)
    draws["warning_effect"] = rng.uniform(*WARNING_EFFECT_RANGE, n)
    draws.index.name = "draw"
    return draws


def monte_carlo(distributions: dict[str, SpeedDistribution], baseline: pd.DataFrame,
                n: int = N_DRAWS, seed: int = SEED) -> pd.DataFrame:
    """Every road type and policy, once per draw, stacked into one table with a draw column."""
    central = Assumptions.central()
    runs = []
    for i, d in draw(n, seed).iterrows():
        assumptions = replace(central, urban_curve=d["urban_curve"],
                              **{name: d[name] for name in ASSUMPTION_RANGES})
        isa = IsaUse(switched_off=d["switched_off"], speed_control_share=d["speed_control_share"],
                     warning_effect=d["warning_effect"],
                     overrides={segment: d[f"override: {segment}"] for segment in OVERRIDE_RANGES})
        result = run(distributions, baseline, assumptions, isa)
        result.insert(0, "draw", i)
        runs.append(result)
    return pd.concat(runs, ignore_index=True)


def percentiles(runs: pd.DataFrame, by: list[str], columns: list[str]) -> pd.DataFrame:
    """The 5th, 50th and 95th percentile of each column, per group."""
    q = runs.groupby(by, sort=False)[columns].quantile([0.05, 0.5, 0.95]).unstack()
    q.columns = [f"{column}_p{round(p * 100)}" for column, p in q.columns]
    return q.reset_index()
