"""Difference-in-differences: did a policy change casualties beyond what happened elsewhere?

The design compares the change in the treated group (Wales) with the change in a control group
(England) over the same months, on the same kind of road. A Poisson regression of monthly
casualty counts per police force has one fixed effect per force (each force's own level) and
one per month (everything that hit both countries that month: weather, traffic, holidays). What
is left for the treated-and-after term is the change beyond the control's: exp(beta) is the
rate ratio, e.g. 0.85 means 15% fewer casualties than the control's trend implies.

The key assumption is parallel trends: without the policy, the treated group would have moved
with the control. It cannot be proved, but an event study (the treated gap, period by period)
should show no gap before the policy. With few treated forces, clustered standard errors are
unreliable, so a permutation test asks how often the same number of control forces, chosen at
random, show a change as large.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

Z95 = 1.959964


def did_poisson(panel: pd.DataFrame, outcome: str, treated: str, post: str,
                unit: str = "police_force_code", time: str = "month") -> dict[str, float]:
    """Two-way fixed-effects Poisson difference-in-differences, standard errors clustered by unit."""
    data = panel.assign(treated_post=(panel[treated] & panel[post]).astype(float))
    fit = smf.glm(f"{outcome} ~ treated_post + C({unit}) + C({time})", data=data,
                  family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": data[unit]})
    coef, se = fit.params["treated_post"], fit.bse["treated_post"]
    return {"rate_ratio": float(np.exp(coef)), "ci_low": float(np.exp(coef - Z95 * se)),
            "ci_high": float(np.exp(coef + Z95 * se)), "p_value": float(fit.pvalues["treated_post"])}


def event_study(panel: pd.DataFrame, outcome: str, treated: str, period: str, reference,
                unit: str = "police_force_code", time: str = "month") -> pd.DataFrame:
    """The treated group's rate ratio in each period, relative to a reference period (ratio 1 by definition)."""
    periods = sorted(panel[period].unique())
    data = panel.copy()
    terms = {}
    for i, p in enumerate(periods):
        if p != reference:
            terms[p] = f"treated_period_{i}"
            data[terms[p]] = (data[treated] & (data[period] == p)).astype(float)
    fit = smf.glm(f"{outcome} ~ {' + '.join(terms.values())} + C({unit}) + C({time})", data=data,
                  family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": data[unit]})
    rows = []
    for p in periods:
        if p == reference:
            rows.append({period: p, "rate_ratio": 1.0, "ci_low": 1.0, "ci_high": 1.0})
            continue
        coef, se = fit.params[terms[p]], fit.bse[terms[p]]
        rows.append({period: p, "rate_ratio": np.exp(coef), "ci_low": np.exp(coef - Z95 * se),
                     "ci_high": np.exp(coef + Z95 * se)})
    return pd.DataFrame(rows)


def ratio_of_ratios(totals: pd.DataFrame, treated_units) -> float:
    """(treated after / treated before) / (control after / control before), from per-unit totals.

    totals has one row per unit, indexed by unit, with columns "before" and "after".
    """
    treated = totals.index.isin(treated_units)
    t, c = totals[treated].sum(), totals[~treated].sum()
    return float((t["after"] / t["before"]) / (c["after"] / c["before"]))


def permutation_test(totals: pd.DataFrame, treated_units, control_units, draws: int, seed: int) -> dict:
    """How unusual is the treated group's ratio of ratios among random sets of control units?

    Each draw pretends that len(treated_units) control units were treated, and compares them with
    the remaining control units. The p-value is the share of draws at least as far from 1
    (on the log scale) as the real one.
    """
    actual = ratio_of_ratios(totals.loc[list(treated_units) + list(control_units)], treated_units)
    controls = totals.loc[list(control_units)]
    rng = np.random.default_rng(seed)
    placebo = np.array([
        ratio_of_ratios(controls, rng.choice(controls.index, size=len(treated_units), replace=False))
        for _ in range(draws)
    ])
    p_value = float(np.mean(np.abs(np.log(placebo)) >= abs(np.log(actual))))
    return {"ratio_of_ratios": actual, "p_value": p_value, "placebo": placebo}


def implied_speed_change_kmh(rate_ratio: float, coefficient_per_kmh: float) -> float:
    """The change in mean speed that the Exponential model, exp(coef x change), needs to give this rate ratio."""
    return float(np.log(rate_ratio) / coefficient_per_kmh)
