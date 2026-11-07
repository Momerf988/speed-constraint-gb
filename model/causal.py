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

Robustness tools for the same design: a version that lets the treated group keep its own earlier
trend; a triple difference against the treated group's other roads; a synthetic control with
placebo-in-space inference; and an exact permutation test restricted to control sets of similar size.
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


def did_poisson_trend(panel: pd.DataFrame, outcome: str, treated: str, post: str, trend: str,
                      unit: str = "police_force_code", time: str = "month") -> dict[str, float]:
    """Difference-in-differences allowing the treated group its own linear trend, fitted before the policy only.

    Every treated month after the policy gets its own term, so the trend term is identified by
    the months before it alone, then carried forward. The effect is the average of those monthly
    terms on the log scale: what is left after the treated group's earlier drift is projected on.
    trend is a numeric column of time, in years.
    """
    data = panel.assign(treated_trend=panel[treated].astype(float) * panel[trend])
    post_months = sorted(data.loc[data[treated] & data[post], time].unique())
    names = []
    for i, month in enumerate(post_months):
        names.append(f"treated_post_{i}")
        data[names[-1]] = (data[treated] & (data[time] == month)).astype(float)
    fit = smf.glm(f"{outcome} ~ treated_trend + {' + '.join(names)} + C({unit}) + C({time})", data=data,
                  family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": data[unit]})
    contrast = np.zeros(len(fit.params))
    for name in names:
        contrast[fit.params.index.get_loc(name)] = 1 / len(names)
    test = fit.t_test(contrast)
    coef, se = float(np.squeeze(test.effect)), float(np.squeeze(test.sd))
    return {"rate_ratio": float(np.exp(coef)), "ci_low": float(np.exp(coef - Z95 * se)),
            "ci_high": float(np.exp(coef + Z95 * se)), "p_value": float(np.squeeze(test.pvalue)),
            "trend_per_year": float(np.exp(fit.params["treated_trend"]))}


def ddd_poisson(panel: pd.DataFrame, outcome: str, treated: str, post: str, group: str, focus,
                unit: str = "police_force_code", time: str = "month") -> dict[str, float]:
    """Triple difference: the treated group's change on one road group, against its change on the others.

    Fixed effects for each unit and road group, each month and road group (anything that hit one
    kind of road nationally), and each month in the treated group (anything that hit all its
    roads at once). exp(beta) is then the change on the focus roads relative to the treated
    group's other roads, net of the same contrast in the control group.
    """
    in_focus = panel[group] == focus
    later = panel[time] != panel[time].min()
    # Month effects common to both road groups, plus extra ones for the focus roads and for the
    # treated group. Each extra set leaves its first month as the reference, so it cannot add up
    # to a road group's or a force's own fixed effect.
    data = panel.assign(
        triple=(panel[treated] & panel[post] & in_focus).astype(float),
        unit_group=panel[unit].astype(str) + "|" + panel[group].astype(str),
        focus_time=np.where(in_focus & later, panel[time].astype(str), "0 reference"),
        treated_time=np.where(panel[treated] & later, panel[time].astype(str), "0 reference"),
    )
    fit = smf.glm(f"{outcome} ~ triple + C(unit_group) + C({time}) + C(focus_time) + C(treated_time)", data=data,
                  family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": data[unit]})
    coef, se = fit.params["triple"], fit.bse["triple"]
    return {"rate_ratio": float(np.exp(coef)), "ci_low": float(np.exp(coef - Z95 * se)),
            "ci_high": float(np.exp(coef + Z95 * se)), "p_value": float(fit.pvalues["triple"])}


def synthetic_weights(target: np.ndarray, donors: np.ndarray) -> np.ndarray:
    """Non-negative donor weights summing to 1 that best reproduce the target before the policy.

    target has one value per month; donors one row per month and one column per donor. The
    sum-to-one condition is added as an extra, heavily weighted row of a non-negative least squares.
    """
    from scipy.optimize import nnls
    scale = 1e3 * max(1.0, float(np.abs(target).max()))
    a = np.vstack([donors, np.full((1, donors.shape[1]), scale)])
    b = np.append(target, scale)
    weights, _ = nnls(a, b)
    return weights / weights.sum()


def synthetic_control(series: pd.DataFrame, treated: str, donors: list, pre: np.ndarray) -> dict:
    """Abadie-style synthetic control on series indexed to each unit's mean before the policy (= 100).

    series has one row per month and one column per unit; pre marks the months before the policy.
    Returns the weights, the treated and synthetic paths, the post-policy ratio of their means,
    and the root mean squared prediction error before and after.
    """
    index = 100 * series / series[pre].mean()
    weights = synthetic_weights(index.loc[pre, treated].to_numpy(), index.loc[pre, donors].to_numpy())
    synthetic = index[donors].to_numpy() @ weights
    actual = index[treated].to_numpy()
    gap = actual - synthetic
    rmspe_pre, rmspe_post = np.sqrt(np.mean(gap[pre] ** 2)), np.sqrt(np.mean(gap[~pre] ** 2))
    return {"weights": dict(zip(donors, weights)), "actual": actual, "synthetic": synthetic,
            "ratio": float(actual[~pre].mean() / synthetic[~pre].mean()),
            "gap_post": float(gap[~pre].mean()), "rmspe_pre": float(rmspe_pre), "rmspe_post": float(rmspe_post),
            "rmspe_ratio": float(rmspe_post / rmspe_pre)}


def synthetic_placebo(series: pd.DataFrame, treated: str, donors: list, pre: np.ndarray) -> dict:
    """Placebo-in-space inference: refit the synthetic control with each donor as if it were treated.

    The p-value is the treated unit's rank by the ratio of post- to pre-policy prediction error,
    among itself and every placebo: (number at least as large) / (number of units).
    """
    actual = synthetic_control(series, treated, donors, pre)
    placebos = {d: synthetic_control(series, d, [x for x in donors if x != d], pre) for d in donors}
    ratios = np.array([p["rmspe_ratio"] for p in placebos.values()])
    rank = int(np.sum(ratios >= actual["rmspe_ratio"])) + 1
    return {**actual, "p_value": rank / (len(donors) + 1), "rank": rank, "units": len(donors) + 1,
            "placebo_ratios": {d: p["ratio"] for d, p in placebos.items()}}


def size_matched_permutation(totals: pd.DataFrame, treated_units, control_units, tolerance: float | None = 0.5) -> dict:
    """Exact permutation test over every set of control units, optionally only those of similar size.

    Each set of len(treated_units) control units is compared with the remaining controls by the
    ratio of ratios. With tolerance, only sets whose "before" total is within that fraction of the
    treated group's are kept, so small forces cannot stand in for a large country.
    """
    from itertools import combinations
    controls = totals.loc[list(control_units)]
    before, after = controls["before"].to_numpy(), controls["after"].to_numpy()
    sets = np.array(list(combinations(range(len(controls)), len(treated_units))))
    set_before, set_after = before[sets].sum(axis=1), after[sets].sum(axis=1)
    placebo = (set_after / set_before) / ((after.sum() - set_after) / (before.sum() - set_before))
    actual = ratio_of_ratios(totals.loc[list(treated_units) + list(control_units)], treated_units)
    treated_before = totals.loc[list(treated_units), "before"].sum()
    keep = np.ones(len(sets), dtype=bool) if tolerance is None else np.abs(set_before / treated_before - 1) <= tolerance
    p_value = float(np.mean(np.abs(np.log(placebo[keep])) >= abs(np.log(actual))))
    return {"ratio_of_ratios": actual, "p_value": p_value, "sets": int(keep.sum())}
