"""Tests for the difference-in-differences tools, on simulated panels with a known effect."""

import numpy as np
import pandas as pd
import pytest

from model.causal import (ddd_poisson, did_poisson, did_poisson_trend, event_study, implied_speed_change_kmh,
                          permutation_test, ratio_of_ratios, size_matched_permutation, synthetic_control,
                          single_unit_placebo, synthetic_placebo)


def simulated_panel(rate_ratio: float, seed: int = 1) -> pd.DataFrame:
    """20 control and 4 treated forces over 24 months; treatment from month 12 multiplies counts by rate_ratio."""
    rng = np.random.default_rng(seed)
    units, months = np.arange(24), pd.date_range("2022-01-01", periods=24, freq="MS")
    level = dict(zip(units, rng.uniform(200, 800, len(units))))
    season = dict(zip(months, 1 + 0.2 * np.sin(np.arange(24) / 2)))
    rows = []
    for unit in units:
        for i, month in enumerate(months):
            treated, after = unit >= 20, i >= 12
            mean = level[unit] * season[month] * (rate_ratio if treated and after else 1)
            rows.append({"police_force_code": unit, "month": month, "treated": treated, "after": after,
                         "quarter": f"{month.year}Q{month.quarter}", "casualties": float(rng.poisson(mean))})
    return pd.DataFrame(rows)


def test_did_recovers_a_known_rate_ratio():
    result = did_poisson(simulated_panel(0.8), "casualties", treated="treated", post="after")
    assert result["rate_ratio"] == pytest.approx(0.8, abs=0.02)
    assert result["ci_low"] < 0.8 < result["ci_high"]


def test_did_finds_nothing_when_there_is_nothing():
    result = did_poisson(simulated_panel(1.0), "casualties", treated="treated", post="after")
    assert result["ci_low"] < 1 < result["ci_high"]


def test_event_study_shows_no_gap_before_treatment():
    study = event_study(simulated_panel(0.8), "casualties", "treated", "quarter", reference="2022Q4")
    before = study[study["quarter"] < "2022Q4"]["rate_ratio"]
    after = study[study["quarter"] >= "2023Q1"]["rate_ratio"]
    assert before.between(0.95, 1.05).all()
    assert after.between(0.75, 0.85).all()


def test_ratio_of_ratios_by_hand():
    totals = pd.DataFrame({"before": [100.0, 200.0, 300.0], "after": [80.0, 200.0, 300.0]}, index=["a", "b", "c"])
    # treated a: 80/100 = 0.8; controls b and c: 500/500 = 1.0
    assert ratio_of_ratios(totals, ["a"]) == pytest.approx(0.8)


def test_permutation_test_flags_a_change_no_control_shows():
    rng = np.random.default_rng(3)
    controls = pd.DataFrame({"before": 1000.0, "after": 1000.0 * rng.uniform(0.95, 1.05, 30)},
                            index=[f"c{i}" for i in range(30)])
    treated = pd.DataFrame({"before": 1000.0, "after": 800.0}, index=["t1", "t2", "t3", "t4"])
    result = permutation_test(pd.concat([treated, controls]), list(treated.index), list(controls.index), 500, seed=1)
    assert result["ratio_of_ratios"] == pytest.approx(0.8, abs=0.02)
    assert result["p_value"] == 0


def test_implied_speed_change_inverts_the_exponential_model():
    assert implied_speed_change_kmh(np.exp(0.06 * -4), 0.06) == pytest.approx(-4)


def simulated_trend_panel(rate_ratio: float, yearly_drift: float, seed: int = 2) -> pd.DataFrame:
    """As simulated_panel, over 48 months, with the treated forces drifting by yearly_drift a year throughout."""
    rng = np.random.default_rng(seed)
    units, months = np.arange(24), pd.date_range("2020-01-01", periods=48, freq="MS")
    level = dict(zip(units, rng.uniform(400, 900, len(units))))
    rows = []
    for unit in units:
        for i, month in enumerate(months):
            treated, after, years = unit >= 20, i >= 36, i / 12
            mean = level[unit] * (1 + 0.2 * np.sin(i / 2)) * (yearly_drift ** years if treated else 1)
            mean *= rate_ratio if treated and after else 1
            rows.append({"police_force_code": unit, "month": month, "treated": treated, "after": after,
                         "years": years, "casualties": float(rng.poisson(mean))})
    return pd.DataFrame(rows)


def test_the_trend_model_removes_an_earlier_drift_that_plain_did_credits_to_the_policy():
    panel = simulated_trend_panel(rate_ratio=1.0, yearly_drift=0.9)
    plain = did_poisson(panel, "casualties", treated="treated", post="after")
    trend = did_poisson_trend(panel, "casualties", treated="treated", post="after", trend="years")
    assert plain["rate_ratio"] < 0.9                                  # the drift looks like an effect
    assert trend["rate_ratio"] == pytest.approx(1.0, abs=0.03)        # and the trend model removes it
    assert trend["trend_per_year"] == pytest.approx(0.9, abs=0.02)


def test_the_trend_model_still_finds_a_real_effect():
    trend = did_poisson_trend(simulated_trend_panel(rate_ratio=0.8, yearly_drift=0.95), "casualties",
                              treated="treated", post="after", trend="years")
    assert trend["rate_ratio"] == pytest.approx(0.8, abs=0.03)


def test_the_triple_difference_separates_an_effect_from_a_shift_on_every_treated_road():
    rng = np.random.default_rng(4)
    rows = []
    for unit in range(24):
        treated = unit >= 20
        for i, month in enumerate(pd.date_range("2022-01-01", periods=24, freq="MS")):
            after = i >= 12
            for group, level in (("focus", 600), ("other", 400)):
                mean = level * (1 + 0.1 * np.sin(i)) * (0.9 if treated and after else 1)   # a shift on all treated roads
                mean *= 0.8 if treated and after and group == "focus" else 1               # and an effect on one group
                rows.append({"police_force_code": unit, "month": month, "treated": treated, "after": after,
                             "road_group": group, "casualties": float(rng.poisson(mean))})
    result = ddd_poisson(pd.DataFrame(rows), "casualties", "treated", "after", "road_group", "focus")
    assert result["rate_ratio"] == pytest.approx(0.8, abs=0.03)


def test_synthetic_control_recovers_known_weights_and_effect():
    months = 30
    pre = np.arange(months) < 20
    rng = np.random.default_rng(5)
    donors = pd.DataFrame({f"d{i}": 100 + rng.normal(0, 8, months).cumsum() for i in range(6)})
    treated = 0.7 * donors["d1"] + 0.3 * donors["d4"]
    treated[~pre] *= 0.85
    series = donors.assign(wales=treated)
    result = synthetic_control(series, "wales", list(donors.columns), pre)
    assert result["weights"]["d1"] > 0.5 and result["weights"]["d4"] > 0.15
    assert sum(result["weights"].values()) == pytest.approx(1)
    assert result["ratio"] == pytest.approx(0.85, abs=0.03)
    assert result["rmspe_ratio"] > 3


def test_synthetic_placebo_ranks_a_large_effect_first():
    months = 30
    pre = np.arange(months) < 20
    rng = np.random.default_rng(6)
    donors = pd.DataFrame({f"d{i}": 100 + rng.normal(0, 3, months) for i in range(12)})
    treated = donors.mean(axis=1) + rng.normal(0, 1, months)
    treated[~pre] *= 0.7
    result = synthetic_placebo(donors.assign(wales=treated), "wales", list(donors.columns), pre)
    assert result["rank"] == 1 and result["p_value"] == pytest.approx(1 / 13)


def test_size_matched_permutation_keeps_only_sets_of_similar_size():
    totals = pd.DataFrame({"before": [100.0, 100.0, 1000.0, 1000.0, 100.0, 100.0],
                           "after": [80.0, 80.0, 1000.0, 1000.0, 100.0, 100.0]},
                          index=["t1", "t2", "c1", "c2", "c3", "c4"])
    every = size_matched_permutation(totals, ["t1", "t2"], ["c1", "c2", "c3", "c4"], tolerance=None)
    matched = size_matched_permutation(totals, ["t1", "t2"], ["c1", "c2", "c3", "c4"], tolerance=0.5)
    assert every["sets"] == 6                     # all pairs of four controls
    assert matched["sets"] == 1                   # only c3 and c4 together are about as big as t1 and t2
    assert matched["ratio_of_ratios"] == pytest.approx(0.8)


def test_single_unit_placebo_ranks_a_clear_effect_first_and_cannot_go_below_one_over_units():
    result = single_unit_placebo(simulated_panel(0.6), "casualties", "treated", "after", tolerance=0.5)
    assert result["rate_ratio"] == pytest.approx(0.6, abs=0.03)
    assert result["all"]["rank"] == 1 and result["all"]["units"] == 21
    assert result["all"]["p_value"] == pytest.approx(1 / 21) == result["all"]["smallest_p"]
    assert result["similar"]["units"] <= 21


def test_single_unit_placebo_finds_nothing_when_there_is_nothing():
    assert single_unit_placebo(simulated_panel(1.0), "casualties", "treated", "after")["all"]["p_value"] > 0.2
