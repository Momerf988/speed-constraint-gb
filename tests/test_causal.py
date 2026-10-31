"""Tests for the difference-in-differences tools, on simulated panels with a known effect."""

import numpy as np
import pandas as pd
import pytest

from model.causal import did_poisson, event_study, implied_speed_change_kmh, permutation_test, ratio_of_ratios


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
