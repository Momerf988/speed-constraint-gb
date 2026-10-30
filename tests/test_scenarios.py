"""Tests for the policies, the casualty models, a full run and the Monte Carlo draws.

They use made-up uniform speed distributions, so they need no data files and every expected
answer can be worked out by hand.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from model.casualties import Assumptions, model_a, model_b
from model.run import EVIDENCE_ORDER, run, worse
from model.scenarios import POLICIES, ROAD_TYPES, apply_policy, isa_working_share, original_speeds
from model.speeds import SpeedDistribution
from model.uncertainty import ASSUMPTION_RANGES, URBAN_CURVES, draw

URBAN_32, URBAN_48, RURAL_97, MOTORWAY = ROAD_TYPES
CENTRAL = Assumptions.central()


def uniform(low_kmh: float, high_kmh: float) -> SpeedDistribution:
    """Cars spread evenly between two speeds, on the same 0.1 km/h grid the model uses."""
    speeds = np.arange(0.05, 250, 0.1)
    weights = ((speeds >= low_kmh) & (speeds < high_kmh)).astype(float)
    return SpeedDistribution(speeds, weights / weights.sum(), f"uniform {low_kmh}-{high_kmh}")


TODAY_48 = uniform(30, 70)


def test_central_assumptions_come_from_the_parameters_file():
    assert CENTRAL.reference_mean == "new"
    assert CENTRAL.free_travel_share == 0.56
    assert (CENTRAL.power_exponent_fatal, CENTRAL.power_exponent_injury) == (5.5, 3.9)


def test_isa_working_share_multiplies_the_two_ways_isa_is_off():
    assert isa_working_share(URBAN_48, switched_off=0.30, override=0.05) == pytest.approx(0.70 * 0.95)


def test_compliance_leaves_nobody_over_the_limit_and_slower_cars_alone():
    after = apply_policy("compliance", TODAY_48, URBAN_48, isa_working=0.665)
    assert after.share_at_or_above(URBAN_48.limit_kmh + 0.05) == 0
    slower = TODAY_48.speeds_kmh < URBAN_48.limit_kmh
    np.testing.assert_array_equal(after.speeds_kmh[slower], TODAY_48.speeds_kmh[slower])


def test_bau_isa_removes_speeding_only_where_isa_is_working():
    working = 0.665
    after = apply_policy("bau_isa", TODAY_48, URBAN_48, working)
    over_today = TODAY_48.share_at_or_above(URBAN_48.limit_kmh + 0.05)
    assert after.share_at_or_above(URBAN_48.limit_kmh + 0.05) == pytest.approx((1 - working) * over_today)
    compliance = apply_policy("compliance", TODAY_48, URBAN_48, working)
    assert compliance.mean() < after.mean() < TODAY_48.mean()


def test_a_cap_above_the_limit_binds_only_where_isa_is_not_working():
    working = 0.665
    after = apply_policy("cap_60", TODAY_48, URBAN_48, working)
    assert after.share_at_or_above(60.05) == 0
    between = TODAY_48.share_at_or_above(URBAN_48.limit_kmh + 0.05) - TODAY_48.share_at_or_above(60)
    assert (after.share_at_or_above(URBAN_48.limit_kmh + 0.05)
            == pytest.approx((1 - working) * (between + TODAY_48.share_at_or_above(60))))


@pytest.mark.parametrize("policy", POLICIES)
def test_original_speeds_line_up_with_every_policy(policy):
    after = apply_policy(policy, TODAY_48, URBAN_48, isa_working=0.665)
    before = original_speeds(policy, TODAY_48)
    assert len(before) == len(after.speeds_kmh)
    assert np.all(after.speeds_kmh <= before)


def test_today_changes_nothing_in_either_model():
    b = model_b(TODAY_48, TODAY_48, TODAY_48.speeds_kmh, URBAN_48, k_ksi=0.9, assumptions=CENTRAL)
    assert b["rho"] == pytest.approx(1) and b["ksi"] == pytest.approx(1) and b["share_clamped"] == 0
    a = model_a(TODAY_48, TODAY_48, k_fatal=0.9, k_casualties=0.9, assumptions=CENTRAL)
    assert a["power_fatal"] == pytest.approx(1) and a["exponential_casualties"] == pytest.approx(1)


def test_compliance_cuts_deaths_most_and_slight_injuries_least():
    after = apply_policy("compliance", TODAY_48, URBAN_48, isa_working=0.665)
    b = model_b(TODAY_48, after, TODAY_48.speeds_kmh, URBAN_48, k_ksi=0.9, assumptions=CENTRAL)
    assert b["fatal"] < b["ksi"] < b["slight"] < 1


def test_the_effect_on_ksi_scales_with_the_free_travel_share():
    after = apply_policy("compliance", TODAY_48, URBAN_48, isa_working=0.665)
    full = model_b(TODAY_48, after, TODAY_48.speeds_kmh, URBAN_48, 0.9, CENTRAL)
    half = model_b(TODAY_48, after, TODAY_48.speeds_kmh, URBAN_48, 0.9,
                   replace(CENTRAL, free_travel_share=CENTRAL.free_travel_share / 2))
    assert 1 - half["ksi"] == pytest.approx((1 - full["ksi"]) / 2)


def test_measuring_from_todays_mean_gives_the_larger_reduction():
    after = apply_policy("compliance", TODAY_48, URBAN_48, isa_working=0.665)
    new = model_b(TODAY_48, after, TODAY_48.speeds_kmh, URBAN_48, 0.9, CENTRAL)
    old = model_b(TODAY_48, after, TODAY_48.speeds_kmh, URBAN_48, 0.9, replace(CENTRAL, reference_mean="old"))
    assert old["rho"] < new["rho"] < 1


def test_cars_slowed_far_below_the_mean_are_counted_as_clamped():
    today = uniform(70, 110)
    after = apply_policy("cap_48", today, RURAL_97, isa_working=0.665)
    b = model_b(today, after, original_speeds("cap_48", today), RURAL_97, 0.9, replace(CENTRAL, reference_mean="old"))
    assert b["share_clamped"] == pytest.approx(1)   # every car slowed to 48, over 10 km/h below a mean of 90


def test_motorways_have_no_individual_curve():
    today = uniform(90, 130)
    assert model_b(today, today, today.speeds_kmh, MOTORWAY, 0.9, CENTRAL) is None


def test_worse_returns_the_weakest_evidence():
    assert worse("within", "beyond", "lower bound") == "beyond"
    assert worse("within", "within") == "within"


def test_a_full_run_measures_every_policy_against_today_and_bau():
    distributions = {URBAN_32.segment: uniform(25, 45), URBAN_48.segment: TODAY_48,
                     RURAL_97.segment: uniform(70, 110), MOTORWAY.segment: uniform(90, 130)}
    baseline = pd.DataFrame({"killed": 100.0, "ksi": 1000.0, "casualties": 5000.0,
                             "k_fatal": 0.9, "k_ksi": 0.9, "k_casualties": 0.9},
                            index=list(distributions))
    table = run(distributions, baseline, CENTRAL).set_index(["segment", "policy"])
    assert len(table) == len(ROAD_TYPES) * len(POLICIES)
    assert set(table["evidence"]) <= set(EVIDENCE_ORDER)
    assert (table.xs("today", level="policy")["killed_prevented"] == 0).all()
    assert (table.xs("bau_isa", level="policy")["killed_prevented_vs_bau"] == 0).all()
    assert (table.xs("compliance", level="policy")["killed_prevented_vs_bau"] > 0).all()
    assert table.loc[(MOTORWAY.segment, "compliance"), "preferred_model"] == "power"


def test_monte_carlo_draws_are_reproducible_and_inside_their_ranges():
    first, second = draw(500, seed=1), draw(500, seed=1)
    pd.testing.assert_frame_equal(first, second)
    for name, (low, high) in ASSUMPTION_RANGES.items():
        assert first[name].between(low, high).all()
    assert set(first["urban_curve"]) <= set(URBAN_CURVES)
