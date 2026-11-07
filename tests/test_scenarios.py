"""Tests for the policies, the casualty models, a full run and the Monte Carlo draws.

They use made-up uniform speed distributions, so they need no data files and every expected
answer can be worked out by hand.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from model.casualties import Assumptions, CarVanShares, model_a, model_b, severity_powers
from model.run import EVIDENCE_ORDER, run, worse
from model.scenarios import (POLICIES, ROAD_TYPES, IsaUse, Traffic, apply_policy, fleet_after, fleet_before,
                             held_to_limit, isa_working_share, mixture, original_speeds)
from model.speeds import MPH_TO_KMH, SpeedDistribution
from model.uncertainty import ASSUMPTION_RANGES, SPEED_CONTROL_RANGE, URBAN_CURVES, WARNING_EFFECT_RANGE, draw

URBAN_32, URBAN_48, RURAL_97, MOTORWAY = ROAD_TYPES
CENTRAL = Assumptions.central()
K = CarVanShares(fatal=0.9, ksi=0.9, slight=0.9)


def uniform(low_kmh: float, high_kmh: float) -> SpeedDistribution:
    """Cars spread evenly between two speeds, on the same 0.1 km/h grid the model uses."""
    speeds = np.arange(0.05, 250, 0.1)
    weights = ((speeds >= low_kmh) & (speeds < high_kmh)).astype(float)
    return SpeedDistribution(speeds, weights / weights.sum(), f"uniform {low_kmh}-{high_kmh}")


TODAY_48 = uniform(30, 70)


def test_central_assumptions_come_from_the_parameters_file():
    assert CENTRAL.reference_mean == "new"
    assert CENTRAL.free_travel_share == 0.56
    assert CENTRAL.free_travel_share_rural == 0.8024    # CR 204: 134 of 167 crashes
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
    b = model_b(TODAY_48, TODAY_48, TODAY_48.speeds_kmh, URBAN_48, K, assumptions=CENTRAL)
    assert b["rho"] == pytest.approx(1) and b["ksi"] == pytest.approx(1) and b["share_clamped"] == 0
    a = model_a(TODAY_48, TODAY_48, k_fatal=0.9, k_casualties=0.9, assumptions=CENTRAL)
    assert a["power_fatal"] == pytest.approx(1) and a["exponential_casualties"] == pytest.approx(1)


def test_compliance_cuts_deaths_most_and_slight_injuries_least():
    after = apply_policy("compliance", TODAY_48, URBAN_48, isa_working=0.665)
    b = model_b(TODAY_48, after, TODAY_48.speeds_kmh, URBAN_48, K, assumptions=CENTRAL)
    assert b["fatal"] < b["ksi"] < b["slight"] < 1


def test_deaths_use_the_share_of_deaths_and_ksi_the_share_of_ksi():
    after = apply_policy("compliance", TODAY_48, URBAN_48, isa_working=0.665)
    fewer_deaths_reachable = CarVanShares(fatal=0.45, ksi=0.9, slight=0.9)
    full = model_b(TODAY_48, after, TODAY_48.speeds_kmh, URBAN_48, K, CENTRAL)
    half = model_b(TODAY_48, after, TODAY_48.speeds_kmh, URBAN_48, fewer_deaths_reachable, CENTRAL)
    assert 1 - half["fatal"] == pytest.approx((1 - full["fatal"]) / 2)
    assert half["ksi"] == pytest.approx(full["ksi"])


def test_the_severity_power_applies_only_to_the_casualties_that_respond():
    after = apply_policy("compliance", TODAY_48, URBAN_48, isa_working=0.665)
    b = model_b(TODAY_48, after, TODAY_48.speeds_kmh, URBAN_48, CarVanShares(0.6, 0.7, 0.8), CENTRAL)
    fatal_power, slight_power = severity_powers("urban")
    f = CENTRAL.free_travel_share
    assert b["ksi"] == pytest.approx(1 - 0.7 * f * (1 - b["rho"]))
    assert b["fatal"] == pytest.approx(1 - 0.6 * f * (1 - b["rho"] ** fatal_power))
    assert b["slight"] == pytest.approx(1 - 0.8 * f * (1 - b["rho"] ** slight_power))
    # Even if every responding casualty vanished, those that cannot respond would remain
    gone = model_b(TODAY_48, TODAY_48.capped(1.0), np.full_like(TODAY_48.speeds_kmh, 1.0), URBAN_48, K,
                   replace(CENTRAL, reference_mean="old"))
    assert gone["fatal"] >= 1 - 0.9 * f


def test_rural_roads_use_the_rural_free_travel_share():
    today = uniform(70, 110)
    after = apply_policy("compliance", today, RURAL_97, isa_working=0.665)
    b = model_b(today, after, today.speeds_kmh, RURAL_97, K, CENTRAL)
    assert b["ksi"] == pytest.approx(1 - 0.9 * CENTRAL.free_travel_share_rural * (1 - b["rho"]))
    changed_urban = model_b(today, after, today.speeds_kmh, RURAL_97, K, replace(CENTRAL, free_travel_share=0.1))
    assert changed_urban["ksi"] == pytest.approx(b["ksi"])


CONTROL_ONLY = IsaUse(switched_off=0.3, speed_control_share=1.0, warning_effect=0.225)


def both(dist: SpeedDistribution, van_share: float = 0.2) -> Traffic:
    """Cars and vans with the same speeds."""
    return Traffic(dist, dist, van_share)


def test_only_single_carriageways_give_vans_a_lower_limit_than_cars():
    assert RURAL_97.van_limit_kmh == pytest.approx(50 * MPH_TO_KMH)
    for road in (URBAN_32, URBAN_48, MOTORWAY):
        assert road.van_limit_kmh == road.limit_kmh
    assert RURAL_97.legal_limit("car") == RURAL_97.limit_kmh
    assert RURAL_97.legal_limit("van") == RURAL_97.van_limit_kmh


def test_compliance_holds_each_vehicle_to_its_own_limit():
    traffic = both(uniform(70, 110))
    cars, vans, mixed = fleet_after("compliance", traffic, RURAL_97, CONTROL_ONLY)
    assert cars.share_at_or_above(RURAL_97.limit_kmh + 0.05) == 0
    assert cars.share_at_or_above(RURAL_97.van_limit_kmh + 0.05) > 0
    assert vans.share_at_or_above(RURAL_97.van_limit_kmh + 0.05) == 0
    assert mixed.mean() == pytest.approx(0.8 * cars.mean() + 0.2 * vans.mean())


def test_isa_holds_a_van_to_the_posted_limit_unless_it_shows_the_vans_own():
    traffic = both(uniform(70, 110))
    _, reads_sign, _ = fleet_after("bau_isa", traffic, RURAL_97, CONTROL_ONLY)
    _, reads_own, _ = fleet_after("bau_isa", traffic, RURAL_97, replace(CONTROL_ONLY, vans_read_own_limit=True))
    assert reads_own.mean() < reads_sign.mean() < traffic.vans.mean()
    between = (traffic.vans.speeds_kmh > RURAL_97.van_limit_kmh) & (traffic.vans.speeds_kmh <= RURAL_97.limit_kmh)
    assert reads_sign.share_at_or_above(RURAL_97.van_limit_kmh + 0.05) >= traffic.vans.probabilities[between].sum()


def test_mixed_traffic_lines_up_with_the_speeds_it_started_from():
    traffic = Traffic(uniform(70, 110), uniform(65, 105), 0.2)
    for policy in POLICIES:
        _, _, mixed = fleet_after(policy, traffic, RURAL_97, CONTROL_ONLY)
        before = fleet_before(policy, traffic)
        assert len(before) == len(mixed.speeds_kmh)
        assert np.all(mixed.speeds_kmh <= before)
        assert mixed.probabilities.sum() == pytest.approx(1)


def test_traffic_mixes_cars_and_vans_by_their_share_of_vehicle_km():
    traffic = Traffic(uniform(80, 80.2), uniform(70, 70.2), 0.25)
    assert traffic.mixed().mean() == pytest.approx(0.75 * 80.1 + 0.25 * 70.1, abs=0.01)


def test_targeted_override_totals_the_trial_share_and_all_of_it_is_above_the_limit():
    held = held_to_limit(TODAY_48, URBAN_48, replace(CONTROL_ONLY, switched_off=0.0))
    overridden = 1 - held
    above = TODAY_48.speeds_kmh > URBAN_48.limit_kmh
    assert np.sum(overridden * TODAY_48.probabilities) == pytest.approx(URBAN_48.override_share)
    assert np.all(overridden[~above] == 0)


def test_override_cannot_exceed_all_the_speeding_there_is():
    hardly_any_speeding = uniform(30, URBAN_48.limit_kmh + 0.5)
    held = held_to_limit(hardly_any_speeding, URBAN_48, replace(CONTROL_ONLY, switched_off=0.0))
    assert np.min(held) == pytest.approx(0)     # every car above the limit overrides
    nobody_speeding = uniform(30, 40)
    assert np.all(held_to_limit(nobody_speeding, URBAN_48, replace(CONTROL_ONLY, switched_off=0.0)) == 1)


def test_random_override_is_one_share_for_every_speed():
    held = held_to_limit(TODAY_48, URBAN_48, replace(CONTROL_ONLY, override_targets_speeding=False))
    assert held == pytest.approx(0.70 * 0.95)


def test_targeted_override_holds_back_less_speeding_than_random_override():
    targeted = apply_policy("bau_isa", TODAY_48, URBAN_48, held_to_limit(TODAY_48, URBAN_48, CONTROL_ONLY))
    random = apply_policy("bau_isa", TODAY_48, URBAN_48,
                          held_to_limit(TODAY_48, URBAN_48, replace(CONTROL_ONLY, override_targets_speeding=False)))
    assert targeted.mean() > random.mean()


def test_a_warning_only_fleet_achieves_kappa_of_speed_control():
    warning = replace(CONTROL_ONLY, speed_control_share=0.0)
    half = replace(CONTROL_ONLY, speed_control_share=0.5)
    control_held = held_to_limit(TODAY_48, URBAN_48, CONTROL_ONLY)
    assert held_to_limit(TODAY_48, URBAN_48, warning) == pytest.approx(0.225 * control_held)
    assert held_to_limit(TODAY_48, URBAN_48, half) == pytest.approx((0.5 + 0.5 * 0.225) * control_held)


def test_a_mixture_takes_one_weight_per_speed_and_still_sums_to_one():
    weights = np.where(TODAY_48.speeds_kmh > 50, 0.2, 0.9)
    mixed = mixture([(weights, TODAY_48.capped(50)), (1 - weights, TODAY_48)], "mixed")
    assert mixed.probabilities.sum() == pytest.approx(1)
    assert len(mixed.speeds_kmh) == 2 * len(TODAY_48.speeds_kmh)
    over = TODAY_48.share_at_or_above(50.05)
    assert mixed.share_at_or_above(50.05) == pytest.approx(0.8 * over)


def test_the_effect_on_ksi_scales_with_the_free_travel_share():
    after = apply_policy("compliance", TODAY_48, URBAN_48, isa_working=0.665)
    full = model_b(TODAY_48, after, TODAY_48.speeds_kmh, URBAN_48, K, CENTRAL)
    half = model_b(TODAY_48, after, TODAY_48.speeds_kmh, URBAN_48, K,
                   replace(CENTRAL, free_travel_share=CENTRAL.free_travel_share / 2))
    assert 1 - half["ksi"] == pytest.approx((1 - full["ksi"]) / 2)


def test_measuring_from_todays_mean_gives_the_larger_reduction():
    after = apply_policy("compliance", TODAY_48, URBAN_48, isa_working=0.665)
    new = model_b(TODAY_48, after, TODAY_48.speeds_kmh, URBAN_48, K, CENTRAL)
    old = model_b(TODAY_48, after, TODAY_48.speeds_kmh, URBAN_48, K, replace(CENTRAL, reference_mean="old"))
    assert old["rho"] < new["rho"] < 1


def test_cars_slowed_far_below_the_mean_are_counted_as_clamped():
    today = uniform(70, 110)
    after = apply_policy("cap_48", today, RURAL_97, isa_working=0.665)
    b = model_b(today, after, original_speeds("cap_48", today), RURAL_97, K, replace(CENTRAL, reference_mean="old"))
    assert b["share_clamped"] == pytest.approx(1)   # every car slowed to 48, over 10 km/h below a mean of 90


def test_motorways_have_no_individual_curve():
    today = uniform(90, 130)
    assert model_b(today, today, today.speeds_kmh, MOTORWAY, K, CENTRAL) is None


def test_worse_returns_the_weakest_evidence():
    assert worse("within", "beyond", "lower bound") == "beyond"
    assert worse("within", "within") == "within"


def test_a_full_run_measures_every_policy_against_today_and_bau():
    distributions = {URBAN_32.segment: both(uniform(25, 45)), URBAN_48.segment: both(TODAY_48),
                     RURAL_97.segment: both(uniform(70, 110)), MOTORWAY.segment: both(uniform(90, 130))}
    baseline = pd.DataFrame({"killed": 100.0, "ksi": 1000.0, "casualties": 5000.0,
                             "k_fatal": 0.9, "k_ksi": 0.9, "k_casualties": 0.9, "k_slight": 0.9},
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
    assert first["speed_control_share"].between(*SPEED_CONTROL_RANGE).all()
    assert first["warning_effect"].between(*WARNING_EFFECT_RANGE).all()
