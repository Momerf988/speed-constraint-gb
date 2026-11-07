"""Tests for the appraisal: the values file, time lost, the £ arithmetic and the extractor.

Time and money tests use made-up uniform speed distributions, so each expected answer can be
worked out by hand. The extractor test re-reads DfT's raw files when they are present.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from model.appraisal import (VALUES_FILE, Values, break_even_free_flow, casualty_benefit, hours_lost_per_km,
                             ksi_from_casualties, load_values, read_values, time_cost, vehicle_hour_values)
from model.scenarios import ROAD_TYPES
from model.speeds import MPH_TO_KMH, SpeedDistribution
from tests.test_scenarios import uniform

URBAN_32, URBAN_48, RURAL_97, MOTORWAY = ROAD_TYPES
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
SIMPLE = Values(per_casualty={"fatal": 1000.0, "serious": 100.0, "slight": 10.0},
                per_collision={"fatal": 50.0, "serious": 20.0, "slight": 5.0},
                car_hour=10.0, van_hour=20.0, unreported_ratio=2.0)
NO_COLLISIONS = {"fatal": 0.0, "serious": 0.0, "slight": 0.0}


def test_values_file_holds_the_tag_values():
    values = load_values()
    assert values.per_casualty["fatal"] == pytest.approx(2_652_796.19, abs=0.01)    # TAG A4.1.1, 2023 £
    assert values.per_collision["fatal"] == pytest.approx(26_560.26 + 16_305.21 + 448.25, abs=0.01)
    assert (values.car_hour, values.van_hour) == (pytest.approx(15.65, abs=0.005), pytest.approx(28.76, abs=0.005))


def test_previous_values_are_lower_than_tag_v2_03():
    assert load_values("previous").per_casualty["fatal"] < load_values().per_casualty["fatal"]


def test_vehicle_hour_value_is_share_times_occupancy_times_value_of_time():
    v = {key: 0.0 for key in read_values()}
    v.update(car_km_share_other=1.0, car_occupancy_other=2.0, vot_other=5.0,
             van_km_share_services=1.0, van_occupancy_work=1.5, vot_van_services=20.0)
    assert vehicle_hour_values(v) == (pytest.approx(10.0), pytest.approx(30.0))


def test_casualty_benefit_splits_ksi_into_deaths_and_serious_injuries():
    # 1 death, 3 KSI (so 2 serious), 10 casualties (so 7 slight)
    assert casualty_benefit(1, 3, 10, SIMPLE, NO_COLLISIONS) == pytest.approx(1000 + 2 * 100 + 7 * 10)
    with_collisions = casualty_benefit(1, 3, 10, SIMPLE, {"fatal": 1.0, "serious": 0.5, "slight": 0.5})
    assert with_collisions == pytest.approx(1050 + 2 * 110 + 7 * 12.5)


def test_unreported_injuries_scale_only_the_non_fatal_benefit():
    reported = casualty_benefit(1, 3, 10, SIMPLE, NO_COLLISIONS)
    everything = casualty_benefit(1, 3, 10, SIMPLE, NO_COLLISIONS, include_unreported=True)
    assert everything == pytest.approx(1000 + (2 * 100 + 7 * 10) * 3)
    assert everything > reported


def test_ksi_from_casualties_keeps_the_baseline_proportion():
    assert ksi_from_casualties(50, baseline_ksi=200, baseline_casualties=1000) == pytest.approx(10)


def test_lawful_time_leaves_out_time_gained_by_speeding():
    today = uniform(70, 110)
    lawful = hours_lost_per_km("cap_80", today, RURAL_97, isa_working=0.665)
    everything = hours_lost_per_km("cap_80", today, RURAL_97, isa_working=0.665, count_speeding_time=True)
    speeds = today.speeds_kmh[today.probabilities > 0]
    expected = np.mean(np.maximum(0, 1 / np.minimum(speeds, 80) - 1 / np.minimum(speeds, RURAL_97.limit_kmh)))
    assert lawful == pytest.approx(expected)
    assert 0 < lawful < everything


def test_compliance_and_caps_above_the_limit_cost_no_lawful_time():
    today = uniform(30, 70)
    assert hours_lost_per_km("compliance", today, URBAN_48, isa_working=0.665) == 0
    assert hours_lost_per_km("cap_60", today, URBAN_48, isa_working=0.665) == 0
    assert hours_lost_per_km("compliance", today, URBAN_48, 0.665, count_speeding_time=True) > 0


def test_lawful_time_lost_does_not_depend_on_how_often_isa_is_switched_off():
    today = uniform(70, 110)
    assert (hours_lost_per_km("cap_80", today, RURAL_97, isa_working=0.2)
            == pytest.approx(hours_lost_per_km("cap_80", today, RURAL_97, isa_working=0.9)))


def test_time_cost_scales_with_traffic_and_free_flow():
    one = time_cost(1e-3, 1e-3, car_billion_km=2.0, van_billion_km=1.0, values=SIMPLE, free_flow_share=0.5)
    assert one == pytest.approx(1e-3 * 0.5 * (2.0 * 10 + 1.0 * 20) * 1e9)
    assert time_cost(1e-3, 1e-3, 2.0, 1.0, SIMPLE, 1.0) == pytest.approx(2 * one)


def test_time_cost_charges_each_vehicle_its_own_hours():
    assert time_cost(1e-3, 0.0, 2.0, 1.0, SIMPLE, 1.0) == pytest.approx(1e-3 * 2.0 * 10 * 1e9)
    assert time_cost(0.0, 1e-3, 2.0, 1.0, SIMPLE, 1.0) == pytest.approx(1e-3 * 1.0 * 20 * 1e9)


def at_one_speed(kmh: float) -> SpeedDistribution:
    return SpeedDistribution(np.array([kmh]), np.array([1.0]), f"everyone at {kmh}")


def test_under_an_80_cap_a_van_at_90_loses_no_lawful_time_and_a_car_does():
    today = at_one_speed(90.0)
    assert hours_lost_per_km("cap_80", today, RURAL_97, 0.0, vehicle="van") == 0
    assert hours_lost_per_km("cap_80", today, RURAL_97, 0.0, vehicle="car") == pytest.approx(1 / 80 - 1 / 90)


def test_a_van_loses_lawful_time_only_below_its_own_limit():
    today = at_one_speed(90.0)
    van_limit = 50 * MPH_TO_KMH
    assert hours_lost_per_km("cap_60", today, RURAL_97, 0.0, vehicle="van") == pytest.approx(1 / 60 - 1 / van_limit)
    assert hours_lost_per_km("compliance", today, RURAL_97, 0.0, vehicle="van") == 0
    assert hours_lost_per_km("cap_80", at_one_speed(70.0), RURAL_97, 0.0, vehicle="van") == 0


def test_a_cap_a_fraction_below_a_limit_is_that_limit_in_practice():
    assert hours_lost_per_km("cap_48", at_one_speed(60.0), URBAN_48, 0.0) == 0     # 48 against 48.28 km/h
    assert hours_lost_per_km("cap_48", at_one_speed(60.0), RURAL_97, 0.0) > 0


def test_at_the_break_even_share_time_cost_equals_benefit():
    share = break_even_free_flow(5e6, 1e-3, 5e-4, 2.0, 1.0, SIMPLE)
    assert time_cost(1e-3, 5e-4, 2.0, 1.0, SIMPLE, share) == pytest.approx(5e6)
    assert np.isinf(break_even_free_flow(5e6, 0.0, 0.0, 2.0, 1.0, SIMPLE))


@pytest.mark.skipif(not (RAW / "tag-data-book-v2-03.xlsm").exists(), reason="raw TAG data book not downloaded")
def test_values_file_matches_dfts_files():
    from ingest.extract_appraisal_values import RAS4001, TAG, ras4001_values, tag_values
    extracted = {row["value_id"]: row["value"] for row in tag_values(TAG) + ras4001_values(RAS4001)}
    with open(VALUES_FILE, newline="") as f:
        saved = {row["value_id"]: float(row["value"]) for row in csv.DictReader(f)}
    assert extracted == pytest.approx(saved)
