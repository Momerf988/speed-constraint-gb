"""Tests for the appraisal: the values file, time lost, the £ arithmetic and the extractor.

Time and money tests use made-up uniform speed distributions, so each expected answer can be
worked out by hand. The extractor test re-reads DfT's raw files when they are present.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from model.appraisal import (VALUES_FILE, Values, break_even_free_flow, capital_cost, casualty_benefit, hours_lost_per_km,
                             ksi_from_casualties, load_fuels, load_values, read_values, running_per_km, time_cost,
                             vehicle_hour_values)
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
    full_cost = time_cost(1e-3, 5e-4, 2.0, 1.0, SIMPLE, 1.0)
    share = break_even_free_flow(5e6, 0.0, full_cost)
    assert time_cost(1e-3, 5e-4, 2.0, 1.0, SIMPLE, share) == pytest.approx(5e6)
    assert np.isinf(break_even_free_flow(5e6, 0.0, 0.0))


def test_running_savings_scale_with_free_flow_in_the_break_even():
    # casualties 100; 40 of running savings and 240 of cost if all traffic flows freely: 100 + 40s = 240s
    assert break_even_free_flow(100.0, 40.0, 240.0) == pytest.approx(0.5)
    assert np.isinf(break_even_free_flow(100.0, 300.0, 240.0))      # savings alone beat the cost at any share


def test_fuel_follows_tags_formula_by_hand_for_a_diesel_car():
    # TAG A1.3.11, 2023, diesel car, rounded here to 7 decimal places
    diesel = load_fuels(2023)["car"][1]
    assert (diesel.a, diesel.b, diesel.c) == (pytest.approx(0.4780137, abs=1e-6), pytest.approx(0.0685353, abs=1e-6),
                                              pytest.approx(-0.0006593, abs=1e-6))
    at_80 = 0.4780137 / 80 + 0.0685353 - 0.0006593 * 80 + 5.2e-06 * 80 ** 2
    at_90 = 0.4780137 / 90 + 0.0685353 - 0.0006593 * 90 + 5.2e-06 * 90 ** 2
    assert diesel.litres_per_km(np.array(80.0)) == pytest.approx(at_80, abs=2e-4)
    assert diesel.litres_per_km(np.array(90.0)) == pytest.approx(at_90, abs=2e-4)
    assert diesel.litres_per_km(np.array(90.0)) > diesel.litres_per_km(np.array(80.0))


def test_fuel_curves_are_held_at_their_published_speed_limits():
    diesel_van = load_fuels(2023)["van"][1]
    assert diesel_van.max_speed == 110
    assert diesel_van.litres_per_km(np.array(150.0)) == diesel_van.litres_per_km(np.array(110.0))


def test_electric_kilometres_save_nothing():
    shares = load_fuels(2023)["car"]
    # TAG's three shares add up to 0.999995
    assert sum(fuel.km_share for fuel in shares) == pytest.approx(1 - read_values()["km_share_electric_car_2023"], abs=1e-4)
    later = load_fuels(2035)["car"]
    assert sum(fuel.km_share for fuel in later) < sum(fuel.km_share for fuel in shares)


def test_running_costs_weight_each_fuel_by_its_share_of_kilometres():
    today = at_one_speed(90.0)
    fuels = load_fuels(2023)["car"]
    expected = sum(fuel.km_share * float(fuel.litres_per_km(np.array(90.0))) * fuel.resource for fuel in fuels)
    expected_co2 = sum(fuel.km_share * float(fuel.litres_per_km(np.array(90.0))) * fuel.co2_kg for fuel in fuels)
    running = running_per_km(today, "car", 2023)
    assert running["fuel"] == pytest.approx(expected)
    assert running["co2"] == pytest.approx(expected_co2)
    assert running["tax"] > 0


def test_slowing_from_90_to_80_saves_fuel_but_slowing_to_48_burns_more():
    def fuel_at(kmh):
        return running_per_km(at_one_speed(kmh), "car", 2023)["fuel"]
    assert fuel_at(80.0) < fuel_at(90.0)
    assert fuel_at(48.0) > fuel_at(80.0)


def test_vehicle_capital_is_charged_on_the_same_hours_as_time():
    values = load_values()
    assert values.car_capital_hour == pytest.approx(0.0955 * 1.893, abs=0.001)
    assert values.van_capital_hour == pytest.approx(0.6114 * 0.656, abs=0.001)
    assert capital_cost(1e-3, 0.0, 2.0, 1.0, values, 0.5) == pytest.approx(1e-3 * 2.0 * values.car_capital_hour * 1e9 * 0.5)
    assert values.carbon_tonne == pytest.approx(338.90, abs=0.01)   # TAG A3.4.1, 2023, central, market prices


@pytest.mark.skipif(not (RAW / "tag-data-book-v2-03.xlsm").exists(), reason="raw TAG data book not downloaded")
def test_values_file_matches_dfts_files():
    from ingest.extract_appraisal_values import RAS4001, TAG, ras4001_values, tag_values
    extracted = {row["value_id"]: row["value"] for row in tag_values(TAG) + ras4001_values(RAS4001)}
    with open(VALUES_FILE, newline="") as f:
        saved = {row["value_id"]: float(row["value"]) for row in csv.DictReader(f)}
    assert extracted == pytest.approx(saved)


def test_a_cap_adds_to_compliance_only_below_every_vehicles_limit():
    from model.appraisal import cap_beyond_compliance
    assert not any(cap_beyond_compliance(URBAN_32, cap) for cap in (48, 60, 80, 97))
    assert [cap_beyond_compliance(URBAN_48, cap) for cap in (48, 60, 80, 97)] == [False, False, False, False]
    assert [cap_beyond_compliance(RURAL_97, cap) for cap in (48, 60, 80, 97)] == [True, True, True, False]
    assert all(cap_beyond_compliance(MOTORWAY, cap) for cap in (48, 60, 80, 97))
    with pytest.raises(ValueError):
        cap_beyond_compliance(RURAL_97, 90)     # between the van's 80 km/h and the car's 97 km/h


def test_beyond_compliance_is_the_cap_minus_compliance_where_it_binds_and_nothing_elsewhere():
    import pandas as pd
    from analysis.rq3 import ADDITIVE, beyond_compliance
    rows = []
    for segment, policy, scale in ((RURAL_97.segment, "compliance", 1.0), (RURAL_97.segment, "cap_80", 3.0),
                                   (RURAL_97.segment, "cap_97", 0.5), (MOTORWAY.segment, "compliance", 2.0),
                                   (MOTORWAY.segment, "cap_97", 5.0)):
        row = {"segment": segment, "policy": policy, "free_flow_share": 0.7}
        row |= {column: scale * (i + 1) for i, column in enumerate(ADDITIVE)}
        if policy == "compliance":
            row |= {"time_cost": 0.0, "capital_cost": 0.0, "cost": 0.0}
        rows.append(row)
    table = pd.DataFrame(rows)
    out = beyond_compliance(table).set_index(["segment", "policy"])
    source = table.set_index(["segment", "policy"])
    for segment, cap in ((RURAL_97.segment, "cap_80"), (MOTORWAY.segment, "cap_97")):
        for column in ADDITIVE:
            assert out.loc[(segment, cap), column] == pytest.approx(
                source.loc[(segment, cap), column] - source.loc[(segment, "compliance"), column])
    assert all(out.loc[(RURAL_97.segment, "cap_97"), column] == 0 for column in ADDITIVE)
    r = out.loc[(RURAL_97.segment, "cap_80")]
    assert r["bcr"] == pytest.approx(r["benefit"] / r["cost"])
    assert r["net"] == pytest.approx(r["benefit"] - r["cost"])


def test_hours_against_today_are_the_hours_beyond_bau_plus_bau_isas_own():
    from analysis.rq3 import hours_table
    from model.scenarios import Traffic
    distributions = {road.segment: Traffic(uniform(0.6 * road.limit_kmh, 1.3 * road.limit_kmh),
                                           uniform(0.6 * road.limit_kmh, 1.2 * road.limit_kmh), 0.2) for road in ROAD_TYPES}
    columns = ["car_hours_per_km", "van_hours_per_km"]
    vs_bau = hours_table(distributions, count_speeding_time=True).set_index(["segment", "policy"])
    vs_today = hours_table(distributions, count_speeding_time=True, against="today").set_index(["segment", "policy"])
    for road in ROAD_TYPES:
        own = vs_today.loc[(road.segment, "bau_isa"), columns].to_numpy()
        assert vs_today.loc[(road.segment, "today"), columns].to_numpy() == pytest.approx([0, 0])
        assert np.all(own > 0)
        for policy in ("compliance", "cap_80", "cap_60"):
            assert vs_today.loc[(road.segment, policy), columns].to_numpy() == pytest.approx(
                vs_bau.loc[(road.segment, policy), columns].to_numpy() + own)


def test_override_shares_are_the_isa_uk_trial_shares_of_distance():
    # Lai and Carsten (2012), Table 5: share of distance driven with ISA overridden, 20, 30, 60 and 70 mph roads
    assert [road.override_share for road in ROAD_TYPES] == [0.133, 0.081, 0.013, 0.163]
