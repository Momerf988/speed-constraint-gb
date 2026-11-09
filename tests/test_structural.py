"""The structural range must report the lowest and highest result and what produced them."""

import pandas as pd
import pytest

from analysis.structural import headlines, number, structural_range
from tests.test_scenarios import CENTRAL, MOTORWAY, RURAL_97, URBAN_32, URBAN_48, TODAY_48, both, uniform


def test_the_range_keeps_the_central_value_and_names_the_extremes():
    central = {"Deaths": 100.0}
    variants = {"flat curve": {"Deaths": 40.0}, "steep curve": {"Deaths": 130.0}, "other": {"Deaths": 90.0}}
    row = structural_range(central, variants).iloc[0]
    assert (row["central"], row["lowest"], row["highest"]) == (100.0, 40.0, 130.0)
    assert (row["lowest_under"], row["highest_under"]) == ("flat curve", "steep curve")


def test_the_central_value_can_itself_be_an_extreme():
    row = structural_range({"Deaths": 100.0}, {"lower": {"Deaths": 50.0}}).iloc[0]
    assert (row["highest"], row["highest_under"]) == (100.0, "central")


def test_numbers_are_written_by_kind():
    assert number("ratio of a cap", 1.834) == "1.83"
    assert number("share of the target", 0.226) == "23%"
    assert number("Deaths", 1234.4) == "1,234"


def appraised_toy_run(model_column_factor: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    from analysis.rq3 import appraise, hours_table
    from model.appraisal import load_values
    from model.run import run
    distributions = {URBAN_32.segment: both(uniform(25, 45)), URBAN_48.segment: both(TODAY_48),
                     RURAL_97.segment: both(uniform(70, 110)), MOTORWAY.segment: both(uniform(90, 130))}
    baseline = pd.DataFrame({"killed": 100.0, "ksi": 1000.0, "casualties": 5000.0, "k_fatal": 0.9, "k_ksi": 0.9,
                             "k_casualties": 0.9, "k_slight": 0.9}, index=list(distributions))
    traffic = pd.DataFrame({"cars": 60.0, "vans": 15.0}, index=list(distributions))
    ratios = {"fatal": 0.9, "serious": 0.9, "slight": 0.75}
    table = appraise(run(distributions, baseline, CENTRAL), hours_table(distributions), traffic, baseline,
                     load_values(), ratios)
    return table, baseline


def test_headlines_add_up_flat_limiters_across_road_types():
    table, baseline = appraised_toy_run()
    result = headlines(table, baseline)
    cap = table[table["policy"] == "cap_80"]
    assert result["Flat 80 km/h limiter on all four road types, ratio"] == pytest.approx(cap["benefit"].sum() / cap["cost"].sum())
    compliance = table[table["policy"] == "compliance"]
    assert result["Compliance beyond ISA, deaths prevented"] == pytest.approx(compliance["killed_prevented_vs_bau"].sum())


def test_model_a_headlines_use_model_a():
    table, baseline = appraised_toy_run()
    power = headlines(table, baseline, "power")
    compliance = table[table["policy"] == "compliance"]
    assert power["Compliance beyond ISA, deaths prevented"] == pytest.approx(compliance["killed_prevented_power_vs_bau"].sum())
    cap = table[(table["policy"] == "cap_80") & (table["segment"] == RURAL_97.segment)].iloc[0]
    assert power["80 km/h cap on 97 km/h roads, ratio"] == pytest.approx(cap["benefit_power"] / cap["cost"])
