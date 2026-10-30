"""The risk functions, fed the parameters file, must reproduce the sources' own published numbers.

This is reconciliation for model code: the same idea as the dbt tests that reproduce DfT's
totals. If a coefficient is mistyped in the CSV, or a formula is wrong, these tests fail.
"""

import numpy as np
import pytest

from model.params import require
from model.risk import exponential_model, intercept_form_relative_risk, power_model, quadratic_relative_risk


def test_kloeden2002_urban_absolute_reproduces_table_2_2():
    a, b = require("kloeden2002_urban_absolute")["kloeden2002_urban_absolute"].values
    speeds = np.array([65, 70, 75, 80])
    published = [1.82, 3.57, 7.63, 17.66]  # CR 207, Table 2.2
    assert np.round(quadratic_relative_risk(speeds - 60, a, b), 2).tolist() == published


def test_absolute_curve_rewrite_matches_the_printed_equation():
    # CR 207 prints the curve as exp(-0.822957835 - 0.083680149 V + 0.001623269 V^2).
    # The parameters file rewrites it in dv = V - 60. Both forms must agree at every speed.
    a, b = require("kloeden2002_urban_absolute")["kloeden2002_urban_absolute"].values
    speeds = np.arange(40, 91)
    printed = intercept_form_relative_risk(speeds, -0.822957835, -0.083680149, 0.001623269)
    rewritten = quadratic_relative_risk(speeds - 60, a, b)
    assert np.allclose(printed, rewritten, rtol=1e-5)


def test_kloeden2002_urban_difference_reproduces_table_2_3():
    a, b = require("kloeden2002_urban_difference")["kloeden2002_urban_difference"].values
    assert np.round(quadratic_relative_risk([10, 20], a, b), 2).tolist() == [4.12, 29.77]  # Table 2.3


def test_kloeden2001_rural_matches_the_summary_page():
    a, b = require("kloeden2001_rural_difference")["kloeden2001_rural_difference"].values
    at_10, at_20 = quadratic_relative_risk([10, 20], a, b)
    assert 2.0 < at_10 < 2.5   # CR 204 summary: "more than twice" the risk at +10 km/h
    assert 5.5 < at_20 < 6.0   # "nearly six times" at +20 km/h


@pytest.mark.parametrize("pid", ["kloeden2002_urban_absolute_se2_5", "kloeden2002_urban_absolute_se7_5"])
def test_sensitivity_curves_give_relative_risk_one_at_60(pid):
    c, a, b = require(pid)[pid].values
    assert intercept_form_relative_risk(60, c, a, b) == pytest.approx(1.0, abs=1e-4)


def test_brenac2015_exponential_at_plus_10():
    coefficient = require("brenac2015_exponential_coef")["brenac2015_exponential_coef"].value
    assert round(float(exponential_model(10, coefficient)), 2) == 1.67


def test_elvik2019_exponential_matches_european_commission_220_percent():
    coefficient = require("exponential_coef_fatalities")["exponential_coef_fatalities"].value
    # European Commission (2021), Figure 1: +10 km/h gives about 220% of the fatal crash risk
    assert round(float(exponential_model(10, coefficient)), 1) == 2.2


def test_elvik2019_power_model_ten_percent_slower():
    exponent = require("power_exponent_fatalities")["power_exponent_fatalities"].value
    assert float(power_model(0.9, 1.0, exponent)) == pytest.approx(0.560, abs=0.001)  # 44% fewer deaths


def test_power_and_exponential_cross_near_74_kmh_for_a_10_kmh_cut():
    # The two aggregate models disagree on which roads gain most. For a 10 km/h cut in mean
    # speed they predict the same change at about 74 km/h: below it the Power Model predicts
    # the bigger cut, above it the Exponential model does. This is why RQ4 runs under both.
    p = require("power_exponent_fatalities", "exponential_coef_fatalities")
    exponent, coefficient = p["power_exponent_fatalities"].value, p["exponential_coef_fatalities"].value
    speeds = np.arange(40.0, 120.0, 0.1)
    gap = power_model(speeds - 10, speeds, exponent) - exponential_model(-10, coefficient)
    crossing = speeds[np.argmin(np.abs(gap))]
    assert 73 < crossing < 75
    assert power_model(50, 60, exponent) < exponential_model(-10, coefficient)    # 60 km/h: Power cuts more
    assert power_model(90, 100, exponent) > exponential_model(-10, coefficient)   # 100 km/h: Exponential cuts more