"""Speed-risk curves from the literature, as plain functions of speed in km/h.

Two families:
  Aggregate models link a change in a road's MEAN speed to a change in casualties
  (Elvik et al. 2019; Elvik 2013).
  Individual models link ONE driver's speed to that driver's relative risk of a casualty
  crash (Kloeden et al. 2001, 2002; Brenac et al. 2015).
The functions only do the arithmetic. Which curve applies where, and whether a scenario
lies inside a curve's fitted range, is decided by the scenario code (Batch 5).
All functions accept single numbers or NumPy arrays.
"""

from __future__ import annotations

import numpy as np


def power_model(speed_after, speed_before, exponent):
    """Aggregate Power Model: casualties after / before = (mean speed after / before) ** exponent."""
    return (np.asarray(speed_after, dtype=float) / np.asarray(speed_before, dtype=float)) ** exponent


def exponential_model(speed_change_kmh, coefficient):
    """Aggregate Exponential model: casualties after / before = exp(coefficient * change in mean km/h)."""
    return np.exp(coefficient * np.asarray(speed_change_kmh, dtype=float))


def quadratic_relative_risk(dv_kmh, a, b):
    """Individual risk curve exp(a*dv + b*dv^2), with dv in km/h.

    Kloeden et al. (2001, 2002) write their curves this way. What dv is measured from differs:
    the urban absolute curve uses dv = speed - 60; the speed-difference curves use
    dv = speed - mean speed of the other traffic at the site.
    """
    dv = np.asarray(dv_kmh, dtype=float)
    return np.exp(a * dv + b * dv**2)


def intercept_form_relative_risk(speed_kmh, c, a, b):
    """Individual risk curve exp(c + a*V + b*V^2), with V the speed in km/h.

    This is how Kloeden et al. (2002) print their absolute curves, including the
    sensitivity versions in Appendix B.
    """
    v = np.asarray(speed_kmh, dtype=float)
    return np.exp(c + a * v + b * v**2)