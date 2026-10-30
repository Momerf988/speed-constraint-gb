"""The parameters file must be well formed, and the guardrail must refuse unverified inputs."""

import pytest

from model.params import load_parameters, require


def test_file_loads_and_every_filled_row_parses():
    parameters = load_parameters()
    assert len(parameters) == 27
    for p in parameters.values():
        if p.status != "to_extract":
            assert len(p.values) >= 1


def test_require_returns_primary_rows():
    p = require("power_exponent_fatalities", "kloeden2001_rural_difference")
    assert p["power_exponent_fatalities"].value == 5.5
    assert p["kloeden2001_rural_difference"].values == (0.07039, 0.0008617)


def test_require_refuses_secondary_rows():
    # The Kloeden 1997 table is a reference, quoted via CR 207: the model may not use it.
    with pytest.raises(ValueError, match="not verified"):
        require("kloeden1997_urban60_rr_65")


def test_require_refuses_unknown_ids():
    with pytest.raises(ValueError, match="not in the file"):
        require("no_such_parameter")


def test_single_value_accessor_rejects_curves():
    with pytest.raises(ValueError, match="use .values"):
        _ = require("kloeden2002_urban_difference")["kloeden2002_urban_difference"].value