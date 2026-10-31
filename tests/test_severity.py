"""Tests for the severity model's data preparation, grouping, scores and SHAP values, on made-up data."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from model.severity import (CATEGORICAL, FEATURES, LEFT_OUT, NUMERIC, expand, fit, folds, logistic_model, predict,
                            prepare, scores, shap_values, xgboost_model)


def made_up_table(n: int = 600, seed: int = 1) -> pd.DataFrame:
    """Casualties with every model field; KSI probability rises with speed limit and for pedestrians."""
    rng = np.random.default_rng(seed)
    table = pd.DataFrame({
        "casualty_key": [f"k{i:05d}" for i in range(n)][::-1],          # deliberately out of order
        "collision_index": [f"c{i // 2:05d}" for i in range(n)],        # two casualties per collision
    })
    for column in CATEGORICAL:
        table[column] = rng.integers(1, 4, n)
    for column in NUMERIC:
        table[column] = rng.normal(40, 10, n)
    table["speed_limit_kmh"] = rng.choice([32, 48, 64, 97], n).astype(float)
    table["number_of_casualties"] = 2                                   # a left-out field that must not be used
    logit = -2 + 0.02 * table["speed_limit_kmh"] + 1.0 * (table["casualty_type"] == 1)
    table["ksi_adjusted"] = 1 / (1 + np.exp(-logit))
    return table


def test_prepare_uses_only_the_listed_fields_in_a_fixed_order():
    features, label, groups = prepare(made_up_table())
    assert list(features.columns) == FEATURES
    assert "number_of_casualties" not in features.columns
    assert not any(field.split(",")[0] in features.columns for field in LEFT_OUT)
    assert label.between(0, 1).all()
    assert (features[CATEGORICAL].dtypes == "category").all()


def test_folds_never_split_a_collision():
    _, _, groups = prepare(made_up_table())
    for train, test in folds(groups):
        assert set(groups.iloc[train]).isdisjoint(groups.iloc[test])


def test_scores_with_hard_labels_match_the_usual_average_precision():
    label = pd.Series([1.0, 0.0, 1.0, 0.0, 0.0])
    prediction = np.array([0.9, 0.2, 0.6, 0.7, 0.1])
    assert scores(label, prediction)["average_precision"] == pytest.approx(average_precision_score(label, prediction))


def test_a_soft_label_counts_as_partly_ksi_and_partly_not():
    y, p, w = expand(pd.Series([0.25]), np.array([0.5]))
    assert list(y) == [1, 0] and list(w) == [0.25, 0.75] and list(p) == [0.5, 0.5]


def test_both_models_fit_probability_labels_and_learn_the_signal():
    features, label, _ = prepare(made_up_table(2000))
    for model in (xgboost_model(), logistic_model(features)):
        prediction = predict(fit(model, features, label), features)
        assert ((prediction > 0) & (prediction < 1)).all()
        assert np.corrcoef(prediction, label)[0, 1] > 0.5


def test_shap_values_add_up_to_each_prediction():
    features, label, _ = prepare(made_up_table())
    model = fit(xgboost_model(), features, label)
    contributions = shap_values(model, features)
    margin = contributions.sum(axis=1).to_numpy()
    assert 1 / (1 + np.exp(-margin)) == pytest.approx(predict(model, features), abs=1e-5)
