"""What makes a road casualty killed or seriously injured, once a crash has happened.

An XGBoost model (gradient-boosted decision trees) learns each casualty's probability of being
killed or seriously injured (KSI) from fields recorded at the scene. The label is DfT's
adjusted KSI probability rather than the police's recorded severity, so the switch of forces
to injury-based recording does not distort it. XGBoost's logistic objective accepts labels
between 0 and 1 directly (through its regressor interface: the classifier insists on 0/1).

Every check keeps the people from one collision together: cross-validation is grouped by
collision, so a crash never sits in both the training and the test data. Settings are tuned by
a random search on 2021-2024 only, stopping each fit early on its validation fold, so 2025
stays untouched as a final test of a later year. SHAP values (from XGBoost's own TreeSHAP)
split each prediction into one contribution per field. They explain the model, not the world,
and related fields (casualty type, vehicle type, casualty class) share credit between them.

Scope: this describes which crashes and casualties end up more severe. It does not estimate
what a speed cap would do, because a cap also changes which crashes happen at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 20261101
FOLDS = 5
TUNED_FILE = Path(__file__).resolve().parent / "parameters" / "severity_xgboost.json"
TUNING_TRIALS = 20
TUNING_FOLDS = 3
MAX_TREES = 2000
EARLY_STOPPING = 50

CATEGORICAL = [
    "casualty_class", "casualty_type", "sex", "vehicle_type", "vehicle_manoeuvre", "first_point_of_impact",
    "road_type", "first_road_class", "urban_rural", "junction_detail", "light_conditions",
    "weather_conditions", "road_surface",
]
NUMERIC = [
    "age", "vehicle_age", "engine_capacity_cc", "driver_age", "number_of_vehicles", "speed_limit_kmh",
    "hour_of_day", "day_of_week", "car_involved", "van_involved", "hgv_involved", "bus_involved",
    "motorcycle_involved", "pedal_cycle_involved",
]
FEATURES = CATEGORICAL + NUMERIC

# Fields deliberately not used, and why: each would let the model learn the outcome, or how a
# police force records outcomes, rather than what makes a crash severe.
LEFT_OUT = {
    "casualty_severity, collision_severity, adjusted probabilities": "the outcome itself",
    "number_of_casualties": "part of the crash's outcome: more people hurt makes a KSI among them likelier by count alone",
    "police_force_code": "reflects how a force records injuries, not how severe the crash was",
    "contributory factors": "judged by the police after the crash, partly from how bad it was",
    "did_police_officer_attend_scene": "police attend serious crashes more often, so it follows the outcome (not loaded)",
    "latitude, longitude, local authority": "place stands in for local recording practice; too fine to generalise",
}

DEFAULT_PARAMS = {                   # a sensible starting point, before tuning
    "objective": "binary:logistic",
    "tree_method": "hist",
    "enable_categorical": True,
    "max_cat_to_onehot": 1,
    "n_estimators": 400,
    "learning_rate": 0.08,
    "max_depth": 6,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
}


def prepare(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Features, soft label and collision groups, in a fixed row order (by casualty)."""
    table = table.sort_values("casualty_key").reset_index(drop=True)
    features = pd.DataFrame(index=table.index)
    for column in CATEGORICAL:
        values = table[column]
        if pd.api.types.is_numeric_dtype(values):     # codes arrive as floats when a column has gaps: 9.0 -> "9"
            values = values.astype("Int64")
        features[column] = values.astype("string").astype("category")
    for column in NUMERIC:
        features[column] = pd.to_numeric(table[column].astype("Float64"), errors="coerce").astype(float)
    return features[FEATURES], table["ksi_adjusted"].astype(float), table["collision_index"]


def expand(label: pd.Series, prediction: np.ndarray, weight: np.ndarray | None = None):
    """A soft label p becomes two weighted rows: a KSI with weight p and a non-KSI with weight 1 - p.

    That lets standard scores for yes/no outcomes (precision-recall, ROC) use probability labels.
    An optional weight per casualty (e.g. how often a bootstrap drew it) multiplies both rows.
    """
    n = len(label)
    times = np.ones(n) if weight is None else np.asarray(weight, dtype=float)
    y = np.concatenate([np.ones(n), np.zeros(n)])
    w = np.concatenate([label.to_numpy() * times, (1 - label.to_numpy()) * times])
    return y, np.concatenate([prediction, prediction]), w


def scores(label: pd.Series, prediction: np.ndarray, weight: np.ndarray | None = None) -> dict[str, float]:
    """Average precision (area under the precision-recall curve), ROC AUC and log loss."""
    y, p, w = expand(label, prediction, weight)
    return {
        "average_precision": float(average_precision_score(y, p, sample_weight=w)),
        "roc_auc": float(roc_auc_score(y, p, sample_weight=w)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), sample_weight=w)),
    }


def folds(groups: pd.Series, n_splits: int = FOLDS):
    """Train and test row positions, with every collision wholly on one side."""
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))


def xgboost_model(params: dict = DEFAULT_PARAMS) -> xgb.XGBRegressor:
    """With the binary:logistic objective, the regressor predicts probabilities from probability labels."""
    return xgb.XGBRegressor(**params)


def tuned_params(path: Path = TUNED_FILE) -> dict:
    """The settings chosen by analysis/severity_tuning.py, on top of the defaults."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `make tune` first")
    return {**DEFAULT_PARAMS, **json.loads(path.read_text())["params"]}


def sample_params(rng: np.random.Generator) -> dict:
    """One random candidate from the search space."""
    return {
        "max_depth": int(rng.choice([4, 6, 8, 10])),
        "learning_rate": float(np.exp(rng.uniform(np.log(0.03), np.log(0.2)))),
        "min_child_weight": int(rng.choice([1, 5, 20, 50, 100])),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.5, 1.0)),
        "reg_lambda": float(np.exp(rng.uniform(np.log(0.1), np.log(10)))),
    }


def tune(features: pd.DataFrame, label: pd.Series, groups: pd.Series, trials: int = TUNING_TRIALS,
         n_folds: int = TUNING_FOLDS, seed: int = SEED, max_trees: int = MAX_TREES) -> pd.DataFrame:
    """Random search: each candidate is scored by grouped cross-validation, each fit stopped early.

    A fit stops when its validation fold's log loss has not improved for EARLY_STOPPING trees;
    the candidate's tree count is the average stopping point. The best candidate is the one with
    the lowest mean log loss, which rewards honest probabilities, not just a good ranking.
    """
    rng = np.random.default_rng(seed)
    splits = folds(groups, n_folds)
    rows = []
    for trial in range(trials):
        candidate = sample_params(rng)
        results = []
        for train, test in splits:
            model = xgb.XGBRegressor(**{**DEFAULT_PARAMS, **candidate, "n_estimators": max_trees,
                                        "early_stopping_rounds": EARLY_STOPPING, "eval_metric": "logloss"})
            model.fit(features.iloc[train], label.iloc[train],
                      eval_set=[(features.iloc[test], label.iloc[test])], verbose=False)
            prediction = model.predict(features.iloc[test], iteration_range=(0, model.best_iteration + 1))
            results.append({**scores(label.iloc[test], prediction), "trees": model.best_iteration + 1})
        mean = pd.DataFrame(results).mean()
        rows.append({"trial": trial, **candidate, "log_loss": mean["log_loss"],
                     "average_precision": mean["average_precision"], "trees": int(round(mean["trees"]))})
    return pd.DataFrame(rows)


def logistic_model(features: pd.DataFrame):
    """The benchmark: logistic regression on the same fields, categories one-hot encoded."""
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical = make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                                OneHotEncoder(handle_unknown="ignore", min_frequency=50))
    encode = ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)])
    return make_pipeline(encode, LogisticRegression(max_iter=2000))


def fit(model, features: pd.DataFrame, label: pd.Series):
    """Fit a classifier to soft labels: XGBoost takes them directly; logistic regression via weighted rows."""
    if isinstance(model, xgb.XGBModel):
        return model.fit(features, label)
    doubled = pd.concat([features, features], ignore_index=True)
    y = np.concatenate([np.ones(len(label)), np.zeros(len(label))])
    weight = np.concatenate([label.to_numpy(), 1 - label.to_numpy()])
    categorical = doubled[CATEGORICAL].astype(object).where(doubled[CATEGORICAL].notna(), None)
    return model.fit(doubled.assign(**categorical), y, logisticregression__sample_weight=weight)


def predict(model, features: pd.DataFrame) -> np.ndarray:
    if isinstance(model, xgb.XGBModel):
        return model.predict(features)
    categorical = features[CATEGORICAL].astype(object).where(features[CATEGORICAL].notna(), None)
    return model.predict_proba(features.assign(**categorical))[:, 1]


def out_of_fold(make_model, features: pd.DataFrame, label: pd.Series, splits) -> np.ndarray:
    """Each casualty's prediction from a model that never saw its collision."""
    prediction = np.empty(len(label))
    for train, test in splits:
        model = fit(make_model(), features.iloc[train], label.iloc[train])
        prediction[test] = predict(model, features.iloc[test])
    return prediction


def shap_values(model: xgb.XGBRegressor, features: pd.DataFrame) -> pd.DataFrame:
    """One contribution per field and casualty, on the log-odds scale; with the bias they add up to the prediction."""
    matrix = xgb.DMatrix(features, enable_categorical=True)
    contributions = model.get_booster().predict(matrix, pred_contribs=True)
    return pd.DataFrame(contributions, columns=FEATURES + ["bias"], index=features.index)


def bootstrap(label: pd.Series, predictions: dict[str, np.ndarray], groups: pd.Series, draws: int,
              seed: int = SEED) -> pd.DataFrame:
    """Scores of each model on resampled test sets: whole collisions are drawn with replacement.

    Drawing collisions, not casualties, keeps the people from one crash together, as the folds do.
    Returns one row per draw and model.
    """
    codes, uniques = pd.factorize(groups)
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(draws):
        times = rng.multinomial(len(uniques), np.full(len(uniques), 1 / len(uniques)))[codes]
        for name, prediction in predictions.items():
            rows.append({"draw": draw, "model": name, **scores(label, prediction, times)})
    return pd.DataFrame(rows)


def by_group(label: pd.Series, prediction: np.ndarray, group: pd.Series) -> pd.DataFrame:
    """Scores within each group. Lift = average precision / KSI share: how far above no skill."""
    rows = []
    for name in group.dropna().unique():
        inside = (group == name).to_numpy()
        share = label[inside].mean()
        result = scores(label[inside], prediction[inside])
        rows.append({"group": name, "casualties": int(inside.sum()), "ksi_share": share,
                     "average_precision": result["average_precision"], "lift": result["average_precision"] / share,
                     "roc_auc": result["roc_auc"], "mean_predicted": prediction[inside].mean()})
    return pd.DataFrame(rows).sort_values("casualties", ascending=False).reset_index(drop=True)


def calibration(label: pd.Series, prediction: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Mean predicted against mean observed KSI probability, in tenths of the predictions."""
    decile = pd.qcut(prediction, bins, labels=False, duplicates="drop")
    return (pd.DataFrame({"decile": decile, "predicted": prediction, "observed": label.to_numpy()})
            .groupby("decile").mean().reset_index())
