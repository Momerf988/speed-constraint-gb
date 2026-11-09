"""Phase 8, part B: what makes a road casualty killed or seriously injured, once a crash has happened.

Trains XGBoost, with the settings tuned by analysis/severity_tuning.py, on every casualty from
2021 to 2025 (data/marts/ml_casualty_severity), judged by 5-fold cross-validation grouped by
collision against two benchmarks: no skill (everyone gets the average KSI probability) and
logistic regression on the same fields. The spread across folds shows how stable each score is.
Then fits one model on all the data and explains it with SHAP values. The test on a later year
is analysis/severity_holdout.py.

How the odds of KSI change with the speed limit, against 48 km/h roads, is given two ways that do
hold the other recorded fields fixed: a partial-dependence contrast from XGBoost (every casualty's
limit set to each value in turn, other fields as recorded, predicted log-odds averaged), and the
adjusted odds ratio from the logistic regression. Both are associations within crashes that
happened, not the effect of changing a limit. (A mean SHAP value by limit is not such a contrast,
so it is no longer reported as one.) Writes:
    results/severity_scores.parquet       scores of each model in each fold
    results/severity_shap.parquet         mean absolute SHAP value per field
    results/severity_speed_limit.parquet  odds of KSI by speed limit: partial dependence, logistic, raw
    results/severity_xgboost.json         the final model
    figures/severity_precision_recall.png
    figures/severity_drivers.png

Usage:
    python -m analysis.severity
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from analysis.rq2_rq4 import GRID, INK, MUTED, SURFACE
from model.severity import (FEATURES, LEFT_OUT, NUMERIC, SEED, calibration, expand, fit, folds, logistic_model,
                            out_of_fold, prepare, scores, shap_values, tuned_params, xgboost_model)

ROOT = Path(__file__).resolve().parent.parent
MARTS = ROOT / "data" / "marts"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

SHAP_SAMPLE = 50_000
TOP_FIELDS = 12
COLOURS = {"XGBoost": "#2a78d6", "Logistic regression": "#1baf7a", "No skill": "#8a8985"}


def chart_precision_recall(label, predictions: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for name, prediction in predictions.items():
        y, p, w = expand(label, prediction)
        precision, recall, _ = precision_recall_curve(y, p, sample_weight=w)
        ax.plot(recall, precision, color=COLOURS[name], lw=2, label=name)
    ax.axhline(label.mean(), color=COLOURS["No skill"], lw=1.5, ls="--", label=f"No skill ({label.mean():.0%} KSI)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall: share of KSI casualties flagged", color=MUTED, fontsize=9)
    ax.set_ylabel("Precision: share of flagged casualties that are KSI", color=MUTED, fontsize=9)
    ax.set_title("Predicting KSI from what is known at the scene (out of fold, 2021-2025)", loc="left", fontsize=10.5, color=INK)
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_drivers(importance: pd.Series, speed: pd.DataFrame, path: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 5), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.3, 1]})
    top = importance.head(TOP_FIELDS)[::-1]
    left.barh(range(len(top)), top.values, color=COLOURS["XGBoost"], height=0.6)
    left.set_yticks(range(len(top)), [name.replace("_", " ") for name in top.index], fontsize=8.5, color=MUTED)
    left.set_xlabel("Mean |SHAP value| (log-odds of KSI)", color=MUTED, fontsize=9)
    left.set_title("Which fields move the prediction most", loc="left", fontsize=10, color=INK)
    right.plot(speed["speed_limit_kmh"], speed["odds_ratio_pd"], "o-", color=COLOURS["XGBoost"], lw=2,
               label="XGBoost, other fields held (partial dependence)")
    right.plot(speed["speed_limit_kmh"], speed["odds_ratio_logistic"], "^-", color=COLOURS["Logistic regression"], lw=1.5,
               label="Logistic regression, adjusted")
    right.plot(speed["speed_limit_kmh"], speed["odds_ratio_raw"], "s--", color=COLOURS["No skill"], lw=1.5,
               label="Raw, other fields not held")
    right.axhline(1, color=MUTED, lw=1)
    right.set_xticks(speed["speed_limit_kmh"], [f"{v:g}" for v in speed["speed_limit_kmh"]])
    right.set_xlabel("Speed limit (km/h)", color=MUTED, fontsize=9)
    right.set_ylabel("Odds of KSI relative to 48 km/h roads", color=MUTED, fontsize=9)
    right.set_title("Speed limit and the odds of KSI", loc="left", fontsize=10, color=INK)
    right.legend(frameon=False, fontsize=8.5, labelcolor=MUTED)
    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x" if ax is left else "y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(fig)


def limit_contrasts(xgboost, logistic, sample: pd.DataFrame, limits, reference: float = 48.0) -> pd.DataFrame:
    """Odds of KSI at each speed limit relative to the reference, with the other recorded fields held as they are.

    XGBoost: set every casualty's limit to each value, average the predicted log-odds (a partial-
    dependence contrast). Logistic regression: its adjusted odds ratio, from the one coefficient on the
    standardised limit.
    """
    margin = {limit: xgboost.predict(sample.assign(speed_limit_kmh=limit), output_margin=True).mean() for limit in limits}
    steps = dict(logistic.steps)
    numeric = steps["columntransformer"].named_transformers_["numeric"]
    scale = numeric.named_steps["standardscaler"].scale_[NUMERIC.index("speed_limit_kmh")]
    per_kmh = steps["logisticregression"].coef_[0][NUMERIC.index("speed_limit_kmh")] / scale
    return pd.DataFrame({"speed_limit_kmh": list(limits),
                         "odds_ratio_pd": [np.exp(margin[limit] - margin[reference]) for limit in limits],
                         "odds_ratio_logistic": [np.exp(per_kmh * (limit - reference)) for limit in limits]})


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    table = pd.read_parquet(MARTS / "ml_casualty_severity.parquet")
    features, label, groups = prepare(table)
    splits = folds(groups)
    params = tuned_params()

    predictions = {
        "No skill": np.concatenate([np.full(len(test), label.iloc[train].mean()) for train, test in splits])[
            np.argsort(np.concatenate([test for _, test in splits]))],
        "Logistic regression": out_of_fold(lambda: logistic_model(features), features, label, splits),
        "XGBoost": out_of_fold(lambda: xgboost_model(params), features, label, splits),
    }
    results = pd.DataFrame([{"model": name, **scores(label, p)} for name, p in predictions.items()])
    per_fold = pd.DataFrame([{"model": name, "fold": i, **scores(label.iloc[test], p[test])}
                             for name, p in predictions.items() for i, (_, test) in enumerate(splits)])
    spread = per_fold.groupby("model", sort=False)[["average_precision", "roc_auc", "log_loss"]].std()
    gain = (per_fold[per_fold["model"] == "XGBoost"].set_index("fold")["average_precision"]
            - per_fold[per_fold["model"] == "Logistic regression"].set_index("fold")["average_precision"])
    calibrated = calibration(label, predictions["XGBoost"])

    final = fit(xgboost_model(params), features, label)
    final.save_model(RESULTS / "severity_xgboost.json")
    sample = features.iloc[np.sort(np.random.default_rng(SEED).choice(len(features), SHAP_SAMPLE, replace=False))]
    contributions = shap_values(final, sample)
    importance = contributions[FEATURES].abs().mean().sort_values(ascending=False)

    by_limit = sample.assign(ksi=label.loc[sample.index])
    speed = by_limit.groupby("speed_limit_kmh").agg(ksi_share=("ksi", "mean")).reset_index()
    reference = speed.set_index("speed_limit_kmh").loc[48.0]
    odds = speed["ksi_share"] / (1 - speed["ksi_share"])
    speed["odds_ratio_raw"] = odds / (reference["ksi_share"] / (1 - reference["ksi_share"]))
    logistic = fit(logistic_model(features), features, label)
    speed = speed.merge(limit_contrasts(final, logistic, sample, speed["speed_limit_kmh"].tolist()), on="speed_limit_kmh")
    speed.to_parquet(RESULTS / "severity_speed_limit.parquet", index=False, compression="zstd")

    per_fold.to_parquet(RESULTS / "severity_scores.parquet", index=False, compression="zstd")
    importance.rename("mean_abs_shap").rename_axis("field").reset_index().to_parquet(
        RESULTS / "severity_shap.parquet", index=False, compression="zstd")
    chart_precision_recall(label, {k: v for k, v in predictions.items() if k != "No skill"},
                           FIGURES / "severity_precision_recall.png")
    chart_drivers(importance, speed, FIGURES / "severity_drivers.png")

    print(f"Phase 8, part B: what makes a casualty killed or seriously injured, once a crash has happened")
    print(f"{len(label):,} casualties in {groups.nunique():,} collisions, 2021-2025; label = DfT's adjusted KSI probability "
          f"(mean {label.mean():.1%}); {len(FEATURES)} fields recorded at the scene")
    print(f"XGBoost settings tuned on 2021-2024 (model/parameters/severity_xgboost.json): "
          f"{params['n_estimators']} trees, depth {params['max_depth']}, learning rate {params['learning_rate']}")
    print(f"5-fold cross-validation grouped by collision: no crash is in both training and test data\n")
    print(f"  {'':22s}{'average precision':>22s}{'ROC AUC':>16s}{'log loss':>16s}   (out of fold, ± spread across folds)")
    for _, r in results.iterrows():
        s = spread.loc[r["model"]]
        print(f"  {r['model']:22s}{r['average_precision']:13.3f} ± {s['average_precision']:.3f}"
              f"{r['roc_auc']:10.3f} ± {s['roc_auc']:.3f}{r['log_loss']:10.3f} ± {s['log_loss']:.3f}")
    print("  Average precision is the area under the precision-recall curve; with no skill it equals the KSI share.")
    print(f"  XGBoost beat logistic regression in {int((gain > 0).sum())} of {len(gain)} folds, by "
          f"{gain.min():.3f} to {gain.max():.3f} in average precision.")
    gap = (calibrated["predicted"] - calibrated["observed"]).abs().max()
    print(f"\n  Calibration (XGBoost, out of fold): across tenths of the predictions, predicted and observed KSI "
          f"probability differ by at most {gap:.3f}")
    print(f"\nFields that move the prediction most (mean |SHAP|, log-odds, {SHAP_SAMPLE:,} casualties):")
    for field, value in importance.head(TOP_FIELDS).items():
        print(f"  {field:24s}{value:.3f}")
    print("\nSpeed limit and the odds of KSI, relative to 48 km/h roads, other recorded fields held as they are:")
    print(f"  {'limit (km/h)':14s}{'XGBoost, partial dependence':>30s}{'logistic, adjusted':>21s}{'raw':>8s}")
    for _, r in speed.iterrows():
        print(f"  {r['speed_limit_kmh']:<14g}{r['odds_ratio_pd']:30.2f}{r['odds_ratio_logistic']:21.2f}{r['odds_ratio_raw']:8.2f}")
    print("  Partial dependence: every casualty's limit set to each value in turn, the other fields as recorded, predicted")
    print("  log-odds averaged. Both columns are associations within crashes that happened, not the effect of a limit.")
    print("\nFields left out, and why:")
    for field, reason in LEFT_OUT.items():
        print(f"  {field}: {reason}")
    print("\nSHAP values explain the model, not the world: related fields (casualty type, vehicle type, casualty class)")
    print("share credit between them, and the speed-limit pattern is an association, not the effect of changing a limit.")
    print("\nScope: this shows which crashes end up more severe. It does not estimate what a speed cap would do,")
    print("because a cap also changes which crashes happen at all.")
    print(f"\nWrote results/severity_scores.parquet, severity_shap.parquet, severity_xgboost.json and two charts to figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
