"""Phase 8, part B: does the severity model hold up on a later year? Train on 2021-2024, test on 2025.

Cross-validation mixes years; a model used in practice predicts crashes that have not happened
yet. So each model is fitted on 2021-2024 and scored once on 2025, which the tuning never saw:
no skill, logistic regression, XGBoost with default settings and XGBoost tuned. A bootstrap
that redraws whole 2025 collisions gives 95% intervals, including for the gaps between models.
The tuned model is then scored within groups: who was hurt, the speed limit and urban or rural,
because a good overall score can hide a group the model serves badly. Writes:
    results/severity_holdout.parquet      scores on 2025 with 95% intervals
    results/severity_subgroups.parquet    the tuned model's scores within each group
    figures/severity_subgroups.png

Usage:
    python -m analysis.severity_holdout
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.rq2_rq4 import GRID, INK, MUTED, SURFACE
from model.severity import (DEFAULT_PARAMS, bootstrap, by_group, calibration, fit, logistic_model, predict, prepare,
                            scores, tuned_params, xgboost_model)

ROOT = Path(__file__).resolve().parent.parent
MARTS = ROOT / "data" / "marts"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

TRAIN_YEARS, TEST_YEAR = (2021, 2024), 2025
BOOTSTRAP_DRAWS = 200
GAPS = (("XGBoost, tuned", "Logistic regression"), ("XGBoost, tuned", "XGBoost, default settings"))


def casualty_group(casualty_type: pd.Series) -> pd.Series:
    """DfT's 22 casualty types, grouped into the five that matter for severity."""
    groups = pd.Series("Other road user", index=casualty_type.index)
    groups[casualty_type.str.contains("Car occupant|Taxi", regex=True)] = "Car or taxi occupant"
    groups[casualty_type.str.contains("Motorcycle|motorcycle", regex=True)] = "Motorcyclist"
    groups[casualty_type == "Cyclist"] = "Cyclist"
    groups[casualty_type == "Pedestrian"] = "Pedestrian"
    return groups


def interval(draws: pd.Series) -> tuple[float, float]:
    return float(draws.quantile(0.025)), float(draws.quantile(0.975))


def chart(subgroups: dict[str, pd.DataFrame], path: Path) -> None:
    """Average precision within each group, against no skill (the group's KSI share)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.2, 1]})
    for ax, (title, table) in zip(axes, subgroups.items()):
        y = np.arange(len(table))[::-1]
        ax.barh(y, table["average_precision"], height=0.6, color="#2a78d6", label="Tuned XGBoost")
        ax.plot(table["ksi_share"], y, "|", ms=18, mew=2.5, color=INK, label="No skill (the group's KSI share)")
        for yi, (_, r) in zip(y, table.iterrows()):
            ax.text(r["average_precision"] + 0.01, yi, f"{r['lift']:.1f}× no skill", va="center", fontsize=8, color=MUTED)
        ax.set_yticks(y, table["group"], fontsize=8.5, color=MUTED)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Average precision on 2025", color=MUTED, fontsize=9)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8.5, labelcolor=MUTED)
    fig.suptitle("Severity model on a later year it never saw (trained 2021-2024, tested 2025)", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    table = pd.read_parquet(MARTS / "ml_casualty_severity.parquet").sort_values("casualty_key").reset_index(drop=True)
    features, label, groups = prepare(table)          # prepared together, so both years share category codes
    train = table["collision_year"].between(*TRAIN_YEARS).to_numpy()
    test = (table["collision_year"] == TEST_YEAR).to_numpy()
    x_train, y_train, x_test, y_test = features[train], label[train], features[test], label[test]

    predictions = {
        "No skill": np.full(int(test.sum()), y_train.mean()),
        "Logistic regression": predict(fit(logistic_model(features), x_train, y_train), x_test),
        "XGBoost, default settings": predict(fit(xgboost_model(DEFAULT_PARAMS), x_train, y_train), x_test),
        "XGBoost, tuned": predict(fit(xgboost_model(tuned_params()), x_train, y_train), x_test),
    }
    point = pd.DataFrame([{"model": name, **scores(y_test, p)} for name, p in predictions.items()]).set_index("model")
    draws = bootstrap(y_test, predictions, groups[test], BOOTSTRAP_DRAWS).pivot_table(
        index="draw", columns="model", values="average_precision")
    for name in predictions:
        point.loc[name, ["ap_low", "ap_high"]] = interval(draws[name])
    tuned = predictions["XGBoost, tuned"]
    checked = calibration(y_test, tuned)

    subgroups = {
        "Who was hurt": by_group(y_test, tuned, casualty_group(table.loc[test, "casualty_type"])),
        "Speed limit": by_group(y_test, tuned, table.loc[test, "speed_limit_kmh"].map(lambda v: f"{v:g} km/h")),
        "Urban or rural": by_group(y_test, tuned, table.loc[test, "urban_rural"].where(
            table.loc[test, "urban_rural"].isin(["Urban", "Rural"]))),
    }
    subgroups["Speed limit"] = subgroups["Speed limit"].sort_values(
        "group", key=lambda g: g.str.split().str[0].astype(float)).reset_index(drop=True)

    point.reset_index().to_parquet(RESULTS / "severity_holdout.parquet", index=False, compression="zstd")
    pd.concat([t.assign(split_by=name) for name, t in subgroups.items()]).to_parquet(
        RESULTS / "severity_subgroups.parquet", index=False, compression="zstd")
    chart({k: subgroups[k] for k in ("Who was hurt", "Speed limit")}, FIGURES / "severity_subgroups.png")

    print(f"Phase 8, part B: the severity model on a later year it never saw")
    print(f"Trained on {TRAIN_YEARS[0]}-{TRAIN_YEARS[1]} ({int(train.sum()):,} casualties), tested on {TEST_YEAR} "
          f"({int(test.sum()):,} casualties, KSI share {y_test.mean():.1%}); 95% intervals from {BOOTSTRAP_DRAWS} "
          f"bootstrap draws of whole collisions\n")
    print(f"  {'':28s}{'average precision':>18s}{'95% interval':>17s}{'ROC AUC':>10s}{'log loss':>10s}")
    for name, r in point.iterrows():
        print(f"  {name:28s}{r['average_precision']:18.3f}{r['ap_low']:10.3f} to {r['ap_high']:.3f}"
              f"{r['roc_auc']:10.3f}{r['log_loss']:10.3f}")
    for better, worse in GAPS:
        low, high = interval(draws[better] - draws[worse])
        print(f"  {better} minus {worse.lower()}: {point.loc[better, 'average_precision'] - point.loc[worse, 'average_precision']:+.3f} "
              f"average precision (95% interval {low:+.3f} to {high:+.3f})")
    print(f"\n  Calibration on {TEST_YEAR}: mean predicted KSI probability {tuned.mean():.3f}, observed {y_test.mean():.3f}; "
          f"across tenths of the predictions they differ by at most {(checked['predicted'] - checked['observed']).abs().max():.3f}")
    for name, sub in subgroups.items():
        print(f"\n  {name}:")
        print(f"    {'':24s}{'casualties':>11s}{'KSI share':>11s}{'avg precision':>15s}{'lift':>7s}{'ROC AUC':>9s}"
              f"{'mean predicted':>16s}")
        for _, r in sub.iterrows():
            print(f"    {r['group']:24s}{r['casualties']:11,d}{r['ksi_share']:11.3f}{r['average_precision']:15.3f}"
                  f"{r['lift']:7.2f}{r['roc_auc']:9.3f}{r['mean_predicted']:16.3f}")
    print("\n  Lift = average precision / KSI share: how many times better than no skill within the group.")
    print(f"\nWrote results/severity_holdout.parquet, results/severity_subgroups.parquet and figures/severity_subgroups.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
