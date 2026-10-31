"""Phase 8, part B: tune the severity model's settings on 2021-2024, leaving 2025 untouched.

A random search tries 20 candidate settings. Each is scored by 3-fold cross-validation grouped
by collision, and each fit stops early once its validation fold's log loss stops improving. The
default settings are scored the same way, for comparison. The winner (lowest mean log loss) is
written to model/parameters/severity_xgboost.json, which analysis/severity.py and
analysis/severity_holdout.py then use. Takes about 10 minutes. Writes:
    model/parameters/severity_xgboost.json   the chosen settings
    results/severity_tuning.parquet          every candidate and its scores

Usage:
    python -m analysis.severity_tuning
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from model.severity import (DEFAULT_PARAMS, EARLY_STOPPING, MAX_TREES, SEED, TUNED_FILE, TUNING_FOLDS,
                            TUNING_TRIALS, fit, folds, predict, prepare, scores, tune, xgboost_model)

ROOT = Path(__file__).resolve().parent.parent
MARTS = ROOT / "data" / "marts"
RESULTS = ROOT / "results"
TRAIN_YEARS = (2021, 2024)
SEARCHED = ["max_depth", "learning_rate", "min_child_weight", "subsample", "colsample_bytree", "reg_lambda"]


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    table = pd.read_parquet(MARTS / "ml_casualty_severity.parquet")
    table = table[table["collision_year"].between(*TRAIN_YEARS)]
    features, label, groups = prepare(table)

    splits = folds(groups, TUNING_FOLDS)
    default = pd.DataFrame([
        scores(label.iloc[test], predict(fit(xgboost_model(), features.iloc[train], label.iloc[train]), features.iloc[test]))
        for train, test in splits
    ]).mean()
    trials = tune(features, label, groups)
    best = trials.loc[trials["log_loss"].idxmin()]
    chosen = {name: (int(best[name]) if name in ("max_depth", "min_child_weight") else round(float(best[name]), 4))
              for name in SEARCHED}
    chosen["n_estimators"] = int(best["trees"])

    TUNED_FILE.write_text(json.dumps({
        "selected_by": f"lowest mean log loss, {TUNING_FOLDS}-fold cross-validation grouped by collision, "
                       f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[1]} only; {TUNING_TRIALS} random candidates, seed {SEED}; "
                       f"each fit stopped after {EARLY_STOPPING} trees without improvement (at most {MAX_TREES})",
        "params": chosen,
    }, indent=2) + "\n")
    trials.to_parquet(RESULTS / "severity_tuning.parquet", index=False, compression="zstd")

    print(f"Phase 8, part B: tuning the severity model on {TRAIN_YEARS[0]}-{TRAIN_YEARS[1]} "
          f"({len(label):,} casualties; 2025 kept back as the final test)")
    print(f"{TUNING_TRIALS} random candidates, {TUNING_FOLDS}-fold cross-validation grouped by collision, "
          f"each fit stopped early\n")
    print(f"  {'':10s}{'depth':>6s}{'learn rate':>11s}{'min child':>10s}{'subsample':>10s}{'columns':>8s}"
          f"{'lambda':>8s}{'trees':>7s}{'log loss':>10s}{'avg precision':>15s}")
    for _, r in trials.sort_values("log_loss").head(5).iterrows():
        print(f"  trial {int(r['trial']):<4d}{int(r['max_depth']):6d}{r['learning_rate']:11.3f}{int(r['min_child_weight']):10d}"
              f"{r['subsample']:10.2f}{r['colsample_bytree']:8.2f}{r['reg_lambda']:8.2f}{int(r['trees']):7d}"
              f"{r['log_loss']:10.4f}{r['average_precision']:15.4f}")
    print(f"  {'defaults':10s}{DEFAULT_PARAMS['max_depth']:6d}{DEFAULT_PARAMS['learning_rate']:11.3f}"
          f"{DEFAULT_PARAMS['min_child_weight']:10d}{DEFAULT_PARAMS['subsample']:10.2f}"
          f"{DEFAULT_PARAMS['colsample_bytree']:8.2f}{1.0:8.2f}{DEFAULT_PARAMS['n_estimators']:7d}"
          f"{default['log_loss']:10.4f}{default['average_precision']:15.4f}")
    print(f"\n  Chosen: trial {int(best['trial'])}. Against the defaults: log loss "
          f"{best['log_loss'] - default['log_loss']:+.4f}, average precision "
          f"{best['average_precision'] - default['average_precision']:+.4f}")
    print(f"\nWrote {TUNED_FILE.relative_to(ROOT)} and results/severity_tuning.parquet ({len(trials)} candidates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
