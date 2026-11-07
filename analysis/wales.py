"""Phase 8, part A: did Wales's default 32 km/h (20 mph) limit cut casualties? A difference-in-differences study.

Wales made 32 km/h the default on its restricted roads on 17 September 2023. This compares
monthly casualties on Welsh 32 and 48 km/h roads with the same roads in England, before and
after, police force by police force (data/marts/agg_casualties_force_month):
    before   January 2022 to August 2023, the 20 months just before the switch
    after    October 2023 to December 2025 (September 2023, the month of the switch, left out)
2020-2021 are left out throughout: the two countries had different COVID rules. Scotland is left
out: it brought in its own 32 km/h limits in 2024-2025. All casualties is the main outcome,
because changes in severity recording cannot move it; adjusted KSI is the second.

The before window was chosen after looking at the event study, which shows Wales falling relative
to England between 2019 and 2022. So the main estimate is one reading, a level shift after a
stable 2022-2023, and two others are reported beside it: a model in which Wales's earlier relative
fall carries on (a Wales-specific trend fitted on 2017-2019 and 2022-2023), and a before window that
includes 2017-2019. Further checks: leaving out each Welsh force; a synthetic control from English
forces with placebo-in-space inference; a triple difference against other Welsh roads, with all
roads together for the net change; and an exact permutation test over every set of four English
forces, and over those of about the size of Wales.

Two English forces have incomplete records inside the windows, by the Department for Transport's
own account ("Road casualty statistics: known data issues", updated 29 May 2025): Staffordshire
under-reported collisions from 2017 to 2023, and Avon and Somerset changed recording systems in 2022.
Staffordshire's casualties fall to about a third of their 2017-2019 level in 2022 and triple by 2024,
while England's barely move. Both stay in the main estimate, as in the first draft, and every check
is repeated without them. The same page notes that some Welsh collisions in 2023 were recorded at
30 mph where the council's limit was 20 mph; grouping the two limits makes the estimate immune to that.
Writes results/wales_did.parquet, results/wales_results.json, figures/wales_event_study.png and
figures/wales_synthetic_control.png.

Usage:
    python -m analysis.wales
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.rq2_rq4 import GRID, INK, MUTED, SURFACE
from model.causal import (ddd_poisson, did_poisson, did_poisson_trend, event_study, implied_speed_change_kmh,
                          permutation_test, ratio_of_ratios, size_matched_permutation, synthetic_placebo)
from model.params import require

ROOT = Path(__file__).resolve().parent.parent
MARTS = ROOT / "data" / "marts"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

BEFORE = ("2022-01-01", "2023-08-01")
EARLIER = ("2017-01-01", "2019-12-01")   # the longer "before" window, sensitivity test only
AFTER = ("2023-10-01", "2025-12-01")
TREATED_ROADS = "32 or 48 km/h roads"
REFERENCE_QUARTER = "2023Q2"          # the last full quarter before the switch
PERMUTATIONS = 2000
SIZE_TOLERANCE = 0.5                  # size-matched permutation: before-period total within 50% of Wales's
FLAGGED_BY_DFT = ("Staffordshire", "Avon and Somerset")   # incomplete records in the windows: see the docstring
SPEED_COEFFICIENTS = (0.034, 0.06, 0.067)   # Exponential model, injury crashes: Elvik (2013), Elvik et al. (2019) central and highest
SEED = 20261101
SERIES = {"casualties": ("All casualties", "#2a78d6"), "ksi_adjusted": ("KSI (adjusted)", "#1baf7a")}


def in_window(months: pd.Series, window: tuple[str, str]) -> pd.Series:
    return months.between(pd.Timestamp(window[0]), pd.Timestamp(window[1]))


def load_panel(marts_dir, road_group: str, include_earlier: bool = False) -> pd.DataFrame:
    """English and Welsh forces, one road group, before and after windows only."""
    panel = pd.read_parquet(marts_dir / "agg_casualties_force_month.parquet")
    panel["month"] = pd.to_datetime(panel["month"])
    before = in_window(panel["month"], BEFORE)
    if include_earlier:
        before |= in_window(panel["month"], EARLIER)
    after = in_window(panel["month"], AFTER)
    panel = panel[panel["country"].isin(["England", "Wales"]) & (panel["road_group"] == road_group) & (before | after)]
    panel = panel.assign(wales=panel["country"] == "Wales", after=after[panel.index],
                         quarter=panel["month"].dt.to_period("Q").astype(str),
                         casualties=panel["casualties"].astype(float), ksi_adjusted=panel["ksi_adjusted"].astype(float))
    return panel.sort_values(["month", "police_force_code"]).reset_index(drop=True)


def both_groups(marts_dir) -> pd.DataFrame:
    """The main windows for both road groups, for the triple difference and the net change on all roads."""
    return pd.concat([load_panel(marts_dir, group) for group in (TREATED_ROADS, "other roads")], ignore_index=True)


def all_roads(panel: pd.DataFrame) -> pd.DataFrame:
    """Both road groups added together, per force and month."""
    keys = ["police_force_code", "month", "wales", "after"]
    return panel.groupby(keys, as_index=False)[["casualties", "ksi_adjusted"]].sum()


def monthly_means(panel: pd.DataFrame) -> dict:
    """Average casualties a month, before and after, by country and road group (Table 1 of the paper)."""
    rows = {}
    for (group, wales, after), part in panel.groupby(["road_group", "wales", "after"]):
        months = part["month"].nunique()
        key = f"{group} | {'Wales' if wales else 'England'} | {'after' if after else 'before'}"
        rows[key] = {"casualties": part["casualties"].sum() / months, "ksi_adjusted": part["ksi_adjusted"].sum() / months}
    return rows


def synthetic(panel: pd.DataFrame, outcome: str) -> tuple[dict, pd.DataFrame]:
    """Synthetic control for Wales as one unit (its four forces added up), from the English forces."""
    series = panel.pivot_table(index="month", columns="police_force_code", values=outcome, aggfunc="sum")
    welsh = sorted(panel.loc[panel["wales"], "police_force_code"].unique())
    english = sorted(panel.loc[~panel["wales"], "police_force_code"].unique())
    series = series[english].assign(Wales=series[welsh].sum(axis=1))
    pre = in_window(series.index.to_series(), BEFORE).to_numpy()
    return synthetic_placebo(series, "Wales", english, pre), series


def p_text(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def recording_check(marts_dir, force_code) -> dict:
    """A force's yearly casualties on all roads, against its 2017-2019 average, beside England's."""
    panel = pd.read_parquet(marts_dir / "agg_casualties_force_month.parquet")
    panel = panel.assign(year=pd.to_datetime(panel["month"]).dt.year)
    england = panel[panel["country"] == "England"].groupby("year")["casualties"].sum()
    force = panel[panel["police_force_code"] == force_code].groupby("year")["casualties"].sum()
    relative = {name: series / series[[2017, 2018, 2019]].mean() for name, series in (("force", force), ("england", england))}
    return {name: {int(y): float(v) for y, v in series.items() if y >= 2017} for name, series in relative.items()}


def inference(panel: pd.DataFrame) -> dict:
    """The main estimate, the exact and size-matched permutation tests and the synthetic control, on one panel."""
    welsh = sorted(panel.loc[panel["wales"], "police_force_code"].unique())
    english = sorted(panel.loc[~panel["wales"], "police_force_code"].unique())
    totals = unit_totals(panel, "casualties")
    synth, series = synthetic(panel, "casualties")
    return {"did": did_poisson(panel, "casualties", "wales", "after"),
            "exact": size_matched_permutation(totals, welsh, english, tolerance=None),
            "matched": size_matched_permutation(totals, welsh, english, tolerance=SIZE_TOLERANCE),
            "synthetic": synth, "series": series}


def unit_totals(panel: pd.DataFrame, outcome: str) -> pd.DataFrame:
    totals = panel.pivot_table(index="police_force_code", columns="after", values=outcome, aggfunc="sum")
    return totals.rename(columns={False: "before", True: "after"})


def estimate(label: str, road_group: str, outcome: str, include_earlier: bool = False) -> dict:
    panel = load_panel(MARTS, road_group, include_earlier)
    did = did_poisson(panel, outcome, treated="wales", post="after")
    totals = unit_totals(panel, outcome)
    welsh = sorted(panel.loc[panel["wales"], "police_force_code"].unique())
    english = sorted(panel.loc[~panel["wales"], "police_force_code"].unique())
    perm = permutation_test(totals, welsh, english, PERMUTATIONS, SEED)
    return {"estimate": label, "road_group": road_group, "outcome": outcome, "include_earlier": include_earlier,
            **did, "ratio_of_ratios": perm["ratio_of_ratios"], "permutation_p": perm["p_value"],
            "welsh_forces": len(welsh), "english_forces": len(english), "months": panel["month"].nunique()}


def chart_synthetic(result: dict, series: pd.DataFrame, path: Path) -> None:
    """Wales and its synthetic control, each indexed to its own 2022-2023 average (= 100)."""
    months = series.index
    fig, ax = plt.subplots(figsize=(10, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(months, result["synthetic"], color=MUTED, lw=1.6, ls="--", label="Synthetic Wales (weighted English forces)")
    ax.plot(months, result["actual"], color="#2a78d6", lw=2, label="Wales")
    switch = pd.Timestamp("2023-09-17")
    ax.axvline(switch, color=INK, lw=1, ls="--")
    ax.text(switch, ax.get_ylim()[1], " 32 km/h default", fontsize=8, color=INK, va="top")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_ylabel("Casualties a month, index (2022 to Aug 2023 = 100)", color=MUTED, fontsize=9)
    ax.set_title("Casualties on 32 and 48 km/h roads: Wales against a synthetic control built from English forces",
                 loc="left", fontsize=10.5, color=INK)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(fig)


def chart(studies: dict[str, pd.DataFrame], path: Path) -> None:
    """The Welsh rate ratio per quarter against England, relative to 2023 Q2, for each outcome."""
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    quarters = sorted(set().union(*(s["quarter"] for s in studies.values())))
    position = {q: i for i, q in enumerate(quarters)}
    for offset, (outcome, study) in zip((-0.12, 0.12), studies.items()):
        label, colour = SERIES[outcome]
        x = study["quarter"].map(position) + offset
        ax.vlines(x, study["ci_low"], study["ci_high"], color=colour, lw=1.5, alpha=0.6)
        ax.plot(x, study["rate_ratio"], "o", ms=5, color=colour, label=label)
    switch = position["2023Q3"] + 0.5
    ax.axvspan(position["2022Q1"] - 0.5, switch, color=GRID, alpha=0.5, lw=0)
    ax.text(position["2022Q1"] - 0.3, 0.62, "'before' in the main estimate", fontsize=8, color=MUTED)
    gap = position["2022Q1"] - 0.5
    ax.axvline(gap, color=MUTED, lw=1, ls=":")
    ax.text(gap - 0.2, 0.62, "2020-21\nleft out", fontsize=8, color=MUTED, ha="right")
    ax.axvline(switch, color=INK, lw=1, ls="--")
    ax.text(switch + 0.2, ax.get_ylim()[1] * 0.97, "32 km/h default\n17 Sep 2023", fontsize=8, color=INK, va="top")
    ax.axhline(1, color=MUTED, lw=1)
    ticks = [q for q in quarters if q.endswith("Q1")]
    ax.set_xticks([position[q] for q in ticks], [q[:4] for q in ticks], fontsize=9, color=MUTED)
    ax.tick_params(axis="y", colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_ylabel(f"Wales vs England, relative to {REFERENCE_QUARTER}", color=MUTED, fontsize=9)
    ax.set_title("Casualties on 32 and 48 km/h roads: Wales against England, quarter by quarter (2020-21 left out)",
                 loc="left", fontsize=10.5, color=INK)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=MUTED)
    fig.text(0.01, 0.01, "Dot: Poisson rate ratio with force and month fixed effects. Line: 95% interval, "
             "clustered by police force (only 4 Welsh forces, so treat as indicative).", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    rows = [
        estimate("All casualties, 32 and 48 km/h roads (main)", TREATED_ROADS, "casualties"),
        estimate("KSI (adjusted), 32 and 48 km/h roads", TREATED_ROADS, "ksi_adjusted"),
        estimate("All casualties, before including 2017-2019", TREATED_ROADS, "casualties", include_earlier=True),
        estimate("Placebo: all casualties, other roads", "other roads", "casualties"),
        estimate("Placebo: KSI (adjusted), other roads", "other roads", "ksi_adjusted"),
    ]
    results = pd.DataFrame(rows)
    history = load_panel(MARTS, TREATED_ROADS, include_earlier=True)
    studies = {outcome: event_study(history, outcome, "wales", "quarter", REFERENCE_QUARTER) for outcome in SERIES}
    results.to_parquet(RESULTS / "wales_did.parquet", index=False, compression="zstd")
    chart(studies, FIGURES / "wales_event_study.png")

    main_row = results.iloc[0]
    coefficient = require("exponential_coef_injury_crashes")["exponential_coef_injury_crashes"].value
    quarterly = studies["casualties"]
    recent = quarterly[(quarterly["quarter"] >= "2022Q1") & (quarterly["quarter"] < REFERENCE_QUARTER)]
    earlier = quarterly[quarterly["quarter"] < "2020Q1"]
    print("Phase 8, part A: Wales's default 32 km/h limit (17 September 2023), difference-in-differences against England")
    print(f"Before: Jan 2022 to Aug 2023; after: Oct 2023 to Dec 2025. "
          f"{int(main_row['welsh_forces'])} Welsh and {int(main_row['english_forces'])} English police forces, "
          f"{int(main_row['months'])} months.\n")
    print(f"  {'':46s}{'rate ratio':>11s}{'95% interval':>16s}{'p, clustered':>14s}{'ratio of ratios':>17s}{'p, permutation':>16s}")
    for _, r in results.iterrows():
        permutation = "<0.001" if r["permutation_p"] == 0 else f"{r['permutation_p']:.3f}"
        print(f"  {r['estimate']:46s}{r['rate_ratio']:11.3f}{r['ci_low']:9.3f} to {r['ci_high']:.3f}"
              f"{r['p_value']:12.3f}{r['ratio_of_ratios']:17.3f}{permutation:>16s}")
    print(f"\n  Rate ratio: casualties in Wales after the switch, relative to what England's change implies "
          f"(0.80 = 20% fewer).")
    print(f"  Permutation p: share of {PERMUTATIONS:,} random sets of 4 English forces with a change at least as large.")
    print(f"\n  Parallel trends (event study, relative to {REFERENCE_QUARTER}): in 2022-2023 the quarterly Welsh gap stayed "
          f"between {recent['rate_ratio'].min():.2f} and {recent['rate_ratio'].max():.2f}; in 2017-2019 it sat higher, "
          f"{earlier['rate_ratio'].min():.2f} to {earlier['rate_ratio'].max():.2f}.")
    print("  So Wales was already falling relative to England between 2019 and 2022, before the policy: the estimate")
    print("  that adds 2017-2019 to 'before' credits the policy with that older fall, and is shown only as a sensitivity.")
    print(f"  Under the Exponential model ({coefficient} per km/h for injury crashes), a rate ratio of "
          f"{main_row['rate_ratio']:.2f} needs mean speeds about "
          f"{-implied_speed_change_kmh(main_row['rate_ratio'], coefficient):.1f} km/h lower on these roads.")

    # ---------- robustness: other readings of the before period, and checks on the design ----------
    history = history.assign(years=(history["month"] - pd.Timestamp("2017-01-01")).dt.days / 365.25)
    trend = {outcome: did_poisson_trend(history, outcome, "wales", "after", "years") for outcome in SERIES}
    panel = load_panel(MARTS, TREATED_ROADS)
    welsh = sorted(panel.loc[panel["wales"], "police_force_code"].unique())
    english = sorted(panel.loc[~panel["wales"], "police_force_code"].unique())
    leave_out = {force: did_poisson(panel[panel["police_force_code"] != force], "casualties", "wales", "after")
                 for force in welsh}
    forces = pd.read_parquet(MARTS / "dim_police_force.parquet").set_index("police_force_code")["police_force"]
    flagged = [forces[forces == name].index[0] for name in FLAGGED_BY_DFT]
    with_all = inference(panel)
    without_break = inference(panel[~panel["police_force_code"].isin(flagged)])
    exact, matched, synth = with_all["exact"], with_all["matched"], with_all["synthetic"]
    groups = both_groups(MARTS)
    triple = {outcome: ddd_poisson(groups, outcome, "wales", "after", "road_group", TREATED_ROADS) for outcome in SERIES}
    net = {outcome: did_poisson(all_roads(groups), outcome, "wales", "after") for outcome in SERIES}
    chart_synthetic(synth, with_all["series"], FIGURES / "wales_synthetic_control.png")
    check = recording_check(MARTS, flagged[0])
    means = monthly_means(groups)
    other = results.iloc[3]
    low_after = means[f"{TREATED_ROADS} | Wales | after"]["casualties"]
    other_after = means["other roads | Wales | after"]["casualties"]
    change_low = low_after - low_after / main_row["rate_ratio"]
    change_other = other_after - other_after / other["rate_ratio"]
    speed = {c: -implied_speed_change_kmh(main_row["rate_ratio"], c) for c in SPEED_COEFFICIENTS}
    speed_ci = (-implied_speed_change_kmh(main_row["ci_high"], coefficient), -implied_speed_change_kmh(main_row["ci_low"], coefficient))
    post = quarterly[quarterly["quarter"] > "2023Q3"]

    print("\nOther readings and checks: all casualties on 32 and 48 km/h roads unless stated")
    print(f"  {'':62s}{'rate ratio':>11s}{'95% interval':>16s}")
    for outcome, (label, _) in SERIES.items():
        r = trend[outcome]
        print(f"  {'Wales keeps its 2017-2023 trend against England: ' + label:62s}{r['rate_ratio']:11.3f}"
              f"{r['ci_low']:9.3f} to {r['ci_high']:.3f}   (Welsh trend {100 * (r['trend_per_year'] - 1):+.1f}% a year)")
    for outcome, (label, _) in SERIES.items():
        r = triple[outcome]
        print(f"  {'Triple difference, against other Welsh roads: ' + label:62s}{r['rate_ratio']:11.3f}"
              f"{r['ci_low']:9.3f} to {r['ci_high']:.3f}")
    for outcome, (label, _) in SERIES.items():
        r = net[outcome]
        print(f"  {'All roads together, the net change: ' + label:62s}{r['rate_ratio']:11.3f}"
              f"{r['ci_low']:9.3f} to {r['ci_high']:.3f}")
    for force, r in leave_out.items():
        print(f"  {'Leaving out ' + forces.get(force, str(force)):62s}{r['rate_ratio']:11.3f}{r['ci_low']:9.3f} to {r['ci_high']:.3f}")
    top = sorted(synth["weights"].items(), key=lambda kv: -kv[1])[:5]
    print(f"  {'Synthetic control: Wales / synthetic Wales, after':62s}{synth['ratio']:11.3f}"
          f"   (gap {synth['gap_post']:+.1f} index points; placebo-in-space rank {synth['rank']} of {synth['units']}, "
          f"p = {synth['p_value']:.3f})")
    print("    weights: " + ", ".join(f"{forces.get(f, str(f))} {w:.2f}" for f, w in top if w >= 0.005)
          + f"; pre-period error {synth['rmspe_pre']:.1f}, after {synth['rmspe_post']:.1f} index points")
    print(f"  Exact permutation over all {exact['sets']:,} sets of 4 English forces: p = {exact['p_value']:.3f}; "
          f"over the {matched['sets']:,} sets within {SIZE_TOLERANCE:.0%} of Wales's size: p = {matched['p_value']:.3f}")
    w = without_break
    print(f"\n  {FLAGGED_BY_DFT[0]}'s casualties on all roads against its 2017-2019 average: "
          + ", ".join(f"{y} {check['force'][y]:.2f}" for y in (2021, 2022, 2023, 2024, 2025))
          + "; England: " + ", ".join(f"{y} {check['england'][y]:.2f}" for y in (2021, 2022, 2023, 2024, 2025)))
    print(f"  Without the forces DfT flags ({', '.join(FLAGGED_BY_DFT)}): rate ratio {w['did']['rate_ratio']:.3f} "
          f"({w['did']['ci_low']:.3f} to {w['did']['ci_high']:.3f}); exact permutation p {p_text(w['exact']['p_value'])} over "
          f"{w['exact']['sets']:,} sets, size-matched p {p_text(w['matched']['p_value'])} over {w['matched']['sets']:,}; "
          f"synthetic control {w['synthetic']['ratio']:.3f}, "
          f"rank {w['synthetic']['rank']} of {w['synthetic']['units']} (p = {w['synthetic']['p_value']:.3f})")
    print(f"\n  Casualties a month in Wales after the switch: {low_after:.1f} on 32 and 48 km/h roads, about "
          f"{-change_low:.0f} fewer than England's change implies; {other_after:.1f} on other roads, about {change_other:+.0f}.")
    print(f"  After the switch the quarterly Welsh gap ran {post['rate_ratio'].min():.2f} to {post['rate_ratio'].max():.2f}.")
    print("  Speed needed under the Exponential model, by coefficient: "
          + ", ".join(f"{c}: {v:.1f} km/h" for c, v in speed.items())
          + f"; at {coefficient}, the 95% interval of the rate ratio gives {speed_ci[0]:.1f} to {speed_ci[1]:.1f} km/h.")

    summary = {
        "main": results.to_dict("records"), "trend": trend, "triple": triple, "net": net,
        "leave_out": {forces.get(f, str(f)): r for f, r in leave_out.items()},
        "synthetic": {k: v for k, v in synth.items() if k not in ("actual", "synthetic", "placebo_ratios")}
                     | {"weights": {forces.get(f, str(f)): w for f, w in synth["weights"].items() if w >= 0.005}},
        "exact_permutation": exact, "size_matched_permutation": matched | {"tolerance": SIZE_TOLERANCE},
        "recording_break": {"forces": list(FLAGGED_BY_DFT), "relative": check,
                            "without": {"did": without_break["did"], "exact": without_break["exact"],
                                        "matched": without_break["matched"],
                                        "synthetic": {k: v for k, v in without_break["synthetic"].items()
                                                      if k in ("ratio", "gap_post", "rmspe_pre", "rmspe_post", "rank", "units", "p_value")}}},
        "monthly_means": means, "change_a_month": {"focus": change_low, "other": change_other},
        "speed_kmh": {str(c): v for c, v in speed.items()}, "speed_ci_kmh": speed_ci, "speed_coefficient": coefficient,
        "event_study": {"recent": [recent["rate_ratio"].min(), recent["rate_ratio"].max()],
                        "earlier": [earlier["rate_ratio"].min(), earlier["rate_ratio"].max()],
                        "after": [post["rate_ratio"].min(), post["rate_ratio"].max()]},
        "forces": {"welsh": len(welsh), "english": len(english)}, "months": int(main_row["months"]),
    }
    (RESULTS / "wales_results.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"\nWrote {len(results)} rows to results/wales_did.parquet, results/wales_results.json and two charts to figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
