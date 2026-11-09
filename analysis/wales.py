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
while England's barely move. So both are left out of the main estimate and of every test (the main
sample), and the estimate with them is a sensitivity. The rule comes from DfT's published list, but it
was applied after the first draft's results were seen; it weakens some results and strengthens others.
The same page lists Greater Manchester's change of system in 2021, before the windows, with totals
"not significantly different", and discrepancies raised in 2025 that are still being examined; those
forces stay in. It also notes that some Welsh collisions in 2023 were recorded at 30 mph where the
council's limit was 20 mph; grouping the two limits makes the estimate immune to that.

Dyfed-Powys moved to injury-based severity reporting in May 2023 (Office for Statistics Regulation;
Senedd Research), which changes how severity is recorded but not how many casualties are. So every
KSI result leaves Dyfed-Powys out, with it put back as a sensitivity; all-casualty results keep it,
with "without Dyfed-Powys" as a named sensitivity.

Inference. Clustered intervals treat the four Welsh forces' shocks as independent, and so do the
permutation tests over sets of English forces. Wales's forces share one government, one camera
partnership and one set of rules, so the test that treats Wales as one unit (single_unit_placebo)
is the main test, and the others are secondary.
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
                          permutation_test, ratio_of_ratios, single_unit_placebo, size_matched_permutation,
                          synthetic_placebo)
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
SEVERITY_BREAK = "Dyfed-Powys"                              # injury-based severity reporting from May 2023
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


def force_codes(marts_dir, names) -> list:
    forces = pd.read_parquet(marts_dir / "dim_police_force.parquet").set_index("police_force_code")["police_force"]
    return [forces[forces == name].index[0] for name in names]


def main_sample(panel: pd.DataFrame, outcome: str, marts_dir=None) -> pd.DataFrame:
    """The main sample: without the forces DfT flags, and for KSI also without Dyfed-Powys."""
    marts_dir = MARTS if marts_dir is None else marts_dir
    dropped = force_codes(marts_dir, FLAGGED_BY_DFT)
    if outcome == "ksi_adjusted":
        dropped += force_codes(marts_dir, [SEVERITY_BREAK])
    return panel[~panel["police_force_code"].isin(dropped)]


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


def inference(panel: pd.DataFrame, outcome: str = "casualties") -> dict:
    """The estimate, the single-unit placebo, the permutation tests and the synthetic control, on one panel."""
    welsh = sorted(panel.loc[panel["wales"], "police_force_code"].unique())
    english = sorted(panel.loc[~panel["wales"], "police_force_code"].unique())
    totals = unit_totals(panel, outcome)
    synth, series = synthetic(panel, outcome)
    return {"did": did_poisson(panel, outcome, "wales", "after"),
            "single_unit": single_unit_placebo(panel, outcome, "wales", "after", tolerance=SIZE_TOLERANCE),
            "exact": size_matched_permutation(totals, welsh, english, tolerance=None),
            "matched": size_matched_permutation(totals, welsh, english, tolerance=SIZE_TOLERANCE),
            "synthetic": synth, "series": series}


def donor_contributions(synth: dict, series: pd.DataFrame) -> dict:
    """Each donor's weight, after-period index (before = 100) and what it adds to synthetic Wales's index."""
    pre = in_window(series.index.to_series(), BEFORE).to_numpy()
    index = series / series[pre].mean() * 100
    return {force: {"weight": w, "index_after": float(index.loc[~pre, force].mean()),
                    "adds": float(w * (index.loc[~pre, force].mean() - 100))}
            for force, w in synth["weights"].items() if w >= 0.005}


def unit_totals(panel: pd.DataFrame, outcome: str) -> pd.DataFrame:
    totals = panel.pivot_table(index="police_force_code", columns="after", values=outcome, aggfunc="sum")
    return totals.rename(columns={False: "before", True: "after"})


def estimate(label: str, panel: pd.DataFrame, outcome: str, include_earlier: bool = False) -> dict:
    road_group = panel["road_group"].iloc[0]
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
    fig.text(0.01, 0.01, "Dot: Poisson rate ratio with force and month fixed effects. Line: 95% interval clustered by police "
             "force, which treats the Welsh forces as independent: indicative only. KSI leaves out Dyfed-Powys.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    forces = pd.read_parquet(MARTS / "dim_police_force.parquet").set_index("police_force_code")["police_force"]
    flagged = force_codes(MARTS, FLAGGED_BY_DFT)
    severity_break = force_codes(MARTS, [SEVERITY_BREAK])
    treated, other = load_panel(MARTS, TREATED_ROADS), load_panel(MARTS, "other roads")
    earlier = load_panel(MARTS, TREATED_ROADS, include_earlier=True)

    def without(panel, codes):
        return panel[~panel["police_force_code"].isin(codes)]

    main_casualties, main_ksi = main_sample(treated, "casualties"), main_sample(treated, "ksi_adjusted")
    rows = [
        estimate("All casualties, 32 and 48 km/h roads (main)", main_casualties, "casualties"),
        estimate("KSI (adjusted), 32 and 48 km/h roads, without Dyfed-Powys", main_ksi, "ksi_adjusted"),
        estimate("All casualties, before including 2017-2019", main_sample(earlier, "casualties"), "casualties", True),
        estimate("Placebo: all casualties, other roads", main_sample(other, "casualties"), "casualties"),
        estimate("Placebo: KSI (adjusted), other roads, without Dyfed-Powys", main_sample(other, "ksi_adjusted"), "ksi_adjusted"),
        estimate("Sensitivity: all casualties, every force", treated, "casualties"),
        estimate("Sensitivity: all casualties, without Dyfed-Powys", without(main_casualties, severity_break), "casualties"),
        estimate("Sensitivity: KSI, with Dyfed-Powys", without(treated, flagged), "ksi_adjusted"),
        estimate("Sensitivity: placebo, all casualties, without Dyfed-Powys",
                 without(main_sample(other, "casualties"), severity_break), "casualties"),
        estimate("Sensitivity: placebo, KSI, with Dyfed-Powys", without(other, flagged), "ksi_adjusted"),
    ]
    results = pd.DataFrame(rows)
    history = {outcome: main_sample(earlier, outcome) for outcome in SERIES}
    studies = {outcome: event_study(history[outcome], outcome, "wales", "quarter", REFERENCE_QUARTER) for outcome in SERIES}
    results.to_parquet(RESULTS / "wales_did.parquet", index=False, compression="zstd")
    chart(studies, FIGURES / "wales_event_study.png")

    main_row = results.iloc[0]
    coefficient = require("exponential_coef_injury_crashes")["exponential_coef_injury_crashes"].value
    quarterly = studies["casualties"]
    recent = quarterly[(quarterly["quarter"] >= "2022Q1") & (quarterly["quarter"] < REFERENCE_QUARTER)]
    before_covid = quarterly[quarterly["quarter"] < "2020Q1"]
    print("Phase 8, part A: Wales's default 32 km/h limit (17 September 2023), difference-in-differences against England")
    print(f"Before: Jan 2022 to Aug 2023; after: Oct 2023 to Dec 2025. Main sample: without the forces DfT flags "
          f"({', '.join(FLAGGED_BY_DFT)}), and for KSI without {SEVERITY_BREAK}.")
    print(f"{int(main_row['welsh_forces'])} Welsh and {int(main_row['english_forces'])} English police forces, "
          f"{int(main_row['months'])} months.\n")
    print(f"  {'':58s}{'rate ratio':>11s}{'95% interval*':>17s}{'p*':>8s}{'ratio of ratios':>17s}{'p, sets*':>10s}")
    for _, r in results.iterrows():
        permutation = "<0.001" if r["permutation_p"] == 0 else f"{r['permutation_p']:.3f}"
        print(f"  {r['estimate']:58s}{r['rate_ratio']:11.3f}{r['ci_low']:9.3f} to {r['ci_high']:.3f}"
              f"{r['p_value']:8.3f}{r['ratio_of_ratios']:17.3f}{permutation:>10s}")
    print("\n  Rate ratio: casualties in Wales after the switch, relative to what England's change implies (0.80 = 20% fewer).")
    print(f"  * Clustered by police force, and permutation over {PERMUTATIONS:,} random sets of English forces: both treat the")
    print("    Welsh forces' shocks as independent, which they are not (one government, one camera partnership), so")
    print("    both overstate certainty. The single-unit test below treats Wales as one place.")
    print(f"\n  Parallel trends (event study, relative to {REFERENCE_QUARTER}): in 2022-2023 the quarterly Welsh gap stayed "
          f"between {recent['rate_ratio'].min():.2f} and {recent['rate_ratio'].max():.2f}; in 2017-2019 it sat higher, "
          f"{before_covid['rate_ratio'].min():.2f} to {before_covid['rate_ratio'].max():.2f}.")
    print("  So Wales was already falling relative to England between 2019 and 2022, before the policy: the estimate")
    print("  that adds 2017-2019 to 'before' credits the policy with that older fall, and is shown only as a sensitivity.")

    # ---------- inference with Wales as one unit, and the other checks ----------
    hist = main_sample(earlier, "casualties")
    trend = {}
    for outcome in SERIES:
        h = history[outcome]
        trend[outcome] = did_poisson_trend(h.assign(years=(h["month"] - pd.Timestamp("2017-01-01")).dt.days / 365.25),
                                           outcome, "wales", "after", "years")
    welsh = sorted(main_casualties.loc[main_casualties["wales"], "police_force_code"].unique())
    english = sorted(main_casualties.loc[~main_casualties["wales"], "police_force_code"].unique())
    leave_out = {force: did_poisson(main_casualties[main_casualties["police_force_code"] != force], "casualties", "wales", "after")
                 for force in welsh}
    main_inf = inference(main_casualties)
    every_force = inference(treated)
    ksi_single = single_unit_placebo(main_ksi, "ksi_adjusted", "wales", "after", tolerance=SIZE_TOLERANCE)
    exact, matched, synth, single = main_inf["exact"], main_inf["matched"], main_inf["synthetic"], main_inf["single_unit"]
    groups = both_groups(MARTS)
    triple = {outcome: ddd_poisson(main_sample(groups, outcome), outcome, "wales", "after", "road_group", TREATED_ROADS)
              for outcome in SERIES}
    # All casualties keep Dyfed-Powys, whose rise on other roads enlarges the triple difference: the same without it
    triple_no_dp = ddd_poisson(without(main_sample(groups, "casualties"), severity_break), "casualties", "wales", "after",
                               "road_group", TREATED_ROADS)
    net = {outcome: did_poisson(all_roads(main_sample(groups, outcome)), outcome, "wales", "after") for outcome in SERIES}
    chart_synthetic(synth, main_inf["series"], FIGURES / "wales_synthetic_control.png")
    check = recording_check(MARTS, flagged[0])
    means = monthly_means(main_sample(groups, "casualties"))
    other_row = results.iloc[3]
    low_after = means[f"{TREATED_ROADS} | Wales | after"]["casualties"]
    other_after = means["other roads | Wales | after"]["casualties"]
    change_low = low_after - low_after / main_row["rate_ratio"]
    change_other = other_after - other_after / other_row["rate_ratio"]
    speed = {c: -implied_speed_change_kmh(main_row["rate_ratio"], c) for c in SPEED_COEFFICIENTS}
    speed_ci = (-implied_speed_change_kmh(main_row["ci_high"], coefficient), -implied_speed_change_kmh(main_row["ci_low"], coefficient))
    post = quarterly[quarterly["quarter"] > "2023Q3"]
    donors_all = donor_contributions(every_force["synthetic"], every_force["series"])

    def named(d):
        return {forces.get(f, str(f)): v for f, v in d.items()}

    print("\nWales as one unit, against single English forces (each refitted as if it alone had the policy)")
    for label, result in (("All casualties, main sample", single), ("All casualties, every force", every_force["single_unit"]),
                          ("KSI, without Dyfed-Powys", ksi_single)):
        a, m = result["all"], result["similar"]
        print(f"  {label:30s} Wales {result['rate_ratio']:.3f}: rank {a['rank']} of {a['units']} units, p = {a['p_value']:.3f}; "
              f"among forces within {SIZE_TOLERANCE:.0%} of Wales's size, rank {m['rank']} of {m['units']}, "
              f"p = {m['p_value']:.3f} (smallest possible {m['smallest_p']:.3f})")
    top = sorted(single["placebo"].items(), key=lambda kv: -abs(np.log(kv[1])))[:4]
    print("  furthest English forces: " + ", ".join(f"{forces.get(f, str(f))} {v:.3f}" for f, v in top))

    print("\nOther readings and checks: main sample, all casualties on 32 and 48 km/h roads unless stated")
    print(f"  {'':62s}{'rate ratio':>11s}{'95% interval*':>17s}")
    for outcome, (label, _) in SERIES.items():
        r = trend[outcome]
        print(f"  {'Wales keeps its 2017-2023 trend against England: ' + label:62s}{r['rate_ratio']:11.3f}"
              f"{r['ci_low']:9.3f} to {r['ci_high']:.3f}   (Welsh trend {100 * (r['trend_per_year'] - 1):+.1f}% a year)")
    for outcome, (label, _) in SERIES.items():
        r = triple[outcome]
        print(f"  {'Triple difference, against other Welsh roads: ' + label:62s}{r['rate_ratio']:11.3f}"
              f"{r['ci_low']:9.3f} to {r['ci_high']:.3f}")
    print(f"  {'Triple difference, all casualties, without ' + SEVERITY_BREAK:62s}{triple_no_dp['rate_ratio']:11.3f}"
          f"{triple_no_dp['ci_low']:9.3f} to {triple_no_dp['ci_high']:.3f}")
    for outcome, (label, _) in SERIES.items():
        r = net[outcome]
        print(f"  {'All roads together, the net change: ' + label:62s}{r['rate_ratio']:11.3f}"
              f"{r['ci_low']:9.3f} to {r['ci_high']:.3f}")
    for force, r in leave_out.items():
        print(f"  {'Leaving out ' + forces.get(force, str(force)):62s}{r['rate_ratio']:11.3f}{r['ci_low']:9.3f} to {r['ci_high']:.3f}")
    print(f"  {'Synthetic control: Wales / synthetic Wales, after':62s}{synth['ratio']:11.3f}"
          f"   (gap {synth['gap_post']:+.1f} index points; placebo-in-space rank {synth['rank']} of {synth['units']}, "
          f"p = {synth['p_value']:.3f})")
    top = sorted(synth["weights"].items(), key=lambda kv: -kv[1])[:5]
    print("    weights: " + ", ".join(f"{forces.get(f, str(f))} {w:.2f}" for f, w in top if w >= 0.005)
          + f"; pre-period error {synth['rmspe_pre']:.1f}, after {synth['rmspe_post']:.1f} index points")
    print(f"  Exact permutation over all {exact['sets']:,} sets of 4 English forces: p {p_text(exact['p_value'])}; "
          f"over the {matched['sets']:,} sets within {SIZE_TOLERANCE:.0%} of Wales's size: p {p_text(matched['p_value'])}  (secondary*)")

    e = every_force
    staffs = donors_all.get(flagged[0], {"weight": 0.0, "index_after": float("nan"), "adds": 0.0})
    avon = donors_all.get(flagged[1], {"weight": 0.0, "index_after": float("nan"), "adds": 0.0})
    synthetic_after = 100 + sum(d["adds"] for d in donors_all.values())
    print(f"\n  {FLAGGED_BY_DFT[0]}'s casualties on all roads against its 2017-2019 average: "
          + ", ".join(f"{y} {check['force'][y]:.2f}" for y in (2021, 2022, 2023, 2024, 2025))
          + "; England: " + ", ".join(f"{y} {check['england'][y]:.2f}" for y in (2021, 2022, 2023, 2024, 2025)))
    print(f"  With every force (sensitivity): rate ratio {e['did']['rate_ratio']:.3f} ({e['did']['ci_low']:.3f} to "
          f"{e['did']['ci_high']:.3f}); exact permutation p {p_text(e['exact']['p_value'])}, size-matched p "
          f"{p_text(e['matched']['p_value'])}; synthetic control {e['synthetic']['ratio']:.3f}, rank {e['synthetic']['rank']} of "
          f"{e['synthetic']['units']} (p = {e['synthetic']['p_value']:.3f})")
    print(f"    in that synthetic control, {FLAGGED_BY_DFT[0]} has weight {staffs['weight']:.3f} and an after-period index of "
          f"{staffs['index_after']:.0f}, adding {staffs['adds']:+.1f} points to synthetic Wales's {synthetic_after:.1f}; "
          f"{FLAGGED_BY_DFT[1]} has weight {avon['weight']:.3f}")
    print(f"\n  Casualties a month in Wales after the switch: {low_after:.1f} on 32 and 48 km/h roads, about "
          f"{-change_low:.0f} fewer than England's change implies; {other_after:.1f} on other roads, about {change_other:+.0f}.")
    print(f"  After the switch the quarterly Welsh gap ran {post['rate_ratio'].min():.2f} to {post['rate_ratio'].max():.2f}.")
    print(f"  Under the Exponential model, a rate ratio of {main_row['rate_ratio']:.3f} needs mean speeds lower by: "
          + ", ".join(f"{v:.1f} km/h at {c}" for c, v in speed.items())
          + f"; at {coefficient}, the 95% interval gives {speed_ci[0]:.1f} to {speed_ci[1]:.1f} km/h.")

    summary = {
        "main": results.to_dict("records"), "trend": trend, "triple": triple, "triple_without_severity_break": triple_no_dp,
        "net": net,
        "leave_out": named(leave_out),
        "single_unit": {"main": {k: v for k, v in single.items() if k != "placebo"} | {"placebo": named(single["placebo"])},
                        "every_force": {k: v for k, v in every_force["single_unit"].items() if k != "placebo"},
                        "ksi": {k: v for k, v in ksi_single.items() if k != "placebo"}},
        "synthetic": {k: v for k, v in synth.items() if k not in ("actual", "synthetic", "placebo_ratios")}
                     | {"weights": {forces.get(f, str(f)): w for f, w in synth["weights"].items() if w >= 0.005}},
        "exact_permutation": exact, "size_matched_permutation": matched | {"tolerance": SIZE_TOLERANCE},
        "sample": {"flagged": list(FLAGGED_BY_DFT), "severity_break": SEVERITY_BREAK},
        "every_force": {"did": e["did"], "exact": e["exact"], "matched": e["matched"],
                        "synthetic": {k: v for k, v in e["synthetic"].items()
                                      if k in ("ratio", "gap_post", "rmspe_pre", "rmspe_post", "rank", "units", "p_value")},
                        "donors": named(donors_all), "synthetic_after_index": synthetic_after},
        "recording_break": {"forces": list(FLAGGED_BY_DFT), "relative": check},
        "monthly_means": means, "change_a_month": {"focus": change_low, "other": change_other},
        "speed_kmh": {str(c): v for c, v in speed.items()}, "speed_ci_kmh": speed_ci, "speed_coefficient": coefficient,
        "event_study": {"recent": [recent["rate_ratio"].min(), recent["rate_ratio"].max()],
                        "earlier": [before_covid["rate_ratio"].min(), before_covid["rate_ratio"].max()],
                        "after": [post["rate_ratio"].min(), post["rate_ratio"].max()]},
        "forces": {"welsh": len(welsh), "english": len(english)}, "months": int(main_row["months"]),
    }
    (RESULTS / "wales_results.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"\nWrote {len(results)} rows to results/wales_did.parquet, results/wales_results.json and two charts to figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
