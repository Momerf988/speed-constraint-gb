"""Phase 8, part A: did Wales's default 32 km/h (20 mph) limit cut casualties? A difference-in-differences study.

Wales made 32 km/h the default on its restricted roads on 17 September 2023. This compares
monthly casualties on Welsh 32 and 48 km/h roads with the same roads in England, before and
after, police force by police force (data/marts/agg_casualties_force_month):
    before   January 2022 to August 2023, the 20 months just before the switch
    after    October 2023 to December 2025 (September 2023, the month of the switch, left out)
The event study shows why "before" starts in 2022: Wales was already falling relative to
England between 2019 and 2022, so adding 2017-2019 would credit the policy with an older
change. That longer window is kept as a sensitivity test. 2020-2021 are left out throughout:
the two countries had different COVID rules. Scotland is left out: it brought in its own
32 km/h limits in 2024-2025. All casualties is the main outcome, because no change in police
severity recording can move it; adjusted KSI is the second.
Writes results/wales_did.parquet and figures/wales_event_study.png.

Usage:
    python -m analysis.wales
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
from model.causal import did_poisson, event_study, implied_speed_change_kmh, permutation_test
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
    fig.savefig(path, dpi=150, facecolor=SURFACE)
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
    print(f"\nWrote {len(results)} rows to results/wales_did.parquet and figures/wales_event_study.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
