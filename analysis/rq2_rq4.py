"""Phase 6: deaths and injuries prevented by holding cars to the limit (RQ4) and by flat caps (RQ2).

Runs the casualty model on 2024 speeds and 2022-2024 casualties in three ways:
    central       the central assumptions, once
    sensitivity   one modelling choice changed at a time
    Monte Carlo   10,000 draws of the uncertain values (model/uncertainty.py)
then compares the results with published estimates, prints the findings and writes:
    results/casualty_model.parquet        central result and 5th/50th/95th percentiles
    results/casualty_model_draws.parquet  every draw, for the cost-benefit analysis (Phase 7)
    figures/rq2_rq4_deaths_prevented.png
    figures/rq2_speeds_97kmh.png

Usage:
    python -m analysis.rq2_rq4
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.casualties import Assumptions
from model.run import PREVENTED_COLUMNS, SAVED_COLUMNS, load_inputs, run, speed_distributions, worse
from model.scenarios import ROAD_TYPES, IsaUse, apply_policy, held_to_limit
from model.uncertainty import N_DRAWS, SEED, monte_carlo, percentiles

ROOT = Path(__file__).resolve().parent.parent
MARTS = ROOT / "data" / "marts"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

ALL_ROADS = "all four road types"
POLICY_NAMES = {
    "today": "Today",
    "bau_isa": "Business as usual (overridable ISA)",
    "compliance": "Posted-limit compliance",
    "cap_97": "Cap at 97 km/h",
    "cap_80": "Cap at 80 km/h",
    "cap_60": "Cap at 60 km/h",
    "cap_48": "Cap at 48 km/h",
}
SHORT_NAMES = {"today": "Today", "bau_isa": "BAU ISA", "compliance": "Compliance", "cap_97": "Cap 97",
               "cap_80": "Cap 80", "cap_60": "Cap 60", "cap_48": "Cap 48"}
SHOWN = ("compliance", "cap_97", "cap_80", "cap_60", "cap_48")
SUMMARY_COLUMNS = ["killed_prevented", "killed_prevented_vs_bau", "ksi_prevented_b", "ksi_prevented_b_vs_bau",
                   "casualties_prevented", "casualties_prevented_vs_bau"]

# Published estimates the model is checked against
ISA_UK_INJURY_REMAINING = {          # Tate and Carsten (2008): injury crashes left under mandatory ISA
    "32 km/h roads": 0.37, "48 km/h roads": 0.60,
    "97 km/h single carriageways": 0.86, "113 km/h motorways": 0.82,
}
VADEBY_SHARE_OF_DEATHS = 0.25        # Vadeby (2023): full compliance in Sweden, Exponential model
LAI_CHOSEN_TO_UNIVERSAL = 12 / 29    # Lai et al. (2012) via Day et al. (2023): serious injuries -12% vs -29%
LAI_ADVISORY_TO_UNIVERSAL = 3 / 29   # the same source: advisory ISA, -3%
TRL_DEATHS_IN_TARGET = 0.1636        # TRL (2024) Table 61: cars, killed, medium, of those in car collisions
LAI_CO2_MANDATORY = 0.058            # Lai et al. (2012) via TRL (2024): CO2 saved on 113 km/h roads, mandatory ISA
LAI_CO2_VOLUNTARY = 0.034            # the same source: voluntary ISA

# Chart colours (light surface): one colour per policy kind, so a policy keeps its colour in every chart
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLOURS = {"today": "#8a8985", "bau_isa": "#1baf7a", "compliance": "#2a78d6", "cap": "#eb6834"}


def colour(policy: str) -> str:
    return COLOURS["cap"] if policy.startswith("cap_") else COLOURS[policy]


def totals(table: pd.DataFrame) -> pd.DataFrame:
    """Sum over the four modelled road types, per policy (and per draw, if there is one)."""
    keys = ["draw", "policy"] if "draw" in table else ["policy"]
    columns = ["baseline_killed", *PREVENTED_COLUMNS, *(f"{c}_vs_bau" for c in PREVENTED_COLUMNS)]
    summed = table.groupby(keys, sort=False)[columns].sum(min_count=1)
    evidence = table.groupby(keys, sort=False).agg(evidence=("evidence", lambda s: worse(*s)),
                                                   evidence_vs_bau=("evidence_vs_bau", lambda s: worse(*s)))
    # Model B has no motorway curve: its totals use the Power Model there, as the preferred estimate does,
    # and Model B's KSI is not totalled
    for column in ("killed_prevented", "casualties_prevented"):
        summed[f"{column}_b"] = summed[column]
        summed[f"{column}_b_vs_bau"] = summed[f"{column}_vs_bau"]
    summed[["ksi_prevented_b", "ksi_prevented_b_vs_bau"]] = np.nan
    return summed.join(evidence).reset_index().assign(segment=ALL_ROADS)


def model_spread(row: pd.Series, suffix: str) -> tuple[float, float]:
    """Lowest and highest deaths prevented across Model B and both Model A forms."""
    values = [row[f"killed_prevented_{m}{suffix}"] for m in ("b", "power", "exp")]
    values = [v for v in values if pd.notna(v)]
    return min(values), max(values)


def describe(row: pd.Series, suffix: str = "_vs_bau") -> str:
    """One result in words: a range for 'within', 'at least' for a lower bound, model spread beyond."""
    evidence = row[f"evidence{suffix}" if suffix else "evidence"]
    central = row[f"killed_prevented{suffix}"]
    low, high = row[f"killed_prevented{suffix}_p5"], row[f"killed_prevented{suffix}_p95"]
    if abs(central) < 0.5 and abs(high) < 0.5:
        return "    0  (no car is affected beyond business as usual)"
    if evidence == "within":
        return f"{central:5.0f}  [{low:4.0f} to {high:4.0f}]"
    if evidence == "lower bound":
        return f"at least {central:.0f}  [{low:.0f} to {high:.0f}], the curve cut short"
    spread_low, spread_high = model_spread(row, suffix)
    return f"beyond the evidence: models give {spread_low:.0f} to {spread_high:.0f}"


def print_findings(summary: pd.DataFrame, baseline: pd.DataFrame, sensitivity: pd.DataFrame,
                   validation: pd.DataFrame) -> None:
    deaths_all = baseline["killed"].sum()
    deaths_modelled = baseline.drop("not modelled")["killed"].sum()
    print(f"Phase 6 casualty model: 2024 speeds, average casualties per year 2022-2024, {N_DRAWS:,} Monte Carlo draws")
    print(f"The four modelled road types have {deaths_modelled:,.0f} of {deaths_all:,.0f} deaths a year "
          f"({deaths_modelled / deaths_all:.0%}).\n")

    print("Mean speed in km/h, and % over their own limit: cars, then vans")
    speeds = summary[summary["segment"] != ALL_ROADS]
    print(f"  {'':36s}" + "".join(f"{SHORT_NAMES[p]:>18s}" for p in ("today", "bau_isa", "compliance", "cap_80")))
    for segment, rows in speeds.groupby("segment", sort=False):
        rows = rows.set_index("policy")
        for vehicle, name in (("car", "cars"), ("van", "vans")):
            cells = "".join(f"{rows.loc[p, f'{vehicle}_mean_after_kmh']:9.1f} ({rows.loc[p, f'pct_{vehicle}s_over_limit_after']:3.0f}%)  "
                            for p in ("today", "bau_isa", "compliance", "cap_80"))
            print(f"  {segment + ', ' + name:36s}{cells}")

    print("\nDeaths prevented per year BEYOND business-as-usual ISA: central [5th to 95th percentile]")
    for segment, rows in summary.groupby("segment", sort=False):
        print(f"  {segment} (today {rows['baseline_killed'].iloc[0]:,.0f} deaths a year)")
        for _, row in rows.set_index("policy").loc[list(SHOWN)].reset_index().iterrows():
            print(f"    {POLICY_NAMES[row['policy']]:28s}{describe(row)}")

    bau_total = summary[(summary["segment"] == ALL_ROADS) & (summary["policy"] == "bau_isa")].iloc[0]
    print(f"\n  Business-as-usual ISA itself, against today, on the four road types: {describe(bau_total, '').strip()}")

    print("\nDeaths prevented per year beyond business-as-usual ISA, all four road types, one choice changed at a time")
    print(f"  {'':32s}" + "".join(f"{SHORT_NAMES[p]:>12s}" for p in SHOWN))
    for name, rows in sensitivity.groupby("sensitivity", sort=False):
        rows = rows.set_index("policy")
        print(f"  {name:32s}" + "".join(f"{rows.loc[p, 'killed_prevented_vs_bau']:12.0f}" for p in SHOWN))
    rows = sensitivity[sensitivity["sensitivity"] == "Central"].set_index("policy")
    print(f"  {'Evidence':32s}" + "".join(f"{rows.loc[p, 'evidence_vs_bau']:>12s}" for p in SHOWN))

    print("\nChecks against published estimates")
    print(f"  {'':56s}{'ours':>8s}{'ours, f=1':>11s}{'published':>11s}")
    for _, row in validation.iterrows():
        f_one = "" if pd.isna(row["ours_f1"]) else f"{row['ours_f1']:.0%}"
        print(f"  {row['check']:56s}{row['ours']:8.0%}{f_one:>11s}{row['published']:11.0%}   {row['source']}")
    print("  f=1: every casualty in a car or van collision responds, not only the free-travel share")


def validate(central: pd.DataFrame, all_responding: pd.DataFrame, speed_control: pd.DataFrame,
             warning_only: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """The model's results next to the published figures they can be compared with.

    all_responding is the same run with f = 1, so the effect of the free-travel share is visible.
    speed_control and warning_only are the same run with one kind of ISA in every car.
    """
    def compliance(table):
        return table[table["policy"] == "compliance"].set_index("segment")

    def injuries_left(table, segment):
        r = compliance(table).loc[segment]
        return 1 - r["casualties_prevented"] / r["baseline_casualties"]

    def deaths_share(table):
        c = compliance(table)
        return c["killed_prevented"].sum() / c["baseline_killed"].sum()

    def isa_share(table):
        c = compliance(table)
        bau = table[table["policy"] == "bau_isa"].set_index("segment")
        curves = c["ksi_prevented_b"].notna()
        return bau.loc[curves, "ksi_prevented_b"].sum() / c.loc[curves, "ksi_prevented_b"].sum()

    rows = [{"check": f"Injuries left at the limit, {segment}",
             "ours": injuries_left(central, segment), "ours_f1": injuries_left(all_responding, segment),
             "published": published, "source": "Tate and Carsten (2008), ISA-UK"}
            for segment, published in ISA_UK_INJURY_REMAINING.items()]
    rows.append({"check": "Deaths prevented by full compliance (modelled roads)",
                 "ours": deaths_share(central), "ours_f1": deaths_share(all_responding),
                 "published": VADEBY_SHARE_OF_DEATHS, "source": "Vadeby (2023), Sweden"})
    rows.append({"check": "ISA with speed control: KSI benefit / mandatory ISA's",
                 "ours": isa_share(speed_control), "ours_f1": np.nan,
                 "published": LAI_CHOSEN_TO_UNIVERSAL, "source": "Lai et al. (2012), voluntary ISA"})
    rows.append({"check": "ISA with warnings only: KSI benefit / mandatory ISA's",
                 "ours": isa_share(warning_only), "ours_f1": np.nan,
                 "published": LAI_ADVISORY_TO_UNIVERSAL, "source": "Lai et al. (2012), advisory ISA"})
    bau_deaths = central[central["policy"] == "bau_isa"].set_index("segment")["killed_prevented"]
    bau_deaths_f1 = all_responding[all_responding["policy"] == "bau_isa"].set_index("segment")["killed_prevented"]
    responding = (baseline["killed"] * baseline["k_fatal"]).loc[bau_deaths.index].sum()
    def motorway_car_co2_saved(table, policy):
        r = table[(table["policy"] == policy) & (table["segment"] == "113 km/h motorways")].iloc[0]
        return r["car_co2_saved_per_km"] / (r["car_co2_per_km"] + r["car_co2_saved_per_km"])

    rows.append({"check": "CO2 saved on motorways, cars: mandatory ISA",
                 "ours": motorway_car_co2_saved(central, "compliance"), "ours_f1": np.nan,
                 "published": LAI_CO2_MANDATORY, "source": "Lai et al. (2012), via TRL (2024)"})
    rows.append({"check": "CO2 saved on motorways, cars: ISA with speed control",
                 "ours": motorway_car_co2_saved(speed_control, "bau_isa"), "ours_f1": np.nan,
                 "published": LAI_CO2_VOLUNTARY, "source": "Lai et al. (2012), via TRL (2024)"})
    rows.append({"check": "Mandated ISA: deaths prevented / deaths in car collisions",
                 "ours": bau_deaths.sum() / responding, "ours_f1": bau_deaths_f1.sum() / responding,
                 "published": TRL_DEATHS_IN_TARGET, "source": "TRL (2024), half warning, half speed control"})
    return pd.DataFrame(rows)


def chart_deaths(summary: pd.DataFrame, path: Path) -> None:
    """Deaths prevented beyond business-as-usual ISA, per road type and policy, with evidence shown."""
    shown = summary[summary["policy"].isin(SHOWN) & (summary["segment"] != ALL_ROADS)]
    fig, ax = plt.subplots(figsize=(9, 8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y, ticks, labels = 0, [], []
    for segment, rows in shown.groupby("segment", sort=False):
        ax.text(0, y - 0.15, segment, fontsize=10, fontweight="bold", color=INK, transform=ax.get_yaxis_transform(),
                ha="left", va="bottom")
        y += 0.9
        for _, row in rows.set_index("policy").loc[list(SHOWN)].reset_index().iterrows():
            c = colour(row["policy"])
            if row["evidence_vs_bau"] == "beyond":
                low, high = model_spread(row, "_vs_bau")
                ax.plot([low, high], [y, y], color=c, lw=6, alpha=0.25, solid_capstyle="round")
            else:
                ax.plot([row["killed_prevented_vs_bau_p5"], row["killed_prevented_vs_bau_p95"]], [y, y], color=c, lw=2)
                lower = row["evidence_vs_bau"] == "lower bound"
                ax.plot(row["killed_prevented_vs_bau"], y, marker=">" if lower else "o", ms=8,
                        color=c, markerfacecolor=SURFACE if lower else c, markeredgewidth=2, linestyle="none")
            ticks.append(y)
            labels.append(POLICY_NAMES[row["policy"]])
            y += 1
        y += 0.6
    ax.set_yticks(ticks, labels, fontsize=9, color=MUTED)
    ax.invert_yaxis()
    ax.set_ylim(y - 0.4, -0.6)
    ax.set_xlim(left=0)
    ax.axvline(0, color=MUTED, lw=1)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Deaths prevented per year, beyond business-as-usual ISA", color=MUTED, fontsize=9)
    ax.set_title("Holding cars to the limit, or capping them, against the ISA GB will mandate",
                 loc="left", fontsize=12, color=INK, pad=28)
    legend = [
        plt.Line2D([], [], color=COLOURS["compliance"], marker="o", lw=2, label="Posted-limit compliance"),
        plt.Line2D([], [], color=COLOURS["cap"], marker="o", lw=2, label="Flat cap"),
        plt.Line2D([], [], color=MUTED, marker=">", markerfacecolor=SURFACE, lw=0, label="Lower bound (curve cut short)"),
        plt.Line2D([], [], color=MUTED, lw=6, alpha=0.25, label="Beyond the evidence: range across models"),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=8.5, labelcolor=MUTED)
    fig.text(0.01, 0.01, f"Dot: central estimate. Line: 5th to 95th percentile of {N_DRAWS:,} Monte Carlo draws. "
             "2024 DfT free-flow speeds; 2022-2024 STATS19 casualties.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_speeds(distributions: dict, path: Path) -> None:
    """Share of cars at or below each speed on 97 km/h single carriageways, under four policies."""
    road = next(r for r in ROAD_TYPES if r.segment == "97 km/h single carriageways")
    today = distributions[road.segment].cars
    working = held_to_limit(today, road, IsaUse.central())
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for policy, style in (("bau_isa", "-"), ("compliance", "-"), ("cap_80", "-"), ("today", "--")):
        dist = apply_policy(policy, today, road, working)
        order = np.argsort(dist.speeds_kmh, kind="stable")
        speeds, cumulative = dist.speeds_kmh[order], 100 * np.cumsum(dist.probabilities[order])
        ax.plot(speeds, cumulative, style, color=colour(policy), lw=2, label=POLICY_NAMES[policy])
    ax.axvline(road.limit_kmh, color=MUTED, lw=1, ls=":")
    ax.text(road.limit_kmh + 1, 4, "limit 97 km/h", color=MUTED, fontsize=8.5)
    ax.set_xlim(40, 130)
    ax.set_ylim(0, 101)
    ax.set_xlabel("Free-flow car speed (km/h)", color=MUTED, fontsize=9)
    ax.set_ylabel("Cars at or below this speed (%)", color=MUTED, fontsize=9)
    ax.set_title("A cap cuts off the fast tail; it does not shift everyone down (97 km/h single carriageways, 2024)",
                 loc="left", fontsize=10.5, color=INK)
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    bands, baseline = load_inputs(MARTS)
    distributions = speed_distributions(bands)
    central_assumptions = Assumptions.central()

    central = run(distributions, baseline, central_assumptions)
    central = pd.concat([central, totals(central)], ignore_index=True)

    central_isa = IsaUse.central()
    sensitivities = {
        "Central": (distributions, central_assumptions, central_isa),
        "Difference from today's mean": (distributions, replace(central_assumptions, reference_mean="old"), central_isa),
        "Normal speed distribution": (speed_distributions(bands, "normal"), central_assumptions, central_isa),
        "Lognormal speed distribution": (speed_distributions(bands, "lognormal"), central_assumptions, central_isa),
        "Brenac's flatter curve": (distributions, replace(central_assumptions, urban_curve="brenac2015_exponential_coef",
                                                          rural_curve="brenac2015_exponential_coef"), central_isa),
        "Override at random": (distributions, central_assumptions, replace(central_isa, override_targets_speeding=False)),
        "Speed control in every car": (distributions, central_assumptions, replace(central_isa, speed_control_share=1.0)),
        "Warning only in every car": (distributions, central_assumptions, replace(central_isa, speed_control_share=0.0)),
        "ISA shows a van its own limit": (distributions, central_assumptions, replace(central_isa, vans_read_own_limit=True)),
    }
    sensitivity = pd.concat([totals(run(d, baseline, a, isa)).assign(sensitivity=name)
                             for name, (d, a, isa) in sensitivities.items()], ignore_index=True)
    vans_like_cars = tuple(replace(road, van_limit_kmh=road.limit_kmh) for road in ROAD_TYPES)
    sensitivity = pd.concat([sensitivity, totals(run(distributions, baseline, central_assumptions, central_isa, vans_like_cars))
                             .assign(sensitivity="Vans have the car limit")], ignore_index=True)

    draws = monte_carlo(distributions, baseline, N_DRAWS, SEED)
    draws = pd.concat([draws, totals(draws)], ignore_index=True)
    ranges = percentiles(draws, ["segment", "policy"], SUMMARY_COLUMNS)
    summary = central.merge(ranges, on=["segment", "policy"], how="left")

    validation = validate(central[central["segment"] != ALL_ROADS],
                          run(distributions, baseline, replace(central_assumptions, free_travel_share=1.0,
                                                               free_travel_share_rural=1.0)),
                          run(distributions, baseline, central_assumptions, replace(central_isa, speed_control_share=1.0)),
                          run(distributions, baseline, central_assumptions, replace(central_isa, speed_control_share=0.0)),
                          baseline)

    summary.to_parquet(RESULTS / "casualty_model.parquet", index=False, compression="zstd")
    keep = ["draw", "segment", "policy", "mean_change_kmh", "evidence", "evidence_vs_bau",
            *PREVENTED_COLUMNS, *(f"{c}_vs_bau" for c in PREVENTED_COLUMNS), *(f"{c}_vs_bau" for c in SAVED_COLUMNS)]
    draws[keep].to_parquet(RESULTS / "casualty_model_draws.parquet", index=False, compression="zstd")
    chart_deaths(summary, FIGURES / "rq2_rq4_deaths_prevented.png")
    chart_speeds(distributions, FIGURES / "rq2_speeds_97kmh.png")

    print_findings(summary, baseline, sensitivity, validation)
    print(f"\nWrote {len(summary)} rows to results/casualty_model.parquet, {len(draws):,} to "
          "results/casualty_model_draws.parquet, and two charts to figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
