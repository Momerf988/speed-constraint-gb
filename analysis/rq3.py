"""Phase 7: do the policies pay for themselves? Casualties prevented against travel time lost (RQ3).

Combines the casualty model (Phase 6), DfT's appraisal values (model/parameters/
appraisal_values.csv) and car and van traffic per road type (data/marts/agg_traffic_segment).
Every policy is measured against business-as-usual ISA, in £ a year. Runs:
    central       the central assumptions
    Monte Carlo   the 10,000 casualty draws from Phase 6, each with a free-flow share drawn too
    sensitivity   one assumption changed at a time, for the 80 km/h cap on 97 km/h roads
then prints the findings and writes:
    results/appraisal.parquet               central result and Monte Carlo percentiles
    figures/rq3_benefit_cost_ratio.png
    figures/rq3_cap80_sensitivity.png

Usage:
    python -m analysis.rq3
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import duckdb
import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.rq2_rq4 import COLOURS, GRID, INK, MUTED, POLICY_NAMES, SHOWN, SURFACE, colour
from model.appraisal import (FREE_FLOW_RANGES, FREE_FLOW_SHARE, break_even_free_flow, casualty_benefit,
                             hours_lost_per_km, ksi_from_casualties, load_values, time_cost)
from model.casualties import Assumptions
from model.run import BASELINE_YEARS, load_inputs, run, speed_distributions
from model.scenarios import POLICIES, ROAD_TYPES, isa_working_share
from model.uncertainty import N_DRAWS

ROOT = Path(__file__).resolve().parent.parent
MARTS = ROOT / "data" / "marts"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

FREE_FLOW_SEED = 20261031
TARGET_CUT = 0.65                       # Road Safety Strategy: 65% fewer KSI by 2035 than in 2022-2024
FOCUS = ("97 km/h single carriageways", "cap_80")   # the cap the evidence supports, used for the sensitivity chart
MODELS = ("b", "power", "exp")


def load_exposure(marts_dir) -> tuple[pd.DataFrame, dict[str, float]]:
    """Car and van billion vehicle-km per road type, and collisions per casualty, averaged over the baseline years."""
    first, last = BASELINE_YEARS
    traffic = duckdb.sql(f"""
        select segment, avg(car_billion_vehicle_km) as cars, avg(van_billion_vehicle_km) as vans
        from read_parquet('{marts_dir}/agg_traffic_segment.parquet')
        where year between {first} and {last}
        group by 1
    """).df().set_index("segment")
    ratios = duckdb.sql(f"""
        select lower(severity) as severity, sum(collisions)::double / sum(casualties) as ratio
        from read_parquet('{marts_dir}/agg_collisions_per_casualty.parquet')
        where year between {first} and {last}
        group by 1
    """).df()
    return traffic, dict(zip(ratios["severity"], ratios["ratio"]))


def benefits(table: pd.DataFrame, baseline: pd.DataFrame, values, ratios, include_unreported=False,
             suffix="_vs_bau") -> pd.DataFrame:
    """£ of casualties prevented, under the preferred model and under each model separately.

    Where a model gives no KSI (Model A, and every model on motorways), KSI prevented are
    estimated from all casualties prevented (model.appraisal.ksi_from_casualties).
    """
    base = baseline.loc[table["segment"]].reset_index(drop=True)
    t = table.reset_index(drop=True)

    def ksi(model_ksi, casualties):
        estimate = ksi_from_casualties(casualties, base["ksi"], base["casualties"])
        return np.where(np.isnan(model_ksi), estimate, model_ksi)

    def value(killed, ksi_prevented, casualties):
        return casualty_benefit(killed, ksi_prevented, casualties, values, ratios, include_unreported)

    nan = np.full(len(t), np.nan)
    out = pd.DataFrame({
        "ksi_prevented": ksi(t[f"ksi_prevented_b{suffix}"], t[f"casualties_prevented{suffix}"]),
        "benefit": value(t[f"killed_prevented{suffix}"],
                         ksi(t[f"ksi_prevented_b{suffix}"], t[f"casualties_prevented{suffix}"]),
                         t[f"casualties_prevented{suffix}"]),
        "benefit_b": value(t[f"killed_prevented_b{suffix}"], t[f"ksi_prevented_b{suffix}"],
                           t[f"casualties_prevented_b{suffix}"]),
    })
    for model in ("power", "exp"):
        casualties = t[f"casualties_prevented_{model}{suffix}"]
        out[f"benefit_{model}"] = value(t[f"killed_prevented_{model}{suffix}"], ksi(nan, casualties), casualties)
    return out


def hours_table(distributions, count_speeding_time=False) -> pd.DataFrame:
    """Hours lost per vehicle-km beyond business-as-usual ISA, per road type and policy.

    Lawful hours do not depend on how often ISA is switched off or overridden: a cap below the
    limit binds every car, and a cap above it costs no lawful time. So the central share serves
    every Monte Carlo draw.
    """
    rows = []
    for road in ROAD_TYPES:
        today, working = distributions[road.segment], isa_working_share(road)
        lost = {p: hours_lost_per_km(p, today, road, working, count_speeding_time) for p in POLICIES}
        rows += [{"segment": road.segment, "policy": p, "hours_per_km": lost[p] - lost["bau_isa"]} for p in POLICIES]
    return pd.DataFrame(rows)


def appraise(casualties: pd.DataFrame, hours: pd.DataFrame, traffic, baseline, values, ratios,
             free_flow=None, include_unreported=False) -> pd.DataFrame:
    """Benefit, time cost, net benefit, benefit-cost ratio and break-even free-flow share, in £ a year."""
    table = casualties.merge(hours, on=["segment", "policy"]).reset_index(drop=True)
    table = pd.concat([table, benefits(table, baseline, values, ratios, include_unreported)], axis=1)
    cars = table["segment"].map(traffic["cars"])
    vans = table["segment"].map(traffic["vans"])
    if free_flow is None:
        free_flow = FREE_FLOW_SHARE
    share = free_flow if isinstance(free_flow, pd.Series) else table["segment"].map(free_flow)
    table["free_flow_share"] = share
    table["time_cost"] = time_cost(table["hours_per_km"], cars, vans, values, share)
    table["net"] = table["benefit"] - table["time_cost"]
    table["bcr"] = np.where(table["time_cost"] > 0, table["benefit"] / table["time_cost"].where(table["time_cost"] > 0), np.inf)
    table["break_even_free_flow"] = break_even_free_flow(table["benefit"].to_numpy(), table["hours_per_km"].to_numpy(),
                                                         cars.to_numpy(), vans.to_numpy(), values)
    for model in MODELS:
        table[f"bcr_{model}"] = table[f"benefit_{model}"] / table["time_cost"].where(table["time_cost"] > 0)
    return table


def monte_carlo(draws: pd.DataFrame, hours, traffic, baseline, values, ratios) -> pd.DataFrame:
    """Net benefit and benefit-cost ratio per draw: the Phase 6 casualty draws, plus a free-flow share each."""
    rng = np.random.default_rng(FREE_FLOW_SEED)
    shares = pd.DataFrame({segment: rng.uniform(low, high, N_DRAWS) for segment, (low, high) in FREE_FLOW_RANGES.items()})
    shares.index.name = "draw"
    draws = draws[draws["segment"].isin(FREE_FLOW_SHARE)].reset_index(drop=True)
    share = pd.Series(shares.to_numpy()[draws["draw"].to_numpy(), [list(shares.columns).index(s) for s in draws["segment"]]])
    return appraise(draws, hours, traffic, baseline, values, ratios, free_flow=share)


def summarise(central: pd.DataFrame, mc: pd.DataFrame) -> pd.DataFrame:
    keys = ["segment", "policy"]
    q = mc.groupby(keys, sort=False)[["benefit", "time_cost", "net", "bcr"]].quantile([0.05, 0.5, 0.95]).unstack()
    q.columns = [f"{column}_p{round(p * 100)}" for column, p in q.columns]
    q["chance_net_positive"] = mc.assign(pays=mc["net"] > 0).groupby(keys, sort=False)["pays"].mean()
    keep = ["segment", "policy", "evidence_vs_bau", "hours_per_km", "free_flow_share", "ksi_prevented",
            "benefit", "benefit_b", "benefit_power", "benefit_exp", "time_cost", "net", "bcr", "bcr_b",
            "bcr_power", "bcr_exp", "break_even_free_flow"]
    return central[keep].merge(q.reset_index(), on=keys, how="left")


def millions(x: float) -> str:
    return f"{x / 1e6:,.0f}"


def model_range(row, prefix: str) -> tuple[float, float]:
    values = [row[f"{prefix}_{m}"] for m in MODELS if pd.notna(row[f"{prefix}_{m}"])]
    return min(values), max(values)


def focus_row(table: pd.DataFrame) -> pd.Series:
    return table[(table["segment"] == FOCUS[0]) & (table["policy"] == FOCUS[1])].iloc[0]


def sensitivity(cases: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Net benefit and benefit-cost ratio of the focus policy, from (benefit, time cost) per case."""
    return pd.DataFrame([{"sensitivity": name, "net": benefit - cost, "bcr": benefit / cost}
                         for name, (benefit, cost) in cases.items()])


def pays_if(share: float) -> str:
    """The break-even free-flow share in words."""
    if share >= 1:
        return "pays even if all traffic flows freely"
    return f"pays if under {share:.0%} of traffic flows freely"


def print_findings(summary, values, traffic, ratios, baseline, compliance_all_time, sens, target):
    print("Phase 7 appraisal: £ a year beyond business-as-usual ISA, 2023 prices and values (TAG data book v2.03),")
    print("every car fitted, 2022-2024 casualty and traffic levels")
    print(f"Values: £{values.per_casualty['fatal']:,.0f} per death, £{values.per_casualty['serious']:,.0f} per serious and "
          f"£{values.per_casualty['slight']:,.0f} per slight injury, plus per-collision costs")
    print(f"        (collisions per casualty: fatal {ratios['fatal']:.3f}, serious {ratios['serious']:.3f}, "
          f"slight {ratios['slight']:.3f}); £{values.car_hour:.2f} per car-hour, £{values.van_hour:.2f} per van-hour")
    print("Car and van traffic, billion vehicle-km a year: " + ", ".join(
        f"{s} {traffic.loc[s, 'cars'] + traffic.loc[s, 'vans']:.1f}" for s in FREE_FLOW_SHARE))
    print("Free-flowing share of traffic (a judgement, no published source): 40% on 32 and 48 km/h roads, "
          "70% on 97 km/h roads and motorways\n")

    print("1. Policies that cost no lawful time: every second they cost was gained by speeding (£m a year)")
    free = summary[(summary["hours_per_km"] <= 1e-12) & (summary["benefit"] > 0.5e6) & summary["policy"].isin(SHOWN)]
    for _, r in free.iterrows():
        label = "at least " if r["evidence_vs_bau"] == "lower bound" else ""
        spread = "" if r["evidence_vs_bau"] == "lower bound" else f"  [{millions(r['benefit_p5'])} to {millions(r['benefit_p95'])}]"
        print(f"   {r['segment']:28s} {POLICY_NAMES[r['policy']]:24s} benefit {label}{millions(r['benefit']):>5s}{spread}")
    compliance = free[free["policy"] == "compliance"]
    print(f"   {'All four road types':28s} {POLICY_NAMES['compliance']:24s} benefit at least {millions(compliance['benefit'].sum())}")

    print("\n2. Caps that cost lawful time (£m a year): central [5th to 95th percentile]")
    costly = summary[(summary["hours_per_km"] > 1e-12) & summary["policy"].isin(SHOWN)]
    for _, r in costly.iterrows():
        head = f"   {r['segment']:28s} {POLICY_NAMES[r['policy']]:16s}"
        if r["evidence_vs_bau"] == "within":
            print(f"{head} benefit {millions(r['benefit'])}, time cost {millions(r['time_cost'])}, "
                  f"net {millions(r['net'])} [{millions(r['net_p5'])} to {millions(r['net_p95'])}], "
                  f"BCR {r['bcr']:.2f} [{r['bcr_p5']:.2f} to {r['bcr_p95']:.2f}], {pays_if(r['break_even_free_flow'])} "
                  f"(chance it pays: {r['chance_net_positive']:.0%})")
        else:
            low, high = model_range(r, "benefit")
            bcr_low, bcr_high = model_range(r, "bcr")
            print(f"{head} beyond the evidence: benefit {millions(low)} to {millions(high)} across models, "
                  f"time cost {millions(r['time_cost'])}, BCR {bcr_low:.2f} to {bcr_high:.2f}")

    print("\n3. If time gained by speeding were counted too: posted-limit compliance (£m a year)")
    for _, r in compliance_all_time.iterrows():
        flag = "at least " if r["evidence_vs_bau"] == "lower bound" else ""
        print(f"   {r['segment']:28s} benefit {flag}{millions(r['benefit']):>5s}, time cost {millions(r['time_cost']):>5s}, "
              f"net {flag}{millions(r['net']):>6s}, {pays_if(r['break_even_free_flow'])}")

    print(f"\n4. {POLICY_NAMES[FOCUS[1]]} on {FOCUS[0]}: net benefit (£m a year), one assumption changed at a time")
    for _, r in sens.iterrows():
        print(f"   {r['sensitivity']:46s} {millions(r['net']):>6s}   BCR {r['bcr']:.2f}")

    print(f"\n5. Share of the 2035 target: {TARGET_CUT:.0%} fewer KSI than {target['baseline']:,.0f} a year "
          f"means {target['required']:,.0f} fewer a year")
    for name, (ksi, p5, p95, note) in target["rows"].items():
        spread = f" [{p5:,.0f} to {p95:,.0f}]" if pd.notna(p5) else ""
        print(f"   {name:58s} {note}{ksi:6,.0f} KSI{spread} = {note}{ksi / target['required']:.0%} of the cut")


def chart_bcr(summary: pd.DataFrame, path: Path) -> None:
    """Benefit-cost ratio of every cap that costs lawful time, with the evidence shown."""
    rows = summary[(summary["hours_per_km"] > 1e-12) & summary["policy"].isin(SHOWN)].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    labels = []
    for y, r in rows.iterrows():
        c = colour(r["policy"])
        if r["evidence_vs_bau"] == "within":
            ax.plot([r["bcr_p5"], r["bcr_p95"]], [y, y], color=c, lw=2)
            ax.plot(r["bcr"], y, "o", ms=8, color=c)
            ax.text(r["bcr_p95"] * 1.15, y, pays_if(r["break_even_free_flow"]), va="center", fontsize=8, color=MUTED)
        else:
            low, high = model_range(r, "bcr")
            ax.plot([low, high], [y, y], color=c, lw=6, alpha=0.25, solid_capstyle="round")
        labels.append(f"{r['segment']}: {POLICY_NAMES[r['policy']].lower()}")
    ax.axvline(1, color=INK, lw=1)
    ax.text(1.1, -0.75, "pays for itself →", fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.01, 1000)
    ax.set_ylim(len(rows) - 0.5, -1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_yticks(range(len(rows)), labels, fontsize=8.5, color=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Benefit-cost ratio beyond business-as-usual ISA (log scale)", color=MUTED, fontsize=9)
    ax.set_title("Caps below the limit: casualty benefit against lawful time lost",
                 loc="left", fontsize=11, color=INK)
    legend = [plt.Line2D([], [], color=COLOURS["cap"], marker="o", lw=2, label="Within the evidence: central, 5th-95th percentile"),
              plt.Line2D([], [], color=COLOURS["cap"], lw=6, alpha=0.25, label="Beyond the evidence: range across models")]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=2, frameon=False, fontsize=8,
              labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def chart_sensitivity(sens: pd.DataFrame, path: Path) -> None:
    """Net benefit of the focus cap with one assumption changed at a time, largest swing first."""
    central = sens.loc[sens["sensitivity"] == "Central", "net"].iloc[0] / 1e6
    rows = sens[sens["sensitivity"] != "Central"].assign(net_m=lambda d: d["net"] / 1e6)
    rows = rows.assign(swing=(rows["net_m"] - central).abs()).sort_values("swing").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for y, r in rows.iterrows():
        ax.barh(y, r["net_m"] - central, left=central, height=0.6, color=COLOURS["cap"])
        ax.text(r["net_m"] + (8 if r["net_m"] >= central else -8), y, f"{r['net_m']:,.0f}", va="center",
                ha="left" if r["net_m"] >= central else "right", fontsize=8, color=MUTED)
    ax.axvline(central, color=INK, lw=1)
    ax.axvline(0, color=MUTED, lw=1, ls=":")
    ax.set_xlim(min(rows["net_m"].min(), central) - 120, max(rows["net_m"].max(), 0) + 80)
    ax.set_yticks(range(len(rows)), rows["sensitivity"], fontsize=8.5, color=MUTED)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(f"Net benefit, £m a year (central {central:,.0f}; dotted line = break-even)", color=MUTED, fontsize=9)
    ax.set_title(f"{POLICY_NAMES[FOCUS[1]]} on {FOCUS[0]}: what moves the answer", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    bands, baseline = load_inputs(MARTS)
    traffic, ratios = load_exposure(MARTS)
    distributions = speed_distributions(bands)
    values = load_values()
    central_assumptions = Assumptions.central()
    hours = hours_table(distributions)

    casualties = run(distributions, baseline, central_assumptions)
    central = appraise(casualties, hours, traffic, baseline, values, ratios)

    draws = pd.read_parquet(RESULTS / "casualty_model_draws.parquet")
    mc = monte_carlo(draws, hours, traffic, baseline, values, ratios)
    summary = summarise(central, mc)

    all_time = appraise(casualties, hours_table(distributions, count_speeding_time=True), traffic, baseline, values, ratios)
    compliance_all_time = all_time[all_time["policy"] == "compliance"]

    focus_road = FOCUS[0]
    focus = focus_row(central)
    focus_mc = mc[(mc["segment"] == focus_road) & (mc["policy"] == FOCUS[1])]
    low_flow, high_flow = FREE_FLOW_RANGES[focus_road]
    central_flow = FREE_FLOW_SHARE[focus_road]

    def benefit_under(assumptions):
        return focus_row(appraise(run(distributions, baseline, assumptions), hours, traffic, baseline, values, ratios))["benefit"]

    sens = sensitivity({
        "Central": (focus["benefit"], focus["time_cost"]),
        f"Free-flowing traffic {low_flow:.0%} (central {central_flow:.0%})": (focus["benefit"], focus["time_cost"] * low_flow / central_flow),
        f"Free-flowing traffic {high_flow:.0%}": (focus["benefit"], focus["time_cost"] * high_flow / central_flow),
        "Traffic on 97 km/h roads 30% lower": (focus["benefit"], focus["time_cost"] * 0.7),
        "Casualty draws: 5th percentile": (focus_mc["benefit"].quantile(0.05), focus["time_cost"]),
        "Casualty draws: 95th percentile": (focus_mc["benefit"].quantile(0.95), focus["time_cost"]),
        "Model A: Power Model": (focus["benefit_power"], focus["time_cost"]),
        "Model A: Exponential model": (focus["benefit_exp"], focus["time_cost"]),
        "Speed difference from today's mean": (benefit_under(replace(central_assumptions, reference_mean="old")), focus["time_cost"]),
        "Brenac's flatter risk curve": (benefit_under(replace(central_assumptions, urban_curve="brenac2015_exponential_coef",
                                                              rural_curve="brenac2015_exponential_coef")), focus["time_cost"]),
        "Previous TAG casualty values (RAS4001)": (
            focus_row(appraise(casualties, hours, traffic, baseline, load_values("previous"), ratios))["benefit"], focus["time_cost"]),
        "Injuries never reported to police counted": (
            focus_row(appraise(casualties, hours, traffic, baseline, values, ratios, include_unreported=True))["benefit"],
            focus["time_cost"]),
        "Time gained by speeding counted": (focus["benefit"], focus_row(all_time)["time_cost"]),
    })

    # Share of the 2035 target. BAU ISA is measured against today; the others against BAU ISA.
    today_benefits = benefits(casualties, baseline, values, ratios, suffix="")
    bau = casualties["policy"].eq("bau_isa").to_numpy()
    compliance_mc = mc[mc["policy"] == "compliance"].groupby("draw")["ksi_prevented"].sum()
    focus_ksi = summary[(summary["segment"] == focus_road) & (summary["policy"] == FOCUS[1])]["ksi_prevented"].iloc[0]
    focus_ksi_mc = focus_mc["ksi_prevented"]
    baseline_ksi = baseline["ksi"].sum()
    target = {
        "baseline": baseline_ksi,
        "required": TARGET_CUT * baseline_ksi,
        "rows": {
            "Business-as-usual ISA, against today": (today_benefits.loc[bau, "ksi_prevented"].sum(), np.nan, np.nan, "at least "),
            "Posted-limit compliance, beyond business as usual": (
                summary[summary["policy"] == "compliance"]["ksi_prevented"].sum(), *compliance_mc.quantile([0.05, 0.95]), "at least "),
            f"{POLICY_NAMES[FOCUS[1]]} on {focus_road}, beyond business as usual": (
                focus_ksi, *focus_ksi_mc.quantile([0.05, 0.95]), ""),
        },
    }

    summary.to_parquet(RESULTS / "appraisal.parquet", index=False, compression="zstd")
    chart_bcr(summary, FIGURES / "rq3_benefit_cost_ratio.png")
    chart_sensitivity(sens, FIGURES / "rq3_cap80_sensitivity.png")
    print_findings(summary, values, traffic, ratios, baseline, compliance_all_time, sens, target)
    print(f"\nWrote {len(summary)} rows to results/appraisal.parquet and two charts to figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
