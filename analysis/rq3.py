"""Phase 7: do the policies pay for themselves? Casualties, fuel and carbon against travel time lost (RQ3).

Combines the casualty model (Phase 6), DfT's appraisal values (model/parameters/
appraisal_values.csv) and car and van traffic per road type (data/marts/agg_traffic_segment).
Every policy is measured against business-as-usual ISA, in £ a year. "Ratio" throughout is annual
benefit over annual time cost: there are no present values and no hardware costs. Runs:
    central       the central assumptions
    Monte Carlo   the 10,000 casualty draws from Phase 6, each with a free-flow share drawn too
    sensitivity   one assumption changed at a time, for the 80 km/h cap on 97 km/h roads
    bound         for caps beyond the evidence, the most they could possibly save
then prints the findings and writes:
    results/appraisal.parquet               central result and Monte Carlo percentiles
    results/appraisal_sensitivity.parquet   the 80 km/h cap on 97 km/h roads, one assumption changed at a time
    results/appraisal_flat.parquet          a flat limiter: the four road types added up
    results/appraisal_bound.parquet         caps beyond the evidence: the best ratio possible
    results/appraisal_all_time.parquet      compliance if time gained by speeding were counted
    results/appraisal_target.parquet        shares of the 2035 target
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

from analysis.rq2_rq4 import COLOURS, GRID, INK, MUTED, POLICY_NAMES, SHOWN, SURFACE, colour, same_as_compliance
from model.appraisal import (FREE_FLOW_RANGES, FREE_FLOW_SHARE, break_even_free_flow, capital_cost, casualty_benefit,
                             hours_lost_per_km, ksi_from_casualties, load_values, read_values, time_cost)
from model.casualties import Assumptions
from model.run import BASELINE_YEARS, load_inputs, run, speed_distributions
from model.scenarios import POLICIES, ROAD_TYPES, IsaUse, held_to_limit
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


def hours_table(distributions, count_speeding_time=False, roads=ROAD_TYPES, isa=None) -> pd.DataFrame:
    """Hours lost per car-km and per van-km beyond business-as-usual ISA, per road type and policy.

    Lawful hours do not depend on how often ISA is switched off or overridden: a cap below a
    vehicle's limit binds it whatever ISA does, and a cap above costs no lawful time. So the central
    shares serve every Monte Carlo draw.
    """
    isa = IsaUse.central() if isa is None else isa
    rows = []
    for road in roads:
        traffic = distributions[road.segment]
        lost = {}
        for vehicle, today in (("car", traffic.cars), ("van", traffic.vans)):
            working = held_to_limit(today, road, isa, vehicle)
            lost[vehicle] = {p: hours_lost_per_km(p, today, road, working, count_speeding_time, vehicle,
                                                  isa.limit_read(road, vehicle)) for p in POLICIES}
        rows += [{"segment": road.segment, "policy": p,
                  "car_hours_per_km": lost["car"][p] - lost["car"]["bau_isa"],
                  "van_hours_per_km": lost["van"][p] - lost["van"]["bau_isa"]} for p in POLICIES]
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

    def per_year(item):
        """£ or kg a year saved beyond business-as-usual ISA by the free-flowing share of traffic."""
        return (table[f"car_{item}_saved_per_km_vs_bau"] * cars + table[f"van_{item}_saved_per_km_vs_bau"] * vans) * 1e9 * share

    table["fuel_benefit"] = per_year("fuel")
    table["carbon_benefit"] = per_year("co2") / 1000 * values.carbon_tonne
    table["tax_lost"] = per_year("tax")
    table["casualty_benefit"] = table["benefit"]
    running = table["fuel_benefit"] + table["carbon_benefit"]
    for column in ("benefit", "benefit_b", "benefit_power", "benefit_exp"):
        table[column] = table[column] + running

    table["costs_lawful_time"] = table["car_hours_per_km"] + table["van_hours_per_km"] > 1e-12
    table["time_cost"] = time_cost(table["car_hours_per_km"], table["van_hours_per_km"], cars, vans, values, share)
    table["capital_cost"] = capital_cost(table["car_hours_per_km"], table["van_hours_per_km"], cars, vans, values, share)
    table["van_time_cost"] = time_cost(0.0, table["van_hours_per_km"], cars, vans, values, share)
    table["cost"] = table["time_cost"] + table["capital_cost"]
    table["net"] = table["benefit"] - table["cost"]
    paid = table["cost"].where(table["cost"] > 0)
    table["bcr"] = np.where(table["cost"] > 0, table["benefit"] / paid, np.inf)
    table["break_even_free_flow"] = break_even_free_flow(table["casualty_benefit"].to_numpy(), (running / share).to_numpy(),
                                                         (table["cost"] / share).to_numpy())
    for model in MODELS:
        table[f"bcr_{model}"] = table[f"benefit_{model}"] / paid
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
    q = mc.groupby(keys, sort=False)[["benefit", "cost", "net", "bcr"]].quantile([0.05, 0.5, 0.95]).unstack()
    q.columns = [f"{column}_p{round(p * 100)}" for column, p in q.columns]
    q["chance_net_positive"] = mc.assign(pays=mc["net"] > 0).groupby(keys, sort=False)["pays"].mean()
    keep = ["segment", "policy", "evidence_vs_bau", "car_hours_per_km", "van_hours_per_km", "costs_lawful_time",
            "free_flow_share", "ksi_prevented", "casualty_benefit", "fuel_benefit", "carbon_benefit", "tax_lost",
            "benefit", "benefit_b", "benefit_power", "benefit_exp", "time_cost", "capital_cost", "van_time_cost", "cost",
            "net", "bcr", "bcr_b",
            "bcr_power", "bcr_exp", "break_even_free_flow"]
    return central[keep].merge(q.reset_index(), on=keys, how="left")


def millions(x: float) -> str:
    return f"{x / 1e6:,.0f}"


def model_range(row, prefix: str) -> tuple[float, float]:
    values = [row[f"{prefix}_{m}"] for m in MODELS if pd.notna(row[f"{prefix}_{m}"])]
    return min(values), max(values)


def focus_row(table: pd.DataFrame) -> pd.Series:
    return table[(table["segment"] == FOCUS[0]) & (table["policy"] == FOCUS[1])].iloc[0]


def best_possible(summary: pd.DataFrame, baseline: pd.DataFrame, values, ratios) -> pd.DataFrame:
    """For caps beyond the evidence: the ratio if the cap prevented every casualty on that road type.

    No speed-risk model is needed: if even this falls short of the time cost, the cap fails whatever
    its safety effect. Fuel and carbon are added at their modelled values. It holds on TAG's convention
    of valuing reported casualties, so the same bound is given with unreported injuries counted too.
    """
    rows = summary[summary["costs_lawful_time"] & (summary["evidence_vs_bau"] == "beyond")
                   & summary["policy"].isin(SHOWN)].copy()
    base = baseline.loc[rows["segment"]]

    def every_casualty(include_unreported):
        return casualty_benefit(base["killed"].to_numpy(), base["ksi"].to_numpy(), base["casualties"].to_numpy(),
                                values, ratios, include_unreported)

    rows["every_casualty"] = every_casualty(False)
    rows["every_casualty_unreported"] = every_casualty(True)
    running = rows["fuel_benefit"] + rows["carbon_benefit"]
    rows["best_ratio"] = (rows["every_casualty"] + running) / rows["cost"]
    rows["best_ratio_unreported"] = (rows["every_casualty_unreported"] + running) / rows["cost"]
    return rows


def sensitivity(cases: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Net benefit and ratio of the focus policy, from (benefit, cost) per case."""
    return pd.DataFrame([{"sensitivity": name, "net": benefit - cost, "bcr": benefit / cost}
                         for name, (benefit, cost) in cases.items()])


def pays_if(share: float) -> str:
    """The break-even free-flow share in words."""
    if share >= 1:
        return "pays even if all traffic flows freely"
    return f"pays if under {share:.0%} of traffic flows freely"


def print_findings(summary, values, traffic, ratios, baseline, compliance_all_time, sens, target, bound, flat):
    print("Phase 7 appraisal: £ a year beyond business-as-usual ISA, 2023 prices and values (TAG data book v2.03),")
    print("every car fitted, 2022-2024 casualty and traffic levels")
    print(f"Values: £{values.per_casualty['fatal']:,.0f} per death, £{values.per_casualty['serious']:,.0f} per serious and "
          f"£{values.per_casualty['slight']:,.0f} per slight injury, plus per-collision costs")
    print(f"        (collisions per casualty: fatal {ratios['fatal']:.3f}, serious {ratios['serious']:.3f}, "
          f"slight {ratios['slight']:.3f}); £{values.car_hour:.2f} per car-hour, £{values.van_hour:.2f} per van-hour,")
    print(f"        £{values.car_capital_hour:.2f} and £{values.van_capital_hour:.2f} of vehicle capital per car- and van-hour; "
          f"£{values.carbon_tonne:.2f} per tonne of CO2e")
    print("Car and van traffic, billion vehicle-km a year: " + ", ".join(
        f"{s} {traffic.loc[s, 'cars'] + traffic.loc[s, 'vans']:.1f}" for s in FREE_FLOW_SHARE))
    print("Free-flowing share of traffic (a judgement, no published source): 40% on 32 and 48 km/h roads, "
          "70% on 97 km/h roads and motorways\n")

    print("Ratio: annual benefit / annual time cost, with no present values and no hardware cost\n")

    print("1. Policies that cost no lawful time: every second they cost was gained by speeding (£m a year)")
    free = summary[~summary["costs_lawful_time"] & (summary["benefit"] > 0.5e6) & summary["policy"].isin(SHOWN)]
    free = free[[not same_as_compliance(s, p) for s, p in zip(free["segment"], free["policy"])]]
    for _, r in free.iterrows():
        note = {"lower bound": ", a lower bound", "beyond": ", beyond the evidence"}.get(r["evidence_vs_bau"], "")
        print(f"   {r['segment']:28s} {POLICY_NAMES[r['policy']]:24s} benefit {millions(r['benefit']):>5s} "
              f"[{millions(r['benefit_p5'])} to {millions(r['benefit_p95'])}]{note}; "
              f"of which fuel {millions(r['fuel_benefit'])}, carbon {millions(r['carbon_benefit'])}; "
              f"tax lost {millions(r['tax_lost'])}")
    compliance = free[free["policy"] == "compliance"]
    print(f"   {'All four road types':28s} {POLICY_NAMES['compliance']:24s} benefit {millions(compliance['benefit'].sum())} "
          f"[{millions(target['compliance_benefit_p5'])} to {millions(target['compliance_benefit_p95'])}]")

    print("\n2. Caps that cost lawful time, by road type (£m a year): central [5th to 95th percentile]")
    costly = summary[summary["costs_lawful_time"] & summary["policy"].isin(SHOWN)]
    for _, r in costly.iterrows():
        head = f"   {r['segment']:28s} {POLICY_NAMES[r['policy']]:16s}"
        if r["evidence_vs_bau"] == "within":
            print(f"{head} casualties {millions(r['casualty_benefit'])} + fuel {millions(r['fuel_benefit'])} + "
                  f"carbon {millions(r['carbon_benefit'])} = {millions(r['benefit'])}; time {millions(r['time_cost'])} + "
                  f"vehicle capital {millions(r['capital_cost'])} = {millions(r['cost'])}; tax lost {millions(r['tax_lost'])}; "
                  f"net {millions(r['net'])} [{millions(r['net_p5'])} to {millions(r['net_p95'])}], "
                  f"ratio {r['bcr']:.2f} [{r['bcr_p5']:.2f} to {r['bcr_p95']:.2f}], {pays_if(r['break_even_free_flow'])} "
                  f"(chance it pays: {r['chance_net_positive']:.0%})")
        else:
            low, high = model_range(r, "benefit")
            bcr_low, bcr_high = model_range(r, "bcr")
            print(f"{head} beyond the evidence: benefit {millions(low)} to {millions(high)} across models, "
                  f"of which fuel {millions(r['fuel_benefit'])}, carbon {millions(r['carbon_benefit'])}; "
                  f"cost {millions(r['cost'])}, ratio {bcr_low:.2f} to {bcr_high:.2f}")

    print("\n   A flat limiter on every road: the four road types added up (£m a year)")
    for policy, r in flat.iterrows():
        print(f"   {POLICY_NAMES[policy]:16s} benefit {millions(r['benefit']):>6s}, cost {millions(r['cost']):>6s}, "
              f"net {millions(r['net']):>7s}, ratio {r['ratio']:.2f}, deaths prevented {r['deaths']:.0f}")

    print("\n   Best possible: caps beyond the evidence, if they prevented every casualty on the road type (£m a year)")
    for _, r in bound.iterrows():
        print(f"   {r['segment']:28s} {POLICY_NAMES[r['policy']]:16s} every casualty {millions(r['every_casualty']):>5s}, "
              f"fuel and carbon {millions(r['fuel_benefit'] + r['carbon_benefit']):>5s}, cost {millions(r['cost']):>6s}, "
              f"ratio at most {r['best_ratio']:.2f}; with unreported injuries {r['best_ratio_unreported']:.2f}")

    print("\n3. If time gained by speeding were counted too: posted-limit compliance (£m a year)")
    for _, r in compliance_all_time.iterrows():
        print(f"   {r['segment']:28s} benefit {millions(r['benefit']):>5s}, cost {millions(r['cost']):>5s}, "
              f"net {millions(r['net']):>6s}, {pays_if(r['break_even_free_flow'])}")

    print(f"\n4. {POLICY_NAMES[FOCUS[1]]} on {FOCUS[0]}: net benefit (£m a year), one assumption changed at a time")
    for _, r in sens.iterrows():
        print(f"   {r['sensitivity']:46s} {millions(r['net']):>6s}   ratio {r['bcr']:.2f}")

    print(f"\n5. Share of the 2035 target: {TARGET_CUT:.0%} fewer KSI than {target['baseline']:,.0f} a year "
          f"means {target['required']:,.0f} fewer a year")
    for name, (ksi, p5, p95) in target["rows"].items():
        print(f"   {name:58s} {ksi:6,.0f} KSI [{p5:,.0f} to {p95:,.0f}] = {ksi / target['required']:.0%} of the cut")
    print("   These are full-fleet steady states at 2022-2024 traffic and casualty levels, not forecasts for 2035.")


def chart_bcr(summary: pd.DataFrame, path: Path) -> None:
    """Benefit-cost ratio of every cap that costs lawful time, with the evidence shown."""
    rows = summary[summary["costs_lawful_time"] & summary["policy"].isin(SHOWN)].reset_index(drop=True)
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
            ax.plot([low * 0.985, high * 1.015], [y, y], color=c, lw=6, alpha=0.25, solid_capstyle="round")     # always visible
        labels.append(f"{r['segment']}: {POLICY_NAMES[r['policy']].lower()}")
    ax.axvline(1, color=INK, lw=1)
    ax.text(1.1, -0.75, "pays for itself →", fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.03, 30)
    ax.set_ylim(len(rows) - 0.5, -1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_yticks(range(len(rows)), labels, fontsize=8.5, color=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Annual benefit / annual time cost, beyond business-as-usual ISA (log scale)", color=MUTED, fontsize=9)
    ax.set_title("Caps below the limit: casualties, fuel and carbon against lawful time lost",
                 loc="left", fontsize=11, color=INK)
    legend = [plt.Line2D([], [], color=COLOURS["cap"], marker="o", lw=2, label="Within the evidence: central, 5th-95th percentile"),
              plt.Line2D([], [], color=COLOURS["cap"], lw=6, alpha=0.25, label="Beyond the evidence: range across models")]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=2, frameon=False, fontsize=8,
              labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor=SURFACE)
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
    fig.savefig(path, dpi=300, facecolor=SURFACE)
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

    flat = central[central["policy"].isin(SHOWN)].groupby("policy", sort=False).agg(
        benefit=("benefit", "sum"), cost=("cost", "sum"), net=("net", "sum"), deaths=("killed_prevented_vs_bau", "sum"))
    flat["ratio"] = flat["benefit"] / flat["cost"].where(flat["cost"] > 0)
    flat = flat.drop("compliance")
    bound = best_possible(summary, baseline, values, ratios)

    focus_road = FOCUS[0]
    focus = focus_row(central)
    focus_mc = mc[(mc["segment"] == focus_road) & (mc["policy"] == FOCUS[1])]
    low_flow, high_flow = FREE_FLOW_RANGES[focus_road]
    central_flow = FREE_FLOW_SHARE[focus_road]

    vans_like_cars = tuple(replace(road, van_limit_kmh=road.limit_kmh) for road in ROAD_TYPES)

    def benefit_under(assumptions):
        return focus_row(appraise(run(distributions, baseline, assumptions), hours, traffic, baseline, values, ratios))["benefit"]

    def scaled(factor):
        """Benefit and cost if the free-flowing traffic were this many times larger: casualties do not scale."""
        return (focus["casualty_benefit"] + (focus["fuel_benefit"] + focus["carbon_benefit"]) * factor, focus["cost"] * factor)

    def with_values(table, **changes):
        return focus_row(appraise(table, hours, traffic, baseline, replace(values, **changes), ratios))

    raw_values = read_values()
    later_fleet = with_values(run(distributions, baseline, central_assumptions, fuel_year=2035),
                              carbon_tonne=raw_values["carbon_central_2035"])
    sens = sensitivity({
        "Central": (focus["benefit"], focus["cost"]),
        f"Free-flowing traffic {low_flow:.0%} (central {central_flow:.0%})": scaled(low_flow / central_flow),
        f"Free-flowing traffic {high_flow:.0%}": scaled(high_flow / central_flow),
        "Traffic on 97 km/h roads 30% lower": scaled(0.7),
        "Traffic on 97 km/h roads 30% higher": scaled(1.3),
        "Casualty draws: 5th percentile": (focus_mc["casualty_benefit"].quantile(0.05) + focus["benefit"] - focus["casualty_benefit"], focus["cost"]),
        "Casualty draws: 95th percentile": (focus_mc["casualty_benefit"].quantile(0.95) + focus["benefit"] - focus["casualty_benefit"], focus["cost"]),
        "Model A: Power Model": (focus["benefit_power"], focus["cost"]),
        "Model A: Exponential model": (focus["benefit_exp"], focus["cost"]),
        "Speed difference from today's mean": (benefit_under(replace(central_assumptions, reference_mean="old")), focus["cost"]),
        "Adelaide's urban free-travel share (0.56)": (
            benefit_under(replace(central_assumptions, free_travel_share_rural=central_assumptions.free_travel_share)), focus["cost"]),
        "Brenac's flatter risk curve": (benefit_under(replace(central_assumptions, urban_curve="brenac2015_exponential_coef",
                                                              rural_curve="brenac2015_exponential_coef")), focus["cost"]),
        "Previous TAG casualty values (RAS4001)": (
            focus_row(appraise(casualties, hours, traffic, baseline, load_values("previous"), ratios))["benefit"], focus["cost"]),
        "Injuries never reported to police counted": (
            focus_row(appraise(casualties, hours, traffic, baseline, values, ratios, include_unreported=True))["benefit"],
            focus["cost"]),
        "Time gained by speeding counted": (focus["benefit"], focus_row(all_time)["cost"]),
        "Vans have the car limit": (focus_row(appraise(run(distributions, baseline, central_assumptions, roads=vans_like_cars),
                                                       hours_table(distributions, roads=vans_like_cars), traffic, baseline,
                                                       values, ratios))["benefit"],
                                    focus_row(appraise(casualties, hours_table(distributions, roads=vans_like_cars), traffic,
                                                       baseline, values, ratios))["cost"]),
        "Fuel and carbon left out": (focus["casualty_benefit"], focus["cost"]),
        "Fleet, prices and carbon value of 2035": (later_fleet["benefit"], focus["cost"]),
        "Carbon at TAG's low value": (with_values(casualties, carbon_tonne=raw_values["carbon_low_2023"])["benefit"], focus["cost"]),
        "Carbon at TAG's high value": (with_values(casualties, carbon_tonne=raw_values["carbon_high_2023"])["benefit"], focus["cost"]),
    })

    # Share of the 2035 target. BAU ISA is measured against today; the others against BAU ISA.
    today_benefits = benefits(casualties, baseline, values, ratios, suffix="")
    bau = casualties["policy"].eq("bau_isa").to_numpy()
    compliance_mc = mc[mc["policy"] == "compliance"].groupby("draw")["ksi_prevented"].sum()
    compliance_benefit_mc = mc[mc["policy"] == "compliance"].groupby("draw")["benefit"].sum()
    bau_mc = draws[(draws["policy"] == "bau_isa") & draws["segment"].isin(FREE_FLOW_SHARE)].reset_index(drop=True)
    bau_ksi_mc = benefits(bau_mc, baseline, values, ratios, suffix="").assign(draw=bau_mc["draw"]).groupby("draw")["ksi_prevented"].sum()
    focus_ksi = summary[(summary["segment"] == focus_road) & (summary["policy"] == FOCUS[1])]["ksi_prevented"].iloc[0]
    focus_ksi_mc = focus_mc["ksi_prevented"]
    baseline_ksi = baseline["ksi"].sum()
    target = {
        "baseline": baseline_ksi,
        "required": TARGET_CUT * baseline_ksi,
        "compliance_benefit_p5": compliance_benefit_mc.quantile(0.05),
        "compliance_benefit_p95": compliance_benefit_mc.quantile(0.95),
        "rows": {
            "Business-as-usual ISA, against today": (today_benefits.loc[bau, "ksi_prevented"].sum(), *bau_ksi_mc.quantile([0.05, 0.95])),
            "Posted-limit compliance, beyond business as usual": (
                summary[summary["policy"] == "compliance"]["ksi_prevented"].sum(), *compliance_mc.quantile([0.05, 0.95])),
            f"{POLICY_NAMES[FOCUS[1]]} on {focus_road}, beyond business as usual": (
                focus_ksi, *focus_ksi_mc.quantile([0.05, 0.95])),
        },
    }

    summary.to_parquet(RESULTS / "appraisal.parquet", index=False, compression="zstd")
    sens.to_parquet(RESULTS / "appraisal_sensitivity.parquet", index=False, compression="zstd")
    flat.reset_index().to_parquet(RESULTS / "appraisal_flat.parquet", index=False, compression="zstd")
    bound.to_parquet(RESULTS / "appraisal_bound.parquet", index=False, compression="zstd")
    compliance_all_time[["segment", "benefit", "cost", "net", "break_even_free_flow"]].to_parquet(
        RESULTS / "appraisal_all_time.parquet", index=False, compression="zstd")
    pd.DataFrame([{"row": name, "ksi": ksi, "p5": p5, "p95": p95, "share": ksi / target["required"],
                   "required": target["required"], "baseline": target["baseline"]}
                  for name, (ksi, p5, p95) in target["rows"].items()]).to_parquet(
        RESULTS / "appraisal_target.parquet", index=False, compression="zstd")
    chart_bcr(summary, FIGURES / "rq3_benefit_cost_ratio.png")
    chart_sensitivity(sens, FIGURES / "rq3_cap80_sensitivity.png")
    print_findings(summary, values, traffic, ratios, baseline, compliance_all_time, sens, target, bound, flat)
    print(f"\nWrote {len(summary)} rows to results/appraisal.parquet and two charts to figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
