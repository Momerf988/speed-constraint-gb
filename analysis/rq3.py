"""Phase 7: do the policies pay for themselves? Casualties, fuel and carbon against travel time lost (RQ3).

Combines the casualty model (Phase 6), DfT's appraisal values (model/parameters/
appraisal_values.csv) and car and van traffic per road type (data/marts/agg_traffic_segment).
Every policy is measured against business-as-usual ISA, in £ a year, and every cap also against
posted-limit compliance, which it contains (beyond_compliance): the comparison the cap must pass. "Ratio" throughout is annual
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
    results/appraisal_all_time_cap.parquet  the 80 km/h cap with all time counted, as Scotland's NSMR counts it
    results/appraisal_beyond_compliance.parquet   every cap judged against compliance, by road type and flat
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
from model.appraisal import (FREE_FLOW_RANGES, FREE_FLOW_SHARE, break_even_free_flow, cap_beyond_compliance, capital_cost,
                             casualty_benefit, hours_lost_per_km, ksi_from_casualties, load_values, read_values, time_cost)
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
FLAT = "All four road types"


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


ADDITIVE = ("killed_prevented_vs_bau", "ksi_prevented", "casualty_benefit", "fuel_benefit", "carbon_benefit", "benefit",
            "benefit_b", "benefit_power", "benefit_exp", "time_cost", "capital_cost", "cost")
ROAD_BY_SEGMENT = {road.segment: road for road in ROAD_TYPES}


def beyond_compliance(table: pd.DataFrame) -> pd.DataFrame:
    """Each cap judged against posted-limit compliance instead of business-as-usual ISA (incremental appraisal).

    Compliance is nested in every cap that binds below the limit, and costs no lawful time, so the cap
    must justify what it adds. For a cap at or above every vehicle's limit that is nothing; otherwise
    it is the cap's result minus compliance's (model.appraisal.cap_beyond_compliance). Works on the
    central table and on the Monte Carlo (one row per draw), keyed by segment, policy and any draw.
    """
    keys = ["draw", "segment"] if "draw" in table.columns else ["segment"]
    caps = table[table["policy"].str.startswith("cap_")].reset_index(drop=True)
    compliance = caps[keys].merge(table[table["policy"] == "compliance"], on=keys, how="left")
    adds = np.array([cap_beyond_compliance(ROAD_BY_SEGMENT[s], float(p.removeprefix("cap_")))
                     for s, p in zip(caps["segment"], caps["policy"])])
    out = caps[keys + ["policy", "free_flow_share"]].copy()
    out["adds_to_compliance"] = adds
    for column in ADDITIVE:
        out[column] = np.where(adds, caps[column].to_numpy() - compliance[column].to_numpy(), 0.0)
    return with_ratios(out)


def with_ratios(table: pd.DataFrame) -> pd.DataFrame:
    """Net benefit, ratio (per model) and break-even free-flow share from summed benefit and cost columns."""
    table = table.copy()
    running = table["fuel_benefit"] + table["carbon_benefit"]
    paid = table["cost"].where(table["cost"] > 1e-6)
    table["net"] = table["benefit"] - table["cost"]
    table["bcr"] = table["benefit"] / paid
    for model in MODELS:
        table[f"bcr_{model}"] = table[f"benefit_{model}"] / paid
    if "free_flow_share" in table:
        share = table["free_flow_share"]
        table["break_even_free_flow"] = break_even_free_flow(table["casualty_benefit"].to_numpy(), (running / share).to_numpy(),
                                                             (table["cost"] / share).to_numpy())
    return table


def flat_beyond_compliance(increments: pd.DataFrame) -> pd.DataFrame:
    """A flat limiter on every road, beyond compliance: the road types added up, per cap (and per draw)."""
    keys = ["draw", "policy"] if "draw" in increments.columns else ["policy"]
    flat = increments.groupby(keys, sort=False)[list(ADDITIVE)].agg(lambda x: x.sum(skipna=False)).reset_index()
    return with_ratios(flat)


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


def compliance_row(table: pd.DataFrame) -> pd.Series:
    return table[(table["segment"] == FOCUS[0]) & (table["policy"] == "compliance")].iloc[0]


def summarise_increments(increments, increments_mc, flat, flat_mc) -> pd.DataFrame:
    """Every cap beyond compliance, by road type and as a flat limiter, with Monte Carlo percentiles."""
    rows = pd.concat([increments, flat.assign(segment=FLAT, adds_to_compliance=True)], ignore_index=True)
    draws = pd.concat([increments_mc, flat_mc.assign(segment=FLAT, adds_to_compliance=True)], ignore_index=True)
    keys = ["segment", "policy"]
    q = draws.groupby(keys, sort=False)[["benefit", "cost", "net", "bcr", "ksi_prevented", "killed_prevented_vs_bau"]] \
        .quantile([0.05, 0.5, 0.95]).unstack()
    q.columns = [f"{column}_p{round(p * 100)}" for column, p in q.columns]
    q["chance_net_positive"] = draws.assign(pays=draws["net"] > 0).groupby(keys, sort=False)["pays"].mean()
    return rows.merge(q.reset_index(), on=keys, how="left")


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


def sensitivity(cases: dict[str, tuple[float, float, float, float]]) -> pd.DataFrame:
    """Net benefit and ratio of the focus cap per case, beyond BAU and beyond compliance.

    Each case gives (benefit, cost) beyond business-as-usual ISA, then (benefit, cost) beyond compliance.
    """
    return pd.DataFrame([{"sensitivity": name, "net": benefit - cost, "bcr": benefit / cost,
                          "net_vs_compliance": extra - extra_cost, "bcr_vs_compliance": extra / extra_cost}
                         for name, (benefit, cost, extra, extra_cost) in cases.items()])


def short_pays(share: float) -> str:
    return "pays at any free-flow share" if share >= 1 else f"pays below {share:.0%} free flow"


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


def print_beyond_compliance(extra: pd.DataFrame, sens: pd.DataFrame, nsmr: pd.DataFrame) -> None:
    print("\n6. Caps judged against compliance, which they contain: what each adds (£m a year)")
    for _, r in extra[extra["adds_to_compliance"].astype(bool)].iterrows():
        if not r["policy"] in SHOWN:
            continue
        ratio = "" if pd.isna(r["bcr"]) else (f"ratio {r['bcr']:.2f} [{r['bcr_p5']:.2f} to {r['bcr_p95']:.2f}], "
                                              f"models {model_range(r, 'bcr')[0]:.2f} to {model_range(r, 'bcr')[1]:.2f}")
        print(f"   {r['segment']:28s} {POLICY_NAMES[r['policy']]:16s} casualties {millions(r['casualty_benefit']):>5s} + fuel "
              f"{millions(r['fuel_benefit'])} + carbon {millions(r['carbon_benefit'])} = {millions(r['benefit']):>5s}; cost "
              f"{millions(r['cost']):>5s}; net {millions(r['net']):>6s}; {ratio}; deaths {r['killed_prevented_vs_bau']:.0f}, "
              f"KSI {r['ksi_prevented']:.0f}; "
              + ("" if pd.isna(r["break_even_free_flow"]) else f"{pays_if(r['break_even_free_flow'])}; ")
              + f"chance it pays {r['chance_net_positive']:.0%}")
    print("   Caps at or above every vehicle's limit add nothing to compliance and are left out.")
    print(f"\n   {POLICY_NAMES[FOCUS[1]]} on {FOCUS[0]}, one assumption changed at a time:  against BAU  |  against compliance")
    for _, r in sens.iterrows():
        print(f"   {r['sensitivity']:46s} ratio {r['bcr']:5.2f}, net {millions(r['net']):>5s}  |  ratio {r['bcr_vs_compliance']:5.2f}, "
              f"net {millions(r['net_vs_compliance']):>5s}")
    n = nsmr.iloc[0]
    print(f"\n7. Counting all time, as Scotland's NSMR does: casualty value / all time lost (time only; with vehicle capital)")
    for label, column in (("Model B", "casualty_benefit"), ("Model A, Power", "casualty_benefit_power"),
                          ("Model A, Exponential", "casualty_benefit_exp")):
        print(f"   {label:22s} {millions(n[column]):>5s} / {millions(n['time_cost'])} = {n[column] / n['time_cost']:.3f}; "
              f"/ {millions(n['cost'])} = {n[column] / n['cost']:.3f}; against lawful time only {n[column] / n['lawful_time_cost']:.3f}")
    print("   NSMR (Transport Scotland 2024, Option 1): 440 / 443 = 0.993 at full compliance; 259 / 223 = 1.161 at realistic compliance")


def chart_bcr(summary: pd.DataFrame, extra: pd.DataFrame, path: Path) -> None:
    """Benefit-cost ratio of every cap that costs lawful time, with the evidence shown."""
    rows = summary[summary["costs_lawful_time"] & summary["policy"].isin(SHOWN)].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    labels = []
    for y, r in rows.iterrows():
        c = colour(r["policy"])
        more = extra[(extra["segment"] == r["segment"]) & (extra["policy"] == r["policy"]) & extra["adds_to_compliance"]]
        if r["evidence_vs_bau"] == "within":
            ax.plot([r["bcr_p5"], r["bcr_p95"]], [y - 0.13, y - 0.13], color=c, lw=2)
            ax.plot(r["bcr"], y - 0.13, "o", ms=8, color=c)
            if len(more):
                m = more.iloc[0]
                ax.plot([m["bcr_p5"], m["bcr_p95"]], [y + 0.17, y + 0.17], color=c, lw=1.2, ls=(0, (3, 2)))
                ax.plot(m["bcr"], y + 0.17, "o", ms=7.5, mfc=SURFACE, mec=c, mew=1.8)
                ax.text(r["bcr_p95"] * 1.12, y - 0.13, f"vs BAU: {short_pays(r['break_even_free_flow'])}", va="center",
                        fontsize=7.5, color=MUTED)
                ax.text(r["bcr_p95"] * 1.12, y + 0.17, f"vs compliance: {short_pays(m['break_even_free_flow'])}", va="center",
                        fontsize=7.5, color=MUTED)
            else:
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
    ax.set_xlabel("Annual benefit / annual lawful time cost (log scale)", color=MUTED, fontsize=9)
    ax.set_title("Caps below the limit: casualties, fuel and carbon against lawful time lost",
                 loc="left", fontsize=11, color=INK)
    legend = [plt.Line2D([], [], color=COLOURS["cap"], marker="o", lw=2, label="Within the evidence, against BAU ISA: central, 5th-95th percentile"),
              plt.Line2D([], [], color=COLOURS["cap"], marker="o", mfc=SURFACE, lw=1.2, ls=(0, (3, 2)),
                         label="The same, against compliance (what the cap adds)"),
              plt.Line2D([], [], color=COLOURS["cap"], lw=6, alpha=0.25, label="Beyond the evidence, against BAU ISA: range across the casualty models")]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=1, frameon=False, fontsize=8,
              labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(fig)


def chart_sensitivity(sens: pd.DataFrame, path: Path) -> None:
    """Net benefit of the focus cap beyond compliance, one assumption changed at a time, largest swing first."""
    central = sens.loc[sens["sensitivity"] == "Central", "net_vs_compliance"].iloc[0] / 1e6
    rows = sens[sens["sensitivity"] != "Central"].assign(net_m=lambda d: d["net_vs_compliance"] / 1e6)
    rows = rows.assign(swing=(rows["net_m"] - central).abs()).sort_values("swing").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for y, r in rows.iterrows():
        ax.barh(y, r["net_m"] - central, left=central, height=0.6, color=COLOURS["cap"])
        ax.text(r["net_m"] + (8 if r["net_m"] >= central else -8), y, f"{r['net_m']:,.0f}", va="center",
                ha="left" if r["net_m"] >= central else "right", fontsize=8, color=MUTED)
    ax.axvline(central, color=INK, lw=1)
    ax.axvline(0, color=MUTED, lw=1, ls=":")
    ax.set_xlim(min(rows["net_m"].min(), central) - 90, max(rows["net_m"].max(), 0) + 90)
    ax.set_yticks(range(len(rows)), rows["sensitivity"], fontsize=8.5, color=MUTED)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(f"Net benefit beyond compliance, £m a year (central {central:,.0f}; dotted line = break-even)", color=MUTED, fontsize=9)
    ax.set_title("80 km/h cap on 97 km/h roads, beyond compliance: what moves the answer", loc="left",
                 fontsize=10.5, color=INK)
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

    increments = beyond_compliance(central)
    increments_mc = beyond_compliance(mc)
    flat_extra = flat_beyond_compliance(increments)
    flat_extra_mc = flat_beyond_compliance(increments_mc)
    extra = summarise_increments(increments, increments_mc, flat_extra, flat_extra_mc)
    focus_extra = focus_row(increments)
    focus_extra_mc = increments_mc[(increments_mc["segment"] == focus_road) & (increments_mc["policy"] == FOCUS[1])]

    vans_like_cars = tuple(replace(road, van_limit_kmh=road.limit_kmh) for road in ROAD_TYPES)

    def case(table, cost_table=None, column="benefit"):
        """(benefit, cost) beyond BAU and beyond compliance for the focus cap, from one appraised table."""
        cost_table = table if cost_table is None else cost_table
        cap, comp = focus_row(table), compliance_row(table)
        cap_cost, comp_cost = focus_row(cost_table)["cost"], compliance_row(cost_table)["cost"]
        return cap[column], cap_cost, cap[column] - comp[column], cap_cost - comp_cost

    def under(assumptions):
        return case(appraise(run(distributions, baseline, assumptions), hours, traffic, baseline, values, ratios))

    def scaled(factor):
        """Free-flowing traffic this many times larger: running savings and time scale, casualties do not."""
        def one(r):
            return r["casualty_benefit"] + (r["fuel_benefit"] + r["carbon_benefit"]) * factor, r["cost"] * factor
        (b, c), (eb, ec) = one(focus), one(focus_extra)
        return b, c, eb, ec

    def with_values(table, **changes):
        return appraise(table, hours, traffic, baseline, replace(values, **changes), ratios)

    def casualty_draw(q):
        cap = focus_mc["casualty_benefit"].quantile(q) + focus["benefit"] - focus["casualty_benefit"]
        more = focus_extra_mc["casualty_benefit"].quantile(q) + focus_extra["benefit"] - focus_extra["casualty_benefit"]
        return cap, focus["cost"], more, focus_extra["cost"]

    raw_values = read_values()
    later_fleet = with_values(run(distributions, baseline, central_assumptions, fuel_year=2035),
                              carbon_tonne=raw_values["carbon_central_2035"])
    vans_hours = hours_table(distributions, roads=vans_like_cars)
    sens = sensitivity({
        "Central": case(central),
        f"Free-flowing traffic {low_flow:.0%} (central {central_flow:.0%})": scaled(low_flow / central_flow),
        "Free-flowing traffic 60%": scaled(0.6 / central_flow),
        f"Free-flowing traffic {high_flow:.0%}": scaled(high_flow / central_flow),
        "Traffic on 97 km/h roads 30% lower": scaled(0.7),
        "Traffic on 97 km/h roads 30% higher": scaled(1.3),
        "Casualty draws: 5th percentile": casualty_draw(0.05),
        "Casualty draws: 95th percentile": casualty_draw(0.95),
        "Model A: Power Model": case(central, column="benefit_power"),
        "Model A: Exponential model": case(central, column="benefit_exp"),
        "Speed difference from today's mean": under(replace(central_assumptions, reference_mean="old")),
        "Adelaide's urban free-travel share (0.56)": under(
            replace(central_assumptions, free_travel_share_rural=central_assumptions.free_travel_share)),
        "Brenac's flatter risk curve": under(replace(central_assumptions, urban_curve="brenac2015_exponential_coef",
                                                     rural_curve="brenac2015_exponential_coef")),
        "Previous TAG casualty values (RAS4001)": case(appraise(casualties, hours, traffic, baseline, load_values("previous"), ratios)),
        "Injuries never reported to police counted": case(
            appraise(casualties, hours, traffic, baseline, values, ratios, include_unreported=True)),
        "Time gained by speeding counted": case(central, all_time),
        "Vans have the car limit": case(appraise(run(distributions, baseline, central_assumptions, roads=vans_like_cars),
                                                 vans_hours, traffic, baseline, values, ratios),
                                        appraise(casualties, vans_hours, traffic, baseline, values, ratios)),
        "Fuel and carbon left out": case(central, column="casualty_benefit"),
        "Fleet, prices and carbon value of 2035": case(later_fleet, central),
        "Carbon at TAG's low value": case(with_values(casualties, carbon_tonne=raw_values["carbon_low_2023"])),
        "Carbon at TAG's high value": case(with_values(casualties, carbon_tonne=raw_values["carbon_high_2023"])),
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
            f"{POLICY_NAMES[FOCUS[1]]} on {focus_road}, beyond compliance": (
                focus_extra["ksi_prevented"], *focus_extra_mc["ksi_prevented"].quantile([0.05, 0.95])),
        },
    }

    # Scotland's NSMR counts all travel time, as a lower limit for every vehicle would remove it. The same
    # convention here: the focus cap's casualty value against all time lost, lawful or gained by speeding.
    all_time_focus = focus_row(all_time)
    running = all_time_focus["fuel_benefit"] + all_time_focus["carbon_benefit"]
    nsmr = pd.DataFrame([{
        "segment": focus_road, "policy": FOCUS[1], "casualty_benefit": all_time_focus["casualty_benefit"],
        "casualty_benefit_power": all_time_focus["benefit_power"] - running,
        "casualty_benefit_exp": all_time_focus["benefit_exp"] - running,
        "time_cost": all_time_focus["time_cost"], "cost": all_time_focus["cost"],
        "lawful_time_cost": focus["time_cost"], "lawful_cost": focus["cost"]}])

    summary.to_parquet(RESULTS / "appraisal.parquet", index=False, compression="zstd")
    sens.to_parquet(RESULTS / "appraisal_sensitivity.parquet", index=False, compression="zstd")
    flat.reset_index().to_parquet(RESULTS / "appraisal_flat.parquet", index=False, compression="zstd")
    bound.to_parquet(RESULTS / "appraisal_bound.parquet", index=False, compression="zstd")
    compliance_all_time[["segment", "benefit", "cost", "net", "break_even_free_flow"]].to_parquet(
        RESULTS / "appraisal_all_time.parquet", index=False, compression="zstd")
    nsmr.to_parquet(RESULTS / "appraisal_all_time_cap.parquet", index=False, compression="zstd")
    extra.to_parquet(RESULTS / "appraisal_beyond_compliance.parquet", index=False, compression="zstd")
    pd.DataFrame([{"row": name, "ksi": ksi, "p5": p5, "p95": p95, "share": ksi / target["required"],
                   "required": target["required"], "baseline": target["baseline"]}
                  for name, (ksi, p5, p95) in target["rows"].items()]).to_parquet(
        RESULTS / "appraisal_target.parquet", index=False, compression="zstd")
    chart_bcr(summary, extra, FIGURES / "rq3_benefit_cost_ratio.png")
    chart_sensitivity(sens, FIGURES / "rq3_cap80_sensitivity.png")
    print_findings(summary, values, traffic, ratios, baseline, compliance_all_time, sens, target, bound, flat)
    print_beyond_compliance(extra, sens, nsmr)
    print(f"\nWrote {len(summary)} rows to results/appraisal.parquet and two charts to figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
