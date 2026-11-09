"""Every headline with three numbers: central, Monte Carlo range and structural range.

The Monte Carlo (model/uncertainty.py) varies values within the central model. This varies the
model: each choice below replaces the central one, one at a time, and the whole model and
appraisal run again. The structural range is the lowest and highest result across those runs. It
is not a mixture of models with weights, because no source says how likely each one is. Nor does it
stack every extreme at once, which would describe no model anyone holds.

    the individual risk curve     Kloeden, or Brenac's flatter one
    what speed is measured from   the new mean, today's mean, or the absolute urban curve
    the casualty model            Model B, or Model A in its Power or Exponential form
    the speed distribution        empirical, normal or lognormal
    how ISA is used               override at random, override at the trial's own rate, every car with speed
                                  control, every car with a warning
    what ISA shows a van          the posted limit, or the van's own

Run after rq2 and rq3, whose Monte Carlo draws it reads. Writes results/structural_range.parquet, and every
variant's headlines to results/structural_variants.parquet.

Usage:
    python -m analysis.structural
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.rq3 import (MARTS, RESULTS, TARGET_CUT, appraise, beyond_compliance, flat_beyond_compliance, hours_table,
                          load_exposure, monte_carlo)
from model.appraisal import FREE_FLOW_SHARE, ksi_from_casualties, load_values
from model.casualties import Assumptions
from model.run import load_inputs, run, speed_distributions
from model.scenarios import TRIAL_SHARE_OVER_LIMIT, IsaUse

RURAL, MOTORWAY = "97 km/h single carriageways", "113 km/h motorways"
MODEL_SUFFIX = {"preferred": "", "power": "_power", "exp": "_exp"}


def headlines(table: pd.DataFrame, baseline: pd.DataFrame, model: str = "preferred") -> dict[str, float]:
    """Every headline from one appraised run, using the preferred model or one form of Model A.

    baseline holds every road type, because the 2035 target covers all roads.
    """
    suffix = MODEL_SUFFIX[model]
    benefit = "benefit" if model == "preferred" else f"benefit_{model}"
    table = table.set_index(["segment", "policy"])

    def on(policy: str) -> pd.DataFrame:
        return table.xs(policy, level="policy")

    def ratio(frame) -> float:
        return float(frame[benefit].sum() / frame["cost"].sum())

    def ksi(policy: str, vs_bau: str) -> float:
        frame = on(policy)
        if model == "preferred" and vs_bau:
            return float(frame["ksi_prevented"].sum())
        base = baseline.loc[frame.index]
        casualties = frame[f"casualties_prevented{suffix}{vs_bau}"]
        estimate = ksi_from_casualties(casualties, base["ksi"], base["casualties"])
        if model == "preferred":
            estimate = frame[f"ksi_prevented_b{vs_bau}"].fillna(estimate)
        return float(estimate.sum())

    required = TARGET_CUT * baseline["ksi"].sum()
    return {
        "Mandated ISA, deaths prevented against today": float(on("bau_isa")[f"killed_prevented{suffix}"].sum()),
        "Mandated ISA, share of the 2035 target": ksi("bau_isa", "") / required,
        "Compliance beyond ISA, deaths prevented": float(on("compliance")[f"killed_prevented{suffix}_vs_bau"].sum()),
        "Compliance beyond ISA, KSI prevented": ksi("compliance", "_vs_bau"),
        "Compliance beyond ISA, share of the 2035 target": ksi("compliance", "_vs_bau") / required,
        "Compliance beyond ISA, benefit, £m a year": float(on("compliance")[benefit].sum()) / 1e6,
        "80 km/h cap on 97 km/h roads, deaths prevented": float(table.loc[(RURAL, "cap_80"), f"killed_prevented{suffix}_vs_bau"]),
        "80 km/h cap on 97 km/h roads, ratio": ratio(table.loc[[(RURAL, "cap_80")]]),
        "60 km/h cap on 97 km/h roads, ratio": ratio(table.loc[[(RURAL, "cap_60")]]),
        "97 km/h cap on motorways, ratio": ratio(table.loc[[(MOTORWAY, "cap_97")]]),
        "Flat 97 km/h limiter on all four road types, ratio": ratio(on("cap_97")),
        "Flat 80 km/h limiter on all four road types, ratio": ratio(on("cap_80")),
        **extra_headlines(table.reset_index(), benefit, suffix),
    }


def extra_headlines(table: pd.DataFrame, benefit: str, suffix: str) -> dict[str, float]:
    """The caps judged against compliance, which they contain (analysis.rq3.beyond_compliance)."""
    increments = beyond_compliance(table)
    flat = flat_beyond_compliance(increments).set_index("policy")
    increments = increments.set_index(["segment", "policy"])

    def ratio(row) -> float:
        return float(row[benefit] / row["cost"])

    cap = increments.loc[(RURAL, "cap_80")]
    deaths = table.set_index(["segment", "policy"])[f"killed_prevented{suffix}_vs_bau"]
    return {
        "80 km/h cap on 97 km/h roads beyond compliance, ratio": ratio(cap),
        "80 km/h cap on 97 km/h roads beyond compliance, deaths prevented":
            float(deaths.loc[(RURAL, "cap_80")] - deaths.loc[(RURAL, "compliance")]),
        "97 km/h cap on motorways beyond compliance, ratio": ratio(increments.loc[(MOTORWAY, "cap_97")]),
        "Flat 97 km/h limiter beyond compliance, ratio": ratio(flat.loc["cap_97"]),
        "Flat 80 km/h limiter beyond compliance, ratio": ratio(flat.loc["cap_80"]),
    }


def structural_range(central: dict[str, float], variants: dict[str, dict[str, float]]) -> pd.DataFrame:
    """For each headline, the lowest and highest result across the variants, the central one included."""
    rows = []
    for headline, value in central.items():
        results = {"central": value} | {name: result[headline] for name, result in variants.items()}
        lowest, highest = min(results, key=results.get), max(results, key=results.get)
        rows.append({"headline": headline, "central": value, "lowest": results[lowest], "lowest_under": lowest,
                     "highest": results[highest], "highest_under": highest})
    return pd.DataFrame(rows)


def monte_carlo_ranges(mc: pd.DataFrame, draws: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """The 5th and 95th percentile of each headline across the Monte Carlo draws."""
    required = TARGET_CUT * baseline["ksi"].sum()

    def spread(series: pd.Series) -> tuple[float, float]:
        return float(series.quantile(0.05)), float(series.quantile(0.95))

    def total(policy: str, column: str) -> pd.Series:
        return mc[mc["policy"] == policy].groupby("draw")[column].sum()

    def flat_ratio(policy: str) -> pd.Series:
        return total(policy, "benefit") / total(policy, "cost")

    def one(segment: str, policy: str, column: str) -> pd.Series:
        return mc[(mc["segment"] == segment) & (mc["policy"] == policy)].set_index("draw")[column]

    increments = beyond_compliance(mc)
    flat_more = flat_beyond_compliance(increments).set_index("policy")

    def more(segment: str, policy: str, column: str) -> pd.Series:
        return increments[(increments["segment"] == segment) & (increments["policy"] == policy)].set_index("draw")[column]

    modelled = draws[draws["segment"].isin(FREE_FLOW_SHARE)]
    bau = modelled[modelled["policy"] == "bau_isa"].reset_index(drop=True)
    base = baseline.loc[bau["segment"]].reset_index(drop=True)
    bau_ksi = bau["ksi_prevented_b"].fillna(ksi_from_casualties(bau["casualties_prevented"], base["ksi"], base["casualties"]))
    bau_ksi = bau_ksi.groupby(bau["draw"]).sum()
    return {
        "Mandated ISA, deaths prevented against today": spread(total("bau_isa", "killed_prevented")),
        "Mandated ISA, share of the 2035 target": spread(bau_ksi / required),
        "Compliance beyond ISA, deaths prevented": spread(total("compliance", "killed_prevented_vs_bau")),
        "Compliance beyond ISA, KSI prevented": spread(total("compliance", "ksi_prevented")),
        "Compliance beyond ISA, share of the 2035 target": spread(total("compliance", "ksi_prevented") / required),
        "Compliance beyond ISA, benefit, £m a year": spread(total("compliance", "benefit") / 1e6),
        "80 km/h cap on 97 km/h roads, deaths prevented": spread(one(RURAL, "cap_80", "killed_prevented_vs_bau")),
        "80 km/h cap on 97 km/h roads, ratio": spread(one(RURAL, "cap_80", "bcr")),
        "60 km/h cap on 97 km/h roads, ratio": spread(one(RURAL, "cap_60", "bcr")),
        "97 km/h cap on motorways, ratio": spread(one(MOTORWAY, "cap_97", "bcr")),
        "Flat 97 km/h limiter on all four road types, ratio": spread(flat_ratio("cap_97")),
        "Flat 80 km/h limiter on all four road types, ratio": spread(flat_ratio("cap_80")),
        "80 km/h cap on 97 km/h roads beyond compliance, ratio": spread(more(RURAL, "cap_80", "bcr")),
        "80 km/h cap on 97 km/h roads beyond compliance, deaths prevented": spread(more(RURAL, "cap_80", "killed_prevented_vs_bau")),
        "97 km/h cap on motorways beyond compliance, ratio": spread(more(MOTORWAY, "cap_97", "bcr")),
        "Flat 97 km/h limiter beyond compliance, ratio": spread(flat_more.loc["cap_97", "bcr"]),
        "Flat 80 km/h limiter beyond compliance, ratio": spread(flat_more.loc["cap_80", "bcr"]),
    }


def number(headline: str, value: float) -> str:
    if "ratio" in headline:
        return f"{value:.2f}"
    if "share" in headline:
        return f"{value:.0%}"
    return f"{value:,.0f}"


def main() -> int:
    speeds, baseline = load_inputs(MARTS)
    traffic, ratios = load_exposure(MARTS)
    values = load_values()
    distributions = speed_distributions(speeds)
    central_assumptions = Assumptions.central()
    central_isa = IsaUse.central()

    def appraised(d=distributions, a=central_assumptions, isa=central_isa):
        return appraise(run(d, baseline, a, isa), hours_table(d, isa=isa), traffic, baseline, values, ratios)

    central_table = appraised()
    central = headlines(central_table, baseline)
    brenac = replace(central_assumptions, urban_curve="brenac2015_exponential_coef", rural_curve="brenac2015_exponential_coef")
    variants = {
        "Brenac's flatter curve": headlines(appraised(a=brenac), baseline),
        "today's mean": headlines(appraised(a=replace(central_assumptions, reference_mean="old")), baseline),
        "the absolute urban curve": headlines(appraised(a=replace(central_assumptions, reference_mean="absolute")), baseline),
        "Model A, Power": headlines(central_table, baseline, "power"),
        "Model A, Exponential": headlines(central_table, baseline, "exp"),
        "a normal shape": headlines(appraised(d=speed_distributions(speeds, "normal")), baseline),
        "a lognormal shape": headlines(appraised(d=speed_distributions(speeds, "lognormal")), baseline),
        "random override": headlines(appraised(isa=replace(central_isa, override_targets_speeding=False)), baseline),
        "override at the trial's own rate": headlines(appraised(isa=replace(central_isa, override_base=TRIAL_SHARE_OVER_LIMIT)),
                                                      baseline),
        "speed control in every vehicle": headlines(appraised(isa=replace(central_isa, speed_control_share=1.0)), baseline),
        "a warning in every vehicle": headlines(appraised(isa=replace(central_isa, speed_control_share=0.0)), baseline),
        "ISA showing a van its own limit": headlines(appraised(isa=replace(central_isa, vans_read_own_limit=True)), baseline),
    }
    table = structural_range(central, variants)

    draws = pd.read_parquet(RESULTS / "casualty_model_draws.parquet")
    hours = hours_table(distributions)
    mc = monte_carlo(draws, hours, traffic, baseline, values, ratios)
    spreads = monte_carlo_ranges(mc, draws, baseline)
    table["mc_p5"] = [spreads[h][0] for h in table["headline"]]
    table["mc_p95"] = [spreads[h][1] for h in table["headline"]]
    table.to_parquet(RESULTS / "structural_range.parquet", index=False, compression="zstd")
    pd.DataFrame([{"variant": name, "headline": h, "value": v} for name, result in ({"central": central} | variants).items()
                  for h, v in result.items()]).to_parquet(RESULTS / "structural_variants.parquet", index=False, compression="zstd")

    print("Every headline: central [5th to 95th percentile of the Monte Carlo] {lowest to highest across the models}")
    for _, r in table.iterrows():
        h = r["headline"]
        print(f"  {h:52s} {number(h, r['central']):>7s} [{number(h, r['mc_p5'])} to {number(h, r['mc_p95'])}] "
              f"{{{number(h, r['lowest'])} to {number(h, r['highest'])}}}")
        print(f"  {'':52s} lowest under {r['lowest_under']}; highest under {r['highest_under']}")

    shown = {"ISA deaths": "Mandated ISA, deaths prevented against today",
             "Compliance deaths": "Compliance beyond ISA, deaths prevented",
             "Compliance £m": "Compliance beyond ISA, benefit, £m a year",
             "Cap 80 deaths": "80 km/h cap on 97 km/h roads, deaths prevented",
             "Cap 80 ratio": "80 km/h cap on 97 km/h roads, ratio",
             "Cap 80 vs comp.": "80 km/h cap on 97 km/h roads beyond compliance, ratio",
             "Flat 80 ratio": "Flat 80 km/h limiter on all four road types, ratio"}
    print("\nEvery variant, one at a time")
    print(f"  {'':34s}" + "".join(f"{name:>19s}" for name in shown))
    for name, result in ({"central": central} | variants).items():
        print(f"  {name:34s}" + "".join(f"{number(h, result[h]):>19s}" for h in shown.values()))
    print(f"\nWrote {len(table)} rows to results/structural_range.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
