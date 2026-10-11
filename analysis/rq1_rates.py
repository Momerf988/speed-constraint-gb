"""RQ1: where does harm happen on Great Britain's roads, once exposure is accounted for?

Reads the Parquet marts (no database needed), prints the numbers behind each chart,
and writes four charts to figures/:
    rq1_death_rate_by_road.png          deaths per billion vehicle-km, 2010-2025
    rq1_ksi_rate_by_road.png            KSI (adjusted) per billion vehicle-km, 2010-2025
    rq1_share_deaths_vs_traffic.png     2025: share of deaths vs share of traffic
    rq1_death_share_by_speed_limit.png  share of deaths by posted limit, 2010-2025

Usage:
    python analysis/rq1_rates.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")  # draw straight to files; no window needed
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RATES = ROOT / "data" / "marts" / "agg_rates_year_road_category.parquet"
CASUALTIES = ROOT / "data" / "marts" / "agg_casualties_year_road.parquet"
FIGURES = ROOT / "figures"

FIRST_YEAR, LAST_YEAR = 2010, 2025
CATEGORIES = ["Rural minor road", "Rural A road", "Urban A road", "Urban minor road", "Motorway"]
COLOURS = dict(zip(CATEGORIES, ["#8c510a", "#d8b365", "#5ab4ac", "#01665e", "#666666"]))


def rate_chart(column: str, title: str, ylabel: str, filename: str) -> None:
    """One line per road category, with the COVID years (DfT note 6) shaded."""
    data = duckdb.sql(f"""
        select year, traffic_road_category, {column} as rate
        from read_parquet('{RATES}')
        where year between {FIRST_YEAR} and {LAST_YEAR}
        order by year
    """).df()
    fig, ax = plt.subplots(figsize=(10, 5))
    for category in CATEGORIES:
        rows = data[data["traffic_road_category"] == category]
        ax.plot(rows["year"], rows["rate"], marker="o", markersize=3, label=category, color=COLOURS[category])
    ax.axvspan(2019.5, 2022.5, color="grey", alpha=0.12, label="COVID years: interpret with caution")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))  # outside, never over the lines
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=150)
    plt.close(fig)


def shares_chart() -> None:
    """2025: each road environment's share of deaths next to its share of traffic."""
    data = duckdb.sql(f"""
        select road_environment,
               100.0 * sum(killed) / sum(sum(killed)) over ()                         as share_of_deaths,
               100.0 * sum(billion_vehicle_km) / sum(sum(billion_vehicle_km)) over () as share_of_traffic
        from read_parquet('{RATES}')
        where year = {LAST_YEAR}
        group by 1
        order by share_of_deaths desc
    """).df()
    print(f"\nShare of deaths vs share of traffic, {LAST_YEAR} (%):")
    print(data.round(1).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    positions = range(len(data))
    deaths = ax.barh([p - 0.2 for p in positions], data["share_of_deaths"], height=0.4, label="Share of deaths", color="#8c510a")
    traffic = ax.barh([p + 0.2 for p in positions], data["share_of_traffic"], height=0.4, label="Share of traffic", color="#bbbbbb")
    ax.bar_label(deaths, fmt="%.0f%%", padding=3, fontsize=8)
    ax.bar_label(traffic, fmt="%.0f%%", padding=3, fontsize=8)
    ax.set_yticks(list(positions), data["road_environment"])
    ax.invert_yaxis()
    ax.set_xlabel("% of Great Britain total")
    ax.set_title(f"Deaths vs traffic by road environment, {LAST_YEAR}")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "rq1_share_deaths_vs_traffic.png", dpi=150)
    plt.close(fig)


def speed_limit_chart() -> None:
    """Share of deaths by posted limit. A share, not a rate: no traffic data exists by limit."""
    data = duckdb.sql(f"""
        select speed_limit_kmh,
               sum(killed) as killed,
               100.0 * sum(killed) / sum(sum(killed)) over () as share_of_deaths
        from read_parquet('{CASUALTIES}')
        where year between {FIRST_YEAR} and {LAST_YEAR}
          and is_standard_speed_limit
        group by 1
        order by 1
    """).df()
    print(f"\nShare of deaths by speed limit, {FIRST_YEAR}-{LAST_YEAR}:")
    print(data.round(1).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([f"{kmh} km/h" for kmh in data["speed_limit_kmh"]], data["share_of_deaths"], color="#8c510a")
    ax.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=8)
    ax.set_ylabel("% of deaths")
    ax.set_title(f"Share of deaths by posted speed limit, {FIRST_YEAR}-{LAST_YEAR}\n(share, not rate: no traffic data exists by speed limit)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "rq1_death_share_by_speed_limit.png", dpi=150)
    plt.close(fig)


def print_rates() -> None:
    data = duckdb.sql(f"""
        select traffic_road_category,
               max(killed_per_bn_vkm) filter (where year = {FIRST_YEAR})       as deaths_rate_{FIRST_YEAR},
               max(killed_per_bn_vkm) filter (where year = {LAST_YEAR})        as deaths_rate_{LAST_YEAR},
               max(ksi_adjusted_per_bn_vkm) filter (where year = {LAST_YEAR})  as ksi_rate_{LAST_YEAR}
        from read_parquet('{RATES}')
        group by 1
        order by deaths_rate_{LAST_YEAR} desc
    """).df()
    data["deaths_rate_change_%"] = 100 * (data[f"deaths_rate_{LAST_YEAR}"] / data[f"deaths_rate_{FIRST_YEAR}"] - 1)
    print(f"Rates per billion vehicle-km:")
    print(data.round(2).to_string(index=False))


def main() -> int:
    FIGURES.mkdir(exist_ok=True)
    print_rates()
    rate_chart("killed_per_bn_vkm", f"Deaths per billion vehicle-km by road type, {FIRST_YEAR}-{LAST_YEAR}",
               "Deaths per billion vehicle-km", "rq1_death_rate_by_road.png")
    rate_chart("ksi_adjusted_per_bn_vkm", f"Killed or seriously injured (adjusted) per billion vehicle-km, {FIRST_YEAR}-{LAST_YEAR}",
               "KSI per billion vehicle-km", "rq1_ksi_rate_by_road.png")
    shares_chart()
    speed_limit_chart()
    print(f"\nCharts written to {FIGURES.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())