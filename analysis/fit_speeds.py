"""Turn DfT's speed bands into continuous speed distributions and check them against DfT.

For every year (2018 onwards) and road type, builds three distributions:
    empirical   reproduces DfT's bands exactly; the central input for the casualty model
    normal      a fitted symmetric shape (sensitivity test)
    lognormal   a fitted skewed shape (sensitivity test)
and compares each with DfT's own published figures: the average speed, the share exceeding
the limit and the share exceeding it by 10 mph (16 km/h) or more.

Writes results/speed_distributions.parquet and figures/speed_distributions_2024.png.

Usage (from the project root):
    python -m analysis.fit_speeds
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.speeds import (FAMILIES, MPH_TO_KMH, band_intervals, empirical_distribution, fit,
                          fitted_distribution)

ROOT = Path(__file__).resolve().parent.parent
BANDS = ROOT / "data" / "marts" / "agg_speed_bands_cars.parquet"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CHART_YEAR = 2024
COLOURS = {"empirical": "#8c510a", "normal": "#5ab4ac", "lognormal": "#666666"}


def build(bands: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows, chart_data = [], {}
    for (year, road_type), group in bands.groupby(["year", "road_type"]):
        intervals, low = band_intervals(group)
        first = group.iloc[0]
        limit_kmh = first["speed_limit_mph"] * MPH_TO_KMH
        shapes = {"empirical": (empirical_distribution(intervals, "empirical"), None, None)}
        for family in FAMILIES:
            location, scale = fit(intervals, family)
            shapes[family] = (fitted_distribution(family, location, scale, family), location, scale)

        known = group[~group["is_low"]]
        lower, upper = known["band_lower_kmh"].to_numpy(), known["band_upper_kmh"].fillna(np.inf).to_numpy()
        for shape, (dist, location, scale) in shapes.items():
            band_shares = np.array([100 * dist.probabilities[(dist.speeds_kmh >= lo) & (dist.speeds_kmh < hi)].sum()
                                    for lo, hi in zip(lower, upper)])
            rows.append({
                "year": year,
                "road_type": road_type,
                "speed_limit_kmh": int(first["speed_limit_kmh"]),
                "shape": shape,
                "location": location,
                "scale": scale,
                "mean_kmh": dist.mean(),
                "published_mean_kmh": first["avg_free_flow_kmh"],
                "pct_exceeding": 100 * dist.share_at_or_above(limit_kmh),
                "published_pct_exceeding": first["pct_exceeding"],
                "pct_exceeding_by_16_kmh": 100 * dist.share_at_or_above(limit_kmh + 10 * MPH_TO_KMH),
                "published_pct_exceeding_by_16_kmh": first["pct_exceeding_by_10_mph"],
                "max_band_error_pct": float(np.max(np.abs(band_shares - known["share_pct"].to_numpy()))),
                "top_low_pct": low["top_low_pct"],
                "bottom_low_pct": low["bottom_low_pct"],
            })
            if year == CHART_YEAR:
                chart_data.setdefault((road_type, limit_kmh), {})[shape] = dist
    return pd.DataFrame(rows), chart_data


def chart(chart_data: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, ((road_type, limit_kmh), shapes) in zip(axes.flat, sorted(chart_data.items(), key=lambda kv: kv[0][1])):
        for shape, dist in shapes.items():
            density = dist.probabilities / (dist.speeds_kmh[1] - dist.speeds_kmh[0])
            ax.plot(dist.speeds_kmh, density, label=shape, color=COLOURS[shape], lw=1.6 if shape == "empirical" else 1.1)
        ax.axvline(limit_kmh, color="black", ls="--", lw=0.8, label=f"limit {limit_kmh:.0f} km/h")
        ax.set_xlim(0, limit_kmh + 60)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{road_type} ({limit_kmh:.0f} km/h)", fontsize=9)
        ax.set_xlabel("Free-flow car speed (km/h)", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes.flat[0].set_ylabel("Share of cars per km/h")
    axes.flat[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Car speed distributions, {CHART_YEAR}: DfT bands reproduced exactly (empirical) vs fitted shapes", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / f"speed_distributions_{CHART_YEAR}.png", dpi=150)
    plt.close(fig)


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    bands = duckdb.sql(f"select * from read_parquet('{BANDS}')").df()
    table, chart_data = build(bands)
    table.to_parquet(RESULTS / "speed_distributions.parquet", index=False)
    chart(chart_data)

    show = table[table["year"] == CHART_YEAR][[
        "road_type", "shape", "mean_kmh", "published_mean_kmh", "pct_exceeding", "published_pct_exceeding",
        "pct_exceeding_by_16_kmh", "published_pct_exceeding_by_16_kmh", "max_band_error_pct"]]
    show = show.assign(road_type=show["road_type"].str.slice(0, 22))
    print(f"Speed distributions vs DfT, {CHART_YEAR}:")
    print(show.round(2).to_string(index=False))
    empirical = table[table["shape"] == "empirical"]
    print(f"\nEmpirical, all years: worst band error {empirical['max_band_error_pct'].max():.4f} points; "
          f"worst mean error {(empirical['mean_kmh'] - empirical['published_mean_kmh']).abs().max():.2f} km/h")
    print(f"Wrote {len(table)} rows to results/speed_distributions.parquet and figures/speed_distributions_{CHART_YEAR}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())