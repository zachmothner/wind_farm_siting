#!/usr/bin/env python3
"""
GridSight — Two-Tier Suitability Scoring (Polars)
===================================================

Reads grid_data_embedded.js, computes slope + LCOE (replicated from client-side),
classifies each cell across two tiers with UK policy-backed thresholds,
and rewrites the JS file with packed tier classifications + composite score.

Tier 1 — Economic Viability:
  Capacity Factor, LCOE, Wind Speed, Grid Distance, Slope, Elevation

Tier 2 — SDG / Social & Environmental:
  Protected Areas, Habitat Score, Residential Distance, CO₂ Displacement, Fuel Poverty

Usage:
  uv run python compute_suitability_tiers.py
"""

import math
import os
import shutil
import polars as pl

JS_FILE = "grid_data_embedded.js"
GRID_STEP = 0.01
CELL_SIZE_M = GRID_STEP * 111000  # ~1110m

# DESNZ 2024 onshore wind assumptions
CAPEX_PER_KW = 1300
OPEX_PER_KW = 45
DISCOUNT_RATE = 0.075
LIFETIME = 25
CRF = (DISCOUNT_RATE * (1 + DISCOUNT_RATE) ** LIFETIME) / (
    (1 + DISCOUNT_RATE) ** LIFETIME - 1
)
ANNUAL_COST_PER_KW = CAPEX_PER_KW * CRF + OPEX_PER_KW


def parse_grid_data_js(filepath):
    """Parse grid_data_embedded.js into a Polars DataFrame."""
    rows = []
    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("],"):
                parts = stripped[1:-2].split(",")
                if len(parts) >= 11:
                    rows.append([float(p) for p in parts[:11]])

    print(f"  Parsed {len(rows)} cells")

    return pl.DataFrame(
        rows,
        schema={
            "lat": pl.Float64,
            "lng": pl.Float64,
            "elevation": pl.Float64,
            "wind_x10": pl.Float64,
            "cf_x1000": pl.Float64,
            "constraints": pl.Float64,
            "co2": pl.Float64,
            "resDist_x10": pl.Float64,
            "gridDist_x10": pl.Float64,
            "habitat": pl.Float64,
            "fuelPov_x10": pl.Float64,
        },
    ).with_columns(
        (pl.col("wind_x10") / 10).alias("wind"),
        (pl.col("cf_x1000") / 1000).alias("cf"),
        (pl.col("resDist_x10") / 10).alias("resDist"),
        (pl.col("gridDist_x10") / 10).alias("gridDist"),
        (pl.col("fuelPov_x10") / 10).alias("fuelPoverty"),
    )


def compute_slope(df):
    """Compute slope from elevation differences to east/north neighbors."""
    lat_i = (pl.col("lat") * 10000).round(0).cast(pl.Int64)
    lng_i = (pl.col("lng") * 10000).round(0).cast(pl.Int64)

    df = df.with_columns(lat_i.alias("lat_i"), lng_i.alias("lng_i"))

    elev_lookup = df.select(["lat_i", "lng_i", "elevation"])

    # East neighbor: same lat, lng + 100 (0.01 * 10000)
    df = df.with_columns((pl.col("lng_i") + 100).alias("east_lng_i"))
    east = elev_lookup.rename({"lng_i": "east_lng_i", "elevation": "elev_east"})
    df = df.join(east, on=["lat_i", "east_lng_i"], how="left")

    # North neighbor: lat + 100, same lng
    df = df.with_columns((pl.col("lat_i") + 100).alias("north_lat_i"))
    north = elev_lookup.rename({"lat_i": "north_lat_i", "elevation": "elev_north"})
    df = df.join(north, on=["north_lat_i", "lng_i"], how="left")

    df = df.with_columns(
        (
            (
                pl.max_horizontal(
                    (pl.col("elev_east").fill_null(pl.col("elevation")) - pl.col("elevation")).abs(),
                    (pl.col("elev_north").fill_null(pl.col("elevation")) - pl.col("elevation")).abs(),
                )
                / CELL_SIZE_M
            ).arctan()
            * (180 / math.pi)
        ).alias("slope")
    )

    df = df.drop(["lat_i", "lng_i", "east_lng_i", "north_lat_i", "elev_east", "elev_north"])

    slopes = df["slope"]
    print(f"  Slope range: {slopes.min():.1f}° – {slopes.max():.1f}°, mean: {slopes.mean():.1f}°")
    return df


def compute_lcoe(df):
    """Compute LCOE (£/MWh) from capacity factor."""
    df = df.with_columns(
        pl.when(pl.col("cf") > 0)
        .then(ANNUAL_COST_PER_KW / (pl.col("cf") * 8760 / 1000))
        .otherwise(999.0)
        .round(1)
        .alias("lcoe")
    )

    valid = df.filter(pl.col("lcoe") < 999)["lcoe"]
    print(f"  LCOE range: £{valid.min():.0f} – £{valid.max():.0f}/MWh, mean: £{valid.mean():.0f}")
    return df


def classify_tier1(df):
    """Classify 6 economic viability metrics: 0=red, 1=amber, 2=green."""

    def classify(col, green_test, amber_test):
        return (
            pl.when(green_test).then(2)
            .when(amber_test).then(1)
            .otherwise(0)
            .cast(pl.Int32)
            .alias(f"t1_{col}")
        )

    df = df.with_columns([
        classify("cf", pl.col("cf") > 0.30, pl.col("cf") >= 0.25),
        classify("lcoe", pl.col("lcoe") < 50, pl.col("lcoe") <= 65),
        classify("wind", pl.col("wind") > 7.0, pl.col("wind") >= 6.0),
        classify("gridDist", pl.col("gridDist") < 10, pl.col("gridDist") <= 25),
        classify("slope", pl.col("slope") < 10, pl.col("slope") <= 15),
        classify("elev", pl.col("elevation") < 400, pl.col("elevation") <= 600),
    ])

    for col in ["cf", "lcoe", "wind", "gridDist", "slope", "elev"]:
        counts = df[f"t1_{col}"].value_counts().sort("count", descending=True)
        dist = {int(r[f"t1_{col}"]): int(r["count"]) for r in counts.iter_rows(named=True)}
        print(f"    T1 {col:8s}: green={dist.get(2,0)}, amber={dist.get(1,0)}, red={dist.get(0,0)}")

    return df


def classify_tier2(df):
    """Classify 5 SDG metrics: 0=red, 1=amber, 2=green."""
    df = df.with_columns([
        # Protected areas: binary (no amber)
        pl.when(pl.col("constraints") == 0).then(2)
        .otherwise(0)
        .cast(pl.Int32)
        .alias("t2_protected"),
        # Habitat score
        pl.when(pl.col("habitat") == 0).then(2)
        .when(pl.col("habitat") <= 4).then(1)
        .otherwise(0)
        .cast(pl.Int32)
        .alias("t2_habitat"),
        # Residential distance
        pl.when(pl.col("resDist") > 0.7).then(2)
        .when(pl.col("resDist") >= 0.5).then(1)
        .otherwise(0)
        .cast(pl.Int32)
        .alias("t2_resDist"),
        # CO2 displacement
        pl.when(pl.col("co2") > 800).then(2)
        .when(pl.col("co2") >= 500).then(1)
        .otherwise(0)
        .cast(pl.Int32)
        .alias("t2_co2"),
        # Fuel poverty (high = green = social benefit)
        # If data is 0 (not available), treat as neutral (amber)
        pl.when(pl.col("fuelPoverty") > 15).then(2)
        .when(pl.col("fuelPoverty") >= 10).then(1)
        .when(pl.col("fuelPoverty") == 0).then(1)
        .otherwise(0)
        .cast(pl.Int32)
        .alias("t2_fuelPov"),
    ])

    for col in ["protected", "habitat", "resDist", "co2", "fuelPov"]:
        counts = df[f"t2_{col}"].value_counts().sort("count", descending=True)
        dist = {int(r[f"t2_{col}"]): int(r["count"]) for r in counts.iter_rows(named=True)}
        print(f"    T2 {col:10s}: green={dist.get(2,0)}, amber={dist.get(1,0)}, red={dist.get(0,0)}")

    return df


def pack_and_score(df):
    """Pack tier classifications into integers and compute composite suitability."""
    # Pack Tier 1: base-3 encoding (cf*1 + lcoe*3 + wind*9 + grid*27 + slope*81 + elev*243)
    df = df.with_columns(
        (
            pl.col("t1_cf")
            + pl.col("t1_lcoe") * 3
            + pl.col("t1_wind") * 9
            + pl.col("t1_gridDist") * 27
            + pl.col("t1_slope") * 81
            + pl.col("t1_elev") * 243
        ).alias("t1Status")
    )

    # Pack Tier 2: base-3 encoding
    df = df.with_columns(
        (
            pl.col("t2_protected")
            + pl.col("t2_habitat") * 3
            + pl.col("t2_resDist") * 9
            + pl.col("t2_co2") * 27
            + pl.col("t2_fuelPov") * 81
        ).alias("t2Status")
    )

    # Composite score: Tier 1 = 60% (10% each of 6), Tier 2 = 40% (8% each of 5)
    t1_cols = ["t1_cf", "t1_lcoe", "t1_wind", "t1_gridDist", "t1_slope", "t1_elev"]
    t2_cols = ["t2_protected", "t2_habitat", "t2_resDist", "t2_co2", "t2_fuelPov"]

    # Per-metric contribution: green=full, amber=half, red=0
    t1_score = pl.lit(0.0)
    for c in t1_cols:
        t1_score = t1_score + pl.col(c).cast(pl.Float64) / 2.0 * 10.0  # max 10 per metric

    t2_score = pl.lit(0.0)
    for c in t2_cols:
        t2_score = t2_score + pl.col(c).cast(pl.Float64) / 2.0 * 8.0  # max 8 per metric

    raw_score = t1_score + t2_score  # max = 60 + 40 = 100

    # If ANY Tier 1 metric is red, cap at 40
    t1_has_red = pl.lit(False)
    for c in t1_cols:
        t1_has_red = t1_has_red | (pl.col(c) == 0)

    df = df.with_columns(
        pl.when(t1_has_red)
        .then(pl.min_horizontal(raw_score, pl.lit(40.0)))
        .otherwise(raw_score)
        .round(0)
        .cast(pl.Int32)
        .alias("suitScore")
    )

    scores = df["suitScore"]
    print(f"\n  Suitability score range: {scores.min()} – {scores.max()}, mean: {scores.mean():.0f}")
    print(f"  Distribution:")
    print(f"    High (≥75):   {df.filter(pl.col('suitScore') >= 75).height}")
    print(f"    Medium (45–74): {df.filter((pl.col('suitScore') >= 45) & (pl.col('suitScore') < 75)).height}")
    print(f"    Low (<45):    {df.filter(pl.col('suitScore') < 45).height}")

    return df


def write_grid_data_js(df, filepath):
    """Rewrite grid_data_embedded.js with 3 new fields appended."""
    backup = filepath + ".bak"
    shutil.copy2(filepath, backup)
    print(f"  Backup saved to {backup}")

    sorted_df = df.sort(["lat", "lng"])
    n = sorted_df.height

    with open(filepath, "w") as f:
        f.write("// GridSight data — auto-generated by compute_suitability_tiers.py\n")
        f.write(f"// {n} cells at {GRID_STEP}° resolution\n")
        f.write("// Fields: elevation, wind, cf, constraints, co2, resDist, gridDist, habitat, fuelPoverty, t1Status, t2Status, suitScore\n")
        f.write(f"var GRID_DATA_STEP = {GRID_STEP};\n")
        f.write("var GRID_DATA = new Map();\n")
        f.write("(function() {\n")
        f.write("  const d = [\n")

        lat_arr = sorted_df["lat"].to_list()
        lng_arr = sorted_df["lng"].to_list()
        elev_arr = sorted_df["elevation"].to_list()
        wind_x10_arr = sorted_df["wind_x10"].to_list()
        cf_x1000_arr = sorted_df["cf_x1000"].to_list()
        constr_arr = sorted_df["constraints"].to_list()
        co2_arr = sorted_df["co2"].to_list()
        rd_x10_arr = sorted_df["resDist_x10"].to_list()
        gd_x10_arr = sorted_df["gridDist_x10"].to_list()
        hab_arr = sorted_df["habitat"].to_list()
        fp_x10_arr = sorted_df["fuelPov_x10"].to_list()
        t1_arr = sorted_df["t1Status"].to_list()
        t2_arr = sorted_df["t2Status"].to_list()
        suit_arr = sorted_df["suitScore"].to_list()

        for i in range(n):
            f.write(
                f"[{lat_arr[i]},{lng_arr[i]},"
                f"{int(elev_arr[i])},{int(wind_x10_arr[i])},{int(cf_x1000_arr[i])},"
                f"{int(constr_arr[i])},{int(co2_arr[i])},"
                f"{int(rd_x10_arr[i])},{int(gd_x10_arr[i])},"
                f"{int(hab_arr[i])},{int(fp_x10_arr[i])},"
                f"{int(t1_arr[i])},{int(t2_arr[i])},{int(suit_arr[i])}],\n"
            )

        f.write("  ];\n")
        f.write("  for (const r of d) GRID_DATA.set(r[0]+','+r[1], {\n")
        f.write("    elevation:r[2], wind:r[3]/10, cf:r[4]/1000, constraints:r[5],\n")
        f.write("    co2:r[6], resDist:r[7]/10, gridDist:r[8]/10, habitat:r[9], fuelPoverty:r[10]/10,\n")
        f.write("    t1Status:r[11], t2Status:r[12], suitScore:r[13]\n")
        f.write("  });\n")
        f.write("})();\n")

    size_mb = os.path.getsize(filepath) / 1024 / 1024
    print(f"  Saved {filepath} ({n} cells, {size_mb:.1f} MB)")


def main():
    print("=" * 60)
    print("GridSight — Two-Tier Suitability Scoring")
    print("=" * 60)

    print("\n[1/5] Parsing grid data...")
    df = parse_grid_data_js(JS_FILE)

    print("\n[2/5] Computing slope from elevation neighbors...")
    df = compute_slope(df)

    print("\n[3/5] Computing LCOE...")
    df = compute_lcoe(df)

    print("\n[4/5] Classifying tiers...")
    print("  Tier 1 — Economic Viability:")
    df = classify_tier1(df)
    print("  Tier 2 — SDG / Social & Environmental:")
    df = classify_tier2(df)
    df = pack_and_score(df)

    print("\n[5/5] Writing output...")
    write_grid_data_js(df, JS_FILE)

    print("\n" + "=" * 60)
    print("Done!")
    print("New fields in GRID_DATA: t1Status, t2Status, suitScore")
    print("Next: open renewable-energy-siting-demo.html to see the radial chart")
    print("=" * 60)


if __name__ == "__main__":
    main()
