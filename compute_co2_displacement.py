#!/usr/bin/env python3
"""
GridSight — Temporally & Regionally Resolved CO₂ Displacement Calculator (v2)
===============================================================================

Computes CO₂ displacement per grid cell by cross-referencing:
  - Hourly wind capacity factors from the atlite cutout (per ERA5 grid cell)
  - Half-hourly REGIONAL carbon intensity from NESO's Carbon Intensity API (14 regions)

This gives a more accurate displacement figure than v1 (which used national CI)
because it captures geographic variation: North West England (heavy wind/nuclear)
has much lower grid CI than South England (more gas), so the same MWh of wind
generation displaces different amounts of CO₂ depending on location.

Data sources:
  - Atlite cutout: england-wales-2024.nc
  - Carbon Intensity API: api.carbonintensity.org.uk (free, no key, CC BY 4.0)

Usage:
  python3 compute_co2_displacement.py
"""

import json
import math
import os
import sys
import time
import urllib.request
import urllib.error
import logging
import numpy as np
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ──
CUTOUT_PATH = "england-wales-2024.nc"
TURBINE = "Vestas_V90_3MW"
RATED_MW = 3.0
LIFECYCLE_EMISSIONS = 11  # gCO2/kWh (IPCC AR6 median for onshore wind)
GRID_STEP = 0.01
CI_CACHE_FILE = "carbon_intensity_2024_regional.json"
YEAR = 2024

# NESO Carbon Intensity API regions with approximate centroids
# Used to map each ERA5 grid cell to the correct CI region via nearest-centroid
CI_REGIONS = {
    1:  {"name": "North Scotland",     "clat": 57.5, "clng": -4.5},
    2:  {"name": "South Scotland",     "clat": 55.8, "clng": -3.5},
    3:  {"name": "North West England", "clat": 54.0, "clng": -2.7},
    4:  {"name": "North East England", "clat": 54.8, "clng": -1.5},
    5:  {"name": "Yorkshire",          "clat": 53.8, "clng": -1.2},
    6:  {"name": "North Wales",        "clat": 53.0, "clng": -3.8},
    7:  {"name": "South Wales",        "clat": 51.7, "clng": -3.4},
    8:  {"name": "West Midlands",      "clat": 52.5, "clng": -2.0},
    9:  {"name": "East Midlands",      "clat": 52.8, "clng": -1.0},
    10: {"name": "East England",       "clat": 52.2, "clng": 0.5},
    11: {"name": "South West England", "clat": 50.8, "clng": -3.5},
    12: {"name": "South England",      "clat": 51.0, "clng": -1.3},
    13: {"name": "London",             "clat": 51.5, "clng": -0.1},
    14: {"name": "South East England", "clat": 51.2, "clng": 0.5},
}


def get_region_for_cell(lat, lng):
    """Assign an ERA5 grid cell to a NESO carbon intensity region via nearest centroid."""
    best_dist = float('inf')
    best_rid = 11
    for rid, info in CI_REGIONS.items():
        d = (lat - info["clat"]) ** 2 + (lng - info["clng"]) ** 2
        if d < best_dist:
            best_dist = d
            best_rid = rid
    return best_rid


def fetch_regional_ci_chunk(start_date):
    """
    Fetch regional carbon intensity for a 24h period using fw24h endpoint.
    Returns dict of {region_id: [list of half-hourly intensity values]}.
    """
    start_str = start_date.strftime("%Y-%m-%dT%H:%MZ")
    url = f"https://api.carbonintensity.org.uk/regional/intensity/{start_str}/fw24h"

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "GridSight/1.0",
            "Accept": "application/json"
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        regional_data = {}
        for entry in data.get("data", []):
            for region in entry.get("regions", []):
                rid = region.get("regionid")
                intensity = region.get("intensity", {}).get("forecast")
                if rid is not None and intensity is not None:
                    if rid not in regional_data:
                        regional_data[rid] = []
                    regional_data[rid].append(intensity)

        return regional_data

    except Exception as e:
        logger.warning(f"  Error fetching regional CI for {start_str}: {e}")
        return {}


def fetch_all_regional_ci():
    """
    Fetch full year regional carbon intensity data.
    Uses fw24h endpoint: one request per day = 366 requests for 2024.
    Returns dict of {region_id: numpy array of half-hourly intensities}.
    """
    if os.path.exists(CI_CACHE_FILE):
        logger.info(f"Loading cached regional CI from {CI_CACHE_FILE}")
        with open(CI_CACHE_FILE) as f:
            raw = json.load(f)
        return {int(k): np.array(v) for k, v in raw.items()}

    logger.info(f"Fetching {YEAR} regional carbon intensity from NESO API...")
    logger.info(f"  (1 request per day, ~366 requests — takes ~5 minutes)")

    all_data = {rid: [] for rid in CI_REGIONS}

    start = datetime(YEAR, 1, 1)
    end_of_year = datetime(YEAR + 1, 1, 1)
    day_count = 0
    consecutive_failures = 0

    while start < end_of_year:
        day_count += 1
        regional = fetch_regional_ci_chunk(start)

        if regional:
            consecutive_failures = 0
            for rid in CI_REGIONS:
                if rid in regional:
                    all_data[rid].extend(regional[rid])
        else:
            consecutive_failures += 1
            if consecutive_failures > 10:
                logger.warning("  Too many consecutive failures, stopping")
                break

        if day_count % 30 == 0:
            sample_rid = 3
            logger.info(f"  Day {day_count}: {start.strftime('%Y-%m-%d')} "
                         f"(region 3 has {len(all_data[sample_rid])} values)")

        start += timedelta(hours=24)
        time.sleep(0.3)

    # Cache
    cache_data = {str(k): v for k, v in all_data.items()}
    with open(CI_CACHE_FILE, "w") as f:
        json.dump(cache_data, f)

    logger.info(f"\n  Cached to {CI_CACHE_FILE}")
    for rid in sorted(CI_REGIONS.keys()):
        vals = all_data[rid]
        if vals:
            arr = np.array(vals)
            logger.info(f"  Region {rid:2d} ({CI_REGIONS[rid]['name']:22s}): "
                         f"{len(vals):5d} values, mean={np.mean(arr):5.0f} gCO2/kWh")

    return {k: np.array(v) for k, v in all_data.items()}


def compute_displacement():
    """
    Cross-reference hourly capacity factors with regional half-hourly
    carbon intensity for temporally and regionally resolved CO₂ displacement.
    """
    import atlite

    if not os.path.exists(CUTOUT_PATH):
        logger.error(f"Cutout not found: {CUTOUT_PATH}")
        logger.error(f"Run compute_capacity_factors.py first to download the {YEAR} ERA5 cutout.")
        sys.exit(1)

    cutout = atlite.Cutout(path=CUTOUT_PATH)
    logger.info(f"Loaded cutout: {cutout}")

    logger.info(f"Computing hourly capacity factors with {TURBINE}...")
    cf_timeseries = cutout.wind(
        turbine=TURBINE,
        capacity_factor_timeseries=True,
    )
    logger.info(f"  Shape: {cf_timeseries.shape} (time x y x x)")

    n_times = len(cf_timeseries.coords['time'])
    lats = cf_timeseries.coords['y'].values
    lons = cf_timeseries.coords['x'].values

    # Get regional carbon intensity
    ci_data = fetch_all_regional_ci()

    # Pre-compute hourly CI arrays per region
    logger.info("  Preparing hourly CI arrays per region...")
    region_hourly_ci = {}
    for rid, ci_halfhourly in ci_data.items():
        n_half = len(ci_halfhourly)
        if n_half >= n_times * 2:
            ci_hourly = (ci_halfhourly[:n_times*2:2] + ci_halfhourly[1:n_times*2:2]) / 2
        elif n_half >= n_times:
            ci_hourly = ci_halfhourly[:n_times]
        elif n_half > 0:
            mean_ci = np.mean(ci_halfhourly)
            ci_hourly = np.full(n_times, mean_ci)
            logger.warning(f"  Region {rid}: only {n_half} values, padding with mean={mean_ci:.0f}")
        else:
            ci_hourly = np.full(n_times, 150.0)
            logger.warning(f"  Region {rid}: no data, using fallback 150 gCO2/kWh")
        region_hourly_ci[rid] = ci_hourly

    # Build region map for ERA5 cells
    region_map = {}
    for iy, lat in enumerate(lats):
        for ix, lng in enumerate(lons):
            region_map[(iy, ix)] = get_region_for_cell(float(lat), float(lng))

    # Log region assignments
    region_counts = {}
    for rid in region_map.values():
        region_counts[rid] = region_counts.get(rid, 0) + 1
    for rid in sorted(region_counts):
        logger.info(f"  Region {rid:2d} ({CI_REGIONS[rid]['name']:22s}): {region_counts[rid]} ERA5 cells")

    # Compute CO₂ displacement per ERA5 cell
    co2_per_cell = {}

    for iy, lat in enumerate(lats):
        for ix, lng in enumerate(lons):
            cf_hourly = cf_timeseries[:, iy, ix].values
            rid = region_map[(iy, ix)]
            ci_hourly = region_hourly_ci.get(rid, np.full(n_times, 150.0))

            min_len = min(len(cf_hourly), len(ci_hourly))
            cf_h = cf_hourly[:min_len]
            ci_h = ci_hourly[:min_len]

            hourly_gen = cf_h * RATED_MW
            hourly_co2 = hourly_gen * ci_h
            hourly_lifecycle = hourly_gen * LIFECYCLE_EMISSIONS
            annual_co2 = (np.sum(hourly_co2) - np.sum(hourly_lifecycle)) / 1000

            co2_per_cell[(float(lat), float(lng))] = round(float(annual_co2), 1)

        logger.info(f"  Row {iy+1}/{len(lats)} (lat={lat:.2f})")

    return co2_per_cell


def interpolate_to_frontend_grid(co2_era5):
    """Interpolate ERA5-resolution CO₂ data to the 0.01° frontend grid."""
    from scipy.interpolate import RegularGridInterpolator

    with open("england_wales_poly_js.json") as f:
        poly_coords = json.load(f)

    def point_in_poly(lat, lng):
        inside = False
        n = len(poly_coords)
        for i in range(n):
            j = (i - 1) % n
            xi, yi = poly_coords[i][1], poly_coords[i][0]
            xj, yj = poly_coords[j][1], poly_coords[j][0]
            if ((xi > lat) != (xj > lat)) and (lng < (yj - yi) * (lat - xi) / (xj - xi) + yi):
                inside = not inside
        return inside

    lats = sorted(set(lat for lat, lng in co2_era5))
    lons = sorted(set(lng for lat, lng in co2_era5))

    grid = np.zeros((len(lats), len(lons)))
    lat_idx = {lat: i for i, lat in enumerate(lats)}
    lon_idx = {lon: i for i, lon in enumerate(lons)}

    for (lat, lng), val in co2_era5.items():
        if lat in lat_idx and lng in lon_idx:
            grid[lat_idx[lat], lon_idx[lng]] = val

    interpolator = RegularGridInterpolator(
        (lats, lons), grid,
        method='linear', bounds_error=False, fill_value=None
    )

    logger.info("Interpolating CO₂ displacement to 0.01° grid...")

    co2_data = {}
    lat = 49.9
    while lat < 55.9:
        lng = -5.8
        while lng < 1.9:
            clat = round(lat + GRID_STEP / 2, 4)
            clng = round(lng + GRID_STEP / 2, 4)
            if point_in_poly(clat, clng):
                val = float(interpolator([clat, clng])[0])
                co2_data[(clat, clng)] = round(max(0, val), 0)
            lng += GRID_STEP
        lat += GRID_STEP

    logger.info(f"  {len(co2_data)} cells interpolated")
    return co2_data


def save_output(co2_data):
    """Save CO₂ displacement JSON. Other scripts handle the JS merge."""
    co2_json = {f"{lat},{lng}": val for (lat, lng), val in co2_data.items()}
    with open("co2_displacement.json", "w") as f:
        json.dump(co2_json, f)
    logger.info(f"Saved co2_displacement.json ({len(co2_json)} cells)")


def main():
    print("=" * 60)
    print("GridSight — Regional CO₂ Displacement Calculator (v2)")
    print("=" * 60)
    print(f"Year: {YEAR}")
    print(f"Using NESO Carbon Intensity API (14 regional zones, half-hourly)")
    print(f"Cross-referenced with atlite hourly capacity factors")
    print(f"Lifecycle emissions deducted: {LIFECYCLE_EMISSIONS} gCO₂/kWh")

    print(f"\n[1/3] Computing regionally-resolved CO₂ displacement...")
    co2_era5 = compute_displacement()

    vals = list(co2_era5.values())
    print(f"\n  ERA5 grid results:")
    print(f"  Range: {min(vals):.0f} – {max(vals):.0f} tCO₂/yr")
    print(f"  Mean: {np.mean(vals):.0f} tCO₂/yr")

    print(f"\n[2/3] Interpolating to 0.01° grid...")
    co2_data = interpolate_to_frontend_grid(co2_era5)

    print(f"\n[3/3] Saving output...")
    save_output(co2_data)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"  Cells: {len(co2_data)}")
    vals = list(co2_data.values())
    print(f"  CO₂ displaced range: {min(vals):.0f} – {max(vals):.0f} tCO₂/yr per turbine")
    print(f"  Mean: {np.mean(vals):.0f} tCO₂/yr per turbine")
    print()
    print("Next: run process_social_layers.py to merge into grid_data_embedded.js")
    print("=" * 60)


if __name__ == "__main__":
    main()