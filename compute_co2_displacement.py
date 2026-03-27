#!/usr/bin/env python3
"""
GridSight — Temporally-Resolved CO₂ Displacement Calculator
=============================================================

Computes CO₂ displacement per grid cell by cross-referencing:
  - Hourly wind capacity factors from the atlite cutout (per ERA5 grid cell)
  - Half-hourly regional carbon intensity from NESO's Carbon Intensity API

This gives a much more accurate displacement figure than a flat rate,
because it accounts for WHEN the turbine generates (high-carbon peak hours
vs low-carbon overnight periods).

Data sources:
  - Atlite cutout: england-wales-2023.nc (already downloaded)
  - Carbon Intensity API: api.carbonintensity.org.uk (free, no key)

Prerequisites:
  pip install atlite xarray numpy

Usage:
  python3 compute_co2_displacement.py

Output:
  co2_displacement.json — per grid cell annual CO₂ displaced (tonnes)
  Updates grid_data_embedded.js with co2 field
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
CUTOUT_PATH = "england-wales-2023.nc"
TURBINE = "Vestas_V90_3MW"
RATED_MW = 3.0  # Vestas V90 rated capacity
LIFECYCLE_EMISSIONS = 11  # gCO2/kWh lifecycle emissions for onshore wind (IPCC AR6)
GRID_STEP = 0.01

# Cache file for carbon intensity data
CI_CACHE_FILE = "carbon_intensity_2023.json"


def fetch_carbon_intensity_chunk(start_date, end_date):
    """
    Fetch national carbon intensity (actual) for a date range (max 14 days).
    Returns list of half-hourly actual intensity values in gCO2/kWh.
    """
    start_str = start_date.strftime("%Y-%m-%dT%H:%MZ")
    end_str = end_date.strftime("%Y-%m-%dT%H:%MZ")

    url = f"https://api.carbonintensity.org.uk/intensity/{start_str}/{end_str}"

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "GridSight/1.0",
            "Accept": "application/json"
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        values = []
        for entry in data.get("data", []):
            intensity = entry.get("intensity", {})
            # Prefer actual, fall back to forecast
            val = intensity.get("actual") or intensity.get("forecast")
            if val is not None:
                values.append(val)

        return values

    except Exception as e:
        logger.warning(f"  Error fetching CI data for {start_str}: {e}")
        return []


def fetch_all_carbon_intensity():
    """
    Fetch full year 2023 national carbon intensity (actual, half-hourly).
    Returns dict with single key 'national' containing numpy array.
    """
    if os.path.exists(CI_CACHE_FILE):
        logger.info(f"Loading cached carbon intensity from {CI_CACHE_FILE}")
        with open(CI_CACHE_FILE) as f:
            raw = json.load(f)
        return {"national": np.array(raw["national"])}

    logger.info("Fetching 2023 national carbon intensity (actual) from NESO API...")
    logger.info("  (14-day chunks, ~26 requests — takes ~1 minute)")

    all_values = []
    start = datetime(2023, 1, 1)
    end_of_year = datetime(2024, 1, 1)
    chunk = 0

    while start < end_of_year:
        chunk_end = min(start + timedelta(days=14), end_of_year)
        chunk += 1

        values = fetch_carbon_intensity_chunk(start, chunk_end)
        all_values.extend(values)

        logger.info(f"  Chunk {chunk}: {start.strftime('%Y-%m-%d')} to "
                     f"{chunk_end.strftime('%Y-%m-%d')} ({len(values)} values, "
                     f"total: {len(all_values)})")

        start = chunk_end
        time.sleep(0.5)

    # Cache
    with open(CI_CACHE_FILE, "w") as f:
        json.dump({"national": all_values}, f)

    arr = np.array(all_values)
    logger.info(f"  Total: {len(all_values)} half-hourly values")
    logger.info(f"  Mean: {np.mean(arr):.0f} gCO2/kWh, "
                 f"Min: {np.min(arr):.0f}, Max: {np.max(arr):.0f}")

    return {"national": arr}


def compute_displacement():
    """
    Main computation: cross-reference hourly capacity factors with
    half-hourly regional carbon intensity to get temporally-resolved
    CO₂ displacement per ERA5 grid cell.
    """
    import atlite

    # Load cutout
    if not os.path.exists(CUTOUT_PATH):
        logger.error(f"Cutout not found: {CUTOUT_PATH}")
        sys.exit(1)

    cutout = atlite.Cutout(path=CUTOUT_PATH)
    logger.info(f"Loaded cutout: {cutout}")

    # Compute hourly capacity factors per grid cell
    logger.info(f"Computing hourly capacity factors with {TURBINE}...")
    cf_timeseries = cutout.wind(
        turbine=TURBINE,
        capacity_factor_timeseries=True,
    )
    logger.info(f"  Shape: {cf_timeseries.shape} (time × y × x)")
    
    n_times = len(cf_timeseries.coords['time'])

    # Get carbon intensity (national, half-hourly)
    ci_data = fetch_all_carbon_intensity()
    ci_halfhourly = ci_data["national"]

    # Convert half-hourly to hourly by averaging pairs
    n_half = len(ci_halfhourly)
    if n_half >= n_times * 2:
        ci_hourly = (ci_halfhourly[:n_times*2:2] + ci_halfhourly[1:n_times*2:2]) / 2
    elif n_half >= n_times:
        ci_hourly = ci_halfhourly[:n_times]
    else:
        logger.warning(f"  Only {n_half} CI values, padding with mean")
        mean_ci = np.mean(ci_halfhourly) if len(ci_halfhourly) > 0 else 160
        ci_hourly = np.full(n_times, mean_ci)

    logger.info(f"  Carbon intensity: {len(ci_hourly)} hourly values, "
                 f"mean={np.mean(ci_hourly):.0f} gCO2/kWh")
    
    lats = cf_timeseries.coords['y'].values
    lons = cf_timeseries.coords['x'].values

    # Compute CO2 displacement per ERA5 cell
    co2_per_cell = {}

    for iy, lat in enumerate(lats):
        for ix, lng in enumerate(lons):
            cf_hourly = cf_timeseries[:, iy, ix].values

            min_len = min(len(cf_hourly), len(ci_hourly))
            cf_h = cf_hourly[:min_len]
            ci_h = ci_hourly[:min_len]

            # Hourly generation in MWh
            hourly_gen = cf_h * RATED_MW

            # CO2 displaced in kg: MWh × gCO2/kWh = kgCO2
            hourly_co2 = hourly_gen * ci_h

            # Subtract lifecycle emissions
            hourly_lifecycle = hourly_gen * LIFECYCLE_EMISSIONS

            # Annual net tonnes
            annual_co2 = (np.sum(hourly_co2) - np.sum(hourly_lifecycle)) / 1000

            co2_per_cell[(float(lat), float(lng))] = round(float(annual_co2), 1)

        logger.info(f"  Row {iy+1}/{len(lats)} (lat={lat:.2f})")

    return co2_per_cell


def interpolate_to_frontend_grid(co2_era5):
    """Interpolate ERA5-resolution CO2 data to the 0.01° frontend grid."""
    from scipy.interpolate import RegularGridInterpolator

    # Load polygon
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

    # Build interpolator
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

    logger.info(f"Interpolating CO₂ displacement to 0.01° grid...")

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
    """Save CO2 displacement data and update grid_data_embedded.js."""

    # Save standalone JSON
    co2_json = {f"{lat},{lng}": val for (lat, lng), val in co2_data.items()}
    with open("co2_displacement.json", "w") as f:
        json.dump(co2_json, f)
    logger.info(f"Saved co2_displacement.json ({len(co2_json)} cells)")

    # Load existing grid data
    existing_data = {}
    if os.path.exists("grid_data.json"):
        with open("grid_data.json") as f:
            existing_data = json.load(f)

    cf_data = {}
    if os.path.exists("capacity_factors.json"):
        with open("capacity_factors.json") as f:
            cf_data = json.load(f)

    # Determine all cells
    all_keys = set(existing_data.keys()) | set(co2_json.keys())
    sorted_cells = sorted(all_keys, key=lambda k: tuple(map(float, k.split(","))))

    # Detect constraint field presence
    has_constraints = False
    sample_js = "grid_data_embedded.js"
    if os.path.exists(sample_js):
        with open(sample_js) as f:
            first_lines = f.read(500)
        has_constraints = "constraints" in first_lines

    logger.info(f"Writing grid_data_embedded.js with {len(sorted_cells)} cells...")

    with open("grid_data_embedded.js", "w") as f:
        f.write("// GridSight data — auto-generated\n")
        f.write(f"// {len(sorted_cells)} cells at {GRID_STEP}° resolution\n")
        f.write("// Includes: elevation, wind, capacity factor, constraints, co2 displacement\n")
        f.write(f"var GRID_DATA_STEP = {GRID_STEP};\n")
        f.write("var GRID_DATA = new Map();\n")
        f.write("(function() {\n")
        f.write("  const d = [\n")

        for key in sorted_cells:
            lat_s, lng_s = key.split(",")
            existing = existing_data.get(key, {})
            elev = existing.get("elevation_m", 0)
            wind = existing.get("wind_speed_ms", 0)
            cf = float(cf_data.get(key, 0))
            co2 = co2_data.get((float(lat_s), float(lng_s)), 0)

            # We don't have constraint flags here easily, default to 0
            # The process_constraints.py script should be re-run after this
            f.write(f"[{lat_s},{lng_s},{int(elev)},{int(round(wind*10))},"
                    f"{int(round(cf*1000))},0,{int(co2)}],\n")

        f.write("  ];\n")
        f.write("  for (const r of d) GRID_DATA.set(r[0]+','+r[1], {")
        f.write(" elevation: r[2], wind: r[3]/10, cf: r[4]/1000,")
        f.write(" constraints: r[5], co2: r[6] });\n")
        f.write("})();\n")

    size_mb = os.path.getsize("grid_data_embedded.js") / 1024 / 1024
    logger.info(f"Saved grid_data_embedded.js ({size_mb:.1f} MB)")


def main():
    print("=" * 60)
    print("GridSight — CO₂ Displacement Calculator")
    print("=" * 60)
    print(f"Using NESO Carbon Intensity API (regional, half-hourly)")
    print(f"Cross-referenced with atlite hourly capacity factors")
    print(f"Lifecycle emissions deducted: {LIFECYCLE_EMISSIONS} gCO2/kWh")

    # Step 1: Compute displacement at ERA5 resolution
    print("\n[1/3] Computing temporally-resolved CO₂ displacement...")
    co2_era5 = compute_displacement()

    vals = list(co2_era5.values())
    print(f"\n  ERA5 grid results:")
    print(f"  Range: {min(vals):.0f} – {max(vals):.0f} tCO₂/yr")
    print(f"  Mean: {np.mean(vals):.0f} tCO₂/yr")

    # Step 2: Interpolate to frontend grid
    print("\n[2/3] Interpolating to 0.01° grid...")
    co2_data = interpolate_to_frontend_grid(co2_era5)

    # Step 3: Save
    print("\n[3/3] Saving output...")
    save_output(co2_data)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"  Cells: {len(co2_data)}")
    vals = list(co2_data.values())
    print(f"  CO₂ displaced range: {min(vals):.0f} – {max(vals):.0f} tCO₂/yr per turbine")
    print(f"  Mean: {np.mean(vals):.0f} tCO₂/yr per turbine")
    print()
    print("Note: Re-run process_constraints.py after this to restore constraint flags.")
    print("Then update the frontend to read cell.co2 from GRID_DATA.")
    print("=" * 60)


if __name__ == "__main__":
    main()
