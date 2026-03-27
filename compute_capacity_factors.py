#!/usr/bin/env python3
"""
GridSight — Atlite Integration Script
======================================

Computes wind turbine capacity factors for England & Wales using atlite + ERA5.

Prerequisites:
  1. pip install atlite geopandas xarray netcdf4 scipy
  2. Register for a free CDS API account:
     https://cds.climate.copernicus.eu/how-to-api
  3. Create ~/.cdsapirc with your API key:
     url: https://cds.climate.copernicus.eu/api
     key: YOUR_UID:YOUR_API_KEY

Usage:
  python3 compute_capacity_factors.py

This script:
  1. Creates an atlite cutout (downloads ERA5 data — ~2-10 GB, takes 10-60 min)
  2. Computes annual mean capacity factors per ERA5 grid cell (~0.25°)
  3. Bilinearly interpolates to your 0.01° frontend grid
  4. Outputs updated grid_data_embedded.js with capacity factor data

The cutout is cached as a NetCDF file — subsequent runs skip the download.
"""

import json
import math
import os
import sys
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ──
CUTOUT_PATH = "england-wales-2023.nc"
CUTOUT_YEAR = "2023"                     # Full year for annual mean
CUTOUT_X = slice(-6.0, 2.0)             # Longitude bounds (with buffer)
CUTOUT_Y = slice(49.5, 56.0)            # Latitude bounds (with buffer)
TURBINE = "Vestas_V90_3MW"             # 2 MW onshore turbine (atlite built-in)
GRID_STEP = 0.01                         # Frontend grid resolution

# Alternative turbines available in atlite:
# "Vestas_V90_2000"   — 2 MW, common onshore reference
# "Vestas_V112_3075"  — 3 MW, modern onshore
# "NREL_ReferenceTurbine_2020ATB_4MW" — 4 MW, next-gen
# "Enercon_E126_7500" — 7.5 MW, large onshore


def check_dependencies():
    """Verify all required packages are installed."""
    missing = []
    for pkg in ['atlite', 'geopandas', 'xarray', 'scipy']:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Install them with: pip install {' '.join(missing)}")
        sys.exit(1)

    # Check CDS API config
    cdsapi_rc = os.path.expanduser("~/.cdsapirc")
    if not os.path.exists(cdsapi_rc):
        print("ERROR: ~/.cdsapirc not found.")
        print("You need a free Copernicus CDS API account to download ERA5 data.")
        print("")
        print("Steps:")
        print("  1. Register at https://cds.climate.copernicus.eu/")
        print("  2. Go to https://cds.climate.copernicus.eu/how-to-api")
        print("  3. Create ~/.cdsapirc with contents:")
        print("     url: https://cds.climate.copernicus.eu/api")
        print("     key: YOUR_UID:YOUR_API_KEY")
        sys.exit(1)


def step1_create_cutout():
    """Create or load the ERA5 cutout for England & Wales."""
    import atlite

    if os.path.exists(CUTOUT_PATH):
        logger.info(f"Loading existing cutout from {CUTOUT_PATH}")
        cutout = atlite.Cutout(path=CUTOUT_PATH)
        logger.info(f"Cutout loaded: {cutout}")
        return cutout

    logger.info("Creating new cutout (this downloads ERA5 data — may take 10-60 minutes)...")
    logger.info(f"  Region: x={CUTOUT_X}, y={CUTOUT_Y}")
    logger.info(f"  Time: {CUTOUT_YEAR}")

    cutout = atlite.Cutout(
        path=CUTOUT_PATH,
        module="era5",
        x=CUTOUT_X,
        y=CUTOUT_Y,
        time=CUTOUT_YEAR,
    )

    # Prepare: downloads wind, height data from ERA5
    # 'wind' feature includes wnd100m (100m wind speed) and roughness
    # 'height' feature includes surface elevation
    cutout.prepare(
        features=['wind', 'height'],
        monthly_requests=True,       # Download month-by-month (more reliable)
        concurrent_requests=False,   # Sequential to avoid rate limits
    )

    logger.info(f"Cutout prepared: {cutout}")
    return cutout


def step2_compute_capacity_factors(cutout):
    """
    Compute per-grid-cell capacity factors using atlite.

    Returns an xarray DataArray with dimensions (y, x) containing
    the annual mean capacity factor (0-1) for each ERA5 grid cell.
    """
    logger.info(f"Computing wind capacity factors with turbine: {TURBINE}")

    cf_timeseries = cutout.wind(
        turbine=TURBINE,
        capacity_factor_timeseries=True,
    )

    # Average over time to get annual mean capacity factor per cell
    cf_mean = cf_timeseries.mean(dim='time')

    logger.info(f"Capacity factor grid shape: {cf_mean.shape}")
    logger.info(f"  Range: {float(cf_mean.min()):.3f} — {float(cf_mean.max()):.3f}")
    logger.info(f"  Mean: {float(cf_mean.mean()):.3f}")

    return cf_mean


def step3_interpolate_to_frontend_grid(cf_mean, cutout):
    """
    Interpolate ERA5-resolution capacity factors (~0.25°) to the
    frontend's 0.01° grid using bilinear interpolation.

    Returns a dict of {(lat, lng): capacity_factor}.
    """
    from scipy.interpolate import RegularGridInterpolator

    # Load the polygon for land masking
    poly_file = "england_wales_poly_js.json"
    if not os.path.exists(poly_file):
        logger.error(f"{poly_file} not found — needed for land masking")
        sys.exit(1)

    with open(poly_file) as f:
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

    # Build the interpolator from the ERA5 grid
    lats = cf_mean.coords['y'].values  # sorted descending in ERA5
    lons = cf_mean.coords['x'].values
    values = cf_mean.values

    # RegularGridInterpolator expects ascending axes
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        values = values[::-1, :]

    interpolator = RegularGridInterpolator(
        (lats, lons), values,
        method='linear',
        bounds_error=False,
        fill_value=None,  # Extrapolate slightly at edges
    )

    logger.info(f"ERA5 grid: {len(lats)} lats × {len(lons)} lons")
    logger.info(f"Interpolating to 0.01° frontend grid...")

    # Generate all frontend grid cells and interpolate
    cf_data = {}
    LAT_MIN, LAT_MAX = 49.9, 55.9
    LNG_MIN, LNG_MAX = -5.8, 1.9

    total_cells = 0
    land_cells = 0

    lat = LAT_MIN
    while lat < LAT_MAX:
        lng = LNG_MIN
        while lng < LNG_MAX:
            clat = round(lat + GRID_STEP / 2, 4)
            clng = round(lng + GRID_STEP / 2, 4)

            if point_in_poly(clat, clng):
                cf_val = float(interpolator([clat, clng])[0])
                cf_val = max(0.0, min(1.0, cf_val))
                cf_data[(clat, clng)] = round(cf_val, 4)
                land_cells += 1

            total_cells += 1
            lng += GRID_STEP
        lat += GRID_STEP

        # Progress
        if int(lat * 10) % 10 == 0:
            logger.info(f"  lat={lat:.1f}, {land_cells} land cells so far")

    logger.info(f"  Total land cells: {land_cells}")
    logger.info(f"  CF range: {min(cf_data.values()):.3f} — {max(cf_data.values()):.3f}")

    return cf_data


def step4_save_output(cf_data):
    """
    Update grid_data_embedded.js to include capacity factors.

    Reads existing data (elevation + wind), adds capacity factors,
    and writes the combined output.
    """
    # Load existing grid data if available
    existing_data = {}
    cache_file = "grid_data.json"
    if os.path.exists(cache_file):
        logger.info(f"Loading existing grid data from {cache_file}")
        with open(cache_file) as f:
            raw = json.load(f)
        for key, val in raw.items():
            lat_s, lng_s = key.split(",")
            existing_data[(float(lat_s), float(lng_s))] = val

    # Merge: add capacity factor to existing data
    all_cells = sorted(set(list(cf_data.keys()) + list(existing_data.keys())))

    logger.info(f"Writing grid_data_embedded.js with {len(all_cells)} cells...")

    with open("grid_data_embedded.js", "w") as f:
        f.write("// GridSight data — auto-generated by compute_capacity_factors.py\n")
        f.write(f"// {len(all_cells)} cells at {GRID_STEP}° resolution\n")
        f.write(f"// Includes: elevation (SRTM), wind speed (ERA5), capacity factor (atlite {TURBINE})\n")
        f.write(f"var GRID_DATA_STEP = {GRID_STEP};\n")
        f.write("var GRID_DATA = new Map();\n")
        f.write("(function() {\n")
        f.write("  const d = [\n")

        for lat, lng in all_cells:
            existing = existing_data.get((lat, lng), {})
            elev = existing.get("elevation_m", 0)
            wind = existing.get("wind_speed_ms", 0)
            cf = cf_data.get((lat, lng), 0)

            # Format: [lat, lng, elevation, wind*10, cf*1000]
            f.write(f"[{lat},{lng},{int(elev)},{int(round(wind * 10))},{int(round(cf * 1000))}],\n")

        f.write("  ];\n")
        f.write("  for (const r of d) GRID_DATA.set(r[0]+','+r[1], {")
        f.write(" elevation: r[2], wind: r[3]/10, cf: r[4]/1000 });\n")
        f.write("})();\n")

    size_mb = os.path.getsize("grid_data_embedded.js") / 1024 / 1024
    logger.info(f"Saved grid_data_embedded.js ({size_mb:.1f} MB)")

    # Also save the capacity factor data separately for inspection
    cf_json = {f"{lat},{lng}": cf for (lat, lng), cf in cf_data.items()}
    with open("capacity_factors.json", "w") as f:
        json.dump(cf_json, f)
    logger.info(f"Saved capacity_factors.json ({len(cf_json)} cells)")


def main():
    print("=" * 60)
    print("GridSight — Atlite Capacity Factor Calculator")
    print("=" * 60)

    check_dependencies()

    # Step 1: Create/load ERA5 cutout
    print("\n[1/4] Preparing ERA5 cutout...")
    cutout = step1_create_cutout()

    # Step 2: Compute capacity factors
    print("\n[2/4] Computing capacity factors...")
    cf_mean = step2_compute_capacity_factors(cutout)

    # Step 3: Interpolate to frontend grid
    print("\n[3/4] Interpolating to 0.01° grid...")
    cf_data = step3_interpolate_to_frontend_grid(cf_mean, cutout)

    # Step 4: Save output
    print("\n[4/4] Saving output files...")
    step4_save_output(cf_data)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"  Turbine: {TURBINE}")
    print(f"  Cells: {len(cf_data)}")
    print(f"  CF range: {min(cf_data.values()):.3f} — {max(cf_data.values()):.3f}")
    print(f"  CF mean: {sum(cf_data.values()) / len(cf_data):.3f}")
    print("")
    print("Next: update the HTML to add a 'Capacity Factor' layer.")
    print("The JS data now includes 'cf' in each cell's GRID_DATA entry.")
    print("=" * 60)


if __name__ == "__main__":
    main()
