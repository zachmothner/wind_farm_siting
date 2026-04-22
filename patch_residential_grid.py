#!/usr/bin/env python3
"""
GridSight — Residential & Grid Connection Layer Upgrade
=========================================================

Patches grid_data_embedded.js in-place to:
  1. Replace grid connection distances (major towns proxy) with real
     NESO Grid Supply Point locations (262 GSPs in England & Wales)
  2. Add population density proxy (count of LSOA centroids within 2km)
     by encoding it into the residential distance field as a combined metric

Data sources:
  - GSP locations: NESO Data Portal CSV (gsp_gnode_directconnect_region_lookup.csv)
  - LSOA centroids: Cached from previous run (lsoa_centroids.json)

Usage:
  python3 patch_residential_grid.py
"""

import json
import math
import os
import sys
import csv
import io
import time
import urllib.request

GSP_CSV_URL = ("https://api.neso.energy/dataset/2810092e-d4b2-472f-b955-d8bea01f9ec0/"
               "resource/bbe2cc72-a6c6-46e6-8f4e-48b879467368/download/"
               "gsp_gnode_directconnect_region_lookup.csv")
GSP_CACHE = "gsp_locations.json"
LSOA_CACHE = "lsoa_centroids.json"
JS_FILE = "grid_data_embedded.js"


def fetch_gsp_locations():
    """Download and parse unique GSP locations from NESO CSV."""
    if os.path.exists(GSP_CACHE):
        print(f"  Loading cached GSP locations from {GSP_CACHE}")
        with open(GSP_CACHE) as f:
            return json.load(f)

    print("  Downloading GSP locations from NESO...")
    req = urllib.request.Request(GSP_CSV_URL, headers={"User-Agent": "GridSight/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode("utf-8-sig")

    reader = csv.DictReader(io.StringIO(text))
    gsps = {}
    for row in reader:
        gid = row.get("gsp_id", "")
        if gid and gid not in gsps:
            try:
                lat = float(row["gsp_lat"])
                lon = float(row["gsp_lon"])
                name = row.get("gsp_name", "")
                if 49.5 < lat < 56.0 and -6.0 < lon < 2.0:
                    gsps[gid] = {"name": name, "lat": lat, "lon": lon}
            except (ValueError, KeyError):
                pass

    gsp_list = [{"id": k, **v} for k, v in gsps.items()]

    with open(GSP_CACHE, "w") as f:
        json.dump(gsp_list, f)
    print(f"  {len(gsp_list)} GSPs in England/Wales")
    return gsp_list


def load_lsoa_centroids():
    """Load LSOA centroids from cache."""
    if not os.path.exists(LSOA_CACHE):
        print(f"  ERROR: {LSOA_CACHE} not found. Run process_social_layers.py first.")
        sys.exit(1)

    with open(LSOA_CACHE) as f:
        features = json.load(f)

    points = []
    for feat in features:
        geom = feat.get("geometry", {})
        if geom and geom.get("type") == "Point":
            lng, lat = geom["coordinates"][:2]
            points.append((lat, lng))

    print(f"  {len(points)} LSOA centroids loaded")
    return points


def build_point_buckets(points, bucket_size=0.5):
    """Build spatial buckets for fast nearest-neighbor queries."""
    buckets = {}
    for lat, lng in points:
        bkey = (int(lat / bucket_size), int(lng / bucket_size))
        if bkey not in buckets:
            buckets[bkey] = []
        buckets[bkey].append((lat, lng))
    return buckets


def nearest_distance(lat, lng, buckets, bucket_size=0.5, search_radius=2):
    """Find distance to nearest point using spatial buckets."""
    bkey = (int(lat / bucket_size), int(lng / bucket_size))

    # Determine how many bucket rings to search
    rings = max(1, int(search_radius / (bucket_size * 111)) + 1)

    best_dist_sq = float('inf')
    for dlat in range(-rings, rings + 1):
        for dlng in range(-rings, rings + 1):
            nkey = (bkey[0] + dlat, bkey[1] + dlng)
            for rlat, rlng in buckets.get(nkey, []):
                dy = (lat - rlat) * 111.0
                dx = (lng - rlng) * 111.0 * math.cos(math.radians(lat))
                d_sq = dx * dx + dy * dy
                if d_sq < best_dist_sq:
                    best_dist_sq = d_sq

    return math.sqrt(best_dist_sq)


def count_within_radius(lat, lng, buckets, radius_km=2.0, bucket_size=0.5):
    """Count how many points are within radius_km of (lat, lng)."""
    bkey = (int(lat / bucket_size), int(lng / bucket_size))
    rings = max(1, int(radius_km / (bucket_size * 111)) + 1)
    radius_sq = radius_km * radius_km
    count = 0

    for dlat in range(-rings, rings + 1):
        for dlng in range(-rings, rings + 1):
            nkey = (bkey[0] + dlat, bkey[1] + dlng)
            for rlat, rlng in buckets.get(nkey, []):
                dy = (lat - rlat) * 111.0
                dx = (lng - rlng) * 111.0 * math.cos(math.radians(lat))
                d_sq = dx * dx + dy * dy
                if d_sq <= radius_sq:
                    count += 1

    return count


def main():
    print("=" * 60)
    print("GridSight — Residential & Grid Connection Upgrade")
    print("=" * 60)

    # Load data sources
    print("\n[1/4] Loading data sources...")
    gsp_list = fetch_gsp_locations()
    gsp_points = [(g["lat"], g["lon"]) for g in gsp_list]
    print(f"  {len(gsp_points)} Grid Supply Points")

    lsoa_points = load_lsoa_centroids()

    # Build spatial buckets
    print("\n[2/4] Building spatial indices...")
    # GSPs need wider search since there are only 262 of them
    gsp_buckets = build_point_buckets(gsp_points, bucket_size=0.5)
    # LSOAs are dense, fine bucket size
    lsoa_buckets = build_point_buckets(lsoa_points, bucket_size=0.2)
    print("  Done")

    # Read existing JS file
    print(f"\n[3/4] Reading {JS_FILE}...")
    with open(JS_FILE) as f:
        lines = f.readlines()

    # Parse and patch
    print("[4/4] Computing new values and patching...")
    out_lines = []
    cell_count = 0
    gsp_dists = []
    res_dists = []
    pop_densities = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("],"):
            parts = stripped[1:-2].split(",")
            if len(parts) >= 10:
                lat = float(parts[0])
                lng = float(parts[1])
                cell_count += 1

                # Compute new grid connection distance (to nearest GSP)
                gsp_dist = nearest_distance(lat, lng, gsp_buckets, bucket_size=0.5, search_radius=3)
                # Fallback brute force if bucket search missed
                if gsp_dist > 200:
                    for glat, glng in gsp_points:
                        dy = (lat - glat) * 111.0
                        dx = (lng - glng) * 111.0 * math.cos(math.radians(lat))
                        d = math.sqrt(dx*dx + dy*dy)
                        if d < gsp_dist:
                            gsp_dist = d

                # Compute residential distance to nearest LSOA
                res_dist = nearest_distance(lat, lng, lsoa_buckets, bucket_size=0.2, search_radius=1)

                # Count LSOAs within 2km (population density proxy)
                pop_density = count_within_radius(lat, lng, lsoa_buckets, radius_km=2.0, bucket_size=0.2)

                gsp_dists.append(gsp_dist)
                res_dists.append(res_dist)
                pop_densities.append(pop_density)

                # Update fields:
                # Index 7 = resDist * 10
                # Index 8 = gridDist * 10
                # We need to add popDensity — append as new field at index 11
                parts[7] = str(int(round(res_dist * 10)))
                parts[8] = str(int(round(gsp_dist * 10)))

                # If there are already 11 fields, replace the last or append
                if len(parts) == 11:
                    # Existing format: [lat,lng,elev,wind,cf,constr,co2,resDist,gridDist,habitat,fuelPov]
                    # Append popDensity as 12th field
                    parts.append(str(pop_density))
                elif len(parts) >= 12:
                    # Already has popDensity field, update it
                    parts[11] = str(pop_density)
                else:
                    parts.append(str(pop_density))

                out_lines.append("[" + ",".join(parts) + "],\n")

                if cell_count % 50000 == 0:
                    print(f"  {cell_count} cells processed")
            else:
                out_lines.append(line)
        elif "for (const r of d)" in line:
            # Update the JS Map assignment to include new fields
            out_lines.append("  for (const r of d) GRID_DATA.set(r[0]+','+r[1], {\n")
            out_lines.append("    elevation:r[2], wind:r[3]/10, cf:r[4]/1000, constraints:r[5],\n")
            out_lines.append("    co2:r[6], resDist:r[7]/10, gridDist:r[8]/10, habitat:r[9], fuelPoverty:r[10]/10,\n")
            out_lines.append("    popDensity:r[11]||0\n")
            out_lines.append("  });\n")
        else:
            out_lines.append(line)

    # Write patched file
    print(f"\n  Writing patched {JS_FILE}...")
    with open(JS_FILE, "w") as f:
        f.writelines(out_lines)

    size_mb = os.path.getsize(JS_FILE) / 1024 / 1024
    print(f"  Saved ({size_mb:.1f} MB)")

    # Summary
    import numpy as np
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Cells updated: {cell_count}")

    print(f"\n  Grid Connection (GSP) distances:")
    print(f"    Range: {min(gsp_dists):.1f} – {max(gsp_dists):.1f} km")
    print(f"    Mean: {np.mean(gsp_dists):.1f} km")
    print(f"    Median: {np.median(gsp_dists):.1f} km")
    print(f"    (was Major Towns proxy, now {len(gsp_points)} real Grid Supply Points)")

    print(f"\n  Residential distances:")
    print(f"    Range: {min(res_dists):.1f} – {max(res_dists):.1f} km")
    print(f"    Mean: {np.mean(res_dists):.1f} km")

    print(f"\n  Population density (LSOAs within 2km):")
    print(f"    Range: {min(pop_densities)} – {max(pop_densities)}")
    print(f"    Mean: {np.mean(pop_densities):.1f}")
    print(f"    Cells with 0 LSOAs within 2km: {sum(1 for p in pop_densities if p == 0)}")
    print(f"    Cells with >10 LSOAs within 2km: {sum(1 for p in pop_densities if p > 10)}")

    print("\n  New field added: popDensity (count of LSOA centroids within 2km)")
    print("  Update frontend to read cell.popDensity from GRID_DATA")
    print("=" * 60)


if __name__ == "__main__":
    main()
