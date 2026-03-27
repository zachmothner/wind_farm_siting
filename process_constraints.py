#!/usr/bin/env python3
"""
GridSight — SSSI Exclusion Zone Processor
==========================================

Downloads SSSI (Sites of Special Scientific Interest) boundaries from
Natural England's Open Data Geoportal and flags grid cells that overlap.

Also downloads AONB/National Landscape and National Park boundaries
as additional constraint layers.

Data source:
  Natural England Open Data Geoportal (ArcGIS REST API)
  https://naturalengland-defra.opendata.arcgis.com/

Usage:
  python3 process_constraints.py

Prerequisites:
  pip install shapely requests

Output:
  Updates grid_data_embedded.js to include constraint flags per cell.
"""

import json
import math
import os
import sys
import time
import urllib.request
import urllib.error

# ── Configuration ──
GRID_STEP = 0.01

# ArcGIS REST API endpoints for Natural England datasets
# These return GeoJSON via the query endpoint
DATASETS = {
    "sssi": {
        "name": "SSSI (Sites of Special Scientific Interest)",
        "url": "https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/SSSI_England/FeatureServer/0/query",
        "cache_file": "sssi_boundaries.json",
    },
    "national_park": {
        "name": "National Parks",
        "url": "https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/National_Parks_England/FeatureServer/0/query",
        "cache_file": "national_park_boundaries.json",
    },
    "aonb": {
        "name": "Areas of Outstanding Natural Beauty (National Landscapes)",
        "url": "https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/Areas_of_Outstanding_Natural_Beauty_England/FeatureServer/0/query",
        "cache_file": "aonb_boundaries.json",
    },
}


def download_features(dataset_key):
    """
    Download all features from a Natural England ArcGIS endpoint.
    The API paginates at 2000 features, so we need to loop.
    Returns a list of GeoJSON features.
    """
    ds = DATASETS[dataset_key]
    cache_file = ds["cache_file"]

    if os.path.exists(cache_file):
        print(f"  Loading cached {ds['name']} from {cache_file}")
        with open(cache_file) as f:
            return json.load(f)

    print(f"  Downloading {ds['name']}...")
    all_features = []
    offset = 0
    batch_size = 2000

    while True:
        params = (
            f"?where=1%3D1"
            f"&outFields=*"
            f"&outSR=4326"  # WGS84 lat/lng
            f"&f=geojson"
            f"&resultOffset={offset}"
            f"&resultRecordCount={batch_size}"
        )
        url = ds["url"] + params

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "GridSight/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())

            features = data.get("features", [])
            all_features.extend(features)
            print(f"    Fetched {len(features)} features (total: {len(all_features)})")

            # Check if there are more
            if len(features) < batch_size:
                break

            offset += batch_size
            time.sleep(1)  # Be polite to the server

        except Exception as e:
            print(f"    Error at offset {offset}: {e}")
            break

    # Cache to disk
    with open(cache_file, "w") as f:
        json.dump(all_features, f)
    print(f"    Saved {len(all_features)} features to {cache_file}")

    return all_features


def build_spatial_index(features):
    """
    Build a simple grid-based spatial index for fast point-in-polygon queries.
    Groups features by which 0.5° grid cells their bounding boxes overlap.
    """
    index = {}
    index_step = 0.5

    for i, feat in enumerate(features):
        geom = feat.get("geometry", {})
        if not geom or geom.get("type") not in ("Polygon", "MultiPolygon"):
            continue

        # Get bounding box from coordinates
        coords = geom.get("coordinates", [])
        if not coords:
            continue

        # Flatten to get all points
        def flatten_coords(c):
            if isinstance(c[0], (int, float)):
                yield c
            else:
                for sub in c:
                    yield from flatten_coords(sub)

        all_pts = list(flatten_coords(coords))
        if not all_pts:
            continue

        lngs = [p[0] for p in all_pts]
        lats = [p[1] for p in all_pts]
        min_lat, max_lat = min(lats), max(lats)
        min_lng, max_lng = min(lngs), max(lngs)

        # Add to all overlapping index cells
        for ilat in range(int(min_lat / index_step), int(max_lat / index_step) + 1):
            for ilng in range(int(min_lng / index_step), int(max_lng / index_step) + 1):
                key = (ilat, ilng)
                if key not in index:
                    index[key] = []
                index[key].append(i)

    return index


def point_in_polygon(lat, lng, coordinates):
    """Ray-casting point-in-polygon for a single ring."""
    inside = False
    ring = coordinates
    n = len(ring)
    for i in range(n):
        j = (i - 1) % n
        xi, yi = ring[i][1], ring[i][0]  # lat, lng
        xj, yj = ring[j][1], ring[j][0]
        if ((xi > lat) != (xj > lat)) and (lng < (yj - yi) * (lat - xi) / (xj - xi) + yi):
            inside = not inside
    return inside


def point_in_feature(lat, lng, feature):
    """Test if a point is inside a GeoJSON feature (Polygon or MultiPolygon)."""
    geom = feature.get("geometry", {})
    gtype = geom.get("type", "")
    coords = geom.get("coordinates", [])

    if gtype == "Polygon":
        # First ring is exterior, rest are holes
        if not coords:
            return False
        if point_in_polygon(lat, lng, coords[0]):
            # Check holes
            for hole in coords[1:]:
                if point_in_polygon(lat, lng, hole):
                    return False
            return True
        return False

    elif gtype == "MultiPolygon":
        for poly in coords:
            if not poly:
                continue
            if point_in_polygon(lat, lng, poly[0]):
                in_hole = False
                for hole in poly[1:]:
                    if point_in_polygon(lat, lng, hole):
                        in_hole = True
                        break
                if not in_hole:
                    return True
        return False

    return False


def flag_cells(cells, features, spatial_index, label):
    """
    For each grid cell, check if it falls inside any feature.
    Returns a set of (lat, lng) tuples that are inside.
    """
    index_step = 0.5
    flagged = set()

    print(f"  Checking {len(cells)} cells against {len(features)} {label} features...")

    for i, (lat, lng) in enumerate(cells):
        # Look up candidate features from spatial index
        ikey = (int(lat / index_step), int(lng / index_step))
        candidates = spatial_index.get(ikey, [])

        for fi in candidates:
            if point_in_feature(lat, lng, features[fi]):
                flagged.add((lat, lng))
                break  # No need to check more features for this cell

        if (i + 1) % 50000 == 0:
            print(f"    {i + 1}/{len(cells)} checked, {len(flagged)} flagged")

    print(f"    Done: {len(flagged)} cells flagged as {label}")
    return flagged


def load_grid_cells():
    """Load cell coordinates from existing grid_data.json."""
    if not os.path.exists("grid_data.json"):
        print("ERROR: grid_data.json not found. Run the preprocessing script first.")
        sys.exit(1)

    with open("grid_data.json") as f:
        raw = json.load(f)

    cells = []
    for key in raw:
        lat_s, lng_s = key.split(",")
        cells.append((float(lat_s), float(lng_s)))

    return cells, raw


def save_output(cells, existing_data, constraints):
    """
    Update grid_data_embedded.js to include constraint flags.
    Format: [lat, lng, elevation, wind*10, cf*1000, constraint_flags]
    where constraint_flags is a bitmask: 1=SSSI, 2=National Park, 4=AONB
    """
    sorted_cells = sorted(cells, key=lambda c: (c[0], c[1]))

    with open("grid_data_embedded.js", "w") as f:
        f.write("// GridSight data — auto-generated\n")
        f.write(f"// {len(sorted_cells)} cells at {GRID_STEP}° resolution\n")
        f.write("// Includes: elevation, wind, capacity factor, constraint flags\n")
        f.write("// Constraint flags bitmask: 1=SSSI, 2=National Park, 4=AONB\n")
        f.write(f"var GRID_DATA_STEP = {GRID_STEP};\n")
        f.write("var GRID_DATA = new Map();\n")
        f.write("(function() {\n")
        f.write("  const d = [\n")

        for lat, lng in sorted_cells:
            key = f"{lat},{lng}"
            existing = existing_data.get(key, {})
            elev = existing.get("elevation_m", 0)
            wind = existing.get("wind_speed_ms", 0)

            # Get CF from capacity_factors.json if it exists
            cf = 0
            if os.path.exists("capacity_factors.json"):
                # Loaded once below
                pass

            # Compute constraint flags
            flags = 0
            if (lat, lng) in constraints.get("sssi", set()):
                flags |= 1
            if (lat, lng) in constraints.get("national_park", set()):
                flags |= 2
            if (lat, lng) in constraints.get("aonb", set()):
                flags |= 4

            f.write(f"[{lat},{lng},{int(elev)},{int(round(wind * 10))},0,{flags}],\n")

        f.write("  ];\n")
        f.write("  for (const r of d) GRID_DATA.set(r[0]+','+r[1], {")
        f.write(" elevation: r[2], wind: r[3]/10, cf: r[4]/1000, constraints: r[5] });\n")
        f.write("})();\n")

    size_mb = os.path.getsize("grid_data_embedded.js") / 1024 / 1024
    print(f"\nSaved grid_data_embedded.js ({len(sorted_cells)} cells, {size_mb:.1f} MB)")


def save_output_with_cf(cells, existing_data, constraints, cf_data):
    """
    Update grid_data_embedded.js with all data including capacity factors and constraints.
    """
    sorted_cells = sorted(cells, key=lambda c: (c[0], c[1]))

    with open("grid_data_embedded.js", "w") as f:
        f.write("// GridSight data — auto-generated\n")
        f.write(f"// {len(sorted_cells)} cells at {GRID_STEP}° resolution\n")
        f.write("// Includes: elevation, wind, capacity factor, constraint flags\n")
        f.write("// Constraint flags bitmask: 1=SSSI, 2=National Park, 4=AONB\n")
        f.write(f"var GRID_DATA_STEP = {GRID_STEP};\n")
        f.write("var GRID_DATA = new Map();\n")
        f.write("(function() {\n")
        f.write("  const d = [\n")

        for lat, lng in sorted_cells:
            key = f"{lat},{lng}"
            existing = existing_data.get(key, {})
            elev = existing.get("elevation_m", 0)
            wind = existing.get("wind_speed_ms", 0)
            cf = cf_data.get(key, 0)

            flags = 0
            if (lat, lng) in constraints.get("sssi", set()):
                flags |= 1
            if (lat, lng) in constraints.get("national_park", set()):
                flags |= 2
            if (lat, lng) in constraints.get("aonb", set()):
                flags |= 4

            f.write(f"[{lat},{lng},{int(elev)},{int(round(wind * 10))},{int(round(cf * 1000))},{flags}],\n")

        f.write("  ];\n")
        f.write("  for (const r of d) GRID_DATA.set(r[0]+','+r[1], {")
        f.write(" elevation: r[2], wind: r[3]/10, cf: r[4]/1000, constraints: r[5] });\n")
        f.write("})();\n")

    size_mb = os.path.getsize("grid_data_embedded.js") / 1024 / 1024
    print(f"\nSaved grid_data_embedded.js ({len(sorted_cells)} cells, {size_mb:.1f} MB)")


def main():
    print("=" * 60)
    print("GridSight — Constraint Layer Processor")
    print("=" * 60)

    # Load existing grid data
    print("\n[1/4] Loading grid cells...")
    cells, existing_data = load_grid_cells()
    print(f"  {len(cells)} cells loaded")

    # Load capacity factors if available
    cf_data = {}
    if os.path.exists("capacity_factors.json"):
        print("  Loading capacity factors...")
        with open("capacity_factors.json") as f:
            cf_data = json.load(f)
        print(f"  {len(cf_data)} capacity factor values loaded")

    # Download constraint datasets
    print("\n[2/4] Downloading constraint boundaries...")
    all_features = {}
    for key in DATASETS:
        all_features[key] = download_features(key)

    # Build spatial indices
    print("\n[3/4] Building spatial indices and flagging cells...")
    constraints = {}
    for key in DATASETS:
        features = all_features[key]
        if features:
            index = build_spatial_index(features)
            flagged = flag_cells(cells, features, index, key.upper())
            constraints[key] = flagged
        else:
            constraints[key] = set()
            print(f"  No features for {key}, skipping")

    # Summary
    print("\n  Summary:")
    for key in DATASETS:
        count = len(constraints[key])
        pct = count / len(cells) * 100
        print(f"    {DATASETS[key]['name']}: {count} cells ({pct:.1f}%)")

    any_constraint = constraints["sssi"] | constraints["national_park"] | constraints["aonb"]
    print(f"    Any constraint: {len(any_constraint)} cells ({len(any_constraint)/len(cells)*100:.1f}%)")

    # Save output
    print("\n[4/4] Saving output...")
    save_output_with_cf(cells, existing_data, constraints, cf_data)

    print("\n" + "=" * 60)
    print("Done!")
    print("Constraint flags are stored as a bitmask in each cell's data:")
    print("  1 = SSSI")
    print("  2 = National Park")
    print("  4 = AONB / National Landscape")
    print("In the frontend, check with: cell.constraints & 1 (SSSI), etc.")
    print("=" * 60)


if __name__ == "__main__":
    main()
