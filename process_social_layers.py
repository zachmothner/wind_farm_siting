#!/usr/bin/env python3
"""
GridSight — Social & Environmental Layers Processor
=====================================================

Computes four new layers for each grid cell:
  1. Residential proximity — distance to nearest population centre (LSOA centroids)
  2. Grid connection proximity — distance to nearest major town (proxy for substations)
  3. Biodiversity value — overlap with Priority Habitats Inventory
  4. Fuel poverty proximity — distance to deprived areas (via optional IMD/fuel poverty CSV)

Data sources:
  - LSOA population-weighted centroids: ONS ArcGIS (34,753 points)
  - Major Towns and Cities: ONS ArcGIS (112 polygons → centroids)
  - Priority Habitats Inventory: Natural England ArcGIS
  - Fuel poverty: gov.uk CSV joined by LSOA code (user-supplied)

Usage:
  python3 process_social_layers.py

Prerequisites:
  pip install numpy

Optional:
  Download fuel poverty data from:
  https://www.gov.uk/government/statistics/sub-regional-fuel-poverty-data-2024
  Save the LSOA-level CSV as 'fuel_poverty_lsoa.csv' in the same directory.
  Expected columns: 'LSOA Code' (or 'lsoa11cd') and 'Proportion of households fuel poor (%)' (or similar)
"""

import json
import math
import os
import sys
import time
import urllib.request
import urllib.error
import csv
import numpy as np

# ── Configuration ──
GRID_STEP = 0.01

# Cache files
LSOA_CACHE = "lsoa_centroids.json"
TOWNS_CACHE = "major_towns_centroids.json"
PHI_CACHE = "priority_habitats.json"


def fetch_arcgis_features(base_url, cache_file, label, out_fields="*", max_features=50000):
    """Generic function to download features from an ArcGIS FeatureServer."""
    if os.path.exists(cache_file):
        print(f"  Loading cached {label} from {cache_file}")
        with open(cache_file) as f:
            return json.load(f)

    print(f"  Downloading {label}...")
    all_features = []
    offset = 0
    batch_size = 2000

    while offset < max_features:
        params = (
            f"?where=1%3D1"
            f"&outFields={out_fields}"
            f"&outSR=4326"
            f"&f=geojson"
            f"&resultOffset={offset}"
            f"&resultRecordCount={batch_size}"
        )
        url = base_url + params

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "GridSight/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())

            features = data.get("features", [])
            all_features.extend(features)
            print(f"    Fetched {len(features)} (total: {len(all_features)})")

            if len(features) < batch_size:
                break

            offset += batch_size
            time.sleep(0.5)

        except Exception as e:
            print(f"    Error at offset {offset}: {e}")
            time.sleep(2)
            # Retry once
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "GridSight/1.0"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read())
                features = data.get("features", [])
                all_features.extend(features)
                print(f"    Retry succeeded: {len(features)} features")
                if len(features) < batch_size:
                    break
                offset += batch_size
            except Exception as e2:
                print(f"    Retry failed: {e2}")
                break

    with open(cache_file, "w") as f:
        json.dump(all_features, f)
    print(f"    Saved {len(all_features)} features to {cache_file}")
    return all_features


# ═══════════════════════════════════════════════
# Layer 1: Residential Proximity
# ═══════════════════════════════════════════════

def fetch_lsoa_centroids():
    """Download LSOA population-weighted centroids from ONS."""
    return fetch_arcgis_features(
        "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
        "LSOA_Dec_2011_PWC_in_England_and_Wales_2022/FeatureServer/0/query",
        LSOA_CACHE,
        "LSOA population-weighted centroids",
        out_fields="lsoa11cd,lsoa11nm",
        max_features=40000,
    )


def extract_points(features):
    """Extract (lat, lng) points from GeoJSON point features."""
    points = []
    for feat in features:
        geom = feat.get("geometry", {})
        if geom and geom.get("type") == "Point":
            lng, lat = geom["coordinates"][:2]
            points.append((lat, lng))
    return points


def extract_polygon_centroids(features):
    """Extract centroids from GeoJSON polygon features."""
    centroids = []
    names = []
    for feat in features:
        geom = feat.get("geometry", {})
        props = feat.get("properties", {})
        if not geom:
            continue

        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])

        def get_ring_centroid(ring):
            lats = [p[1] for p in ring]
            lngs = [p[0] for p in ring]
            return sum(lats) / len(lats), sum(lngs) / len(lngs)

        if gtype == "Polygon" and coords:
            lat, lng = get_ring_centroid(coords[0])
            centroids.append((lat, lng))
            names.append(props.get("TCITY15NM", ""))
        elif gtype == "MultiPolygon" and coords:
            # Use largest polygon
            largest = max(coords, key=lambda p: len(p[0]) if p else 0)
            lat, lng = get_ring_centroid(largest[0])
            centroids.append((lat, lng))
            names.append(props.get("TCITY15NM", ""))

    return centroids, names


def compute_nearest_distances(cells, reference_points):
    """
    For each grid cell, compute distance to nearest reference point.
    Uses a simple spatial bucket approach for performance.
    Returns dict of {(lat, lng): distance_km}.
    """
    # Build spatial buckets for reference points
    BUCKET_SIZE = 0.5  # degrees
    buckets = {}
    for rlat, rlng in reference_points:
        bkey = (int(rlat / BUCKET_SIZE), int(rlng / BUCKET_SIZE))
        if bkey not in buckets:
            buckets[bkey] = []
        buckets[bkey].append((rlat, rlng))

    distances = {}
    for i, (lat, lng) in enumerate(cells):
        bkey = (int(lat / BUCKET_SIZE), int(lng / BUCKET_SIZE))

        # Check this bucket and 8 neighbors
        best_dist_sq = float('inf')
        for dlat in [-1, 0, 1]:
            for dlng in [-1, 0, 1]:
                nkey = (bkey[0] + dlat, bkey[1] + dlng)
                for rlat, rlng in buckets.get(nkey, []):
                    # Approximate km distance
                    dy = (lat - rlat) * 111.0
                    dx = (lng - rlng) * 111.0 * math.cos(math.radians(lat))
                    d_sq = dx * dx + dy * dy
                    if d_sq < best_dist_sq:
                        best_dist_sq = d_sq

        if best_dist_sq == float('inf'):
            # Fallback: brute force search all reference points
            for rlat, rlng in reference_points:
                dy = (lat - rlat) * 111.0
                dx = (lng - rlng) * 111.0 * math.cos(math.radians(lat))
                d_sq = dx * dx + dy * dy
                if d_sq < best_dist_sq:
                    best_dist_sq = d_sq

        distances[(lat, lng)] = round(math.sqrt(best_dist_sq), 2)

        if (i + 1) % 50000 == 0:
            print(f"    {i + 1}/{len(cells)} computed")

    return distances


# ═══════════════════════════════════════════════
# Layer 2: Grid Connection Proximity
# ═══════════════════════════════════════════════

def fetch_major_towns():
    """Download Major Towns and Cities from ONS."""
    return fetch_arcgis_features(
        "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
        "Major_Towns_and_Cities_December_2015_Boundaries/FeatureServer/0/query",
        TOWNS_CACHE,
        "Major Towns and Cities",
        out_fields="TCITY15NM",
        max_features=200,
    )


# ═══════════════════════════════════════════════
# Layer 3: Biodiversity (Priority Habitats)
# ═══════════════════════════════════════════════

def fetch_priority_habitats():
    """
    Download Priority Habitats Inventory from Natural England.
    This is a large dataset (~hundreds of thousands of polygons).
    We'll use a simplified approach: query for features that intersect
    our bounding box, using the ArcGIS spatial query.
    """
    if os.path.exists(PHI_CACHE):
        print(f"  Loading cached Priority Habitats from {PHI_CACHE}")
        with open(PHI_CACHE) as f:
            return json.load(f)

    print("  Downloading Priority Habitats Inventory...")
    print("  (This is a large dataset — may take several minutes)")

    base_url = ("https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/"
                "Priority_Habitats_Inventory_England/FeatureServer/0/query")

    all_features = []
    offset = 0
    batch_size = 2000
    max_features = 500000  # PHI is very large

    while offset < max_features:
        params = (
            f"?where=1%3D1"
            f"&outFields=OBJECTID"
            f"&outSR=4326"
            f"&f=geojson"
            f"&resultOffset={offset}"
            f"&resultRecordCount={batch_size}"
            f"&returnGeometry=true"
        )
        url = base_url + params

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "GridSight/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())

            features = data.get("features", [])
            all_features.extend(features)

            if (offset // batch_size + 1) % 10 == 0 or len(features) < batch_size:
                print(f"    Fetched {len(all_features)} features so far...")

            if len(features) < batch_size:
                break

            offset += batch_size
            time.sleep(0.3)

        except Exception as e:
            print(f"    Error at offset {offset}: {e}")
            time.sleep(5)
            continue

    with open(PHI_CACHE, "w") as f:
        json.dump(all_features, f)
    print(f"    Saved {len(all_features)} features to {PHI_CACHE}")
    return all_features


def build_habitat_spatial_index(features):
    """Build a spatial index for habitat polygons."""
    index = {}
    index_step = 0.1  # finer than constraints since habitats are smaller

    for i, feat in enumerate(features):
        geom = feat.get("geometry", {})
        if not geom or geom.get("type") not in ("Polygon", "MultiPolygon"):
            continue

        coords = geom.get("coordinates", [])
        if not coords:
            continue

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

        for ilat in range(int(min_lat / index_step), int(max_lat / index_step) + 1):
            for ilng in range(int(min_lng / index_step), int(max_lng / index_step) + 1):
                key = (ilat, ilng)
                if key not in index:
                    index[key] = []
                index[key].append(i)

    return index


def point_in_polygon_ring(lat, lng, ring):
    """Ray-casting point-in-polygon for a single ring."""
    inside = False
    n = len(ring)
    for i in range(n):
        j = (i - 1) % n
        xi, yi = ring[i][1], ring[i][0]
        xj, yj = ring[j][1], ring[j][0]
        if ((xi > lat) != (xj > lat)) and (lng < (yj - yi) * (lat - xi) / (xj - xi) + yi):
            inside = not inside
    return inside


def point_in_habitat(lat, lng, feature):
    """Test if a point is inside a habitat polygon."""
    geom = feature.get("geometry", {})
    gtype = geom.get("type", "")
    coords = geom.get("coordinates", [])

    if gtype == "Polygon":
        if coords and point_in_polygon_ring(lat, lng, coords[0]):
            for hole in coords[1:]:
                if point_in_polygon_ring(lat, lng, hole):
                    return False
            return True
    elif gtype == "MultiPolygon":
        for poly in coords:
            if poly and point_in_polygon_ring(lat, lng, poly[0]):
                in_hole = False
                for hole in poly[1:]:
                    if point_in_polygon_ring(lat, lng, hole):
                        in_hole = True
                        break
                if not in_hole:
                    return True
    return False


def flag_habitat_cells(cells, features, spatial_index):
    """Flag cells that overlap priority habitats."""
    index_step = 0.1
    flagged = set()

    print(f"  Checking {len(cells)} cells against {len(features)} habitat features...")

    for i, (lat, lng) in enumerate(cells):
        ikey = (int(lat / index_step), int(lng / index_step))
        candidates = spatial_index.get(ikey, [])

        for fi in candidates:
            if point_in_habitat(lat, lng, features[fi]):
                flagged.add((lat, lng))
                break

        if (i + 1) % 50000 == 0:
            print(f"    {i + 1}/{len(cells)} checked, {len(flagged)} in priority habitat")

    print(f"    Done: {len(flagged)} cells in priority habitat ({len(flagged)/len(cells)*100:.1f}%)")
    return flagged


# ═══════════════════════════════════════════════
# Layer 4: Fuel Poverty
# ═══════════════════════════════════════════════

def load_fuel_poverty_csv():
    """
    Load fuel poverty rates by LSOA from a user-supplied CSV.
    Expected format: CSV with LSOA code and fuel poverty proportion columns.
    Returns dict of {lsoa_code: fuel_poverty_rate}.
    """
    csv_candidates = [
        "fuel_poverty_lsoa.csv",
        "sub-regional-fuel-poverty-2024-tables.csv",
        "fuel-poverty-sub-regional-tables-2024.csv",
    ]

    for filename in csv_candidates:
        if os.path.exists(filename):
            print(f"  Loading fuel poverty data from {filename}")
            fp_data = {}

            with open(filename, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames

                # Find the LSOA code column
                code_col = None
                rate_col = None
                for h in headers:
                    hl = h.lower().strip()
                    if 'lsoa' in hl and ('code' in hl or 'cd' in hl):
                        code_col = h
                    elif 'proportion' in hl and 'fuel' in hl:
                        rate_col = h
                    elif 'fuel poor' in hl and '%' in hl:
                        rate_col = h

                if not code_col or not rate_col:
                    print(f"    Could not identify columns. Headers: {headers[:10]}")
                    print(f"    Looking for LSOA code column and fuel poverty rate column")
                    continue

                print(f"    Using columns: code='{code_col}', rate='{rate_col}'")

                for row in reader:
                    try:
                        code = row[code_col].strip()
                        rate = float(row[rate_col].strip().replace('%', ''))
                        fp_data[code] = rate
                    except (ValueError, KeyError):
                        continue

            print(f"    Loaded {len(fp_data)} LSOA fuel poverty rates")
            return fp_data

    print("  No fuel poverty CSV found. Skipping fuel poverty layer.")
    print("  To enable, download from:")
    print("  https://www.gov.uk/government/statistics/sub-regional-fuel-poverty-data-2024")
    print("  and save as 'fuel_poverty_lsoa.csv'")
    return None


def compute_fuel_poverty_scores(cells, lsoa_features, fp_data):
    """
    For each grid cell, find the nearest LSOA and assign its fuel poverty rate.
    """
    if not fp_data:
        return {}

    # Build lookup: LSOA centroid → fuel poverty rate
    fp_points = []
    for feat in lsoa_features:
        geom = feat.get("geometry", {})
        props = feat.get("properties", {})
        if geom and geom.get("type") == "Point":
            lng, lat = geom["coordinates"][:2]
            code = props.get("lsoa11cd", "")
            if code in fp_data:
                fp_points.append((lat, lng, fp_data[code]))

    if not fp_points:
        print("  No matching LSOA codes found between centroids and fuel poverty data")
        return {}

    print(f"  Matching {len(fp_points)} LSOAs with fuel poverty data to grid cells...")

    # Bucket the fuel poverty points
    BUCKET_SIZE = 0.2
    buckets = {}
    for rlat, rlng, rate in fp_points:
        bkey = (int(rlat / BUCKET_SIZE), int(rlng / BUCKET_SIZE))
        if bkey not in buckets:
            buckets[bkey] = []
        buckets[bkey].append((rlat, rlng, rate))

    fp_scores = {}
    for i, (lat, lng) in enumerate(cells):
        bkey = (int(lat / BUCKET_SIZE), int(lng / BUCKET_SIZE))

        best_dist_sq = float('inf')
        best_rate = 0

        for dlat in [-1, 0, 1]:
            for dlng in [-1, 0, 1]:
                nkey = (bkey[0] + dlat, bkey[1] + dlng)
                for rlat, rlng, rate in buckets.get(nkey, []):
                    dy = (lat - rlat) * 111.0
                    dx = (lng - rlng) * 111.0 * math.cos(math.radians(lat))
                    d_sq = dx * dx + dy * dy
                    if d_sq < best_dist_sq:
                        best_dist_sq = d_sq
                        best_rate = rate

        fp_scores[(lat, lng)] = round(best_rate, 1)

        if (i + 1) % 50000 == 0:
            print(f"    {i + 1}/{len(cells)} computed")

    return fp_scores


# ═══════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════

def save_output(cells, existing_data, cf_data, co2_data,
                res_distances, grid_distances, habitat_cells,
                fp_scores, constraint_data):
    """Write updated grid_data_embedded.js with all layers."""

    sorted_cells = sorted(cells, key=lambda c: (c[0], c[1]))

    print(f"\nWriting grid_data_embedded.js with {len(sorted_cells)} cells...")

    with open("grid_data_embedded.js", "w") as f:
        f.write("// GridSight data — auto-generated by process_social_layers.py\n")
        f.write(f"// {len(sorted_cells)} cells at {GRID_STEP}° resolution\n")
        f.write("// Fields: elevation, wind, cf, constraints, co2, res_dist, grid_dist, habitat, fuel_poverty\n")
        f.write(f"var GRID_DATA_STEP = {GRID_STEP};\n")
        f.write("var GRID_DATA = new Map();\n")
        f.write("(function() {\n")
        f.write("  const d = [\n")

        for lat, lng in sorted_cells:
            key = f"{lat},{lng}"
            existing = existing_data.get(key, {})
            elev = existing.get("elevation_m", 0)
            wind = existing.get("wind_speed_ms", 0)
            cf = float(cf_data.get(key, 0))
            co2 = float(co2_data.get(key, 0))
            constraints = constraint_data.get((lat, lng), 0)
            res_dist = res_distances.get((lat, lng), 99)
            grid_dist = grid_distances.get((lat, lng), 99)
            habitat = 1 if (lat, lng) in habitat_cells else 0
            fp = fp_scores.get((lat, lng), 0)

            # Format: [lat, lng, elev, wind*10, cf*1000, constraints, co2, res_dist*10, grid_dist*10, habitat, fp*10]
            f.write(f"[{lat},{lng},{int(elev)},{int(round(wind*10))},"
                    f"{int(round(cf*1000))},{constraints},{int(co2)},"
                    f"{int(round(res_dist*10))},{int(round(grid_dist*10))},"
                    f"{habitat},{int(round(fp*10))}],\n")

        f.write("  ];\n")
        f.write("  for (const r of d) GRID_DATA.set(r[0]+','+r[1], {\n")
        f.write("    elevation:r[2], wind:r[3]/10, cf:r[4]/1000, constraints:r[5],\n")
        f.write("    co2:r[6], resDist:r[7]/10, gridDist:r[8]/10, habitat:r[9], fuelPoverty:r[10]/10\n")
        f.write("  });\n")
        f.write("})();\n")

    size_mb = os.path.getsize("grid_data_embedded.js") / 1024 / 1024
    print(f"Saved grid_data_embedded.js ({size_mb:.1f} MB)")


def load_existing_data():
    """Load all existing processed data."""
    existing = {}
    if os.path.exists("grid_data.json"):
        with open("grid_data.json") as f:
            existing = json.load(f)

    cf_data = {}
    if os.path.exists("capacity_factors.json"):
        with open("capacity_factors.json") as f:
            cf_data = json.load(f)

    co2_data = {}
    if os.path.exists("co2_displacement.json"):
        with open("co2_displacement.json") as f:
            co2_data = json.load(f)

    return existing, cf_data, co2_data


def load_constraint_flags(cells):
    """Recompute constraint flags from cached boundary JSON files."""
    constraints = {(lat, lng): 0 for lat, lng in cells}
    
    boundary_files = {
        1: "sssi_boundaries.json",
        2: "national_park_boundaries.json",
        4: "aonb_boundaries.json",
    }
    
    for flag_bit, filename in boundary_files.items():
        if not os.path.exists(filename):
            print(f"  Warning: {filename} not found, skipping")
            continue
        
        with open(filename) as f:
            features = json.load(f)
        
        if not features:
            continue
        
        # Build spatial index
        index = {}
        index_step = 0.5
        for i, feat in enumerate(features):
            geom = feat.get("geometry", {})
            if not geom or geom.get("type") not in ("Polygon", "MultiPolygon"):
                continue
            coords = geom.get("coordinates", [])
            if not coords:
                continue
            
            def flatten(c):
                if isinstance(c[0], (int, float)):
                    yield c
                else:
                    for sub in c:
                        yield from flatten(sub)
            
            pts = list(flatten(coords))
            if not pts:
                continue
            lats_f = [p[1] for p in pts]
            lngs_f = [p[0] for p in pts]
            for ilat in range(int(min(lats_f) / index_step), int(max(lats_f) / index_step) + 1):
                for ilng in range(int(min(lngs_f) / index_step), int(max(lngs_f) / index_step) + 1):
                    key = (ilat, ilng)
                    if key not in index:
                        index[key] = []
                    index[key].append(i)
        
        # Check each cell
        flagged = 0
        for lat, lng in cells:
            ikey = (int(lat / index_step), int(lng / index_step))
            for fi in index.get(ikey, []):
                feat = features[fi]
                geom = feat.get("geometry", {})
                gtype = geom.get("type", "")
                gcoords = geom.get("coordinates", [])
                
                hit = False
                if gtype == "Polygon" and gcoords:
                    hit = _pip_ring(lat, lng, gcoords[0])
                    if hit:
                        for hole in gcoords[1:]:
                            if _pip_ring(lat, lng, hole):
                                hit = False
                                break
                elif gtype == "MultiPolygon":
                    for poly in gcoords:
                        if poly and _pip_ring(lat, lng, poly[0]):
                            in_hole = False
                            for hole in poly[1:]:
                                if _pip_ring(lat, lng, hole):
                                    in_hole = True
                                    break
                            if not in_hole:
                                hit = True
                                break
                
                if hit:
                    constraints[(lat, lng)] |= flag_bit
                    flagged += 1
                    break
        
        label = filename.replace("_boundaries.json", "").upper()
        print(f"  {label}: {flagged} cells flagged from {len(features)} features")
    
    total_flagged = sum(1 for v in constraints.values() if v > 0)
    print(f"  Total constrained: {total_flagged} cells")
    return constraints


def _pip_ring(lat, lng, ring):
    """Ray-casting point-in-polygon for a single ring."""
    inside = False
    n = len(ring)
    for i in range(n):
        j = (i - 1) % n
        xi, yi = ring[i][1], ring[i][0]
        xj, yj = ring[j][1], ring[j][0]
        if ((xi > lat) != (xj > lat)) and (lng < (yj - yi) * (lat - xi) / (xj - xi) + yi):
            inside = not inside
    return inside


def main():
    print("=" * 60)
    print("GridSight — Social & Environmental Layers")
    print("=" * 60)

    # Load existing data
    print("\n[1/6] Loading existing data...")
    existing_data, cf_data, co2_data = load_existing_data()

    cells = []
    for key in existing_data:
        lat_s, lng_s = key.split(",")
        cells.append((float(lat_s), float(lng_s)))
    print(f"  {len(cells)} cells, {len(cf_data)} capacity factors, {len(co2_data)} CO2 values")

    constraint_data = load_constraint_flags(cells)

    # Layer 1: Residential proximity
    print("\n[2/6] Computing residential proximity...")
    lsoa_features = fetch_lsoa_centroids()
    lsoa_points = extract_points(lsoa_features)
    print(f"  {len(lsoa_points)} LSOA centroids loaded")
    res_distances = compute_nearest_distances(cells, lsoa_points)
    vals = list(res_distances.values())
    print(f"  Distance range: {min(vals):.1f} – {max(vals):.1f} km, mean: {np.mean(vals):.1f} km")

    # Layer 2: Grid connection proximity
    print("\n[3/6] Computing grid connection proximity...")
    town_features = fetch_major_towns()
    town_centroids, town_names = extract_polygon_centroids(town_features)
    print(f"  {len(town_centroids)} major towns loaded")
    print(f"  Sample towns: {', '.join(town_names[:5])}")
    grid_distances = compute_nearest_distances(cells, town_centroids)
    vals = list(grid_distances.values())
    print(f"  Distance range: {min(vals):.1f} – {max(vals):.1f} km, mean: {np.mean(vals):.1f} km")

    # Layer 3: Biodiversity
    print("\n[4/6] Computing biodiversity (Priority Habitats)...")
    phi_features = fetch_priority_habitats()
    if phi_features:
        phi_index = build_habitat_spatial_index(phi_features)
        habitat_cells = flag_habitat_cells(cells, phi_features, phi_index)
    else:
        habitat_cells = set()
        print("  No habitat data available, skipping")

    # Layer 4: Fuel poverty
    print("\n[5/6] Computing fuel poverty proximity...")
    fp_data = load_fuel_poverty_csv()
    fp_scores = compute_fuel_poverty_scores(cells, lsoa_features, fp_data) if fp_data else {}

    # Save
    print("\n[6/6] Saving output...")
    save_output(cells, existing_data, cf_data, co2_data,
                res_distances, grid_distances, habitat_cells,
                fp_scores, constraint_data)

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Residential proximity: {min(res_distances.values()):.1f}–{max(res_distances.values()):.1f} km")
    print(f"  Grid connection proximity: {min(grid_distances.values()):.1f}–{max(grid_distances.values()):.1f} km")
    print(f"  Priority habitat cells: {len(habitat_cells)} ({len(habitat_cells)/len(cells)*100:.1f}%)")
    if fp_scores:
        fp_vals = list(fp_scores.values())
        print(f"  Fuel poverty rates: {min(fp_vals):.1f}–{max(fp_vals):.1f}%")
    else:
        print(f"  Fuel poverty: not available (no CSV provided)")
    print()
    print("New fields in GRID_DATA: resDist, gridDist, habitat, fuelPoverty")
    print("=" * 60)


if __name__ == "__main__":
    main()
