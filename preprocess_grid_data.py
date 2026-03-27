#!/usr/bin/env python3
"""
Preprocessor for GridSight: fetches real elevation and wind data
for a 0.01° (~1km) grid over England & Wales.

Data sources:
  - Elevation: Open Topo Data API (SRTM 30m) — https://www.opentopodata.org
  - Wind speed: Open-Meteo Climate API (ERA5 annual mean 10m wind) — https://open-meteo.com

Output: grid_data.json — a JSON file keyed by "lat,lng" with elevation_m and wind_speed_ms
        grid_data.bin — a compact binary file for fast frontend loading

Usage:
  python3 preprocess_grid_data.py

Note: This will make many API calls. With batching and rate limiting:
  - Open Topo Data: ~2,100 batch requests (100 locations each)
  - Open-Meteo: ~213,000 individual requests (but they're fast)

For a demo, you can reduce GRID_STEP to 0.05 for a quick test run (~2k cells).
"""

import json
import math
import struct
import time
import urllib.request
import urllib.error
import sys
import os

# ── Configuration ──
GRID_STEP = 0.01  # ~1km resolution. Use 0.05 for fast testing.
BATCH_SIZE_ELEV = 100  # Open Topo Data allows up to 100 locations per request
RATE_LIMIT_PAUSE = 1.1  # seconds between Open Topo Data batches (1 req/sec free tier)
SKIP_ELEVATION = True  # Set to False to re-fetch elevation data from API
OPEN_METEO_BATCH = 50  # Process wind in chunks for progress reporting

# ── Load the England+Wales boundary polygon ──
# This is the same polygon from the frontend, extracted from the world-geojson npm package.
POLY_FILE = "england_wales_poly_js.json"
if os.path.exists(POLY_FILE):
    with open(POLY_FILE) as f:
        ENGLAND_POLY = json.load(f)
else:
    print(f"ERROR: {POLY_FILE} not found. Run the previous setup first.")
    sys.exit(1)


def point_in_poly(lat, lng, poly=ENGLAND_POLY):
    """Ray-casting point-in-polygon test. poly is [[lng, lat], ...]"""
    inside = False
    n = len(poly)
    for i in range(n):
        j = (i - 1) % n
        xi, yi = poly[i][1], poly[i][0]  # lat, lng of vertex
        xj, yj = poly[j][1], poly[j][0]
        if ((xi > lat) != (xj > lat)) and (lng < (yj - yi) * (lat - xi) / (xj - xi) + yi):
            inside = not inside
    return inside


# ── Step 1: Generate all grid cell centers that fall on land ──
def generate_grid_cells():
    """Generate all (lat, lng) cell centers inside England+Wales."""
    LAT_MIN, LAT_MAX = 49.9, 55.9
    LNG_MIN, LNG_MAX = -5.8, 1.9

    cells = []
    lat = LAT_MIN
    while lat < LAT_MAX:
        lng = LNG_MIN
        while lng < LNG_MAX:
            clat = round(lat + GRID_STEP / 2, 4)
            clng = round(lng + GRID_STEP / 2, 4)
            if point_in_poly(clat, clng):
                cells.append((clat, clng))
            lng += GRID_STEP
        lat += GRID_STEP

    return cells


# ── Step 2: Fetch elevation data from Open Topo Data ──
def fetch_elevation_batch(locations):
    """
    Fetch elevation for a batch of (lat, lng) pairs.
    Uses Open Topo Data SRTM 30m endpoint.
    Returns dict of {(lat, lng): elevation_m}
    """
    loc_str = "|".join(f"{lat},{lng}" for lat, lng in locations)
    url = f"https://api.opentopodata.org/v1/srtm30m?locations={loc_str}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "GridSight/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        if data.get("status") != "OK":
            print(f"  Warning: API returned status={data.get('status')}")
            return {}

        result = {}
        for item in data.get("results", []):
            lat = item["location"]["lat"]
            lng = item["location"]["lng"]
            elev = item.get("elevation")
            if elev is not None:
                result[(round(lat, 4), round(lng, 4))] = round(elev)
            else:
                result[(round(lat, 4), round(lng, 4))] = 0  # ocean/void
        return result

    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
        print(f"  Error fetching elevation: {e}")
        return {}


def fetch_all_elevations(cells):
    """Fetch elevation for all cells in batches."""
    elevations = {}
    total_batches = math.ceil(len(cells) / BATCH_SIZE_ELEV)

    print(f"\nFetching elevation for {len(cells)} cells in {total_batches} batches...")
    print(f"Estimated time: ~{total_batches * RATE_LIMIT_PAUSE / 60:.0f} minutes\n")

    for i in range(0, len(cells), BATCH_SIZE_ELEV):
        batch = cells[i:i + BATCH_SIZE_ELEV]
        batch_num = i // BATCH_SIZE_ELEV + 1

        result = fetch_elevation_batch(batch)
        elevations.update(result)

        if batch_num % 50 == 0 or batch_num == total_batches:
            print(f"  Batch {batch_num}/{total_batches} — {len(elevations)} elevations fetched")

        # Rate limiting
        if batch_num < total_batches:
            time.sleep(RATE_LIMIT_PAUSE)

    return elevations


# ── Step 3: Fetch wind speed from Open-Meteo ERA5 Historical ──
WIND_CACHE_FILE = "wind_samples_cache.json"


def fetch_wind_era5(lat, lng, retries=3):
    """
    Fetch annual mean wind speed at 100m from Open-Meteo ERA5 historical API.
    Uses 100m wind (close to typical turbine hub height) for a full year.
    Returns wind speed in m/s or None on failure.
    Includes retry logic with exponential backoff.
    """
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lng}"
        f"&start_date=2023-01-01&end_date=2023-12-31"
        f"&hourly=wind_speed_100m&wind_speed_unit=ms"
        f"&timezone=UTC"
    )

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "GridSight/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())

            # Check for API error responses
            if data.get("error"):
                reason = data.get("reason", "unknown")
                if "rate limit" in reason.lower() or "too many" in reason.lower():
                    wait = 10 * (attempt + 1)
                    print(f"    Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                # If 100m not available, try 10m
                break

            hourly = data.get("hourly", {}).get("wind_speed_100m", [])
            if hourly:
                valid = [v for v in hourly if v is not None]
                if valid:
                    return round(sum(valid) / len(valid), 2)

            # If we got a response but no data, break (no point retrying)
            break

        except urllib.error.HTTPError as e:
            if e.code == 429:  # Too Many Requests
                wait = 15 * (attempt + 1)
                print(f"    HTTP 429 rate limit, waiting {wait}s...")
                time.sleep(wait)
                continue
            elif e.code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            break
        except (urllib.error.URLError, json.JSONDecodeError, ConnectionError) as e:
            time.sleep(3 * (attempt + 1))
            continue

    # Fallback: try 10m wind
    for attempt in range(2):
        try:
            url2 = (
                f"https://archive-api.open-meteo.com/v1/archive?"
                f"latitude={lat}&longitude={lng}"
                f"&start_date=2023-01-01&end_date=2023-12-31"
                f"&hourly=wind_speed_10m&wind_speed_unit=ms"
                f"&timezone=UTC"
            )
            req2 = urllib.request.Request(url2, headers={"User-Agent": "GridSight/1.0"})
            with urllib.request.urlopen(req2, timeout=30) as resp2:
                data2 = json.loads(resp2.read())

            if data2.get("error"):
                break

            hourly2 = data2.get("hourly", {}).get("wind_speed_10m", [])
            if hourly2:
                valid2 = [v for v in hourly2 if v is not None]
                if valid2:
                    ws_10m = sum(valid2) / len(valid2)
                    # Power law extrapolation to 100m (exponent 0.143 for typical open terrain)
                    return round(ws_10m * (100 / 10) ** 0.143, 2)
            break
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            break
        except Exception:
            break

    return None


def load_wind_cache():
    """Load previously fetched wind samples from disk."""
    if os.path.exists(WIND_CACHE_FILE):
        with open(WIND_CACHE_FILE) as f:
            raw = json.load(f)
        # Keys are stored as "lat,lng" strings in JSON
        return {tuple(map(float, k.split(","))): v for k, v in raw.items()}
    return {}


def save_wind_cache(samples):
    """Save wind samples to disk for resume capability."""
    raw = {f"{lat},{lng}": v for (lat, lng), v in samples.items()}
    with open(WIND_CACHE_FILE, "w") as f:
        json.dump(raw, f)


def bilinear_interpolate(lat, lng, samples, sample_step):
    """
    Bilinear interpolation from a dict of {(lat, lng): value} sampled
    on a regular grid with the given step size.
    """
    # Find the four surrounding sample points
    lat0 = math.floor(lat / sample_step) * sample_step + sample_step / 2
    lng0 = math.floor(lng / sample_step) * sample_step + sample_step / 2

    # Ensure lat0/lng0 are the SW corner of the enclosing sample cell
    lat0 = round(lat0 - (sample_step if lat < lat0 else 0), 4)
    lng0 = round(lng0 - (sample_step if lng < lng0 else 0), 4)

    lat1 = round(lat0 + sample_step, 4)
    lng1 = round(lng0 + sample_step, 4)

    # Get corner values
    q00 = samples.get((lat0, lng0))
    q10 = samples.get((lat1, lng0))
    q01 = samples.get((lat0, lng1))
    q11 = samples.get((lat1, lng1))

    # Count how many corners we have
    corners = [(q00, 0), (q10, 1), (q01, 2), (q11, 3)]
    valid_corners = [(v, i) for v, i in corners if v is not None]

    if not valid_corners:
        return None

    if len(valid_corners) < 3:
        # Not enough for bilinear — use inverse-distance weighted average
        total_w = 0
        total_v = 0
        ref_points = [(lat0, lng0, q00), (lat1, lng0, q10),
                      (lat0, lng1, q01), (lat1, lng1, q11)]
        for plat, plng, val in ref_points:
            if val is not None:
                d = max(0.0001, math.sqrt((lat - plat) ** 2 + (lng - plng) ** 2))
                w = 1 / d
                total_w += w
                total_v += w * val
        return total_v / total_w if total_w > 0 else None

    # Fill missing corner with average of others for robustness
    avg = sum(v for v, _ in valid_corners) / len(valid_corners)
    if q00 is None: q00 = avg
    if q10 is None: q10 = avg
    if q01 is None: q01 = avg
    if q11 is None: q11 = avg

    # Fractional position within the cell
    t_lat = (lat - lat0) / sample_step if sample_step > 0 else 0
    t_lng = (lng - lng0) / sample_step if sample_step > 0 else 0
    t_lat = max(0, min(1, t_lat))
    t_lng = max(0, min(1, t_lng))

    # Bilinear formula
    val = (q00 * (1 - t_lat) * (1 - t_lng) +
           q10 * t_lat * (1 - t_lng) +
           q01 * (1 - t_lat) * t_lng +
           q11 * t_lat * t_lng)

    return val


def fetch_all_wind(cells, elevations=None):
    """
    Fetch wind speed for all cells using ERA5 100m wind data.

    Strategy:
    1. Sample at 0.05° (~5km) with caching and resume
    2. Bilinear interpolation to each 1km cell
    3. Elevation-based perturbation: adjust for local terrain vs. sample mean
    """
    WIND_SAMPLE_STEP = 0.05  # Sample every 0.05° (~5km)

    # Build the set of sample grid points that cover the data
    sample_lats = set()
    sample_lngs = set()
    for lat, lng in cells:
        base_lat = math.floor(lat / WIND_SAMPLE_STEP) * WIND_SAMPLE_STEP + WIND_SAMPLE_STEP / 2
        base_lng = math.floor(lng / WIND_SAMPLE_STEP) * WIND_SAMPLE_STEP + WIND_SAMPLE_STEP / 2
        for dl in [-WIND_SAMPLE_STEP, 0, WIND_SAMPLE_STEP]:
            sample_lats.add(round(base_lat + dl, 4))
            sample_lngs.add(round(base_lng + dl, 4))

    # Generate sample points
    sample_points = []
    for slat in sorted(sample_lats):
        for slng in sorted(sample_lngs):
            sample_points.append((slat, slng))

    # Filter to a reasonable bounding box
    sample_points = [(la, lo) for la, lo in sample_points
                     if 49.5 <= la <= 56.5 and -6.5 <= lo <= 2.5]

    # Load cache from previous partial runs
    wind_samples = load_wind_cache()
    cached_count = len(wind_samples)
    if cached_count > 0:
        print(f"\n  Loaded {cached_count} cached wind samples from {WIND_CACHE_FILE}")

    # Figure out which points still need fetching
    remaining = [(la, lo) for la, lo in sample_points if (la, lo) not in wind_samples]

    print(f"\nFetching wind speed for {len(remaining)} sample points ({len(sample_points)} total, {cached_count} cached)")
    print(f"Using Open-Meteo ERA5 historical API (100m wind, 2023 annual mean)")
    print(f"Will bilinearly interpolate to {len(cells)} cells.")
    if remaining:
        est_minutes = len(remaining) * 0.25 / 60  # ~0.25s per request with rate limiting
        print(f"Estimated time: ~{max(1, est_minutes):.0f} minutes\n")

    failures = 0
    consecutive_failures = 0
    for i, (lat, lng) in enumerate(remaining):
        ws = fetch_wind_era5(lat, lng)
        if ws is not None:
            wind_samples[(lat, lng)] = ws
            consecutive_failures = 0
        else:
            failures += 1
            consecutive_failures += 1

            # If we get many consecutive failures, we're probably rate limited
            # Back off aggressively
            if consecutive_failures == 10:
                print(f"    10 consecutive failures — backing off 30s...")
                time.sleep(30)
            elif consecutive_failures == 30:
                print(f"    30 consecutive failures — backing off 60s...")
                time.sleep(60)
            elif consecutive_failures >= 50:
                print(f"    50+ consecutive failures — saving cache and stopping wind fetch.")
                print(f"    Re-run the script to resume from where we left off.")
                save_wind_cache(wind_samples)
                break

        if (i + 1) % 100 == 0 or i == len(remaining) - 1:
            print(f"  {i + 1}/{len(remaining)} — {len(wind_samples)} total samples, {failures} failures")

        # Save cache periodically
        if (i + 1) % 500 == 0:
            save_wind_cache(wind_samples)

        # Rate limiting: ~4 requests/sec to stay well under Open-Meteo's limits
        time.sleep(0.25)

    # Final cache save
    save_wind_cache(wind_samples)
    print(f"\n  Total wind samples: {len(wind_samples)} ({failures} failures this run)")

    # Compute mean elevation per sample cell for perturbation
    sample_mean_elev = {}
    if elevations:
        sample_cell_elevs = {}
        for (clat, clng), elev in elevations.items():
            skey = (
                round(math.floor(clat / WIND_SAMPLE_STEP) * WIND_SAMPLE_STEP + WIND_SAMPLE_STEP / 2, 4),
                round(math.floor(clng / WIND_SAMPLE_STEP) * WIND_SAMPLE_STEP + WIND_SAMPLE_STEP / 2, 4),
            )
            if skey not in sample_cell_elevs:
                sample_cell_elevs[skey] = []
            sample_cell_elevs[skey].append(elev)

        for skey, elev_list in sample_cell_elevs.items():
            sample_mean_elev[skey] = sum(elev_list) / len(elev_list)

    # Interpolate to each 1km cell
    print(f"  Bilinear interpolation + elevation adjustment to {len(cells)} cells...")
    wind_data = {}
    fallback_count = 0

    for lat, lng in cells:
        ws = bilinear_interpolate(lat, lng, wind_samples, WIND_SAMPLE_STEP)

        if ws is None:
            # Last resort: nearest sample
            best_dist = float('inf')
            ws = None
            for (plat, plng), val in wind_samples.items():
                d = (plat - lat) ** 2 + (plng - lng) ** 2
                if d < best_dist:
                    best_dist = d
                    ws = val
            if ws is None:
                ws = 7.0  # absolute last fallback
            fallback_count += 1

        # Elevation-based perturbation
        if elevations and (lat, lng) in elevations:
            cell_elev = elevations[(lat, lng)]
            skey = (
                round(math.floor(lat / WIND_SAMPLE_STEP) * WIND_SAMPLE_STEP + WIND_SAMPLE_STEP / 2, 4),
                round(math.floor(lng / WIND_SAMPLE_STEP) * WIND_SAMPLE_STEP + WIND_SAMPLE_STEP / 2, 4),
            )
            mean_elev = sample_mean_elev.get(skey, cell_elev)
            elev_diff = cell_elev - mean_elev

            # ~0.3 m/s per 100m above/below local mean
            perturbation = elev_diff * 0.003
            ws = ws + perturbation

        wind_data[(lat, lng)] = round(max(1.0, min(18.0, ws)), 1)

    print(f"  Done. {fallback_count} cells used nearest-neighbor fallback.")
    return wind_data


# ── Step 4: Output ──
def save_json(cells, elevations, wind_data, filename="grid_data.json"):
    """Save as JSON for easy inspection."""
    output = {}
    for lat, lng in cells:
        key = f"{lat},{lng}"
        output[key] = {
            "elevation_m": elevations.get((lat, lng), 0),
            "wind_speed_ms": wind_data.get((lat, lng), 5.0),
        }

    with open(filename, "w") as f:
        json.dump(output, f)

    size_mb = os.path.getsize(filename) / 1024 / 1024
    print(f"\nSaved {filename} ({len(output)} cells, {size_mb:.1f} MB)")


def save_binary(cells, elevations, wind_data, filename="grid_data.bin"):
    """
    Save as compact binary for fast frontend loading.
    Format:
      Header: 4 bytes magic "GS01"
              4 bytes uint32 cell_count
              8 bytes float64 grid_step
      Per cell: 4 bytes float32 lat
                4 bytes float32 lng
                2 bytes int16 elevation_m
                2 bytes uint16 wind_speed_ms * 10 (e.g. 72 = 7.2 m/s)
    Total per cell: 12 bytes
    ~213k cells = ~2.5 MB
    """
    with open(filename, "wb") as f:
        # Header
        f.write(b"GS01")
        f.write(struct.pack("<I", len(cells)))
        f.write(struct.pack("<d", GRID_STEP))

        for lat, lng in cells:
            elev = elevations.get((lat, lng), 0)
            wind = wind_data.get((lat, lng), 5.0)

            f.write(struct.pack("<f", lat))
            f.write(struct.pack("<f", lng))
            f.write(struct.pack("<h", max(-500, min(9000, int(elev)))))
            f.write(struct.pack("<H", int(round(wind * 10))))

    size_mb = os.path.getsize(filename) / 1024 / 1024
    print(f"Saved {filename} ({len(cells)} cells, {size_mb:.1f} MB)")


def save_js_embedded(cells, elevations, wind_data, filename="grid_data_embedded.js"):
    """
    Save as a JS file that can be directly <script src>'d in the HTML.
    Uses a typed array approach for compact representation.
    """
    # Sort cells by lat then lng for consistent ordering
    sorted_cells = sorted(cells, key=lambda c: (c[0], c[1]))

    lines = [
        "// Auto-generated by preprocess_grid_data.py",
        f"// {len(sorted_cells)} cells at {GRID_STEP}° resolution",
        "// Format: [lat, lng, elevation_m, wind_speed_ms*10]",
        f"var GRID_DATA_STEP = {GRID_STEP};",
        f"var GRID_DATA = new Map();",
        "(function() {",
        "  const d = [",
    ]

    for lat, lng in sorted_cells:
        elev = elevations.get((lat, lng), 0)
        wind = wind_data.get((lat, lng), 5.0)
        lines.append(f"    [{lat},{lng},{int(elev)},{int(round(wind * 10))}],")

    lines.append("  ];")
    lines.append("  for (const r of d) {")
    lines.append("    GRID_DATA.set(r[0]+','+r[1], { elevation: r[2], wind: r[3]/10 });")
    lines.append("  }")
    lines.append("})();")

    with open(filename, "w") as f:
        f.write("\n".join(lines))

    size_mb = os.path.getsize(filename) / 1024 / 1024
    print(f"Saved {filename} ({len(sorted_cells)} cells, {size_mb:.1f} MB)")


# ── Main ──
def main():
    print("=" * 60)
    print("GridSight Data Preprocessor")
    print("=" * 60)
    print(f"Grid resolution: {GRID_STEP}° (~{GRID_STEP * 111:.0f} km)")

    # Step 1: Generate grid
    print("\n[1/4] Generating grid cells...")
    cells = generate_grid_cells()
    print(f"  {len(cells)} land cells found")

    # Step 2: Fetch elevation
    if SKIP_ELEVATION and os.path.exists("grid_data.json"):
        print("\n[2/4] Loading cached elevation data from grid_data.json...")
        with open("grid_data.json") as f:
            cached = json.load(f)
        elevations = {}
        for key, val in cached.items():
            lat_s, lng_s = key.split(",")
            elevations[(float(lat_s), float(lng_s))] = val["elevation_m"]
        print(f"  {len(elevations)} elevation values loaded from cache")
    else:
        print("\n[2/4] Fetching elevation data (SRTM 30m via Open Topo Data)...")
        elevations = fetch_all_elevations(cells)
        print(f"  {len(elevations)} elevation values retrieved")

    # Step 3: Fetch wind
    print("\n[3/4] Fetching wind speed data (Open-Meteo ERA5 100m, 2023)...")
    wind_data = fetch_all_wind(cells, elevations=elevations)
    print(f"  {len(wind_data)} wind values (sampled + bilinear interpolated + elevation adjusted)")

    # Step 4: Save
    print("\n[4/4] Saving output files...")
    save_json(cells, elevations, wind_data)
    save_binary(cells, elevations, wind_data)
    save_js_embedded(cells, elevations, wind_data)

    print("\n" + "=" * 60)
    print("Done! Place grid_data_embedded.js next to the HTML and refresh.")
    print("=" * 60)


if __name__ == "__main__":
    main()
