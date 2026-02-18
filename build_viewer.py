#!/usr/bin/env python3
"""
Generate viewer_data.json for the Leaflet tile viewer.

Reads the county grid metadata and converts State Plane coordinates
to WGS84 lat/lon for each tile. Run once after tile_county.py finishes
(or while it's running -- it picks up whatever PNGs exist).

Usage:
    python build_viewer.py
"""

import json
from pathlib import Path

from pyproj import Transformer

TILE_SIZE = 10560  # feet
CRS = "EPSG:3735"
OUTPUT_DIR = Path("results/county_tiles")
META_PATH = OUTPUT_DIR / "metadata.json"

KNOWN_SITES = {
    "Great Hopewell Road": (1979967.0, 738723.0),
    "Great Circle Earthworks": (1987788.0, 743540.0),
}


def main():
    with open(META_PATH) as f:
        meta = json.load(f)

    cb = meta["county_bounds"]
    n_rows = meta["grid_rows"]
    n_cols = meta["grid_cols"]

    transformer = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)

    processed = set()
    for png in OUTPUT_DIR.glob("R??_C??.png"):
        processed.add(png.stem)

    tiles = []
    for r in range(n_rows):
        for c in range(n_cols):
            x_min = cb["x_min"] + c * TILE_SIZE
            x_max = x_min + TILE_SIZE
            y_min = cb["y_min"] + r * TILE_SIZE
            y_max = y_min + TILE_SIZE

            tid = f"R{r:02d}_C{c:02d}"

            lon_sw, lat_sw = transformer.transform(x_min, y_min)
            lon_ne, lat_ne = transformer.transform(x_max, y_max)

            tiles.append({
                "id": tid,
                "row": r,
                "col": c,
                "bounds": [[lat_sw, lon_sw], [lat_ne, lon_ne]],
                "has_png": tid in processed,
                "png": f"{tid}.png",
            })

    sites = []
    for name, (sx, sy) in KNOWN_SITES.items():
        lon, lat = transformer.transform(sx, sy)
        sites.append({"name": name, "lat": lat, "lon": lon})

    viewer_data = {
        "tiles": tiles,
        "sites": sites,
        "grid_rows": n_rows,
        "grid_cols": n_cols,
        "total_processed": len(processed),
        "total_tiles": n_rows * n_cols,
    }

    out_path = OUTPUT_DIR / "viewer_data.json"
    with open(out_path, "w") as f:
        json.dump(viewer_data, f, indent=2)

    print(f"Generated {out_path}")
    print(f"  {len(processed)} / {n_rows * n_cols} tiles have PNGs")
    print(f"  {len(sites)} known sites")
    print(f"\nOpen viewer.html in a browser to view the map.")


if __name__ == "__main__":
    main()
