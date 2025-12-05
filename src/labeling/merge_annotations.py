"""Merge edited tile annotations back into the GeoTIFF coordinate frame."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import pandas as pd
import rasterio

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import (
    ensure_directory,
    pixel_to_map_geometry,
    tile_origin_from_name,
)


def _load_tiling_metadata(tiles_dir: Path) -> Tuple[Optional[int], Optional[int]]:
    metadata_file = tiles_dir / "tiling_metadata.json"
    if not metadata_file.exists():
        return None, None

    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid tiling metadata JSON at {metadata_file}") from exc

    return metadata.get("tile_size"), metadata.get("overlap")


def merge_annotations(
    input_tif,
    tiles_dir,
    annotations_dir,
    output_gpkg,
    *,
    tile_size: Optional[int] = None,
    overlap: Optional[int] = None,
    layer_name: str = "tornado_damage_labels",
):
    """Merge per-tile annotation GeoJSON files into a georeferenced GeoPackage."""

    tiles_path = Path(tiles_dir)
    annotations_path = Path(annotations_dir)
    out_path = Path(output_gpkg)

    inferred_tile_size, inferred_overlap = _load_tiling_metadata(tiles_path)
    tile_size = tile_size or inferred_tile_size or 400
    overlap = overlap or inferred_overlap or 128

    with rasterio.open(input_tif) as src:
        transform = src.transform
        crs = src.crs

    layers = []
    for geojson_file in sorted(annotations_path.glob("*.geojson")):
        stem = geojson_file.stem
        candidate_names = [f"{stem}.png", f"{stem}.jpg"]
        tile_file = next((tiles_path / name for name in candidate_names if (tiles_path / name).exists()), None)
        if tile_file is None:
            continue

        x0, y0 = tile_origin_from_name(tile_file.name, tile_size=tile_size, overlap=overlap)

        gdf = gpd.read_file(geojson_file)
        if gdf.empty:
            continue

        gdf["geometry"] = gdf["geometry"].translate(xoff=x0, yoff=y0)
        gdf["geometry"] = gdf["geometry"].apply(lambda geom: pixel_to_map_geometry(geom, transform))
        gdf.crs = crs
        layers.append(gdf)

    if not layers:
        print("[WARN] No annotations found.")
        return

    merged = gpd.GeoDataFrame(pd.concat(layers, ignore_index=True))
    merged.crs = crs
    ensure_directory(out_path.parent)
    merged.to_file(out_path, layer=layer_name, driver="GPKG")
    print(f"[OK] Wrote georeferenced annotations: {out_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge edited annotations into a GeoPackage")
    parser.add_argument("input_tif", help="Original GeoTIFF used for tiling")
    parser.add_argument("tiles_dir", help="Directory containing image tiles")
    parser.add_argument("annotations_dir", help="Directory with edited GeoJSON annotations")
    parser.add_argument("output_gpkg", help="Path to the output GeoPackage")
    parser.add_argument("--tile-size", type=int, help="Tile size in pixels (overrides metadata)")
    parser.add_argument("--overlap", type=int, help="Tile overlap in pixels (overrides metadata)")
    parser.add_argument("--layer", default="tornado_damage_labels", help="Layer name for the GeoPackage output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    merge_annotations(
        args.input_tif,
        args.tiles_dir,
        args.annotations_dir,
        args.output_gpkg,
        tile_size=args.tile_size,
        overlap=args.overlap,
        layer_name=args.layer,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
