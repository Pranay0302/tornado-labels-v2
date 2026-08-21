"""Orthomosaic tiling for the tornado labeling pipeline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from PIL import Image
from rasterio.warp import transform_bounds, transform_geom
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.windows import transform as window_transform
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import ensure_directory, is_blank_tile


def process_tile(
    src_path: str | Path,
    window: Window,
    tile_name: str,
    output_path: Path,
    is_floating: bool,
    nodata: float | None,
    *,
    tile_size: int = 640,
    blank_threshold: float = 0.95,
    variance_threshold: float = 1e-4,
    sharpness_threshold: float = 0.0,
    max_nodata_fraction: float = 0.5,
    min_coverage: float = 1.0,
) -> bool:
    """Read a single window from the raster, apply quality filters, and save if it passes.

    Quality filters applied in order (cheapest first):

    1. *min_coverage* — skip edge tiles whose area is less than this fraction
       of the full ``tile_size × tile_size`` square.
    2. *max_nodata_fraction* — skip tiles where more than this fraction of
       pixels are masked / nodata according to the raster's dataset mask.
    3. :func:`is_blank_tile` — skip near-uniform, all-white, all-black, or
       (optionally) blurry tiles.
    """
    # ------------------------------------------------------------------ #
    # 1. Minimum coverage — skip tiny edge slivers before reading pixels. #
    # ------------------------------------------------------------------ #
    coverage = (window.width * window.height) / (tile_size * tile_size)
    if coverage < min_coverage:
        return False

    with rasterio.open(src_path) as src:
        image = src.read(window=window)
        # dataset_mask: 0 = masked/nodata, 255 = valid data.
        dataset_mask = src.dataset_mask(window=window)

    # ------------------------------------------------------------------ #
    # 2. Nodata fraction — skip tiles dominated by missing data.          #
    # ------------------------------------------------------------------ #
    nodata_fraction = float(np.mean(dataset_mask == 0))
    if nodata_fraction > max_nodata_fraction:
        return False

    tile_array = image.transpose(1, 2, 0)
    if tile_array.shape[2] >= 3:
        tile_array = tile_array[:, :, :3]
    elif tile_array.shape[2] == 1:
        tile_array = tile_array.repeat(3, axis=2)

    if is_floating:
        if nodata is not None:
            mask = np.isclose(tile_array, nodata)
            tile_array[mask] = 0
        tile_array = (np.clip(tile_array, 0, 1) * 255).astype(np.uint8)

    # ------------------------------------------------------------------ #
    # 3. Content quality — blank / uniform / blurry detection.            #
    # ------------------------------------------------------------------ #
    if is_blank_tile(
        tile_array,
        blank_threshold=blank_threshold,
        variance_threshold=variance_threshold,
        sharpness_threshold=sharpness_threshold,
    ):
        return False

    Image.fromarray(tile_array).save(output_path / tile_name)
    return True


def _affine_to_list(transform) -> list[float]:
    """Serialise a rasterio/affine ``Affine`` as ``[a, b, c, d, e, f]``.

    This is affine order (``x = a·col + b·row + c``, ``y = d·col + e·row + f``);
    ``(c, f)`` is the top-left corner. It matches ``list(Affine)[:6]`` and lets a
    consumer rebuild the transform with ``rasterio.Affine(*values)``.
    """
    return [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f]


def _tile_feature(window: Window, tile_name: str, row: int, col: int, src_transform, src_crs) -> dict:
    """Build one GeoJSON ``Feature`` describing a single saved tile.

    The footprint polygon is reprojected to WGS84 (lon/lat) when the source has a
    CRS so the index opens correctly in any GIS / web map; the native-CRS pixel
    ``window``, per-tile ``transform`` and ``bounds`` needed to reassemble the
    tiles later are carried in ``properties``.
    """
    tile_transform = window_transform(window, src_transform)
    left, bottom, right, top = window_bounds(window, src_transform)

    # Counter-clockwise, closed exterior ring in the source CRS.
    ring = [[left, bottom], [right, bottom], [right, top], [left, top], [left, bottom]]
    geometry: dict = {"type": "Polygon", "coordinates": [ring]}
    if src_crs is not None:
        geometry = transform_geom(src_crs, "EPSG:4326", geometry)

    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": {
            "name": tile_name,
            "row": row,
            "col": col,
            "window": [int(window.col_off), int(window.row_off), int(window.width), int(window.height)],
            "transform": _affine_to_list(tile_transform),
            "bounds": [left, bottom, right, top],
        },
    }


def _build_tile_index(
    saved_entries: list[tuple[Window, str, int, int]],
    *,
    source: str | Path,
    src_crs,
    src_transform,
    width: int,
    height: int,
    tile_size: int,
    overlap: int,
    image_format: str,
    saved_count: int,
    skipped_count: int,
) -> dict:
    """Assemble the GeoJSON ``FeatureCollection`` tile index.

    The ``metadata`` header carries the source georeferencing; combined with each
    feature's pixel ``window`` it is sufficient to reconstruct every tile's exact
    position, which is what makes a later merge possible.
    """
    features = [
        _tile_feature(window, name, row, col, src_transform, src_crs)
        for window, name, row, col in saved_entries
    ]
    return {
        "type": "FeatureCollection",
        "metadata": {
            "source": str(Path(source).resolve()),
            "source_crs": src_crs.to_string() if src_crs is not None else None,
            "source_transform": _affine_to_list(src_transform),
            "width": width,
            "height": height,
            "tile_size": tile_size,
            "overlap": overlap,
            "image_format": image_format,
            "geometry_crs": "EPSG:4326" if src_crs is not None else None,
            "saved_tiles": saved_count,
            "skipped_tiles": skipped_count,
        },
        "features": features,
    }


def _read_valid_values(dataset, bounds: list[float], bounds_crs) -> tuple[np.ndarray, int]:
    """Read band-1 pixels of *dataset* inside geographic *bounds*.

    *bounds* is ``[left, bottom, right, top]`` in *bounds_crs* (the tile/source
    CRS); it is reprojected to the dataset CRS when they differ, so a CHM/NDVI on
    a different grid or resolution is sampled by geography, not pixel index.

    Returns ``(valid_values, total_pixels_read)`` where nodata / non-finite pixels
    are removed from ``valid_values`` but still counted in ``total_pixels_read``.
    Any non-overlap or read error yields ``(empty, 0)`` rather than raising.
    """
    left, bottom, right, top = bounds
    ds_crs = dataset.crs
    if bounds_crs is not None and ds_crs is not None and bounds_crs != ds_crs:
        left, bottom, right, top = transform_bounds(bounds_crs, ds_crs, left, bottom, right, top)

    try:
        win = window_from_bounds(left, bottom, right, top, dataset.transform)
    except Exception:
        return np.empty(0), 0

    # Clamp to the dataset extent (a partly-outside tile keeps its overlap).
    col_off = max(0, int(math.floor(win.col_off)))
    row_off = max(0, int(math.floor(win.row_off)))
    col_end = min(dataset.width, int(math.ceil(win.col_off + win.width)))
    row_end = min(dataset.height, int(math.ceil(win.row_off + win.height)))
    if col_end <= col_off or row_end <= row_off:
        return np.empty(0), 0

    try:
        data = dataset.read(1, window=Window(col_off, row_off, col_end - col_off, row_end - row_off))
    except Exception:
        return np.empty(0), 0

    total = int(data.size)
    arr = data.astype("float64", copy=False).ravel()
    if dataset.nodata is not None:
        arr = arr[arr != dataset.nodata]
    arr = arr[np.isfinite(arr)]
    return arr, total


def _chm_stats(dataset, bounds: list[float], bounds_crs) -> dict:
    """Canopy-height summary for one tile footprint (heights in the CHM's units)."""
    values, total = _read_valid_values(dataset, bounds, bounds_crs)
    if values.size == 0:
        return {
            "chm_mean": None,
            "chm_max": None,
            "chm_p95": None,
            "chm_valid_frac": (0.0 if total else None),
        }
    return {
        "chm_mean": round(float(values.mean()), 3),
        "chm_max": round(float(values.max()), 3),
        "chm_p95": round(float(np.percentile(values, 95)), 3),
        "chm_valid_frac": round(values.size / total, 4),
    }


def _ndvi_stats(dataset, bounds: list[float], bounds_crs, *, veg_threshold: float) -> dict:
    """NDVI summary for one tile footprint; *veg_threshold* sets the vegetation cut."""
    values, _ = _read_valid_values(dataset, bounds, bounds_crs)
    if values.size == 0:
        return {"ndvi_mean": None, "ndvi_veg_frac": None}
    return {
        "ndvi_mean": round(float(values.mean()), 4),
        "ndvi_veg_frac": round(float((values >= veg_threshold).mean()), 4),
    }


def _enrich_with_band_stats(
    index: dict,
    *,
    bounds_crs,
    chm_path: str | Path | None = None,
    ndvi_path: str | Path | None = None,
    ndvi_veg_threshold: float = 0.3,
) -> dict:
    """Attach per-tile CHM/NDVI stats to every feature of *index*, in place.

    Each source is opened once and sampled at every tile's native ``bounds``.
    Enrichment is best-effort: a source that cannot be opened is recorded but
    skipped, and per-tile read failures degrade to ``null`` stats (see
    :func:`_read_valid_values`) — tiling output is never lost to this step.
    """
    md = index["metadata"]

    for label, path, sampler in (
        ("chm", chm_path, lambda ds, b: _chm_stats(ds, b, bounds_crs)),
        ("ndvi", ndvi_path, lambda ds, b: _ndvi_stats(ds, b, bounds_crs, veg_threshold=ndvi_veg_threshold)),
    ):
        if not path:
            continue
        md[f"{label}_source"] = str(Path(path).resolve())
        if label == "ndvi":
            md["ndvi_veg_threshold"] = ndvi_veg_threshold
        try:
            dataset = rasterio.open(path)
        except Exception as exc:  # missing / unreadable source: record, skip
            print(f"WARNING: could not open {label.upper()} {path}: {exc}")
            continue
        try:
            for feature in index["features"]:
                feature["properties"].update(sampler(dataset, feature["properties"]["bounds"]))
        finally:
            dataset.close()

    return index


def tile_raster(
    input_tif: str | Path,
    output_dir: str | Path,
    *,
    tile_size: int = 640,
    overlap: int = 160,
    image_format: Literal["png", "jpg"] = "png",
    write_metadata: bool = True,
    write_geojson: bool = True,
    chm_path: str | Path | None = None,
    ndvi_path: str | Path | None = None,
    ndvi_veg_threshold: float = 0.3,
    num_workers: int = 8,
    # Quality-filter parameters forwarded to process_tile / is_blank_tile.
    blank_threshold: float = 0.95,
    variance_threshold: float = 1e-4,
    sharpness_threshold: float = 0.0,
    max_nodata_fraction: float = 0.5,
    min_coverage: float = 1.0,
) -> dict[str, int | str]:
    """Tile *input_tif* and write image tiles into *output_dir*.

    Returns a metadata dictionary containing basic statistics that can be
    persisted alongside the tiles.

    Quality-filter parameters
    -------------------------
    blank_threshold:
        Fraction of near-white or near-black pixels that marks a tile blank.
    variance_threshold:
        Minimum normalised luminance variance; tiles below this are skipped.
    sharpness_threshold:
        Minimum Laplacian proxy variance; ``0`` disables the sharpness check.
    max_nodata_fraction:
        Tiles with more than this fraction of nodata pixels are skipped.
    min_coverage:
        Tiles whose area is less than this fraction of ``tile_size²`` are
        skipped. Defaults to ``1.0`` so only full ``tile_size × tile_size``
        tiles are kept and partial edge tiles are dropped.
    """

    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be in [0, tile_size)")

    output_path = ensure_directory(output_dir)
    ext = ".jpg" if image_format.lower() == "jpg" else ".png"

    with rasterio.open(input_tif) as src:
        width, height = src.width, src.height
        is_floating = np.issubdtype(src.dtypes[0], np.floating)
        nodata = src.nodata
        src_crs = src.crs
        src_transform = src.transform
        step = tile_size - overlap
        x_steps = math.ceil((width - overlap) / step)
        y_steps = math.ceil((height - overlap) / step)

    print(f"Creating tiles: {x_steps}x{y_steps} = {x_steps * y_steps} total tiles")

    tasks = []
    for row in range(y_steps):
        for col in range(x_steps):
            x0 = col * step
            y0 = row * step
            window_width = min(tile_size, width - x0)
            window_height = min(tile_size, height - y0)
            if window_width <= 0 or window_height <= 0:
                continue

            window = Window(x0, y0, window_width, window_height)
            tile_name = f"tile_y{row:04d}_x{col:04d}{ext}"
            tasks.append((window, tile_name, row, col))

    saved_count = 0
    skipped_count = 0
    saved_entries: list[tuple[Window, str, int, int]] = []

    filter_kwargs = dict(
        tile_size=tile_size,
        blank_threshold=blank_threshold,
        variance_threshold=variance_threshold,
        sharpness_threshold=sharpness_threshold,
        max_nodata_fraction=max_nodata_fraction,
        min_coverage=min_coverage,
    )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {
            executor.submit(
                process_tile,
                input_tif,
                window,
                tile_name,
                output_path,
                is_floating,
                nodata,
                **filter_kwargs,
            ): (window, tile_name, row, col)
            for window, tile_name, row, col in tasks
        }

        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Tiling"):
            window, tile_name, row, col = future_to_task[future]
            if future.result():
                saved_count += 1
                saved_entries.append((window, tile_name, row, col))
            else:
                skipped_count += 1

    # Deterministic top-to-bottom, left-to-right ordering of the index.
    saved_entries.sort(key=lambda entry: (entry[2], entry[3]))

    # Georeferenced tile index: per-tile footprints + pixel windows so the tiles
    # can be reassembled into a georeferenced mosaic later if wanted.
    geojson_index: str | None = None
    if write_geojson:
        index = _build_tile_index(
            saved_entries,
            source=input_tif,
            src_crs=src_crs,
            src_transform=src_transform,
            width=width,
            height=height,
            tile_size=tile_size,
            overlap=overlap,
            image_format=image_format.lower(),
            saved_count=saved_count,
            skipped_count=skipped_count,
        )
        if chm_path or ndvi_path:
            _enrich_with_band_stats(
                index,
                bounds_crs=src_crs,
                chm_path=chm_path,
                ndvi_path=ndvi_path,
                ndvi_veg_threshold=ndvi_veg_threshold,
            )
        geojson_name = "tiles_index.geojson"
        (output_path / geojson_name).write_text(json.dumps(index, indent=2), encoding="utf-8")
        geojson_index = geojson_name

    metadata = {
        "input": str(Path(input_tif).resolve()),
        "tile_size": tile_size,
        "overlap": overlap,
        "image_format": image_format.lower(),
        "crs": src_crs.to_string() if src_crs is not None else None,
        "source_transform": _affine_to_list(src_transform),
        "saved_tiles": saved_count,
        "skipped_tiles": skipped_count,
        "geojson_index": geojson_index,
        "filter": {
            "blank_threshold": blank_threshold,
            "variance_threshold": variance_threshold,
            "sharpness_threshold": sharpness_threshold,
            "max_nodata_fraction": max_nodata_fraction,
            "min_coverage": min_coverage,
        },
    }

    if write_metadata:
        metadata_path = output_path / "tiling_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[OK] Tiles written to: {output_path}")
    print(f"[STATS] Saved: {saved_count}, Skipped (blank/nodata/edge): {skipped_count}")
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tile an orthomosaic into square image chips")
    parser.add_argument("input_tif", help="Source GeoTIFF to tile")
    parser.add_argument("output_dir", help="Directory where image tiles will be stored")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile size in pixels (default: 640)")
    parser.add_argument("--overlap", type=int, default=160, help="Overlap between tiles in pixels (default: 160)")
    parser.add_argument(
        "--image-format",
        choices=["png", "jpg"],
        default="png",
        help="Image format for saved tiles (default: png)",
    )
    parser.add_argument("--no-metadata", action="store_true", help="Do not write tiling metadata JSON")
    parser.add_argument(
        "--no-geojson",
        action="store_true",
        help="Do not write the georeferenced tiles_index.geojson tile index",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel workers (default: 8)")

    bands = parser.add_argument_group("per-tile band stats (added to tiles_index.geojson)")
    bands.add_argument("--chm", metavar="PATH", help="CHM raster to sample per-tile height stats from")
    bands.add_argument("--ndvi", metavar="PATH", help="NDVI raster to sample per-tile vegetation stats from")
    bands.add_argument(
        "--ndvi-veg-threshold",
        type=float,
        default=0.3,
        metavar="VAL",
        help="NDVI value at/above which a pixel counts as vegetation (default: 0.3)",
    )

    qf = parser.add_argument_group("quality filters")
    qf.add_argument(
        "--blank-threshold",
        type=float,
        default=0.95,
        metavar="FRAC",
        help="Fraction of near-white or near-black pixels that marks a tile blank (default: 0.95)",
    )
    qf.add_argument(
        "--variance-threshold",
        type=float,
        default=1e-4,
        metavar="VAR",
        help="Minimum normalised luminance variance; tiles below this are skipped (default: 1e-4)",
    )
    qf.add_argument(
        "--sharpness-threshold",
        type=float,
        default=0.0,
        metavar="VAR",
        help="Minimum Laplacian proxy variance for sharpness check; 0 disables it (default: 0)",
    )
    qf.add_argument(
        "--max-nodata-fraction",
        type=float,
        default=0.5,
        metavar="FRAC",
        help="Skip tiles with more than this fraction of nodata pixels (default: 0.5)",
    )
    qf.add_argument(
        "--min-coverage",
        type=float,
        default=1.0,
        metavar="FRAC",
        help="Skip edge tiles smaller than this fraction of tile_size²; 1.0 keeps only full tiles (default: 1.0)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    metadata = tile_raster(
        args.input_tif,
        args.output_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
        image_format=args.image_format,
        write_metadata=not args.no_metadata,
        write_geojson=not args.no_geojson,
        chm_path=args.chm,
        ndvi_path=args.ndvi,
        ndvi_veg_threshold=args.ndvi_veg_threshold,
        num_workers=args.num_workers,
        blank_threshold=args.blank_threshold,
        variance_threshold=args.variance_threshold,
        sharpness_threshold=args.sharpness_threshold,
        max_nodata_fraction=args.max_nodata_fraction,
        min_coverage=args.min_coverage,
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
