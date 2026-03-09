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
from rasterio.windows import Window
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
    tile_size: int = 400,
    blank_threshold: float = 0.95,
    variance_threshold: float = 1e-4,
    sharpness_threshold: float = 0.0,
    max_nodata_fraction: float = 0.5,
    min_coverage: float = 0.25,
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


def tile_raster(
    input_tif: str | Path,
    output_dir: str | Path,
    *,
    tile_size: int = 400,
    overlap: int = 128,
    image_format: Literal["png", "jpg"] = "png",
    write_metadata: bool = True,
    num_workers: int = 8,
    # Quality-filter parameters forwarded to process_tile / is_blank_tile.
    blank_threshold: float = 0.95,
    variance_threshold: float = 1e-4,
    sharpness_threshold: float = 0.0,
    max_nodata_fraction: float = 0.5,
    min_coverage: float = 0.25,
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
        skipped (catches tiny edge slivers).
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
            tasks.append((window, tile_name))

    saved_count = 0
    skipped_count = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_tile = {
            executor.submit(
                process_tile,
                input_tif,
                window,
                tile_name,
                output_path,
                is_floating,
                nodata,
                tile_size=tile_size,
                blank_threshold=blank_threshold,
                variance_threshold=variance_threshold,
                sharpness_threshold=sharpness_threshold,
                max_nodata_fraction=max_nodata_fraction,
                min_coverage=min_coverage,
            ): tile_name
            for window, tile_name in tasks
        }

        for future in tqdm(as_completed(future_to_tile), total=len(tasks), desc="Tiling"):
            if future.result():
                saved_count += 1
            else:
                skipped_count += 1

    metadata = {
        "input": str(Path(input_tif).resolve()),
        "tile_size": tile_size,
        "overlap": overlap,
        "image_format": image_format.lower(),
        "saved_tiles": saved_count,
        "skipped_tiles": skipped_count,
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
    parser.add_argument("--tile-size", type=int, default=400, help="Tile size in pixels (default: 400)")
    parser.add_argument("--overlap", type=int, default=128, help="Overlap between tiles in pixels (default: 128)")
    parser.add_argument(
        "--image-format",
        choices=["png", "jpg"],
        default="png",
        help="Image format for saved tiles (default: png)",
    )
    parser.add_argument("--no-metadata", action="store_true", help="Do not write tiling metadata JSON")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel workers (default: 8)")

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
        default=0.25,
        metavar="FRAC",
        help="Skip edge tiles whose area is less than this fraction of tile_size² (default: 0.25)",
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
