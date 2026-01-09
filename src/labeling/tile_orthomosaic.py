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
) -> bool:
    """Read a single window from the raster, check if it's blank, and save if not."""
    with rasterio.open(src_path) as src:
        image = src.read(window=window)

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

    if is_blank_tile(tile_array):
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
) -> dict[str, int | str]:
    """Tile *input_tif* and write image tiles into *output_dir*.

    Returns a metadata dictionary containing basic statistics that can be
    persisted alongside the tiles.
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
                process_tile, input_tif, window, tile_name, output_path, is_floating, nodata
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
    }

    if write_metadata:
        metadata_path = output_path / "tiling_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[OK] Tiles written to: {output_path}")
    print(f"[STATS] Saved: {saved_count}, Skipped (blank): {skipped_count}")
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
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
