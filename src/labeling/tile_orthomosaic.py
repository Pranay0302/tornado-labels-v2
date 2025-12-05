"""Orthomosaic tiling for the tornado labeling pipeline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Literal

import rasterio
from PIL import Image
from rasterio.windows import Window
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import ensure_directory, is_blank_tile


def tile_raster(
    input_tif: str | Path,
    output_dir: str | Path,
    *,
    tile_size: int = 400,
    overlap: int = 128,
    image_format: Literal["png", "jpg"] = "png",
    write_metadata: bool = True,
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

    skipped_count = 0
    saved_count = 0

    with rasterio.open(input_tif) as src:
        width, height = src.width, src.height
        step = tile_size - overlap
        x_steps = math.ceil((width - overlap) / step)
        y_steps = math.ceil((height - overlap) / step)

        print(f"Creating tiles: {x_steps}x{y_steps} = {x_steps * y_steps} total tiles")

        for row in tqdm(range(y_steps), desc="Rows"):
            for col in range(x_steps):
                x0 = col * step
                y0 = row * step
                window_width = min(tile_size, width - x0)
                window_height = min(tile_size, height - y0)
                if window_width <= 0 or window_height <= 0:
                    continue

                window = Window(x0, y0, window_width, window_height)
                image = src.read(window=window)
                tile_array = image.transpose(1, 2, 0)
                if tile_array.shape[2] >= 3:
                    tile_array = tile_array[:, :, :3]
                elif tile_array.shape[2] == 1:
                    tile_array = tile_array.repeat(3, axis=2)

                if is_blank_tile(tile_array):
                    skipped_count += 1
                    continue

                tile_name = f"tile_y{row:04d}_x{col:04d}{ext}"
                Image.fromarray(tile_array).save(output_path / tile_name)
                saved_count += 1

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
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
