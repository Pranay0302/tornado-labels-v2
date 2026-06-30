import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.labeling.tile_orthomosaic import _build_parser, tile_raster


def _make_tif(path: Path, size: int) -> None:
    data = np.random.randint(0, 255, size=(3, size, size), dtype=np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=3,
        dtype="uint8", crs="EPSG:32614", transform=from_origin(500000, 4000000, 1, 1),
    ) as dst:
        dst.write(data)


def test_defaults_are_big_and_full_only():
    args = _build_parser().parse_args(["in.tif", "out"])
    assert args.tile_size == 640
    assert args.overlap == 160
    assert args.min_coverage == 1.0


def test_drops_partial_edge_tiles(tmp_path):
    tif = tmp_path / "site.tif"
    # 800px with 640px tiles, no overlap -> one full 640 tile per axis; the
    # 160px remainder is a partial edge tile that must be dropped.
    _make_tif(tif, 800)
    out = tmp_path / "tiles"

    meta = tile_raster(tif, out, tile_size=640, overlap=0)

    pngs = sorted(out.glob("*.png"))
    assert meta["saved_tiles"] == len(pngs) == 1
    assert meta["skipped_tiles"] >= 1  # the edge remainder(s) were dropped
    for png in pngs:
        with Image.open(png) as im:
            assert im.size == (640, 640)  # every saved tile is full-size


def test_full_grid_no_drops(tmp_path):
    tif = tmp_path / "site.tif"
    _make_tif(tif, 1280)  # exactly 2x2 full 640 tiles, no overlap
    out = tmp_path / "tiles"

    meta = tile_raster(tif, out, tile_size=640, overlap=0)

    assert meta["saved_tiles"] == 4
    assert meta["skipped_tiles"] == 0
