"""Tests for the GUI raster-inspection core (metadata, stats, reads, render)."""

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.gui import raster_inspect as ri


def _make_tif(path, *, size=256, count=3, res=0.5, nodata=None, descriptions=None):
    """Write a small georeferenced GeoTIFF (EPSG:32614) for inspection tests."""
    data = np.zeros((count, size, size), dtype="uint8")
    for b in range(count):
        data[b] = (b + 1) * 10  # band 1 -> 10, band 2 -> 20, ...
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=count,
        dtype="uint8", crs="EPSG:32614",
        transform=from_origin(500000, 4000000, res, res), nodata=nodata,
    ) as dst:
        dst.write(data)
        if descriptions:
            for i, desc in enumerate(descriptions, start=1):
                dst.set_band_description(i, desc)


# --------------------------------------------------------------------------- #
# raster_metadata
# --------------------------------------------------------------------------- #
def test_metadata_core_dimensions(tmp_path):
    tif = tmp_path / "ortho.tif"
    _make_tif(tif, size=256, count=3)

    md = ri.raster_metadata(tif)

    assert md["width"] == 256
    assert md["height"] == 256
    assert md["band_count"] == 3
    assert md["dtypes"] == ["uint8", "uint8", "uint8"]


def test_metadata_crs_resolution_and_bounds(tmp_path):
    tif = tmp_path / "ortho.tif"
    _make_tif(tif, size=256, res=0.5)

    md = ri.raster_metadata(tif)

    assert md["epsg"] == 32614
    assert md["res_x"] == 0.5 and md["res_y"] == 0.5
    # from_origin(500000, 4000000, 0.5, 0.5) over 256px -> 128 CRS units wide.
    assert md["bounds"] == [500000.0, 3999872.0, 500128.0, 4000000.0]
    assert len(md["transform"]) == 6
    assert md["units"] in ("metre", "meter", "m")


def test_metadata_reprojects_bounds_to_lonlat(tmp_path):
    tif = tmp_path / "ortho.tif"
    _make_tif(tif)

    md = ri.raster_metadata(tif)

    assert md["bounds_lonlat"] is not None
    left, bottom, right, top = md["bounds_lonlat"]
    assert -100.0 < left < -98.0        # UTM 14N easting 500000 ~ -99 lon
    assert 35.0 < bottom < 37.0


def test_metadata_band_descriptions(tmp_path):
    tif = tmp_path / "ortho.tif"
    _make_tif(tif, count=3, descriptions=["Red", "Green", "Blue"])

    md = ri.raster_metadata(tif)

    assert md["band_descriptions"] == ["Red", "Green", "Blue"]
    assert len(md["color_interpretations"]) == 3


def test_metadata_extra_details_present(tmp_path):
    tif = tmp_path / "ortho.tif"
    _make_tif(tif)

    md = ri.raster_metadata(tif)

    assert md["driver"] == "GTiff"
    assert isinstance(md["is_tiled"], bool)
    assert isinstance(md["overview_levels"], list)
    assert md["file_size_bytes"] > 0
    assert "compression" in md
    assert md["nodata"] is None


def test_metadata_nodata_reported(tmp_path):
    tif = tmp_path / "ortho.tif"
    _make_tif(tif, nodata=0.0)

    md = ri.raster_metadata(tif)

    assert md["nodata"] == 0.0


# --------------------------------------------------------------------------- #
# band_statistics
# --------------------------------------------------------------------------- #
def test_band_statistics_on_constant_band(tmp_path):
    tif = tmp_path / "ortho.tif"
    _make_tif(tif, count=3)  # band 2 is uniformly 20

    stats = ri.band_statistics(tif, band=2)

    assert stats["min"] == 20.0
    assert stats["max"] == 20.0
    assert stats["mean"] == 20.0
    assert stats["std"] == 0.0
    assert stats["valid_fraction"] == 1.0


def test_band_statistics_excludes_nodata(tmp_path):
    tif = tmp_path / "ortho.tif"
    # Band 1 is uniformly 10; declare 10 as nodata -> every pixel is masked out.
    _make_tif(tif, count=1, nodata=10.0)

    stats = ri.band_statistics(tif, band=1)

    assert stats["valid_fraction"] == 0.0
    assert stats["mean"] is None


# --------------------------------------------------------------------------- #
# read_overview / reads (memory safety)
# --------------------------------------------------------------------------- #
def test_read_overview_is_bounded_by_max_dim(tmp_path):
    tif = tmp_path / "ortho.tif"
    _make_tif(tif, size=512, count=3)

    arr, decimation = ri.read_overview(tif, max_dim=64)

    assert arr.ndim == 3 and arr.shape[0] == 3     # (count, h, w)
    assert max(arr.shape[1], arr.shape[2]) <= 64
    assert decimation >= 1.0


def test_read_tile_reads_only_the_window(tmp_path):
    tif = tmp_path / "ortho.tif"
    _make_tif(tif, size=256, count=3)

    arr = ri.read_tile(tif, Window(0, 0, 64, 48))

    assert arr.shape == (3, 48, 64)                # (count, h, w)


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def test_percentile_stretch_spans_full_uint8_range(tmp_path):
    gradient = np.linspace(0, 1000, 256, dtype="float32").reshape(16, 16)

    out = ri.percentile_stretch(gradient)

    assert out.dtype == np.uint8
    assert out.min() == 0 and out.max() == 255


def test_render_single_band_returns_rgb(tmp_path):
    band = np.linspace(0, 255, 64 * 64, dtype="float32").reshape(64, 64)

    rgb = ri.render_single_band(band, cmap="viridis")

    assert rgb.shape == (64, 64, 3)
    assert rgb.dtype == np.uint8


def test_render_rgb_composite_returns_bounded_rgb(tmp_path):
    tif = tmp_path / "ortho.tif"
    _make_tif(tif, size=512, count=3)

    rgb = ri.render_rgb_composite(tif, r=1, g=2, b=3, max_dim=64)

    assert rgb.ndim == 3 and rgb.shape[2] == 3
    assert max(rgb.shape[0], rgb.shape[1]) <= 64
    assert rgb.dtype == np.uint8
