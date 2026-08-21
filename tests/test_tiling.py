import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rasterio.windows import transform as window_transform

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


def _make_ungeoref_tif(path: Path, size: int) -> None:
    """A GeoTIFF with a pixel geotransform but no CRS (nothing to reproject)."""
    data = np.random.randint(0, 255, size=(3, size, size), dtype=np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=3, dtype="uint8",
        transform=from_origin(0, size, 1, 1),
    ) as dst:
        dst.write(data)


def _make_band_tif(path: Path, array: np.ndarray, *, nodata: float, size: int = 1280) -> None:
    """Single-band float32 raster on the same grid as ``_make_tif`` (EPSG:32614)."""
    h, w = array.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32",
        crs="EPSG:32614", transform=from_origin(500000, 4000000, 1, 1), nodata=nodata,
    ) as dst:
        dst.write(array.astype("float32"), 1)


def _read_geojson(tiles_dir: Path) -> dict:
    return json.loads((tiles_dir / "tiles_index.geojson").read_text(encoding="utf-8"))


def _by_rowcol(gj: dict) -> dict:
    return {(f["properties"]["row"], f["properties"]["col"]): f["properties"] for f in gj["features"]}


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


def test_geojson_index_has_one_feature_per_saved_tile(tmp_path):
    tif = tmp_path / "site.tif"
    _make_tif(tif, 1280)
    out = tmp_path / "tiles"

    meta = tile_raster(tif, out, tile_size=640, overlap=0)

    assert (out / "tiles_index.geojson").exists()
    assert meta["geojson_index"] == "tiles_index.geojson"

    gj = _read_geojson(out)
    assert gj["type"] == "FeatureCollection"
    # one feature per tile actually written to disk
    names = {f["properties"]["name"] for f in gj["features"]}
    assert names == {p.name for p in out.glob("*.png")}
    assert len(gj["features"]) == meta["saved_tiles"] == 4


def test_geojson_feature_schema_and_header(tmp_path):
    tif = tmp_path / "site.tif"
    _make_tif(tif, 1280)
    out = tmp_path / "tiles"

    tile_raster(tif, out, tile_size=640, overlap=0)
    gj = _read_geojson(out)

    md = gj["metadata"]
    assert md["source_crs"] == "EPSG:32614"
    assert md["geometry_crs"] == "EPSG:4326"
    assert md["tile_size"] == 640 and md["overlap"] == 0
    assert md["width"] == 1280 and md["height"] == 1280
    assert len(md["source_transform"]) == 6

    feat = gj["features"][0]
    assert feat["type"] == "Feature"
    props = feat["properties"]
    assert set(props) >= {"name", "row", "col", "window", "transform", "bounds"}
    assert len(props["window"]) == 4          # col_off, row_off, width, height
    assert len(props["transform"]) == 6
    assert len(props["bounds"]) == 4          # left, bottom, right, top

    geom = feat["geometry"]
    assert geom["type"] == "Polygon"
    ring = geom["coordinates"][0]
    assert len(ring) == 5                      # closed ring
    assert ring[0] == ring[-1]
    # geometry is reprojected to WGS84 lon/lat (UTM 14N easting 500000 -> ~ -99 lon)
    for lon, lat in ring:
        assert -100.0 < lon < -98.0
        assert 35.0 < lat < 37.0


def test_geojson_metadata_is_sufficient_to_reconstruct_tiles(tmp_path):
    """Round-trip: source_transform + window must reproduce each tile's native
    georeferencing, and the saved windows must tile the raster with no gaps."""
    tif = tmp_path / "site.tif"
    _make_tif(tif, 1280)
    out = tmp_path / "tiles"

    tile_raster(tif, out, tile_size=640, overlap=0)
    gj = _read_geojson(out)

    a, b, c, d, e, f = gj["metadata"]["source_transform"]
    src_transform = rasterio.Affine(a, b, c, d, e, f)

    offsets = set()
    for feat in gj["features"]:
        col_off, row_off, width, height = feat["properties"]["window"]
        offsets.add((col_off, row_off))
        assert (width, height) == (640, 640)

        win = Window(col_off, row_off, width, height)
        # native bounds reconstructed purely from header transform + window
        expected_bounds = list(window_bounds(win, src_transform))
        assert feat["properties"]["bounds"] == expected_bounds
        # per-tile transform reconstructed the same way
        expected_tf = list(window_transform(win, src_transform))[:6]
        assert feat["properties"]["transform"] == expected_tf

    # the four windows partition the 1280x1280 raster with no gap/overlap
    assert offsets == {(0, 0), (640, 0), (0, 640), (640, 640)}


def test_geojson_can_be_disabled(tmp_path):
    tif = tmp_path / "site.tif"
    _make_tif(tif, 1280)
    out = tmp_path / "tiles"

    meta = tile_raster(tif, out, tile_size=640, overlap=0, write_geojson=False)

    assert not (out / "tiles_index.geojson").exists()
    assert meta.get("geojson_index") is None

    # the flag is exposed on the CLI and defaults to writing the index
    args = _build_parser().parse_args(["in.tif", "out"])
    assert args.no_geojson is False


def test_geojson_for_ungeoreferenced_source(tmp_path):
    tif = tmp_path / "plain.tif"
    _make_ungeoref_tif(tif, 1280)
    out = tmp_path / "tiles"

    tile_raster(tif, out, tile_size=640, overlap=0)
    gj = _read_geojson(out)

    md = gj["metadata"]
    assert md["source_crs"] is None
    assert md["geometry_crs"] is None          # nothing to reproject into
    assert len(gj["features"]) == 4
    # still a usable pixel-space index
    feat = gj["features"][0]
    assert feat["geometry"]["type"] == "Polygon"
    assert len(feat["properties"]["window"]) == 4


def test_no_band_paths_leaves_geojson_unchanged(tmp_path):
    tif = tmp_path / "site.tif"
    _make_tif(tif, 1280)
    out = tmp_path / "tiles"

    tile_raster(tif, out, tile_size=640, overlap=0)
    gj = _read_geojson(out)

    assert "chm_source" not in gj["metadata"]
    assert "ndvi_source" not in gj["metadata"]
    for feat in gj["features"]:
        assert not any(k.startswith(("chm_", "ndvi_")) for k in feat["properties"])


def test_chm_stats_added_per_tile(tmp_path):
    tif = tmp_path / "site.tif"
    _make_tif(tif, 1280)
    chm = tmp_path / "site_CHM.tif"
    _make_band_tif(chm, np.full((1280, 1280), 3.0), nodata=-9999.0)
    out = tmp_path / "tiles"

    tile_raster(tif, out, tile_size=640, overlap=0, chm_path=chm)
    gj = _read_geojson(out)

    assert gj["metadata"]["chm_source"] == str(chm.resolve())
    for props in _by_rowcol(gj).values():
        assert props["chm_mean"] == 3.0
        assert props["chm_max"] == 3.0
        assert props["chm_p95"] == 3.0
        assert props["chm_valid_frac"] == 1.0


def test_chm_nodata_is_excluded(tmp_path):
    tif = tmp_path / "site.tif"
    _make_tif(tif, 1280)
    # Whole CHM is nodata except the top-half rows of the top-left tile.
    arr = np.full((1280, 1280), -9999.0)
    arr[0:320, 0:640] = 8.0
    chm = tmp_path / "site_CHM.tif"
    _make_band_tif(chm, arr, nodata=-9999.0)
    out = tmp_path / "tiles"

    tile_raster(tif, out, tile_size=640, overlap=0, chm_path=chm)
    props = _by_rowcol(_read_geojson(out))

    top_left = props[(0, 0)]
    assert top_left["chm_mean"] == 8.0                 # nodata pixels excluded
    assert top_left["chm_valid_frac"] == 0.5           # only half the tile is valid

    all_nodata = props[(1, 1)]
    assert all_nodata["chm_mean"] is None
    assert all_nodata["chm_valid_frac"] == 0.0


def test_ndvi_stats_and_threshold(tmp_path):
    tif = tmp_path / "site.tif"
    _make_tif(tif, 1280)
    ndvi = tmp_path / "site_NDVI.tif"
    _make_band_tif(ndvi, np.full((1280, 1280), 0.6), nodata=-32767.0)
    out = tmp_path / "tiles"

    # Default threshold 0.3 -> all pixels count as vegetation.
    tile_raster(tif, out, tile_size=640, overlap=0, ndvi_path=ndvi)
    gj = _read_geojson(out)
    assert gj["metadata"]["ndvi_source"] == str(ndvi.resolve())
    assert gj["metadata"]["ndvi_veg_threshold"] == 0.3
    for props in _by_rowcol(gj).values():
        assert props["ndvi_mean"] == 0.6
        assert props["ndvi_veg_frac"] == 1.0

    # Raise threshold above the constant value -> nothing counts as vegetation.
    out2 = tmp_path / "tiles2"
    tile_raster(tif, out2, tile_size=640, overlap=0, ndvi_path=ndvi, ndvi_veg_threshold=0.7)
    gj2 = _read_geojson(out2)
    assert gj2["metadata"]["ndvi_veg_threshold"] == 0.7
    for props in _by_rowcol(gj2).values():
        assert props["ndvi_veg_frac"] == 0.0


def test_tile_outside_chm_extent_gets_null_stats(tmp_path):
    tif = tmp_path / "site.tif"
    _make_tif(tif, 1280)
    # CHM only covers the top-left 640x640 (the (0,0) tile); other tiles miss it.
    chm = tmp_path / "small_CHM.tif"
    _make_band_tif(chm, np.full((640, 640), 2.0), nodata=-9999.0)
    out = tmp_path / "tiles"

    tile_raster(tif, out, tile_size=640, overlap=0, chm_path=chm)
    props = _by_rowcol(_read_geojson(out))

    assert props[(0, 0)]["chm_mean"] == 2.0
    assert props[(1, 1)]["chm_mean"] is None
    assert props[(1, 1)]["chm_valid_frac"] is None


def test_band_flags_exposed_on_cli():
    args = _build_parser().parse_args(["in.tif", "out"])
    assert args.chm is None
    assert args.ndvi is None
    assert args.ndvi_veg_threshold == 0.3
