import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_pipeline


def _make_tif(path: Path, size: int = 600) -> None:
    data = np.random.randint(0, 255, size=(3, size, size), dtype=np.uint8)
    transform = from_origin(500000, 4000000, 1, 1)
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=3,
        dtype="uint8", crs="EPSG:32614", transform=transform,
    ) as dst:
        dst.write(data)


def _summary(tiles_dir: Path) -> dict:
    return json.loads((tiles_dir.parent / "pipeline_run_summary.json").read_text())


def test_pipeline_skip_roboflow(tmp_path):
    tif = tmp_path / "site-a.tif"
    _make_tif(tif)
    tiles_dir = tmp_path / "run" / "tiles"
    rc = run_pipeline.main([
        str(tif), "--skip-roboflow",
        "--tile-size", "200", "--overlap", "0",
        "--tiles-dir", str(tiles_dir),
    ])
    assert rc == 0
    assert list(tiles_dir.glob("*.png"))
    stage = _summary(tiles_dir)["stages"]["roboflow"]
    assert stage["skipped"] is True
    assert stage["reason"] == "--skip-roboflow"


def test_pipeline_roboflow_dry_run(tmp_path):
    tif = tmp_path / "site-a.tif"
    _make_tif(tif)
    tiles_dir = tmp_path / "run" / "tiles"
    rc = run_pipeline.main([
        str(tif), "--rf-dry-run",
        "--tile-size", "200", "--overlap", "0",
        "--tiles-dir", str(tiles_dir),
    ])
    assert rc == 0
    stage = _summary(tiles_dir)["stages"]["roboflow"]
    assert stage["dry_run"] is True
    assert stage["executed"] is True
    assert stage["uploaded"] == len(list(tiles_dir.glob("*.png")))
    assert stage["batch"].startswith("site-a_")


def test_pipeline_max_tiles_caps_upload_only(tmp_path):
    tif = tmp_path / "site-a.tif"
    _make_tif(tif)  # 600px -> 9 full tiles at 200px / no overlap
    tiles_dir = tmp_path / "run" / "tiles"
    rc = run_pipeline.main([
        str(tif), "--rf-dry-run",
        "--tile-size", "200", "--overlap", "0",
        "--max-tiles", "3", "--sample-seed", "0",
        "--tiles-dir", str(tiles_dir),
    ])
    assert rc == 0
    # all 9 tiles saved locally; only 3 selected for Roboflow
    assert len(list(tiles_dir.glob("*.png"))) == 9
    stage = _summary(tiles_dir)["stages"]["roboflow"]
    assert stage["tiles_found"] == 9
    assert stage["sampled"] is True
    assert stage["uploaded"] == 3


def test_pipeline_summary_points_at_geojson_index(tmp_path):
    tif = tmp_path / "site-a.tif"
    _make_tif(tif)
    tiles_dir = tmp_path / "run" / "tiles"
    rc = run_pipeline.main([
        str(tif), "--skip-roboflow",
        "--tile-size", "200", "--overlap", "0",
        "--tiles-dir", str(tiles_dir),
    ])
    assert rc == 0
    tile_stage = _summary(tiles_dir)["stages"]["tile"]
    geojson_file = Path(tile_stage["geojson_index_file"])
    assert geojson_file == (tiles_dir / "tiles_index.geojson").resolve()
    assert geojson_file.exists()


def test_pipeline_rejects_non_tif(tmp_path):
    bad = tmp_path / "image.png"
    bad.write_bytes(b"x")
    rc = run_pipeline.main([str(bad), "--skip-roboflow"])
    assert rc == 1
