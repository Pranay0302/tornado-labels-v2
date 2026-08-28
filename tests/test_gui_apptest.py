"""End-to-end check that the Streamlit app runs and renders against a real raster.

Uses Streamlit's AppTest to execute the whole script headlessly and drive the
source path widget, so we exercise metadata + overview reads + all three tabs
without a browser.
"""

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

APP_PATH = REPO_ROOT / "src" / "gui" / "app.py"


def _by_key(widgets, key):
    """Address an AppTest widget by its explicit key (robust across re-runs)."""
    return next(w for w in widgets if w.key == key)


def _make_tif(path, *, size=256, count=3):
    data = np.zeros((count, size, size), dtype="uint8")
    for b in range(count):
        data[b] = np.random.randint(0, 255, size=(size, size), dtype="uint8")
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=count,
        dtype="uint8", crs="EPSG:32614", transform=from_origin(500000, 4000000, 0.5, 0.5),
    ) as dst:
        dst.write(data)


def test_app_starts_without_a_file():
    at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not at.exception
    # No source chosen yet -> guidance shown, no crash.
    assert any("Pick a" in info.value for info in at.info)


def test_app_inspects_a_selected_raster(tmp_path):
    tif = tmp_path / "ortho.tif"
    _make_tif(tif, size=256, count=3)

    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not at.exception

    # Drive the "paste an absolute path" text input to our synthetic raster.
    _by_key(at.text_input, "src_pasted_path").set_value(str(tif)).run()
    assert not at.exception

    # Metadata metrics should now be populated.
    labels = [m.label for m in at.metric]
    assert "Width" in labels and "Bands" in labels
    # Tiles tab computed the grid without error.
    assert "Grid total" in labels


def test_run_tiling_button_writes_tiles(tmp_path):
    tif = tmp_path / "ortho.tif"
    _make_tif(tif, size=256, count=3)
    out_dir = tmp_path / "tiles_out"

    at = AppTest.from_file(str(APP_PATH), default_timeout=90).run()
    _by_key(at.text_input, "src_pasted_path").set_value(str(tif)).run()
    # Small tiles so full tiles fit in a 256px raster (overlap first: default 160
    # would exceed a 128px tile size and short-circuit the tab).
    _by_key(at.number_input, "overlap").set_value(32).run()
    _by_key(at.number_input, "tile_size").set_value(128).run()
    _by_key(at.text_input, "tile_out_dir").set_value(str(out_dir)).run()

    _by_key(at.button, "run_tiling").click().run()
    assert not at.exception

    assert any("Saved" in s.value for s in at.success)
    assert list(out_dir.glob("*.png"))       # chips actually written
    assert (out_dir / "tiling_metadata.json").exists()
