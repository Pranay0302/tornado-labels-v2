"""Tests for the extracted grid/window helpers shared by the tiler and the GUI."""

import math
import sys
from pathlib import Path

import pytest
from rasterio.windows import Window

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.labeling.tile_orthomosaic import iter_tile_windows, tile_grid


def _expected_grid(width, height, tile_size, overlap):
    step = tile_size - overlap
    return (
        math.ceil((width - overlap) / step),
        math.ceil((height - overlap) / step),
    )


@pytest.mark.parametrize(
    "width,height,tile_size,overlap",
    [
        (1280, 1280, 640, 0),
        (800, 800, 640, 0),
        (1280, 1280, 640, 160),
        (1000, 500, 256, 64),
    ],
)
def test_tile_grid_matches_formula(width, height, tile_size, overlap):
    assert tile_grid(width, height, tile_size, overlap) == _expected_grid(
        width, height, tile_size, overlap
    )


def test_iter_tile_windows_matches_original_loop():
    """iter_tile_windows must yield exactly the windows tile_raster used to build
    inline: the same row/col ordering, offsets, and clamped edge sizes."""
    width = height = 800
    tile_size, overlap = 640, 0
    got = list(iter_tile_windows(width, height, tile_size, overlap))

    expected = [
        (0, 0, Window(0, 0, 640, 640)),
        (0, 1, Window(640, 0, 160, 640)),   # right edge clamped
        (1, 0, Window(0, 640, 640, 160)),   # bottom edge clamped
        (1, 1, Window(640, 640, 160, 160)), # corner
    ]
    assert len(got) == len(expected)
    for (r, c, win), (er, ec, ewin) in zip(got, expected):
        assert (r, c) == (er, ec)
        assert (win.col_off, win.row_off, win.width, win.height) == (
            ewin.col_off, ewin.row_off, ewin.width, ewin.height,
        )


def test_iter_tile_windows_count_equals_grid_when_no_zero_area():
    x_steps, y_steps = tile_grid(1280, 1280, 640, 160)
    windows = list(iter_tile_windows(1280, 1280, 640, 160))
    assert len(windows) == x_steps * y_steps


def test_tile_grid_rejects_bad_params():
    with pytest.raises(ValueError):
        tile_grid(100, 100, 0, 0)          # tile_size must be positive
    with pytest.raises(ValueError):
        tile_grid(100, 100, 64, 64)        # overlap >= tile_size
    with pytest.raises(ValueError):
        tile_grid(100, 100, 64, -1)        # overlap < 0
