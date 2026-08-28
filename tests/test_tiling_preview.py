"""Tests for the GUI tiling-preview core (grid summary + grid overlay)."""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.gui.tiling_preview import grid_summary, overlay_grid
from src.labeling.tile_orthomosaic import iter_tile_windows


def test_grid_summary_full_grid_has_no_edge_tiles():
    s = grid_summary(1280, 1280, 640, 0)
    assert (s["x_steps"], s["y_steps"]) == (2, 2)
    assert s["total"] == 4
    assert s["full_tiles"] == 4
    assert s["edge_tiles"] == 0


def test_grid_summary_counts_partial_edge_tiles():
    # 800px with 640 tiles: one full tile, three clamped edge tiles.
    s = grid_summary(800, 800, 640, 0)
    assert s["total"] == 4
    assert s["full_tiles"] == 1
    assert s["edge_tiles"] == 3


def test_grid_summary_is_consistent_with_iter_tile_windows():
    width, height, ts, ov = 1000, 700, 256, 64
    windows = list(iter_tile_windows(width, height, ts, ov))
    full = sum(1 for _, _, w in windows if w.width == ts and w.height == ts)

    s = grid_summary(width, height, ts, ov)
    assert s["total"] == len(windows)
    assert s["full_tiles"] == full
    assert s["edge_tiles"] == len(windows) - full


def test_overlay_grid_preserves_shape_and_draws_lines():
    preview = np.zeros((64, 64, 3), dtype=np.uint8)
    out = overlay_grid(preview, 1280, 1280, 640, 0)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8
    assert out.sum() > 0          # grid lines were drawn onto the black preview


def test_overlay_grid_highlight_changes_output():
    preview = np.zeros((64, 64, 3), dtype=np.uint8)
    base = overlay_grid(preview, 1280, 1280, 640, 0)
    highlighted = overlay_grid(preview, 1280, 1280, 640, 0, selected=(0, 0))
    assert not np.array_equal(base, highlighted)


def test_overlay_grid_does_not_mutate_input():
    preview = np.zeros((64, 64, 3), dtype=np.uint8)
    overlay_grid(preview, 1280, 1280, 640, 0)
    assert preview.sum() == 0     # original preview untouched
