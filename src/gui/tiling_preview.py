"""Tile-grid preview helpers for the GUI.

Pure and Streamlit-free: :func:`grid_summary` reports how many tiles the current
settings will produce, and :func:`overlay_grid` draws the tile boundaries onto a
downsampled orthomosaic preview. Both are built on the tiler's own
:func:`~src.labeling.tile_orthomosaic.iter_tile_windows`, so what the user sees
before running matches what the pipeline actually writes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.labeling.tile_orthomosaic import iter_tile_windows, tile_grid

# Colours (RGB) for the drawn overlay.
_GRID_COLOR = (255, 214, 0)      # amber grid lines
_SELECT_COLOR = (0, 200, 255)    # cyan highlight for the selected tile


def grid_summary(width: int, height: int, tile_size: int, overlap: int) -> dict[str, int]:
    """Summarise the tile grid for a raster of ``width`` x ``height`` pixels.

    Returns ``x_steps``/``y_steps`` (grid dimensions), ``total`` (windows walked),
    ``full_tiles`` (exactly ``tile_size`` square - what survives ``min_coverage``
    ``= 1.0``) and ``edge_tiles`` (clamped remainder that is dropped by default).
    Blank / nodata filtering can still reduce the final saved count below
    ``full_tiles``; that is only known after a run.
    """
    x_steps, y_steps = tile_grid(width, height, tile_size, overlap)

    total = 0
    full_tiles = 0
    for _row, _col, window in iter_tile_windows(width, height, tile_size, overlap):
        total += 1
        if window.width == tile_size and window.height == tile_size:
            full_tiles += 1

    return {
        "x_steps": x_steps,
        "y_steps": y_steps,
        "total": total,
        "full_tiles": full_tiles,
        "edge_tiles": total - full_tiles,
    }


def overlay_grid(
    preview: np.ndarray,
    full_width: int,
    full_height: int,
    tile_size: int,
    overlap: int,
    *,
    selected: tuple[int, int] | None = None,
    line_width: int = 1,
) -> np.ndarray:
    """Draw tile boundaries onto a downsampled ``preview`` RGB array.

    ``preview`` is an ``(h, w, 3)`` uint8 array of the whole orthomosaic scaled
    down; boundaries from :func:`iter_tile_windows` are mapped from full-raster
    pixels into preview pixels using the preview's own scale. When ``selected``
    ``(row, col)`` is given, that tile is highlighted. The input array is never
    mutated; a new array is returned.
    """
    if preview.ndim != 3 or preview.shape[2] != 3:
        raise ValueError("preview must be an (h, w, 3) RGB array")

    ph, pw = preview.shape[:2]
    sx = pw / full_width
    sy = ph / full_height

    img = Image.fromarray(preview.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)

    for row, col, window in iter_tile_windows(full_width, full_height, tile_size, overlap):
        left = window.col_off * sx
        top = window.row_off * sy
        right = (window.col_off + window.width) * sx - 1
        bottom = (window.row_off + window.height) * sy - 1
        is_selected = selected is not None and (row, col) == selected
        color = _SELECT_COLOR if is_selected else _GRID_COLOR
        width = line_width + 2 if is_selected else line_width
        draw.rectangle([left, top, right, bottom], outline=color, width=width)

    return np.asarray(img)
