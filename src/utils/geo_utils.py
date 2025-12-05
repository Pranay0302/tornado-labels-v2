"""Geospatial helper routines shared across the labeling pipeline."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from rasterio.transform import Affine
from shapely.affinity import affine_transform
from shapely.geometry.base import BaseGeometry


_TILE_PATTERN = re.compile(r"tile_y(?P<row>\d+)_x(?P<col>\d+)", re.IGNORECASE)


def is_blank_tile(
    tile: NDArray[np.generic],
    *,
    threshold: float = 0.95,
    variance_threshold: float = 1.0,
) -> bool:
    """Return ``True`` when *tile* appears visually empty.

    The heuristic checks for extremely low variance and for tiles that are
    dominated by very bright (white) or very dark (black) pixels once the
    image is normalised to the ``0..1`` range.
    """

    array = np.asarray(tile)
    if array.ndim == 3:
        gray = array.mean(axis=2, dtype=np.float32)
    elif array.ndim == 2:
        gray = array.astype(np.float32, copy=False)
    else:
        raise ValueError("Expected a 2-D or 3-D array for tile")

    if gray.size == 0:
        return True

    if np.var(gray) < variance_threshold:
        return True

    minimum = float(np.min(gray))
    maximum = float(np.max(gray))
    if maximum - minimum <= 1e-6:
        return True

    normalised = (gray - minimum) / (maximum - minimum)
    if float((normalised > 0.98).mean()) > threshold:
        return True
    if float((normalised < 0.02).mean()) > threshold:
        return True

    return False


def tile_origin_from_name(
    tile_name: str | Path,
    *,
    tile_size: int,
    overlap: int,
) -> Tuple[int, int]:
    """Return the pixel origin for the given tile filename.

    Parameters
    ----------
    tile_name:
        Name of the tile file (the extension is ignored). Filenames are
        expected to follow the convention ``tile_y<row>_x<col>``.
    tile_size:
        Tile dimension that was used during tiling (in pixels).
    overlap:
        Overlap that was used between adjacent tiles (in pixels).

    Returns
    -------
    tuple[int, int]
        ``(x0, y0)`` pixel coordinates of the upper-left corner of the tile in
        the full image.
    """

    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be in [0, tile_size)")

    stem = Path(tile_name).stem
    match = _TILE_PATTERN.search(stem)
    if not match:
        raise ValueError(
            f"Cannot parse tile indices from '{tile_name}'. Expected pattern 'tile_y####_x####'."
        )

    row = int(match.group("row"))
    col = int(match.group("col"))
    stride = tile_size - overlap
    x0 = col * stride
    y0 = row * stride
    return x0, y0


def affine_to_shapely_tuple(transform: Affine | Sequence[float]) -> Tuple[float, float, float, float, float, float]:
    """Convert a :class:`~rasterio.transform.Affine` to the tuple Shapely expects."""

    if isinstance(transform, Affine):
        return (transform.a, transform.b, transform.d, transform.e, transform.c, transform.f)

    values = tuple(transform)
    if len(values) != 6:
        raise ValueError("Affine transform sequences must contain exactly six values")
    return values  # type: ignore[return-value]


def pixel_to_map_geometry(
    geometry: BaseGeometry,
    transform: Affine | Sequence[float],
) -> BaseGeometry:
    """Project *geometry* from pixel to map coordinates using *transform*."""

    return affine_transform(geometry, affine_to_shapely_tuple(transform))


