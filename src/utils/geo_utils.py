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
    blank_threshold: float = 0.95,
    variance_threshold: float = 1e-4,
    sharpness_threshold: float = 0.0,
) -> bool:
    """Return ``True`` when *tile* appears visually empty.

    Checks (in order):

    1. Empty array.
    2. Normalised variance below *variance_threshold* → near-uniform tile.
    3. Pixel range too small → perfectly uniform tile.
    4. More than *blank_threshold* fraction of pixels near white or black.
    5. Laplacian variance below *sharpness_threshold* → blurry/soft tile
       (disabled when *sharpness_threshold* is ``0``).

    Parameters
    ----------
    blank_threshold:
        Fraction of pixels that must be near-white (>0.98) or near-black
        (<0.02) in the locally contrast-stretched image for the tile to be
        considered blank.  Default ``0.95``.
    variance_threshold:
        Minimum acceptable variance of the luminance channel after
        normalising pixel values to ``[0, 1]``.  Default ``1e-4`` (~1 % std).
    sharpness_threshold:
        Minimum acceptable sum of second-order-difference variances (a
        Laplacian proxy).  Set to ``0`` (default) to skip the sharpness
        check entirely.
    """

    array = np.asarray(tile)
    if array.ndim == 3:
        if array.shape[2] >= 3:
            # Rec.601 luminance weights give perceptually accurate greyscale.
            gray = (
                0.299 * array[:, :, 0].astype(np.float32)
                + 0.587 * array[:, :, 1].astype(np.float32)
                + 0.114 * array[:, :, 2].astype(np.float32)
            )
        else:
            gray = array[:, :, 0].astype(np.float32)
    elif array.ndim == 2:
        gray = array.astype(np.float32, copy=False)
    else:
        raise ValueError("Expected a 2-D or 3-D array for tile")

    if gray.size == 0:
        return True

    # Normalise to [0, 1] using the dtype's full scale so that all thresholds
    # are scale-independent regardless of bit depth.
    if np.issubdtype(array.dtype, np.integer):
        dtype_max = float(np.iinfo(array.dtype).max)
    else:
        dtype_max = float(np.max(gray)) or 1.0
    gray_norm = gray / dtype_max

    if float(np.var(gray_norm)) < variance_threshold:
        return True

    minimum = float(np.min(gray_norm))
    maximum = float(np.max(gray_norm))
    if maximum - minimum <= 1e-6:
        return True

    normalised = (gray_norm - minimum) / (maximum - minimum)
    if float((normalised > 0.98).mean()) > blank_threshold:
        return True
    if float((normalised < 0.02).mean()) > blank_threshold:
        return True

    if sharpness_threshold > 0.0:
        # Approximate Laplacian via second-order finite differences — pure
        # NumPy, no extra dependencies.  Low variance → blurry / featureless.
        lap_var = float(np.var(np.diff(gray_norm, 2, axis=0))) + float(
            np.var(np.diff(gray_norm, 2, axis=1))
        )
        if lap_var < sharpness_threshold:
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


