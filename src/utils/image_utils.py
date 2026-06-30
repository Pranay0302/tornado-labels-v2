"""Image helper routines for the tiling step."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


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
