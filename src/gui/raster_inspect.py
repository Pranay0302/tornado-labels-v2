"""Memory-safe raster inspection for the GUI.

Pure and Streamlit-free. Every full-extent read is decimated through rasterio's
``out_shape`` so a multi-GB orthomosaic is never loaded whole; per-tile reads use
a :class:`~rasterio.windows.Window`. Functions take a path (or anything
``rasterio.open`` accepts) and return plain Python / NumPy values, so they are
trivially unit-tested and reusable from any front end.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
from matplotlib import colormaps
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import Window


def raster_metadata(path: str | Path) -> dict:
    """Extract inspection metadata from a GeoTIFF without reading pixel data.

    Covers dimensions, CRS/EPSG, resolution, native + lon/lat bounds, affine
    transform, nodata, per-band dtype/description/colour-interpretation, plus the
    storage details that matter for windowed reads (driver, compression, internal
    tiling + block size, overview levels, on-disk size, linear units).
    """
    path = Path(path)
    with rasterio.open(path) as src:
        crs = src.crs
        transform = src.transform
        bounds = src.bounds

        epsg = None
        bounds_lonlat = None
        units = None
        if crs is not None:
            try:
                epsg = crs.to_epsg()
            except Exception:
                epsg = None
            try:
                units = crs.linear_units
            except Exception:
                units = None
            try:
                left, bottom, right, top = transform_bounds(
                    crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top
                )
                bounds_lonlat = [left, bottom, right, top]
            except Exception:
                bounds_lonlat = None

        try:
            block_rows, block_cols = src.block_shapes[0]
        except Exception:
            block_rows = block_cols = None
        # Internally tiled TIFFs use square-ish blocks; strip TIFFs use blocks
        # that span the full image width. (Avoids the deprecated src.is_tiled.)
        is_tiled = block_cols is not None and block_cols != src.width

        try:
            overview_levels = list(src.overviews(1))
        except Exception:
            overview_levels = []

        try:
            compression = src.compression.name if src.compression is not None else None
        except Exception:
            compression = None

        return {
            "path": str(path),
            "width": src.width,
            "height": src.height,
            "band_count": src.count,
            "dtypes": [str(dt) for dt in src.dtypes],
            "band_descriptions": list(src.descriptions),
            "color_interpretations": [ci.name for ci in src.colorinterp],
            "crs": crs.to_string() if crs is not None else None,
            "epsg": epsg,
            "res_x": float(src.res[0]),
            "res_y": float(src.res[1]),
            "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
            "bounds_lonlat": bounds_lonlat,
            "transform": [transform.a, transform.b, transform.c,
                          transform.d, transform.e, transform.f],
            "nodata": src.nodata,
            "driver": src.driver,
            "compression": compression,
            "is_tiled": is_tiled,
            "block_width": block_cols,
            "block_height": block_rows,
            "overview_levels": overview_levels,
            "file_size_bytes": path.stat().st_size if path.exists() else None,
            "units": units,
        }


def _decimated_shape(width: int, height: int, max_dim: int) -> tuple[int, int, float]:
    """Return ``(out_width, out_height, decimation)`` capping the longest side to
    ``max_dim``; never upsamples (decimation >= 1)."""
    scale = min(1.0, max_dim / max(width, height))
    ow = max(1, int(width * scale))
    oh = max(1, int(height * scale))
    return ow, oh, width / ow


def band_statistics(path: str | Path, band: int, *, max_pixels: int = 4_000_000) -> dict:
    """Basic statistics for one band from a decimated read (nodata excluded).

    The band is read at a resolution capped to ``max_pixels`` samples, so stats
    stay cheap on huge rasters. Returns min/max/mean/std (``None`` when no valid
    pixels remain) and the fraction of sampled pixels that were valid.
    """
    with rasterio.open(path) as src:
        width, height, nodata = src.width, src.height, src.nodata
        total_px = width * height
        if total_px > max_pixels:
            scale = math.sqrt(max_pixels / total_px)
            ow, oh = max(1, int(width * scale)), max(1, int(height * scale))
        else:
            ow, oh = width, height
        data = src.read(band, out_shape=(oh, ow)).astype("float64")

    flat = data.ravel()
    sampled = flat.size
    if nodata is not None:
        flat = flat[flat != nodata]
    flat = flat[np.isfinite(flat)]
    valid_fraction = round(flat.size / sampled, 4) if sampled else 0.0

    if flat.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None,
                "valid_fraction": valid_fraction}
    return {
        "min": round(float(flat.min()), 4),
        "max": round(float(flat.max()), 4),
        "mean": round(float(flat.mean()), 4),
        "std": round(float(flat.std()), 4),
        "valid_fraction": valid_fraction,
    }


def read_overview(path: str | Path, max_dim: int = 1024) -> tuple[np.ndarray, float]:
    """Decimated full-extent read of every band, longest side <= ``max_dim``.

    Returns ``(array, decimation)`` where ``array`` is ``(count, h, w)`` and
    ``decimation`` is how many source pixels map to one preview pixel (>= 1).
    """
    with rasterio.open(path) as src:
        ow, oh, decimation = _decimated_shape(src.width, src.height, max_dim)
        arr = src.read(out_shape=(src.count, oh, ow), resampling=Resampling.average)
    return arr, decimation


def read_tile(path: str | Path, window: Window) -> np.ndarray:
    """Native-resolution read of a single tile window - ``(count, h, w)``."""
    with rasterio.open(path) as src:
        return src.read(window=window)


def percentile_stretch(arr: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Contrast-stretch a 2-D array to ``uint8`` using the ``[lo, hi]`` percentiles.

    Non-finite pixels are ignored when computing the range; a flat array returns
    all zeros.
    """
    a = np.asarray(arr, dtype="float64")
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros(a.shape, dtype=np.uint8)

    vmin, vmax = np.percentile(finite, [lo, hi])
    if vmax <= vmin:
        vmin, vmax = float(finite.min()), float(finite.max())
    if vmax <= vmin:
        return np.zeros(a.shape, dtype=np.uint8)

    scaled = np.clip((a - vmin) / (vmax - vmin), 0.0, 1.0)
    return (scaled * 255).round().astype(np.uint8)


def render_single_band(arr: np.ndarray, cmap: str = "gray") -> np.ndarray:
    """Render a 2-D band as an ``(h, w, 3)`` uint8 RGB image via a colormap."""
    stretched = percentile_stretch(arr).astype("float32") / 255.0
    rgba = colormaps[cmap](stretched)
    return (rgba[..., :3] * 255).round().astype(np.uint8)


def render_rgb_composite(
    path: str | Path, r: int, g: int, b: int, max_dim: int = 1024
) -> np.ndarray:
    """Decimated RGB composite from three band indices, per-band stretched.

    Returns an ``(h, w, 3)`` uint8 image with the longest side <= ``max_dim``.
    """
    with rasterio.open(path) as src:
        ow, oh, _ = _decimated_shape(src.width, src.height, max_dim)
        bands = src.read((r, g, b), out_shape=(3, oh, ow),
                         resampling=Resampling.average).astype("float32")

    out = np.zeros((bands.shape[1], bands.shape[2], 3), dtype=np.uint8)
    for i in range(3):
        out[..., i] = percentile_stretch(bands[i])
    return out
