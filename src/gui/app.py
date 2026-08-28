"""Streamlit GUI for inspecting GeoTIFF orthomosaics and previewing the tiling.

Thin presentation layer: all raster/tiling logic lives in ``raster_inspect`` and
``tiling_preview`` (and the pipeline's own ``tile_orthomosaic``). Run with::

    streamlit run src/gui/app.py

The UI is inside :func:`main`, so importing this module has no side effects.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.gui import raster_inspect as ri
from src.gui.tiling_preview import grid_summary, overlay_grid
from src.labeling.tile_orthomosaic import iter_tile_windows, tile_raster

DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
COLORMAPS = ["gray", "viridis", "magma", "plasma", "inferno", "cividis", "terrain"]


# --------------------------------------------------------------------------- #
# Source discovery + small pure helpers (unit-testable, no Streamlit needed)
# --------------------------------------------------------------------------- #
def discover_tif_files(*roots: Path) -> list[Path]:
    """Return sorted unique ``.tif``/``.tiff`` files found under ``roots``."""
    found: set[Path] = set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
            found.update(root.rglob(pattern))
    return sorted(found)


def label_for(path: Path) -> str:
    """Human label for a discovered raster: repo-relative when possible, else absolute.

    Files living outside the repo (e.g. a 50 GB orthomosaic on an external drive)
    keep their full absolute path so they remain selectable.
    """
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def to_display_rgb(overview: np.ndarray, bands: tuple[int, int, int] | None = None) -> np.ndarray:
    """Turn a decimated ``(count, h, w)`` read into an ``(h, w, 3)`` uint8 preview.

    Multi-band rasters use the given ``bands`` (default 1,2,3) as R/G/B; a single
    band is shown greyscale. Each channel is percentile-stretched for contrast.
    """
    count = overview.shape[0]
    if count >= 3:
        r, g, b = bands or (1, 2, 3)
        channels = [overview[r - 1], overview[g - 1], overview[b - 1]]
    else:
        channels = [overview[0]] * 3
    return np.stack([ri.percentile_stretch(c) for c in channels], axis=-1)


# --------------------------------------------------------------------------- #
# Cached wrappers — keyed on (path, mtime) so edits bust the cache but repeated
# widget interactions on the same file are instant.
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _cached_metadata(path: str, mtime: float) -> dict:
    return ri.raster_metadata(path)


@st.cache_data(show_spinner=False)
def _cached_overview(path: str, mtime: float, max_dim: int):
    return ri.read_overview(path, max_dim=max_dim)


@st.cache_data(show_spinner=False)
def _cached_band_stats(path: str, mtime: float, band: int) -> dict:
    return ri.band_statistics(path, band)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


# --------------------------------------------------------------------------- #
# Sidebar: choose a source raster + tiling / preview settings
# --------------------------------------------------------------------------- #
def _select_source() -> Path | None:
    st.sidebar.header("Source")
    st.sidebar.caption(
        "Large orthomosaics (tens of GB) are read in place with windowed / "
        "downsampled reads — pick from the list or paste a path. Upload is for "
        "small samples only."
    )

    scan_dir = st.sidebar.text_input(
        "Folder to scan for .tif (optional)", "", key="src_scan_dir",
        help="Point this at wherever your large orthomosaics live; matches appear below.",
    ).strip()
    roots = [DATA_DIR, OUTPUTS_DIR]
    if scan_dir:
        roots.append(Path(scan_dir).expanduser())

    label_to_path = {label_for(p): p for p in discover_tif_files(*roots)}
    labels = ["— none —"] + list(label_to_path)
    picked = st.sidebar.selectbox("Discovered rasters", labels, index=0)

    pasted = st.sidebar.text_input(
        "…or paste an absolute path to a .tif", "", key="src_pasted_path"
    ).strip()

    with st.sidebar.expander("Upload a small sample (not for large files)"):
        st.caption(
            "Browser upload buffers the whole file in memory and is size-capped — "
            "use the path/list above for big orthomosaics."
        )
        uploaded = st.file_uploader("Upload .tif/.tiff", type=["tif", "tiff"], key="src_upload")

    if uploaded is not None:
        tmp_dir = Path(tempfile.gettempdir()) / "geotiff_inspector_uploads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / uploaded.name
        tmp_path.write_bytes(uploaded.getbuffer())
        return tmp_path
    if pasted:
        return Path(pasted).expanduser()
    if picked != "— none —":
        return label_to_path[picked]
    return None


# --------------------------------------------------------------------------- #
# Tab renderers
# --------------------------------------------------------------------------- #
def _fmt(value, nd: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.{nd}f}"
    return str(value)


def _render_metadata_tab(md: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Width", f"{md['width']:,}px")
    c2.metric("Height", f"{md['height']:,}px")
    c3.metric("Bands", md["band_count"])
    size_mb = (md["file_size_bytes"] or 0) / 1e6
    c4.metric("File size", f"{size_mb:,.1f} MB")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CRS", md["crs"] or "—")
    c2.metric("EPSG", md["epsg"] or "—")
    c3.metric("Resolution", f"{_fmt(md['res_x'])} × {_fmt(md['res_y'])} {md['units'] or ''}".strip())
    c4.metric("NoData", _fmt(md["nodata"]))

    st.subheader("Extent")
    left, bottom, right, top = md["bounds"]
    st.write(
        {
            "bounds (native)": f"[{_fmt(left)}, {_fmt(bottom)}, {_fmt(right)}, {_fmt(top)}]",
            "bounds (lon/lat)": (
                "[" + ", ".join(_fmt(v, 6) for v in md["bounds_lonlat"]) + "]"
                if md["bounds_lonlat"] else "—"
            ),
            "affine transform": "[" + ", ".join(_fmt(v, 6) for v in md["transform"]) + "]",
        }
    )

    st.subheader("Bands")
    st.dataframe(
        {
            "band": list(range(1, md["band_count"] + 1)),
            "dtype": md["dtypes"],
            "color interp": md["color_interpretations"],
            "description": [d or "—" for d in md["band_descriptions"]],
        },
        width="stretch",
        hide_index=True,
    )

    st.subheader("Storage")
    st.write(
        {
            "driver": md["driver"],
            "compression": md["compression"] or "none",
            "internally tiled": md["is_tiled"],
            "block size": f"{md['block_width']} × {md['block_height']}",
            "overview levels": md["overview_levels"] or "none",
        }
    )


def _render_bands_tab(path: Path, md: dict, overview: np.ndarray, max_dim: int) -> None:
    band_count = md["band_count"]
    mode = st.radio("View", ["Single band", "RGB composite"], horizontal=True)

    if mode == "Single band":
        col_a, col_b = st.columns([1, 3])
        with col_a:
            band = st.selectbox("Band", list(range(1, band_count + 1)))
            cmap = st.selectbox("Colormap", COLORMAPS)
        with col_b:
            image = ri.render_single_band(overview[band - 1], cmap=cmap)
            st.image(image, caption=f"Band {band} ({cmap}) — downsampled preview",
                     width="stretch")

        stats = _cached_band_stats(str(path), _mtime(path), band)
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("min", _fmt(stats["min"]))
        s2.metric("max", _fmt(stats["max"]))
        s3.metric("mean", _fmt(stats["mean"]))
        s4.metric("std", _fmt(stats["std"]))
        s5.metric("valid", f"{stats['valid_fraction'] * 100:.1f}%")

        values = overview[band - 1].ravel()
        values = values[np.isfinite(values)]
        if md["nodata"] is not None:
            values = values[values != md["nodata"]]
        if values.size:
            counts, edges = np.histogram(values, bins=64)
            st.bar_chart({"count": counts}, x_label="value bin", y_label="pixels")
    else:
        if band_count < 3:
            st.info("RGB composite needs at least 3 bands.")
            return
        c1, c2, c3 = st.columns(3)
        r = c1.selectbox("R band", list(range(1, band_count + 1)), index=0)
        g = c2.selectbox("G band", list(range(1, band_count + 1)), index=1)
        b = c3.selectbox("B band", list(range(1, band_count + 1)), index=2)
        image = to_display_rgb(overview, bands=(r, g, b))
        st.image(image, caption=f"RGB = bands {r}/{g}/{b} — downsampled preview",
                 width="stretch")


def _render_tiles_tab(
    path: Path, md: dict, overview: np.ndarray, tile_size: int, overlap: int
) -> None:
    width, height = md["width"], md["height"]

    if overlap >= tile_size or tile_size <= 0:
        st.error("Overlap must be in [0, tile size) and tile size must be positive.")
        return

    summary = grid_summary(width, height, tile_size, overlap)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tiles along X", summary["x_steps"])
    c2.metric("Tiles along Y", summary["y_steps"])
    c3.metric("Grid total", summary["total"])
    c4.metric("Full tiles", summary["full_tiles"])
    c5.metric("Edge dropped", summary["edge_tiles"])
    st.caption(
        f"Grid is {summary['x_steps']} × {summary['y_steps']} = {summary['total']} cells. "
        f"With min_coverage=1.0 the {summary['edge_tiles']} partial edge tiles are dropped, "
        f"leaving {summary['full_tiles']} full tiles; blank/nodata filtering may reduce the "
        f"final saved count further."
    )

    preview_rgb = to_display_rgb(overview)

    st.subheader("Grid overlay")
    show_single = st.checkbox("Preview a single tile", value=False)
    selected = None
    if show_single:
        sc1, sc2 = st.columns(2)
        sel_row = sc1.number_input("row", 0, max(summary["y_steps"] - 1, 0), 0)
        sel_col = sc2.number_input("col", 0, max(summary["x_steps"] - 1, 0), 0)
        selected = (int(sel_row), int(sel_col))

    overlay = overlay_grid(preview_rgb, width, height, tile_size, overlap, selected=selected)
    st.image(overlay, caption="Orthomosaic with tile grid (downsampled)",
             width="stretch")

    if show_single and selected is not None:
        window = _window_for(width, height, tile_size, overlap, selected)
        if window is None:
            st.warning("That tile position is empty (off the raster).")
        else:
            tile = ri.read_tile(path, window)
            st.image(
                to_display_rgb(tile),
                caption=f"Tile (row {selected[0]}, col {selected[1]}) — "
                        f"{int(window.width)}×{int(window.height)} px, native resolution",
            )

    st.subheader("Run tiling")
    default_out = OUTPUTS_DIR / f"{path.stem}_tiles"
    out_dir = st.text_input("Output directory", str(default_out), key="tile_out_dir")
    fmt = st.selectbox("Tile format", ["png", "jpg"], key="tile_format")
    if st.button("Run tiling", type="primary", key="run_tiling"):
        with st.spinner("Tiling… (writing chips + metadata + GeoJSON index)"):
            meta = tile_raster(
                path, out_dir, tile_size=tile_size, overlap=overlap, image_format=fmt
            )
        st.success(
            f"Saved {meta['saved_tiles']} tiles "
            f"(skipped {meta['skipped_tiles']}) → {out_dir}"
        )
        st.json(meta)


def _window_for(width, height, tile_size, overlap, selected):
    """Return the Window for a chosen (row, col), or None if it was skipped."""
    for row, col, window in iter_tile_windows(width, height, tile_size, overlap):
        if (row, col) == selected:
            return window
    return None


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="GeoTIFF Inspector", page_icon="🛰️", layout="wide")
    st.title("🛰️ GeoTIFF Orthomosaic Inspector")
    st.caption("Inspect metadata, bands, and the tiling grid — with memory-safe windowed reads.")

    path = _select_source()

    st.sidebar.header("Tiling")
    tile_size = int(st.sidebar.number_input("Tile size (px)", min_value=1, value=640, step=32,
                                            key="tile_size"))
    overlap = int(st.sidebar.number_input("Overlap (px)", min_value=0, value=160, step=16,
                                          key="overlap"))

    st.sidebar.header("Preview")
    max_dim = int(st.sidebar.slider("Max preview dimension (px)", 256, 4096, 1024, step=256))

    if path is None:
        st.info("Pick a `.tif`/`.tiff` from the sidebar (browse, paste a path, or upload) to begin.")
        return
    if not path.exists():
        st.error(f"File not found: {path}")
        return

    try:
        md = _cached_metadata(str(path), _mtime(path))
    except Exception as exc:  # unreadable / not a raster
        st.error(f"Could not open raster: {exc}")
        return

    with st.spinner("Reading downsampled overview…"):
        overview, _decimation = _cached_overview(str(path), _mtime(path), max_dim)

    st.success(f"Loaded **{path.name}** — {md['width']:,} × {md['height']:,} px, "
               f"{md['band_count']} band(s)")

    tab_meta, tab_bands, tab_tiles = st.tabs(["Metadata", "Bands", "Tiles"])
    with tab_meta:
        _render_metadata_tab(md)
    with tab_bands:
        _render_bands_tab(path, md, overview, max_dim)
    with tab_tiles:
        _render_tiles_tab(path, md, overview, tile_size, overlap)


if __name__ == "__main__":
    main()
