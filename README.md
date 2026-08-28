# Tornado Labeling Pipeline

**Author:** Pranay Kumar Andra | **Advisor:** Dr. Melissa Wagner

Tiles a tornado-damage orthomosaic (`.tif`) into uniform image chips and uploads
them to [Roboflow](https://app.roboflow.com) for **instance-segmentation**
annotation and dataset management. A Streamlit GUI lets you inspect the raster and
preview the tiling before you run.

```mermaid
flowchart LR
    A[orthomosaic.tif] --> B[tile_orthomosaic.py<br/>640px full tiles]
    B --> C[roboflow_upload.py<br/>upload to Roboflow]
    C --> D[Roboflow<br/>annotate, version, export]
    A -. optional preview .-> G[GeoTIFF Inspector GUI<br/>inspect + preview tiling]
    G -.-> B
```

One command does both steps:

```bash
python3 scripts/run_pipeline.py data/orthomosaic.tif
```

---

## Installation

```bash
git clone <repository-url>
cd tornado-labels-v2
conda env create -f environment.yml -n tornado-labels
conda activate tornado-labels
```

Or with pip: `pip install -r requirements.txt`.

---

## Setup: Roboflow key

Copy the template and add your private API key (`.env` is gitignored):

```bash
cp .env.example .env
# edit .env  ->  ROBOFLOW_API_KEY=...
```

| Variable | Default | Purpose |
|---|---|---|
| `ROBOFLOW_API_KEY` | — | Your private Roboflow key (required to upload) |
| `ROBOFLOW_WORKSPACE` | `tornado-ml` | Target workspace |

If no key is set, tiling still runs and the upload is skipped with a warning.

---

## Usage

Put your orthomosaic in `data/`, then run:

```bash
# tile + upload (recommended: dry-run first to check tile counts)
python3 scripts/run_pipeline.py data/site-a.tif --rf-dry-run
python3 scripts/run_pipeline.py data/site-a.tif
```

> **Big orthomosaic?** Tiling always saves the **full** set of chips to
> `outputs/tiles/`. `--max-tiles` only limits how many are **uploaded** to
> Roboflow — sampled randomly from across the whole image (not one corner) — so
> you keep the complete local dataset but push just a test sample:
>
> ```bash
> python3 scripts/run_pipeline.py data/site-a.tif --max-tiles 50 --sample-seed 0
> ```

Each run lands in Roboflow as:

- **Project:** one per dataset, named after the input file (e.g. `site-a`),
  created as an instance-segmentation project if it doesn't exist.
- **Batch:** one per run, e.g. `site-a_20260630_142210`.

### Options

| Flag | Description |
|---|---|
| `--tile-size` / `--overlap` | Tile size / overlap in px (default `640` / `160`) |
| `--tile-format` | `png` (default) or `jpg` |
| `--max-tiles N` | Upload only N tiles (sampled randomly); all tiles still saved locally |
| `--sample-seed` | Make `--max-tiles` upload sampling reproducible |
| `--skip-roboflow` | Tile only, don't upload |
| `--rf-dry-run` | Count tiles without uploading |
| `--rf-project` / `--rf-workspace` / `--rf-batch` | Override the Roboflow target |
| `--rf-project-type` | `instance-segmentation` (default) |

Tiles are written to `outputs/tiles/` with a `tiling_metadata.json`, a
georeferenced `tiles_index.geojson`, and a `pipeline_run_summary.json` alongside.

### Upload an existing tiles folder

```bash
python3 src/labeling/roboflow_upload.py outputs/tiles --project site-a
```

---

## Tiling

- **640 px tiles, 160 px overlap** by default — large enough to keep whole
  structures in one chip, which makes instance-segmentation masks clean.
- **Full tiles only** (`--min-coverage 1.0`): partial edge tiles are dropped, so
  every uploaded chip is exactly `tile_size × tile_size`.
- Blank / near-uniform / mostly-nodata tiles are filtered out automatically.
- **Geotagged for re-merging.** Every run writes `tiles_index.geojson` next to the
  tiles: a `FeatureCollection` with one WGS84 footprint per saved tile, carrying
  its pixel `window`, per-tile affine `transform`, and native `bounds`, plus the
  source CRS/transform in the header. That's enough to reassemble the tiles into a
  georeferenced mosaic later, and the footprints load directly as a layer in QGIS.
  (Disable with `--no-geojson` on `tile_orthomosaic.py`.)
- **Optional CHM / NDVI stats.** Pass `--chm PATH` and/or `--ndvi PATH` to
  `tile_orthomosaic.py` to sample each tile's footprint from the height / vegetation
  rasters (matched by geography, so differing resolutions are fine) and add per-tile
  `chm_mean/chm_max/chm_p95/chm_valid_frac` and `ndvi_mean/ndvi_veg_frac` to the
  GeoJSON — handy for filtering tiles (e.g. skip open fields, prioritise structures)
  or routing tree vs structural damage. `--ndvi-veg-threshold` sets the vegetation
  cut (default `0.3`).

  ```bash
  python3 src/labeling/tile_orthomosaic.py site_ortho.tif outputs/tiles \
    --chm site_CHM.tif --ndvi site_NDVI.tif
  ```

---

## GeoTIFF Inspector (GUI)

A Streamlit app to inspect an orthomosaic and preview the tiling before a run.

Set up the environment (once):

```bash
conda env create -f environment.yml -n tornado-labels
```

Then activate it and start the app:

```bash
conda activate tornado-labels
streamlit run src/gui/app.py
```

Pick a `.tif`/`.tiff` in the sidebar: browse files under `data/`/`outputs/`, point
**Folder to scan** at your own data, or paste an absolute path. Big files are read
in place, so select them by path; upload is for small samples only.

Three tabs:

- **Metadata**: size, CRS/EPSG, resolution, bounds, NoData, per-band dtype, and storage details.
- **Bands**: view a single band with a colormap (min/max/mean/std + histogram) or an RGB composite.
- **Tiles**: set tile size/overlap to preview the grid and tile counts, then **Run tiling** to write
  the chips with the same `tile_raster()` the pipeline uses (Roboflow upload stays on the CLI).

Previews use windowed/downsampled reads, so multi-GB GeoTIFFs are never loaded whole
into memory, and the tile counts match what a run produces.

---

## Labeling Schema

Annotate polygons in Roboflow using the 8 classes in `schemas/classes.txt`:

| Class | Description |
|---|---|
| `Structural_Detailed_NoDamage` | No structural damage |
| `Structural_Detailed_MinorDamage` | Minor structural damage |
| `Structural_Detailed_ModerateDamage` | Moderate structural damage |
| `Structural_Detailed_MajorDamage` | Major structural damage |
| `Structural_Detailed_Destroyed` | Structure destroyed |
| `Tree_Quick_NoDamage` | No tree damage |
| `Tree_Quick_MinorDamage` | Minor tree damage |
| `Tree_Quick_MajorDamage` | Major tree damage |

Use Roboflow's smart-polygon tool to speed up annotation, then generate a dataset
version and export to YOLOv8-seg, COCO, SAM, etc.

---

## File Structure

```text
tornado-labels-v2/
├── data/                         # Input orthomosaics (.tif; gitignored)
├── outputs/                      # Tiles + metadata (auto-generated, gitignored)
├── schemas/
│   └── classes.txt               # 8-class instance-segmentation schema
├── scripts/
│   └── run_pipeline.py           # tile + upload entry point
├── src/
│   ├── gui/
│   │   ├── app.py                # Streamlit GeoTIFF inspector (streamlit run)
│   │   ├── raster_inspect.py     # metadata + memory-safe reads + rendering
│   │   └── tiling_preview.py     # tile-grid summary + overlay
│   ├── labeling/
│   │   ├── tile_orthomosaic.py   # tiling (+ tile_grid / iter_tile_windows)
│   │   └── roboflow_upload.py    # Roboflow upload
│   └── utils/
│       ├── image_utils.py        # blank-tile detection
│       └── io_utils.py           # filesystem helpers
├── tests/
├── .env.example                  # Roboflow credentials template
├── environment.yml
└── requirements.txt
```

---

## Testing

```bash
python3 -m pytest tests/ -v
```

Tests cover tiling (big/full-only) and the Roboflow upload (dry-run + mocked SDK);
no live API key or network access is required.
