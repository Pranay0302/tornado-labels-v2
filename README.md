# Tornado Labeling Pipeline

**Author:** Pranay Kumar Andra | **Advisor:** Dr. Melissa Wagner

End-to-end pipeline for labeling tornado damage from orthomosaics. Accepts `.tev`/`.tif` inputs, tiles them, proposes damage polygons via SAM, and exports georeferenced GeoPackage labels after human review.

---

## Installation

```bash
git clone <repository-url>
cd tornado-labels
conda env create -f environment.yml -n tornado-labels
conda activate tornado-labels
```

---

## Quick Start

Place orthomosaics in `data/`, then submit via Slurm:

```bash
sbatch scripts/run_pipeline.slurm
```

Or check available options first:

```bash
python3 scripts/run_pipeline.py --help
```

Outputs go to `outputs/<dataset>/<timestamp>_job<ID>/` with a `pipeline_run_summary.json`.

---

## Manual Workflow (Local / Debugging)

```bash
# 1. Convert to TIF (if needed)
python3 src/labeling/convert_to_tif.py data/orthomosaic.tev data/

# 2. Tile the orthomosaic
python3 src/labeling/tile_orthomosaic.py data/orthomosaic.tif outputs/tiles 2048 128

# 3. Generate SAM proposals (optional)
python3 src/labeling/samgeo_propose.py outputs/tiles outputs/proposals

# 4. Edit proposals with a labeling tool (X-AnyLabeling, QGIS, etc.)
#    Save annotations as GeoJSON in outputs/edited/

# 5. Merge annotations into final GeoPackage
python3 src/labeling/merge_annotations.py data/orthomosaic.tif outputs/tiles outputs/edited outputs/labels.gpkg
```

---

## Labeling Schema

Currently uses `schemas/classes.txt` (8 classes):

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

A more detailed schema (`schemas/labels.txt`, 60+ classes) covering structures, trees, vegetation, vehicles, debris, and land cover is planned for a future version.

---

## Labeling Tool: X-AnyLabeling

Connect with X11 forwarding, then:

```bash
# Recommended — auto-detects latest tiles directory
./run_anylabeling.sh

# Or specify paths explicitly
./run_anylabeling.sh outputs/<dataset>/<timestamp>_job<ID>/tiles schemas/classes.txt
```

SSH with X11:
```bash
ssh -YC <SSH-LOGIN-IDENTIFIER>
```

---

## Output Format

Final output is a GeoPackage (`.gpkg`) with layer `tornado_damage_labels`:

- `label` — damage class
- `confidence` — optional score
- `tile` — source tile name
- `proposal_id` — SAM proposal ID (if applicable)

---

## File Structure

```
tornado-labels/
├── data/                         # Input orthomosaics
├── outputs*/                     # Pipeline artefacts (auto-generated)
│   └── <dataset>/<timestamp>_jobID>/
│       ├── tiles/
│       ├── proposals/
│       ├── edited/
│       └── pipeline_run_summary.json
├── schemas/
│   ├── classes.txt               # Current label schema (8 classes)
│   ├── labels.txt                # Extended schema (60+ classes, future)
│   └── labels_schema.json
├── scripts/
│   ├── run_pipeline.py
│   ├── run_pipeline.slurm
│   ├── submit_pipeline_jobs.py
│   └── verify_pipeline_outputs.py
├── src/
│   ├── labeling/
│   │   ├── convert_to_tif.py
│   │   ├── merge_annotations.py
│   │   ├── samgeo_propose.py
│   │   └── tile_orthomosaic.py
│   └── utils/
│       ├── geo_utils.py
│       └── io_utils.py
├── run_anylabeling.sh
├── setup_groundingdino.py
├── environment.yml
└── requirements.txt
```

> `*` = gitignored

---

## Recent Changes

### Mar 2026 — Tile quality filtering
- `tile_orthomosaic.py`: tiles now go through a multi-stage quality filter before being saved:
  1. **Min coverage** — skips edge tiles smaller than 25% of the target area
  2. **Nodata fraction** — skips tiles with >50% masked/nodata pixels
  3. **Blank detection** — skips near-uniform, all-white, all-black, or (optionally) blurry tiles
- `geo_utils.py`: `is_blank_tile()` helper added with configurable variance, range, and Laplacian sharpness thresholds; tiling now uses `ThreadPoolExecutor` for parallel tile processing

