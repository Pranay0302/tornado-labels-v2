# Tornado Labeling Pipeline

Author: Pranay Kumar Andra  
Advisor: Dr. Melissa Wagner

End-to-end labeling pipeline for tornado damage assessment from orthomosaics:

- Accepts `.tev` or `.tif` orthomosaics
- Tiles large images for fast viewing and processing
- Uses **SAMGeo** to automatically propose damage polygons
- Human-in-the-loop editing via **X-AnyLabeling** or any polygon tool
- Exports **GeoPackage** (georeferenced) with 5 tornado-damage classes

## Features

- **Automatic Tiling**: Breaks large orthomosaics into manageable tiles
- **AI-Powered Proposals**: Uses Segment Anything Model (SAM) to propose damage areas
- **Human-in-the-Loop**: Edit and refine AI proposals with any labeling tool
- **Georeferenced Output**: Final labels maintain spatial reference for GIS analysis
- **5-Class System**: Categorizes damage from "No/Very Minor" to "Destroyed"

## Damage Classes

| Class | Name | Color | Description |
|-------|------|-------|-------------|
| 0 | NoOrVeryMinor | Green | No visible damage or very minor damage |
| 1 | Minor | Light Green | Minor damage, some visible effects |
| 2 | Moderate | Yellow | Moderate damage, clear visible effects |
| 3 | Severe | Orange | Severe damage, significant structural impact |
| 4 | Destroyed | Red | Complete destruction or severe damage |

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd tornado-labels

# Create conda environment
conda env create -f environment.yml -n tornado-labels
conda activate tornado-labels
```

## Quick Start

> **Data layout tip:** Store all source orthomosaics (`.tif`/`.tev`) inside the
> top-level `data/` directory. The pipeline will populate `outputs/` with
> derived artefacts (`tiles/`, `proposals/`, `edited/`, final labels) while
> leaving the raw imagery untouched.

### Recommended: Run via Slurm

1. Copy your orthomosaics into `data/`.
2. Review available runtime options:
   ```bash
   cd /home/<USER>/dev/tornado-labels && python3 scripts/run_pipeline.py --help
   ```
3. Submit the batch job:
   ```bash
   cd /home/<USER>/dev/tornado-labels && sbatch /home/<USER>/dev/tornado-labels/scripts/run_pipeline.slurm
   ```
4. Monitor the job and review the verification summary from `slurm-<jobid>.out` when it completes.

Outputs are written to `outputs/<dataset>/<timestamp>_job<ID>/` with a `pipeline_run_summary.json` describing the results.

### Running Under Slurm

Use the provided Slurm script for individual runs or the submission helper for batch processing.

**Inspect available flags**

```bash
cd /home/<USER>/dev/tornado-labels && python3 scripts/run_pipeline.py --help
```

**Submit a single job**

```bash
cd /home/<USER>/dev/tornado-labels && sbatch /home/<USER>/dev/tornado-labels/scripts/run_pipeline.slurm
```

**Monitor progress**

```bash
squeue -j <jobid>
tail -f slurm-<jobid>.out
```

After the job leaves the queue, read the verification summary printed at the end of the Slurm log and review the generated artefacts under `outputs/<dataset>/<timestamp>_job<ID>/`.

The project also includes Slurm helpers for running multiple orthomosaics on an HPC cluster:

- `scripts/run_pipeline.slurm` executes the complete workflow for one
  orthomosaic, writes outputs into
  `outputs/<dataset>/<timestamp>_job<ID>/`, and stores a
  `pipeline_run_summary.json` describing stage outcomes.
- `scripts/submit_pipeline_jobs.py` locates GeoTIFFs (default: `data/*.tif`) and
  submits one job per file with `afterok` dependencies so runs occur
  sequentially. Configuration flags are forwarded as environment variables to
  the batch script.

### Optional: Manual Workflow (Local Debugging)

```bash
# Step 1: Convert to TIF (if needed)
python3 src/labeling/convert_to_tif.py data/orthomosaic.tev data/

# Step 2: Create tiles
python3 src/labeling/tile_orthomosaic.py data/orthomosaic.tif outputs/tiles 2048 128

# Step 3: Generate SAM proposals (optional)
python3 src/labeling/samgeo_propose.py outputs/tiles outputs/proposals

# Step 4: Edit proposals with labeling tool
# Save edited annotations as GeoJSON files in outputs/edited/

# Step 5: Merge final annotations
python3 src/labeling/merge_annotations.py data/orthomosaic.tif outputs/tiles outputs/edited outputs/labels.gpkg
```

Each Slurm run finishes by calling `scripts/verify_pipeline_outputs.py`, which
checks for expected artefacts (tiles, optional proposals, merged labels) and
reports discrepancies via the job log. A verification failure returns a non-zero
exit code so that Slurm marks the job as failed.

## File Structure

```
tornado-labels/
|-- data/                         # Input orthomosaics used for runs
|   `-- any_orthomosaic.tif
|-- outputs*/                      # Pipeline artefacts (tiles, proposals, edits, logs)
|   |-- <dataset>/
|   |   `-- <timestamp>_job<ID>/
|   |       |-- edited/
|   |       |-- proposals/
|   |       |-- tiles/
|   |       |-- work/
|   |       |-- pipeline_run_summary.json
|   |       `-- requirements.lock
|   `-- slurm_runs/
|       `-- <timestamp>/
|-- progress_updates*/             # Periodic notes or reports from me
|-- schemas/
|   `-- labels_schema.json        # Label definitions for labeling tools
|-- scripts/                      
|   |-- run_pipeline.py
|   |-- run_pipeline.slurm
|   |-- submit_pipeline_jobs.py
|   `-- verify_pipeline_outputs.py
|-- src/
|   |-- labeling/
|   |   |-- convert_to_tif.py
|   |   |-- merge_annotations.py
|   |   |-- samgeo_propose.py
|   |   `-- tile_orthomosaic.py
|   `-- utils/
|       |-- geo_utils.py
|       `-- io_utils.py
|-- environment.yml
|-- requirements.txt
|-- README.md
`-- slurm-<jobID>.out*
```

> `*` means it's a hidden (adding this for better clarity)

## Labeling Tools

### X-AnyLabeling (Recommended)
```bash
pip install x-anylabeling

# Launch with tiles and labels file
# Note: x-anylabeling takes directory as positional argument
# The labels file should contain one label per line
x-anylabeling outputs/tiles --labels schemas/labels.txt
```

### QGIS
1. Open QGIS
2. Add tiles as raster layers
3. Create new vector layer for annotations
4. Use polygon tools to draw damage areas
5. Export as GeoJSON files


## Output Format

The final output is a GeoPackage (`.gpkg`) file containing:

- **Layer**: `tornado_damage_labels`
- **Geometry**: Polygons in the original coordinate system
- **Attributes**: 
  - `label`: Damage class (0-4)
  - `confidence`: Optional confidence score
  - `tile`: Source tile name
  - `proposal_id`: Original proposal ID (if from SAM)


## Advanced Usage

### Custom Tile Parameters
```bash
# High overlap for better edge handling
python3 scripts/run_pipeline.py data/orthomosaic.tif --tile-size 2048 --overlap 256

# Small tiles for detailed work
python3 scripts/run_pipeline.py data/orthomosaic.tif --tile-size 512 --overlap 32
```

### Batch Processing
```bash
# Process multiple orthomosaics
for file in data/*.tif; do
    python3 scripts/run_pipeline.py "$file" --output-gpkg "outputs/$(basename "$file" .tif)_labels.gpkg"
done
```
