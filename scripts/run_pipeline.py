#!/usr/bin/env python3
"""Tile an orthomosaic and upload the chips to Roboflow for instance segmentation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.labeling.tile_orthomosaic import tile_raster
from src.utils import ensure_directory


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tile a .tif orthomosaic and upload the chips to Roboflow"
    )
    parser.add_argument("input_file", help="Input orthomosaic (.tif)")
    parser.add_argument("--tiles-dir", default="./outputs/tiles", help="Directory for generated tiles")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile size in pixels")
    parser.add_argument("--overlap", type=int, default=160, help="Tile overlap in pixels")
    parser.add_argument(
        "--tile-format",
        choices=["png", "jpg"],
        default="png",
        help="Image format for tiles (default: png)",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        metavar="N",
        help="Upload at most N tiles to Roboflow (sampled randomly); all tiles are still saved locally",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        metavar="SEED",
        help="Seed for --max-tiles upload sampling (default: non-deterministic)",
    )

    rf = parser.add_argument_group("Roboflow upload")
    rf.add_argument("--skip-roboflow", action="store_true", help="Skip Roboflow upload after tiling")
    rf.add_argument("--rf-project", help="Roboflow project name (default: input file stem)")
    rf.add_argument("--rf-workspace", help="Roboflow workspace (default: env ROBOFLOW_WORKSPACE or tornado-ml)")
    rf.add_argument("--rf-batch", help="Roboflow batch name (default: <dataset>_<timestamp>)")
    rf.add_argument(
        "--rf-project-type",
        default="instance-segmentation",
        choices=["instance-segmentation", "object-detection", "semantic-segmentation"],
        help="Roboflow project type used when creating the project",
    )
    rf.add_argument("--rf-dry-run", action="store_true", help="Count tiles for Roboflow but do not upload")
    return parser


def _print_step(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"STEP: {title}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    input_path = Path(args.input_file).expanduser()
    if not input_path.exists():
        print(f"ERROR: Input file {input_path} does not exist")
        return 1
    if input_path.suffix.lower() != ".tif":
        print(f"ERROR: Only .tif inputs are supported (got {input_path.suffix or 'no extension'})")
        return 1

    ensure_directory(args.tiles_dir)

    print("Tornado Labeling Pipeline")
    print(f"Input file: {input_path}")
    print(f"Tiles directory: {args.tiles_dir}")

    summary: dict[str, object] = {
        "input_file": str(input_path),
        "tiles_dir": str(Path(args.tiles_dir).resolve()),
        "stages": {
            "tile": {"executed": False, "metadata": None},
            "roboflow": {"executed": False, "skipped": False},
        },
        "pipeline_completed": False,
    }

    _print_step("Tile orthomosaic")
    tile_metadata = tile_raster(
        input_path,
        args.tiles_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
        image_format=args.tile_format,
    )
    geojson_index = tile_metadata.get("geojson_index")
    summary["stages"]["tile"] = {
        "executed": True,
        "metadata": tile_metadata,
        "metadata_file": str((Path(args.tiles_dir) / "tiling_metadata.json").resolve()),
        "geojson_index_file": (
            str((Path(args.tiles_dir) / geojson_index).resolve()) if geojson_index else None
        ),
    }

    if not args.skip_roboflow:
        _print_step("Upload tiles to Roboflow")
        dataset = input_path.stem
        try:
            from src.labeling.roboflow_upload import upload_tiles

            rf_summary = upload_tiles(
                args.tiles_dir,
                project=args.rf_project or dataset,
                workspace=args.rf_workspace,
                batch_name=args.rf_batch,
                project_type=args.rf_project_type,
                dataset_name=dataset,
                dry_run=args.rf_dry_run,
                max_tiles=args.max_tiles,
                sample_seed=args.sample_seed,
            )
            summary["stages"]["roboflow"] = rf_summary
            if rf_summary.get("skipped"):
                print(f"Roboflow upload skipped: {rf_summary.get('reason')}")
            else:
                print(
                    f"Roboflow: project={rf_summary['project']} batch={rf_summary['batch']} "
                    f"uploaded={rf_summary['uploaded']} failed={rf_summary['failed']}"
                    + (" (dry-run)" if rf_summary.get("dry_run") else "")
                )
        except ModuleNotFoundError:
            print("WARNING: 'roboflow' not installed; skipping upload (pip install roboflow)")
            summary["stages"]["roboflow"] = {"executed": False, "skipped": True, "reason": "roboflow not installed"}
        except Exception as exc:  # never let upload kill the pipeline
            print(f"WARNING: Roboflow upload failed: {exc}")
            summary["stages"]["roboflow"] = {"executed": False, "skipped": True, "reason": str(exc)}
    else:
        print("Skipping Roboflow upload (per --skip-roboflow)")
        summary["stages"]["roboflow"] = {"executed": False, "skipped": True, "reason": "--skip-roboflow"}

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED")
    print("=" * 60)
    print(f"Tiles: {args.tiles_dir}")
    summary["pipeline_completed"] = True

    try:
        run_root = Path(args.tiles_dir).resolve().parent
        summary_path = run_root / "pipeline_run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Summary written to {summary_path}")
    except OSError as exc:
        print(f"WARNING: Failed to write pipeline summary: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
