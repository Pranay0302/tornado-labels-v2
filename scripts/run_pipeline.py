#!/usr/bin/env python3
"""Entry point for the tornado labeling pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.labeling.convert_to_tif import convert_to_tif
from src.labeling.merge_annotations import merge_annotations
from src.labeling.tile_orthomosaic import tile_raster
from src.utils import ensure_directories, ensure_directory


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the tornado labeling pipeline end-to-end")
    parser.add_argument("input_file", help="Input orthomosaic (.tif or .tev)")
    parser.add_argument("--workdir", default="./data", help="Working directory for intermediate files")
    parser.add_argument("--tiles-dir", default="./outputs/tiles", help="Directory for generated tiles")
    parser.add_argument("--proposals-dir", default="./outputs/proposals", help="Directory for SAM proposals")
    parser.add_argument("--edited-dir", default="./outputs/edited", help="Directory containing edited GeoJSONs")
    parser.add_argument("--output-gpkg", default="./outputs/labels.gpkg", help="Final GeoPackage output path")
    parser.add_argument("--tile-size", type=int, default=400, help="Tile size in pixels")
    parser.add_argument("--overlap", type=int, default=128, help="Tile overlap in pixels")
    parser.add_argument(
        "--tile-format",
        choices=["png", "jpg"],
        default="png",
        help="Image format for tiles (default: png)",
    )
    parser.add_argument("--skip-proposals", action="store_true", help="Skip SAM proposal generation step")
    parser.add_argument("--sam-model", default="vit_h", choices=["vit_h", "vit_l", "vit_b"], help="SAM model variant")
    parser.add_argument("--sam-timeout", type=int, default=60, help="Timeout per tile in seconds for SAM")
    parser.add_argument("--sam-max-tiles", type=int, help="Limit the number of tiles processed by SAM")
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

    ensure_directory(args.workdir)
    ensure_directories([args.tiles_dir, args.proposals_dir, args.edited_dir, Path(args.output_gpkg).parent])

    print("Tornado Labeling Pipeline")
    print(f"Input file: {input_path}")
    print(f"Working directory: {args.workdir}")
    print(f"Tiles directory: {args.tiles_dir}")
    print(f"Output GeoPackage: {args.output_gpkg}")

    summary: dict[str, object] = {
        "input_file": str(input_path),
        "skip_proposals": bool(args.skip_proposals),
        "workdir": str(Path(args.workdir).resolve()),
        "tiles_dir": str(Path(args.tiles_dir).resolve()),
        "proposals_dir": str(Path(args.proposals_dir).resolve()),
        "edited_dir": str(Path(args.edited_dir).resolve()),
        "output_gpkg": str(Path(args.output_gpkg).resolve()),
        "stages": {
            "convert_to_tif": {"executed": False, "output": None},
            "tile": {"executed": False, "metadata": None},
            "proposals": {"executed": False, "generated_files": 0},
            "merge": {"executed": False, "output": None},
        },
        "pipeline_completed": False,
    }

    tif_path = input_path
    if input_path.suffix.lower() != ".tif":
        _print_step("Convert to GeoTIFF")
        tif_path = convert_to_tif(input_path, args.workdir)
        print(f"GeoTIFF prepared at: {tif_path}")
        summary["stages"]["convert_to_tif"] = {
            "executed": True,
            "output": str(Path(tif_path).resolve()),
        }
    else:
        summary["stages"]["convert_to_tif"] = {
            "executed": False,
            "output": str(Path(tif_path).resolve()),
        }

    _print_step("Tile orthomosaic")
    tile_metadata = tile_raster(
        tif_path,
        args.tiles_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
        image_format=args.tile_format,
    )
    summary["stages"]["tile"] = {
        "executed": True,
        "metadata": tile_metadata,
        "metadata_file": str((Path(args.tiles_dir) / "tiling_metadata.json").resolve()),
    }

    if not args.skip_proposals:
        _print_step("Generate SAM proposals")
        try:
            from src.labeling.samgeo_propose import generate_sam_proposals
        except ModuleNotFoundError as exc:
            missing_pkg = exc.name or "samgeo"
            print(
                f"ERROR: Optional dependency '{missing_pkg}' is required for proposal generation."
                " Install it or rerun with --skip-proposals."
            )
            return 1
        generate_sam_proposals(
            args.tiles_dir,
            args.proposals_dir,
            max_tiles=args.sam_max_tiles,
            timeout_seconds=args.sam_timeout,
            model_type=args.sam_model,
        )
        proposal_files = list(Path(args.proposals_dir).glob("*.geojson"))
        summary["stages"]["proposals"] = {
            "executed": True,
            "generated_files": len(proposal_files),
        }
    else:
        print("Skipping SAM proposal generation (per --skip-proposals)")
        proposal_files = []
        summary["stages"]["proposals"] = {
            "executed": False,
            "generated_files": 0,
        }

    edited_dir = Path(args.edited_dir)
    edited_files = list(edited_dir.glob("*.geojson"))
    if edited_files:
        _print_step("Merge edited annotations")
        merge_annotations(
            tif_path,
            args.tiles_dir,
            args.edited_dir,
            args.output_gpkg,
            tile_size=tile_metadata.get("tile_size"),
            overlap=tile_metadata.get("overlap"),
        )
        summary["stages"]["merge"] = {
            "executed": True,
            "output": str(Path(args.output_gpkg).resolve()),
            "merged_files": sorted(str(path.resolve()) for path in edited_files),
        }
    else:
        print(f"\nNo edited annotations found in {edited_dir}")
        print("To finalise labels:")
        print(f"1. Review tiles in {args.tiles_dir}")
        if not args.skip_proposals:
            print(f"2. Refine proposals located in {args.proposals_dir}")
        print(f"3. Export edited GeoJSON files into {edited_dir}")
        print("4. Re-run this script to merge the final annotations")
        summary["stages"]["merge"] = {
            "executed": False,
            "output": None,
            "merged_files": [],
        }

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED")
    print("=" * 60)
    print(f"Tiles: {args.tiles_dir}")
    if not args.skip_proposals:
        print(f"Proposals: {args.proposals_dir}")
    if edited_files:
        print(f"Merged labels: {args.output_gpkg}")
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
