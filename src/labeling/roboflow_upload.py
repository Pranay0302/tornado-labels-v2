"""Upload tiled image chips to Roboflow (one project per dataset, batch per run)."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path

try:  # optional dependency; only needed when actually uploading
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - dotenv is a declared dep
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def slugify(name: str) -> str:
    """Lowercase, hyphenated, Roboflow-safe identifier."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(name).strip().lower())
    return slug.strip("-") or "dataset"


def make_batch_name(dataset: str, timestamp: datetime | None = None) -> str:
    """``<dataset>_<YYYYmmdd_HHMMSS>``, plus ``_job<ID>`` when SLURM_JOB_ID is set."""
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    name = f"{slugify(dataset)}_{stamp}"
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        name = f"{name}_job{job_id}"
    return name


def resolve_api_key(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    load_dotenv()
    return os.environ.get("ROBOFLOW_API_KEY") or None


def resolve_workspace(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    load_dotenv()
    return os.environ.get("ROBOFLOW_WORKSPACE") or "tornado-ml"


def _get_workspace(api_key: str, workspace_name: str | None = None):
    """Build a Roboflow workspace handle. Raises ModuleNotFoundError if SDK absent."""
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    return rf.workspace(workspace_name) if workspace_name else rf.workspace()


def get_or_create_project(
    workspace,
    project_id: str,
    *,
    project_type: str = "instance-segmentation",
    license: str = "Private",
    annotation_group: str = "tornado-damage",
):
    """Return the project, creating an instance-segmentation project if missing."""
    try:
        return workspace.project(project_id)
    except Exception:
        workspace.create_project(
            project_name=project_id,
            project_type=project_type,
            project_license=license,
            annotation=annotation_group,
        )
        return workspace.project(project_id)


def _list_tiles(tiles_dir: Path) -> list[Path]:
    return sorted(p for p in tiles_dir.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def upload_tiles(
    tiles_dir: str | Path,
    *,
    project: str,
    workspace: str | None = None,
    api_key: str | None = None,
    batch_name: str | None = None,
    project_type: str = "instance-segmentation",
    dataset_name: str | None = None,
    dry_run: bool = False,
    split: str = "train",
    num_retry: int = 3,
    max_tiles: int | None = None,
    sample_seed: int | None = None,
) -> dict:
    """Upload image chips from *tiles_dir* into a named Roboflow batch.

    All tiles remain on disk; when *max_tiles* is set, only that many tiles are
    uploaded — sampled in random order across the whole folder (seeded by
    *sample_seed*) so the uploaded sample is spatially representative.

    Returns a summary dict. Never raises for a missing key / no tiles — those are
    reported as ``skipped``. Per-image upload errors are collected in ``failures``.
    """
    tiles_path = Path(tiles_dir)
    images = _list_tiles(tiles_path)
    dataset = dataset_name or project
    batch = batch_name or make_batch_name(dataset)
    ws_name = resolve_workspace(workspace)

    # Every tile stays on disk; only a sample is uploaded when max_tiles is set.
    to_upload = images
    sampled = False
    if max_tiles is not None and len(images) > max_tiles:
        to_upload = sorted(random.Random(sample_seed).sample(images, max_tiles))
        sampled = True

    summary: dict = {
        "executed": False,
        "dry_run": dry_run,
        "skipped": False,
        "workspace": ws_name,
        "project": slugify(project),
        "batch": batch,
        "tiles_found": len(images),
        "selected": len(to_upload),
        "sampled": sampled,
        "max_tiles": max_tiles,
        "uploaded": 0,
        "failed": 0,
        "failures": [],
    }

    if not images:
        summary["skipped"] = True
        summary["reason"] = f"no image tiles found in {tiles_path}"
        return summary

    if dry_run:
        summary["executed"] = True
        summary["uploaded"] = len(to_upload)  # would-be uploads
        return summary

    key = resolve_api_key(api_key)
    if not key:
        summary["skipped"] = True
        summary["reason"] = "ROBOFLOW_API_KEY not set"
        return summary

    ws = _get_workspace(key, ws_name)
    proj = get_or_create_project(ws, slugify(project), project_type=project_type)

    for image in to_upload:
        try:
            proj.upload(
                image_path=str(image),
                batch_name=batch,
                split=split,
                num_retry_uploads=num_retry,
            )
            summary["uploaded"] += 1
        except Exception as exc:  # collect, keep going
            summary["failed"] += 1
            summary["failures"].append({"file": image.name, "error": str(exc)})

    summary["executed"] = True
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload tiles to Roboflow")
    parser.add_argument("tiles_dir", help="Directory of image chips to upload")
    parser.add_argument("--project", help="Roboflow project (default: tiles dir parent name)")
    parser.add_argument("--workspace", help="Roboflow workspace (default: env or tornado-ml)")
    parser.add_argument("--batch", help="Batch name (default: <dataset>_<timestamp>)")
    parser.add_argument(
        "--project-type",
        default="instance-segmentation",
        choices=["instance-segmentation", "object-detection", "semantic-segmentation"],
    )
    parser.add_argument("--dry-run", action="store_true", help="Count tiles, no upload")
    parser.add_argument(
        "--max-tiles",
        type=int,
        metavar="N",
        help="Upload at most N tiles, sampled randomly across the folder (all tiles stay on disk)",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        metavar="SEED",
        help="Seed for --max-tiles sampling (default: non-deterministic)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    tiles_dir = Path(args.tiles_dir)
    project = args.project or tiles_dir.resolve().parent.name
    try:
        summary = upload_tiles(
            tiles_dir,
            project=project,
            workspace=args.workspace,
            batch_name=args.batch,
            project_type=args.project_type,
            dataset_name=project,
            dry_run=args.dry_run,
            max_tiles=args.max_tiles,
            sample_seed=args.sample_seed,
        )
    except ModuleNotFoundError:
        print("ERROR: the 'roboflow' package is required. Install it: pip install roboflow")
        return 1
    print(json.dumps(summary, indent=2))
    if summary.get("skipped"):
        print(f"[SKIP] {summary.get('reason')}")
    return 0 if summary.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
