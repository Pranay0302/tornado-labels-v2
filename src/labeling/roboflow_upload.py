"""Upload tiled image chips to Roboflow (one project per dataset, batch per run)."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

try:  # optional dependency; only needed when actually uploading
    from dotenv import find_dotenv
    from dotenv import load_dotenv as _load_dotenv

    def load_dotenv() -> bool:
        """Load ``.env`` from the working-directory tree, robustly.

        Uses ``find_dotenv(usecwd=True)`` so it doesn't rely on dotenv's
        call-stack walking, which raises in some contexts (e.g. ``python -c``
        or stdin). Any failure degrades to "no .env loaded".
        """
        try:
            return _load_dotenv(find_dotenv(usecwd=True))
        except Exception:
            return False
except ModuleNotFoundError:  # pragma: no cover - dotenv is a declared dep
    def load_dotenv() -> bool:
        return False

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def slugify(name: str) -> str:
    """Lowercase, hyphenated, Roboflow-safe identifier."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(name).strip().lower())
    return slug.strip("-") or "dataset"


def project_slug(name: str) -> str:
    """Roboflow project id: slugified and guaranteed to start with a letter.

    Roboflow stores but cannot serve (HTTP 404) projects whose slug starts with
    a digit, so a name derived from a date-leading filename (e.g.
    ``2025_06_20_...``) is prefixed with ``tornado-`` to make it retrievable.
    """
    slug = slugify(name)
    if not slug[0].isalpha():
        slug = f"tornado-{slug}"
    return slug


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
    """Return the project, creating an instance-segmentation project if missing.

    Uses the ``Project`` returned by ``create_project`` directly rather than
    re-querying by slug — re-querying immediately after creation is both
    unnecessary and racy.
    """
    try:
        return workspace.project(project_id)
    except Exception:
        return workspace.create_project(
            project_name=project_id,
            project_type=project_type,
            project_license=license,
            annotation=annotation_group,
        )


def _list_tiles(tiles_dir: Path) -> list[Path]:
    return sorted(p for p in tiles_dir.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def _format_eta(seconds: float) -> str:
    """Human ETA: ``M:SS`` (or ``H:MM:SS`` past an hour)."""
    total = int(max(seconds, 0))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class _NullProgress:
    """No-op reporter used when progress is disabled or there is nothing to do."""

    def update(self, failed: int) -> None:  # noqa: D401 - trivial
        pass

    def close(self) -> None:
        pass


class _TqdmProgress:
    """Live ``tqdm`` bar for interactive terminals."""

    def __init__(self, total: int, desc: str) -> None:
        from tqdm import tqdm

        self._bar = tqdm(total=total, desc=desc, unit="tile")
        self._failed = 0

    def update(self, failed: int) -> None:
        if failed != self._failed:
            self._failed = failed
            self._bar.set_postfix(failed=failed, refresh=False)
        self._bar.update(1)

    def close(self) -> None:
        self._bar.close()


class _LogProgress:
    """Throttled plain-text progress for non-TTY output (e.g. SLURM logs).

    Emits at most one line every *interval* seconds during the run, plus a
    guaranteed final line on :meth:`close`. The clock and interval are
    injectable so throttling is deterministically testable.
    """

    def __init__(
        self,
        total: int,
        desc: str,
        *,
        interval: float = 30.0,
        clock=time.monotonic,
        out=None,
    ) -> None:
        self._total = total
        self._desc = desc
        self._interval = interval
        self._clock = clock
        self._out = out if out is not None else sys.stdout
        self._done = 0
        self._failed = 0
        self._start = clock()
        self._last_emit = self._start

    def _emit(self) -> None:
        elapsed = max(self._clock() - self._start, 1e-9)
        rate = self._done / elapsed
        pct = (self._done / self._total * 100) if self._total else 100.0
        if rate > 0 and self._done < self._total:
            eta = _format_eta((self._total - self._done) / rate)
        else:
            eta = "0:00"
        print(
            f"{self._desc}: {pct:3.0f}%  {self._done}/{self._total}  "
            f"{rate:.1f} tile/s  ETA {eta}  failed={self._failed}",
            file=self._out,
            flush=True,
        )
        self._last_emit = self._clock()

    def update(self, failed: int) -> None:
        self._done += 1
        self._failed = failed
        if self._clock() - self._last_emit >= self._interval:
            self._emit()

    def close(self) -> None:
        self._emit()


def _make_progress(total: int, desc: str, *, show: bool):
    """Pick a progress backend: tqdm for a TTY, plain log lines otherwise."""
    if not show or total <= 0:
        return _NullProgress()
    if sys.stderr.isatty():
        try:
            return _TqdmProgress(total, desc)
        except Exception:  # tqdm import/init trouble -> degrade, never block upload
            return _LogProgress(total, desc)
    return _LogProgress(total, desc)


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
    show_progress: bool = True,
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
        "project": project_slug(project),
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
    proj = get_or_create_project(ws, project_slug(project), project_type=project_type)

    progress = _make_progress(
        len(to_upload), f"Uploading {summary['project']}", show=show_progress
    )
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
        progress.update(summary["failed"])
    progress.close()

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
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable the upload progress bar / progress log lines",
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
            show_progress=not args.quiet,
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
