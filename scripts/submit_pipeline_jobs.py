#!/usr/bin/env python3
"""Submit tornado labeling pipeline jobs to Slurm sequentially."""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SubmissionResult:
    input_file: Path
    output_root: Path
    job_id: str | None
    command: list[str]


def _resolve_paths(pattern: str) -> list[Path]:
    pattern_path = Path(pattern)
    if pattern_path.is_absolute():
        search_pattern = str(pattern_path)
    else:
        search_pattern = str((REPO_ROOT / pattern_path).resolve())
    matches = sorted(Path(match).resolve() for match in glob.glob(search_pattern))
    return matches


def _build_export_env(env: dict[str, str]) -> str:
    return "--export=" + ",".join(f"{key}={value}" for key, value in env.items())


def submit_jobs(
    inputs: Iterable[Path],
    *,
    base_output: Path,
    sbatch: str,
    skip_proposals: bool,
    tile_size: int | None,
    overlap: int | None,
    tile_format: str | None,
    sam_model: str | None,
    sam_timeout: int | None,
    sam_max_tiles: int | None,
    job_name_prefix: str,
    dry_run: bool,
) -> list[SubmissionResult]:
    results: list[SubmissionResult] = []
    previous_job_id: str | None = None

    for index, input_file in enumerate(inputs, start=1):
        dataset_stem = input_file.stem
        output_root = base_output / dataset_stem

        env_vars: dict[str, str] = {
            "INPUT_FILE": str(input_file),
            "OUTPUT_ROOT": str(output_root),
        }
        if skip_proposals:
            env_vars["SKIP_PROPOSALS"] = "1"
        if tile_size is not None:
            env_vars["TILE_SIZE"] = str(tile_size)
        if overlap is not None:
            env_vars["OVERLAP"] = str(overlap)
        if tile_format is not None:
            env_vars["TILE_FORMAT"] = tile_format
        if sam_model is not None:
            env_vars["SAM_MODEL"] = sam_model
        if sam_timeout is not None:
            env_vars["SAM_TIMEOUT"] = str(sam_timeout)
        if sam_max_tiles is not None:
            env_vars["SAM_MAX_TILES"] = str(sam_max_tiles)

        export_arg = _build_export_env(env_vars)

        cmd: list[str] = [sbatch, "--parsable", export_arg, "--job-name", f"{job_name_prefix}-{dataset_stem}"]
        if previous_job_id:
            cmd.extend(["--dependency", f"afterok:{previous_job_id}"])
        cmd.append(str(REPO_ROOT / "scripts" / "run_pipeline.slurm"))

        job_id: str | None = None
        if dry_run:
            print(f"[DRY-RUN] {' '.join(cmd)}")
        else:
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                job_id = result.stdout.strip().split(";", maxsplit=1)[0]
                print(f"Submitted job {job_id} for {input_file.name} (#{index})")
            except subprocess.CalledProcessError as exc:
                print(f"ERROR: Failed to submit job for {input_file}: {exc.stderr}", file=sys.stderr)
                raise

        results.append(SubmissionResult(input_file=input_file, output_root=output_root, job_id=job_id, command=cmd))
        if job_id:
            previous_job_id = job_id

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Submit tornado labeling pipeline jobs sequentially via Slurm")
    parser.add_argument("--input-glob", default="data/*.tif", help="Glob pattern (relative to repo) for input GeoTIFFs")
    parser.add_argument("--base-output", default="outputs/slurm_runs", help="Base directory for pipeline outputs")
    parser.add_argument("--sbatch", default="sbatch", help="Path to sbatch executable")
    parser.add_argument("--skip-proposals", action="store_true", help="Skip SAM proposals across all jobs")
    parser.add_argument("--tile-size", type=int, help="Tile size override for all jobs")
    parser.add_argument("--overlap", type=int, help="Tile overlap override for all jobs")
    parser.add_argument("--tile-format", choices=["png", "jpg"], help="Tile image format override")
    parser.add_argument("--sam-model", choices=["vit_h", "vit_l", "vit_b"], help="SAM model override")
    parser.add_argument("--sam-timeout", type=int, help="Timeout per tile for SAM (seconds)")
    parser.add_argument("--sam-max-tiles", type=int, help="Limit the number of tiles processed by SAM")
    parser.add_argument("--job-name-prefix", default="tornado", help="Prefix for submitted Slurm job names")
    parser.add_argument("--dry-run", action="store_true", help="Print sbatch commands without submitting")
    args = parser.parse_args(argv)

    input_files = _resolve_paths(args.input_glob)
    if not input_files:
        print(f"No input files matched pattern '{args.input_glob}'.", file=sys.stderr)
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_path = Path(args.base_output)
    if not base_output_path.is_absolute():
        base_output_path = (REPO_ROOT / base_output_path).resolve()
    base_output_path = base_output_path / timestamp
    base_output_path.mkdir(parents=True, exist_ok=True)

    print("Submitting the following datasets in sequence:")
    for path in input_files:
        print(f" - {path}")

    try:
        results = submit_jobs(
            input_files,
            base_output=base_output_path,
            sbatch=args.sbatch,
            skip_proposals=args.skip_proposals,
            tile_size=args.tile_size,
            overlap=args.overlap,
            tile_format=args.tile_format,
            sam_model=args.sam_model,
            sam_timeout=args.sam_timeout,
            sam_max_tiles=args.sam_max_tiles,
            job_name_prefix=args.job_name_prefix,
            dry_run=args.dry_run,
        )
    except subprocess.CalledProcessError:
        return 1

    print("\nSubmission summary:")
    for result in results:
        job_repr = result.job_id or "(dry-run)"
        print(f" - {result.input_file.name} -> {job_repr}, output: {result.output_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

