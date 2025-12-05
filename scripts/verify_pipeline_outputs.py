#!/usr/bin/env python3
"""Validate artefacts produced by the tornado labeling pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


def _collect_files(directory: Path, patterns: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(directory.glob(pattern))
    return sorted({path.resolve() for path in files})


def _load_summary(summary_path: Path) -> dict[str, object] | None:
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[WARN] Could not parse summary JSON at {summary_path}: {exc}")
        return None


def _add(env: list[str], prefix: str, message: str) -> None:
    env.append(f"[{prefix}] {message}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify outputs from the tornado labeling pipeline")
    parser.add_argument("--run-root", required=True, help="Root directory for a single pipeline run")
    parser.add_argument("--tiles-dir", help="Directory containing generated tiles")
    parser.add_argument("--proposals-dir", help="Directory containing SAM proposals")
    parser.add_argument("--edited-dir", help="Directory expected to contain edited GeoJSONs")
    parser.add_argument("--expected-gpkg", help="Expected GeoPackage output path")
    parser.add_argument("--skip-proposals", action="store_true", help="Set if the pipeline skipped proposal generation")
    args = parser.parse_args(argv)

    run_root = Path(args.run_root).resolve()
    tiles_dir = Path(args.tiles_dir).resolve() if args.tiles_dir else run_root / "tiles"
    proposals_dir = Path(args.proposals_dir).resolve() if args.proposals_dir else run_root / "proposals"
    edited_dir = Path(args.edited_dir).resolve() if args.edited_dir else run_root / "edited"
    expected_gpkg = Path(args.expected_gpkg).resolve() if args.expected_gpkg else run_root / "labels.gpkg"

    summary = _load_summary(run_root / "pipeline_run_summary.json")
    successes: list[str] = []
    warnings: list[str] = []
    failures: list[str] = []

    if not run_root.exists():
        _add(failures, "FAIL", f"Run root does not exist: {run_root}")
    else:
        _add(successes, "OK", f"Run root located at {run_root}")

    # Tiles validation
    if tiles_dir.exists():
        metadata_path = tiles_dir / "tiling_metadata.json"
        tile_images = _collect_files(tiles_dir, ("*.png", "*.jpg"))
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                saved_tiles = metadata.get("saved_tiles", 0)
                _add(successes, "OK", f"Tiling metadata present (saved_tiles={saved_tiles})")
            except json.JSONDecodeError as exc:
                _add(failures, "FAIL", f"Invalid tiling metadata JSON: {exc}")
        else:
            _add(failures, "FAIL", f"Missing tiling metadata at {metadata_path}")

        if tile_images:
            _add(successes, "OK", f"Found {len(tile_images)} tile image(s)")
        else:
            _add(failures, "FAIL", f"No tile images detected in {tiles_dir}")
    else:
        _add(failures, "FAIL", f"Tiles directory missing: {tiles_dir}")

    # Proposals validation
    proposals_expected = not args.skip_proposals
    summary_proposals = None
    if summary:
        summary_proposals = summary.get("stages", {}).get("proposals") if isinstance(summary.get("stages"), dict) else None

    if proposals_expected:
        if proposals_dir.exists():
            proposal_files = _collect_files(proposals_dir, ("*.geojson",))
            if proposal_files:
                _add(successes, "OK", f"Found {len(proposal_files)} proposal GeoJSON file(s)")
            else:
                _add(failures, "FAIL", f"No proposal GeoJSON files found in {proposals_dir}")
        else:
            _add(failures, "FAIL", f"Proposals directory missing: {proposals_dir}")

        if summary_proposals and summary_proposals.get("executed") is False:
            _add(failures, "FAIL", "Summary indicates proposals stage was skipped unexpectedly")
    else:
        _add(warnings, "SKIP", "Proposals step marked as skipped by configuration")

    # Edited annotations directory
    if edited_dir.exists():
        edited_files = _collect_files(edited_dir, ("*.geojson",))
        _add(successes, "OK", f"Edited annotations directory present ({len(edited_files)} file(s))")
    else:
        _add(warnings, "WARN", f"Edited annotations directory not found: {edited_dir}")

    # Output GeoPackage validation
    summary_merge = None
    if summary:
        summary_merge = summary.get("stages", {}).get("merge") if isinstance(summary.get("stages"), dict) else None

    if expected_gpkg.exists():
        _add(successes, "OK", f"GeoPackage exists at {expected_gpkg}")
    else:
        if summary_merge and summary_merge.get("executed"):
            _add(failures, "FAIL", f"Merge stage reported success but GeoPackage missing: {expected_gpkg}")
        else:
            _add(warnings, "WARN", f"GeoPackage not found (merge stage likely skipped): {expected_gpkg}")

    # Summary diagnostics
    if summary:
        if summary.get("pipeline_completed"):
            _add(successes, "OK", "Pipeline summary indicates completion")
        else:
            _add(warnings, "WARN", "Pipeline summary found but marked incomplete")
    else:
        _add(warnings, "WARN", "No pipeline_run_summary.json found; limited validation performed")

    print("=== PIPELINE OUTPUT VERIFICATION ===")
    for line in successes:
        print(line)
    for line in warnings:
        print(line)
    for line in failures:
        print(line)
    print("====================================")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

