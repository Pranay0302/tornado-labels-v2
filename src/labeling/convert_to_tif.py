"""Convert photogrammetry exports into GeoTIFF files."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.io_utils import ensure_directory, resolve_existing_path


def convert_to_tif(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    gdal_translate_args: Iterable[str] | None = None,
) -> Path:
    """Prepare a GeoTIFF copy of *input_path* inside *output_dir*.

    ``.tif`` files are copied verbatim, while ``.tev`` files are converted via
    ``gdal_translate``. Returns the path to the resulting GeoTIFF.
    """

    src_path = resolve_existing_path(input_path)
    dest_dir = ensure_directory(output_dir)

    if src_path.suffix.lower() == ".tif":
        destination = dest_dir / src_path.name
        if overwrite or not destination.exists():
            shutil.copy2(src_path, destination)
        return destination

    destination = dest_dir / f"{src_path.stem}.tif"
    if destination.exists() and not overwrite:
        return destination

    translate_cmd = [
        "gdal_translate",
        "-of",
        "GTiff",
        *([] if gdal_translate_args is None else list(gdal_translate_args)),
        str(src_path),
        str(destination),
    ]

    try:
        subprocess.check_call(translate_cmd)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "gdal_translate failed; ensure GDAL is installed and the input file is supported"
        ) from exc

    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert an input raster to GeoTIFF")
    parser.add_argument("input", help="Input raster (.tev or .tif)")
    parser.add_argument("output_dir", help="Directory where the GeoTIFF will be stored")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument(
        "--gdal-arg",
        dest="gdal_args",
        action="append",
        help="Extra arguments to pass to gdal_translate",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    tif_path = convert_to_tif(
        args.input,
        args.output_dir,
        overwrite=args.overwrite,
        gdal_translate_args=args.gdal_args,
    )
    print(f"[OK] GeoTIFF ready: {tif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
