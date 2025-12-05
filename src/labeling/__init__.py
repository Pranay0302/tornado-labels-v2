"""Labeling pipeline modules."""

from __future__ import annotations

from .convert_to_tif import convert_to_tif
from .merge_annotations import merge_annotations
from .tile_orthomosaic import tile_raster

__all__ = [
    "convert_to_tif",
    "generate_sam_proposals",
    "merge_annotations",
    "tile_raster",
]


def __getattr__(name: str):
    if name == "generate_sam_proposals":
        from .samgeo_propose import generate_sam_proposals

        return generate_sam_proposals
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

