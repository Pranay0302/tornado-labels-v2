"""Filesystem helpers for the tornado labeling pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def ensure_directory(path: Path | str) -> Path:
    """Create *path* if needed and return it as a :class:`Path` instance."""
    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_directories(paths: Iterable[Path | str]) -> tuple[Path, ...]:
    """Ensure that every directory in *paths* exists and return them."""
    return tuple(ensure_directory(path) for path in paths)
