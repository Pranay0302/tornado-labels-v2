"""Input/output helper utilities for the tornado labeling pipeline.

These helpers centralise common filesystem operations such as ensuring
directories exist, resolving user-provided paths and creating structured
output folders.  They are intentionally lightweight so they can be imported
from both library code and command-line entry points without pulling in
heavy dependencies.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable


def ensure_directory(path: Path | str) -> Path:
    """Create *path* if needed and return it as a :class:`Path` instance.

    Parameters
    ----------
    path:
        Directory path to ensure. Both string and ``Path`` instances are
        accepted. The path is expanded (``~`` support) before creation.

    Returns
    -------
    Path
        The resolved directory path. The directory is guaranteed to exist
        when the function returns.
    """

    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_directories(paths: Iterable[Path | str]) -> tuple[Path, ...]:
    """Ensure that every directory in *paths* exists.

    This is a convenience wrapper around :func:`ensure_directory` that returns
    the resolved paths so the caller can keep using them without re-creating
    ``Path`` objects.
    """

    return tuple(ensure_directory(path) for path in paths)


def create_timestamped_subdir(
    base_dir: Path | str,
    *,
    prefix: str | None = None,
    timestamp: datetime | None = None,
) -> Path:
    """Create and return a timestamped sub-directory inside *base_dir*.

    The resulting directory name follows the pattern
    ``<prefix>_<YYYYmmdd_HHMMSS>``. When ``prefix`` is ``None`` or empty, the
    leading underscore is omitted.
    """

    base_path = ensure_directory(base_dir)
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    if prefix:
        name = f"{prefix}_{stamp}"
    else:
        name = stamp
    return ensure_directory(base_path / name)


def resolve_existing_path(path: Path | str) -> Path:
    """Resolve *path* and ensure it exists.

    Raises
    ------
    FileNotFoundError
        If the path does not exist on disk.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved


