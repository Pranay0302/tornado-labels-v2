"""Smoke test for the Streamlit GUI module.

The Streamlit runtime is not exercised here; we only confirm the module imports
cleanly (its UI lives in ``main()`` and is not run on import) and that its core
helpers are wired up.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_app_imports_and_exposes_main():
    from src.gui import app

    assert callable(app.main)


def test_app_discovers_tif_sources(tmp_path):
    from src.gui import app

    (tmp_path / "a.tif").write_bytes(b"")
    (tmp_path / "b.tiff").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")

    found = {p.name for p in app.discover_tif_files(tmp_path)}
    assert found == {"a.tif", "b.tiff"}


def test_discover_recurses_and_dedupes_overlapping_roots(tmp_path):
    from src.gui import app

    (tmp_path / "top.tif").write_bytes(b"")
    nested = tmp_path / "sub" / "deep"
    nested.mkdir(parents=True)
    (nested / "buried.tif").write_bytes(b"")

    # Overlapping roots must not double-count the same file.
    found = app.discover_tif_files(tmp_path, tmp_path / "sub")
    names = sorted(p.name for p in found)
    assert names == ["buried.tif", "top.tif"]


def test_label_for_repo_relative_and_external_paths():
    from src.gui import app

    inside = app.REPO_ROOT / "data" / "site.tif"
    assert app.label_for(inside) == "data/site.tif"

    # A 50GB orthomosaic living outside the repo keeps its absolute path label.
    outside = Path("/mnt/bigdrive/2023_03_Ortho.tif")
    assert app.label_for(outside) == "/mnt/bigdrive/2023_03_Ortho.tif"
