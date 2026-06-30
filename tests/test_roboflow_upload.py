import sys
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.labeling import roboflow_upload as rfu


def _make_tiles(directory: Path, n: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (directory / f"tile_y0000_x{i:04d}.png").write_bytes(b"fake-png")
    (directory / "tiling_metadata.json").write_text("{}")  # must be ignored


def test_slugify():
    assert rfu.slugify("Site A!! 2026") == "site-a-2026"
    assert rfu.slugify("   ") == "dataset"


def test_make_batch_name_format():
    ts = datetime(2026, 6, 30, 14, 22, 10)
    assert rfu.make_batch_name("Site A", ts) == "site-a_20260630_142210"


def test_make_batch_name_slurm(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    ts = datetime(2026, 6, 30, 14, 22, 10)
    assert rfu.make_batch_name("site-a", ts) == "site-a_20260630_142210_job123"


def test_upload_tiles_dry_run(tmp_path, monkeypatch):
    _make_tiles(tmp_path, 3)
    monkeypatch.setattr(rfu, "_get_workspace", lambda *a, **k: pytest.fail("network!"))
    summary = rfu.upload_tiles(tmp_path, project="site-a", dry_run=True)
    assert summary["dry_run"] is True
    assert summary["tiles_found"] == 3
    assert summary["uploaded"] == 3
    assert summary["batch"].startswith("site-a_")


def test_upload_tiles_no_key(tmp_path, monkeypatch):
    _make_tiles(tmp_path, 2)
    monkeypatch.setattr(rfu, "resolve_api_key", lambda *a, **k: None)
    monkeypatch.setattr(rfu, "_get_workspace", lambda *a, **k: pytest.fail("network!"))
    summary = rfu.upload_tiles(tmp_path, project="site-a")
    assert summary["skipped"] is True
    assert "ROBOFLOW_API_KEY" in summary["reason"]


def test_upload_tiles_empty(tmp_path):
    tmp_path.joinpath("only").mkdir()
    summary = rfu.upload_tiles(tmp_path, project="site-a", dry_run=True)
    assert summary["skipped"] is True
    assert summary["tiles_found"] == 0


class _FakeProject:
    def __init__(self):
        self.uploads = []

    def upload(self, image_path, batch_name, split, num_retry_uploads):
        if image_path.endswith("0001.png"):
            raise RuntimeError("boom")
        self.uploads.append({"path": image_path, "batch": batch_name})


class _FakeWorkspace:
    def __init__(self, exists=False):
        self._project = _FakeProject() if exists else None
        self.created = None

    def project(self, project_id):
        if self._project is None:
            raise RuntimeError("not found")
        return self._project

    def create_project(self, project_name, project_type, project_license, annotation):
        self.created = {
            "name": project_name,
            "type": project_type,
            "license": project_license,
            "annotation": annotation,
        }
        self._project = _FakeProject()


def test_upload_tiles_mocked_sdk(tmp_path, monkeypatch):
    _make_tiles(tmp_path, 3)  # x0000, x0001 (fails), x0002
    ws = _FakeWorkspace(exists=False)
    monkeypatch.setattr(rfu, "resolve_api_key", lambda *a, **k: "key")
    monkeypatch.setattr(rfu, "_get_workspace", lambda *a, **k: ws)
    summary = rfu.upload_tiles(tmp_path, project="Site A", batch_name="b1")
    assert ws.created["type"] == "instance-segmentation"  # auto-created
    assert ws.created["name"] == "site-a"
    assert summary["uploaded"] == 2
    assert summary["failed"] == 1
    assert summary["failures"][0]["file"].endswith("0001.png")
    assert all(u["batch"] == "b1" for u in ws._project.uploads)


def test_get_or_create_existing():
    ws = _FakeWorkspace(exists=True)
    proj = rfu.get_or_create_project(ws, "site-a")
    assert proj is ws._project
    assert ws.created is None


def test_upload_tiles_max_tiles_samples(tmp_path):
    _make_tiles(tmp_path, 10)
    summary = rfu.upload_tiles(tmp_path, project="site-a", dry_run=True, max_tiles=3, sample_seed=0)
    assert summary["tiles_found"] == 10   # all tiles still on disk
    assert summary["selected"] == 3       # only 3 chosen for upload
    assert summary["sampled"] is True
    assert summary["uploaded"] == 3


def test_upload_tiles_max_tiles_reproducible(tmp_path, monkeypatch):
    _make_tiles(tmp_path, 10)
    monkeypatch.setattr(rfu, "resolve_api_key", lambda *a, **k: "key")

    class _Recorder:
        def __init__(self):
            self.paths = []

        def upload(self, image_path, batch_name, split, num_retry_uploads):
            self.paths.append(Path(image_path).name)

    class _WS:
        def __init__(self):
            self._p = _Recorder()

        def project(self, project_id):
            return self._p

    def run(seed):
        ws = _WS()
        monkeypatch.setattr(rfu, "_get_workspace", lambda *a, **k: ws)
        rfu.upload_tiles(tmp_path, project="site-a", batch_name="b", max_tiles=4, sample_seed=seed)
        return sorted(ws._p.paths)

    first, second = run(7), run(7)
    assert first == second  # same seed -> same uploaded tiles
    assert len(first) == 4
