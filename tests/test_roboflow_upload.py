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


def test_project_slug_must_start_with_letter():
    # Roboflow 404s on digit-leading project slugs -> prefix them.
    assert rfu.project_slug("2025_06_20_Enderlin_ND") == "tornado-2025-06-20-enderlin-nd"
    # letter-leading names are left untouched
    assert rfu.project_slug("Site A") == "site-a"


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
        return self._project  # SDK returns the new Project; get_or_create uses it


def test_upload_tiles_mocked_sdk(tmp_path, monkeypatch):
    _make_tiles(tmp_path, 3)  # x0000, x0001 (fails), x0002
    ws = _FakeWorkspace(exists=False)
    monkeypatch.setattr(rfu, "resolve_api_key", lambda *a, **k: "key")
    monkeypatch.setattr(rfu, "_get_workspace", lambda *a, **k: ws)
    summary = rfu.upload_tiles(tmp_path, project="Site A", batch_name="b1", show_progress=False)
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
        rfu.upload_tiles(
            tmp_path, project="site-a", batch_name="b", max_tiles=4, sample_seed=seed,
            show_progress=False,
        )
        return sorted(ws._p.paths)

    first, second = run(7), run(7)
    assert first == second  # same seed -> same uploaded tiles
    assert len(first) == 4


class _Clock:
    """Manually advanced monotonic clock for deterministic throttle tests."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_format_eta():
    assert rfu._format_eta(30) == "0:30"
    assert rfu._format_eta(90) == "1:30"
    assert rfu._format_eta(3725) == "1:02:05"
    assert rfu._format_eta(-5) == "0:00"  # never negative


def test_log_progress_throttle_and_final_line():
    import io

    clock = _Clock()
    out = io.StringIO()
    p = rfu._LogProgress(total=5, desc="Uploading site-a", interval=30.0, clock=clock, out=out)

    clock.t = 10; p.update(0)   # 10s since start -> under interval, no emit
    clock.t = 20; p.update(0)   # still under interval
    assert out.getvalue() == ""  # nothing emitted yet

    clock.t = 45; p.update(0)   # 45s >= 30s -> emit
    mid = out.getvalue()
    assert "3/5" in mid
    assert "60%" in mid

    p.close()                    # close always emits a final line
    lines = out.getvalue().strip().splitlines()
    assert len(lines) == 2       # one throttled + one final


def test_log_progress_line_format():
    import io

    clock = _Clock()
    out = io.StringIO()
    # total=10, 4 tiles done in 2s -> rate 2.0 tile/s, 6 left -> ETA 0:03.
    # Numbers chosen so rate/ETA are exact in float (no fragile rounding).
    p = rfu._LogProgress(total=10, desc="Uploading site-a", interval=30.0, clock=clock, out=out)
    clock.t = 2.0
    for _ in range(4):
        p.update(2)  # all under the 30s interval -> no mid-run emit
    p.close()        # final line reflects done=4, failed=2, elapsed=2s
    line = out.getvalue().strip().splitlines()[-1]
    assert line.startswith("Uploading site-a:")
    assert "40%" in line
    assert "4/10" in line
    assert "2.0 tile/s" in line
    assert "ETA 0:03" in line
    assert "failed=2" in line


class _RecordProgress:
    def __init__(self):
        self.updates = []
        self.closed = 0

    def update(self, failed):
        self.updates.append(failed)

    def close(self):
        self.closed += 1


def test_upload_reports_progress_per_tile(tmp_path, monkeypatch):
    _make_tiles(tmp_path, 3)  # x0000 ok, x0001 fails, x0002 ok
    ws = _FakeWorkspace(exists=True)
    rec = _RecordProgress()
    monkeypatch.setattr(rfu, "resolve_api_key", lambda *a, **k: "key")
    monkeypatch.setattr(rfu, "_get_workspace", lambda *a, **k: ws)
    monkeypatch.setattr(rfu, "_make_progress", lambda *a, **k: rec)
    rfu.upload_tiles(tmp_path, project="site-a", batch_name="b1")
    assert rec.updates == [0, 1, 1]  # cumulative failed count, one call per tile
    assert rec.closed == 1


def test_make_progress_disabled_and_empty():
    assert isinstance(rfu._make_progress(0, "x", show=True), rfu._NullProgress)
    assert isinstance(rfu._make_progress(5, "x", show=False), rfu._NullProgress)
