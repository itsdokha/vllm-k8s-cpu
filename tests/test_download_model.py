import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "docker" / "model-downloader" / "download_model.py"


@pytest.fixture
def dl(monkeypatch):
    # huggingface_hub в тестах не нужен: подменяем snapshot_download заглушкой,
    # которая кладёт в каталог один файл и запоминает вызовы.
    calls = []

    def fake_snapshot_download(repo_id, revision, local_dir, **kwargs):
        calls.append((repo_id, revision))
        Path(local_dir, "model.safetensors").write_bytes(b"weights")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=fake_snapshot_download))
    spec = importlib.util.spec_from_file_location("download_model", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.calls = calls
    return module


def setenv(monkeypatch, root, name="model-a", revision="abc", **extra):
    monkeypatch.setenv("MODEL_ID", f"org/{name}")
    monkeypatch.setenv("MODEL_REVISION", revision)
    monkeypatch.setenv("MODEL_DIR", str(root / name))
    for key, value in extra.items():
        monkeypatch.setenv(key, value)


def test_first_run_downloads_and_writes_marker(dl, monkeypatch, tmp_path):
    setenv(monkeypatch, tmp_path)

    assert dl.main() == 0
    assert dl.calls == [("org/model-a", "abc")]
    marker = json.loads((tmp_path / "model-a" / dl.MARKER).read_text())
    assert marker["model_id"] == "org/model-a" and marker["revision"] == "abc"
    assert not (tmp_path / "model-a" / f"{dl.MARKER}.tmp").exists()


def test_second_run_does_not_touch_network(dl, monkeypatch, tmp_path):
    setenv(monkeypatch, tmp_path)
    dl.main()
    dl.calls.clear()

    assert dl.main() == 0
    assert dl.calls == []


def test_new_revision_triggers_download(dl, monkeypatch, tmp_path):
    setenv(monkeypatch, tmp_path)
    dl.main()
    setenv(monkeypatch, tmp_path, revision="def")

    dl.main()
    assert dl.calls[-1] == ("org/model-a", "def")


def test_partial_files_are_removed(dl, monkeypatch, tmp_path):
    setenv(monkeypatch, tmp_path)
    partial = tmp_path / "model-a" / ".cache" / "huggingface" / "download" / "x.incomplete"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"half")

    dl.main()
    assert not partial.exists()


def test_prune_removes_only_our_old_models(dl, monkeypatch, tmp_path):
    setenv(monkeypatch, tmp_path, name="model-old")
    dl.main()
    (tmp_path / "compile-cache").mkdir()

    setenv(monkeypatch, tmp_path, name="model-new", PRUNE_OTHER_MODELS="true", CACHE_DIR=str(tmp_path / "compile-cache"))
    dl.main()

    assert not (tmp_path / "model-old").exists()
    assert (tmp_path / "model-new" / dl.MARKER).exists()
    assert (tmp_path / "compile-cache").exists()
