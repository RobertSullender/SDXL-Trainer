"""Native offscreen workflow tests; no model weights, GPU, or network required."""
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog
from PIL import Image

from sdxl_trainer import app as desktop
from sdxl_trainer.captioning import CaptionError
from sdxl_trainer.dataset import DatasetProject, verify_finalized
from sdxl_trainer.runtime import RemoteTokenCounter
from sdxl_trainer.tokens import TokenCounts
from sdxl_trainer.training.config import TrainingConfig


@pytest.fixture(scope="session")
def qt_app():
    return QApplication.instance() or QApplication([])


def wait_until(qt_app, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        qt_app.processEvents()
        if time.monotonic() > deadline:
            raise AssertionError("The native UI operation did not finish in time.")
        time.sleep(0.005)
    qt_app.processEvents()


@pytest.fixture
def window(qt_app, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    errors = []
    qt_exceptions = []
    monkeypatch.setattr(desktop.Window, "error", lambda self, text: errors.append(text))
    monkeypatch.setattr(sys, "excepthook", lambda *args: qt_exceptions.append(args))
    result = desktop.Window()
    result.test_errors = errors
    result.show()
    qt_app.processEvents()
    yield result
    if result.task:
        result.task.cancel.set()
        wait_until(qt_app, lambda: result.task is None)
    if result.process:
        result.process.kill()
        wait_until(qt_app, lambda: result.process is None)
    result.close()
    qt_app.processEvents()
    assert not qt_exceptions, qt_exceptions


def imported_project(tmp_path, count=2):
    project = DatasetProject.create(tmp_path / "draft")
    originals = []
    for i in range(count):
        path = tmp_path / f"original-{i}.jpg"
        Image.new("RGB", (96 + i * 20, 64), (100 + i, 30, 80)).save(path)
        originals.append(path)
    project.import_images(originals)
    return project, originals


def test_application_loads_all_pages_and_all_backend_controls(window, qt_app):
    assert window.pages.count() == 6
    assert set(window.training_controls) == set(TrainingConfig.__dataclass_fields__)
    assert set(window.caption_form.controls) == set(desktop.CaptionConfig.__dataclass_fields__)
    for index in range(6):
        window.navigation.setCurrentRow(index)
        qt_app.processEvents()
        assert window.pages.currentIndex() == index
    assert not window.test_errors


def test_new_import_edit_switch_and_reopen_native_workflow(window, qt_app, tmp_path, monkeypatch):
    root = tmp_path / "draft"
    root.mkdir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(root))
    window.new_project()
    assert window.project.root == root
    originals = []
    original_bytes = []
    for index in range(2):
        path = tmp_path / f"source-{index}.jpg"
        Image.new("RGB", (160, 100 + index * 20), "blue").save(path)
        originals.append(str(path))
        original_bytes.append(path.read_bytes())
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *a, **k: (originals, ""))
    window.import_images(False)
    wait_until(qt_app, lambda: window.task is None)
    assert window.images.count() == 2 and window.entry_id == 2
    window.images.setCurrentRow(0)
    window.caption.setPlainText("First reviewed image caption.")
    window.images.setCurrentRow(1)
    assert window.project.caption_text(1) == "First reviewed image caption."
    window.caption.setPlainText("Second reviewed image caption.")
    wait_until(qt_app, lambda: window.project.caption_text(2) == "Second reviewed image caption.")
    window.open_project()
    assert window.images.count() == 2
    assert window.project.caption_text(1) == "First reviewed image caption."
    assert window.project.caption_text(2) == "Second reviewed image caption."
    assert [Path(p).read_bytes() for p in originals] == original_bytes
    assert not window.test_errors


def test_opening_empty_project_clears_old_image_and_caption(window, qt_app, tmp_path):
    project, _ = imported_project(tmp_path)
    window.set_project(project)
    assert not window.loaded_pixmap.isNull()
    window.caption.setPlainText("Saved caption.")
    empty = DatasetProject.create(tmp_path / "empty")
    window.set_project(empty)
    assert window.entry_id is None
    assert window.loaded_pixmap.isNull()
    assert not window.caption.toPlainText()
    assert "Import images" in window.preview.text()
    assert project.caption_text(1) == "Saved caption."


def test_failed_caption_batch_reloads_completed_caption_before_any_save(window, qt_app, tmp_path, monkeypatch):
    project, _ = imported_project(tmp_path)
    window.set_project(project)
    monkeypatch.setattr(desktop.CaptionConfig, "validate", lambda self: [])

    class Server:
        def __init__(self, config):
            self.calls = 0
        def start(self, cancel):
            pass
        def caption(self, path, cancel):
            self.calls += 1
            if self.calls == 2:
                raise CaptionError("Simulated second-image failure")
            return "Authoritative generated caption."
        def stop(self):
            pass

    monkeypatch.setattr(desktop, "CaptionServer", Server)
    window.auto_caption(True)
    wait_until(qt_app, lambda: window.task is None)
    assert "Simulated second-image failure" in window.test_errors[0]
    assert window.caption.toPlainText() == "Authoritative generated caption."
    window.save_caption()
    assert project.caption_text(1) == "Authoritative generated caption."
    assert project.caption_text(2) == ""


def test_finalize_creates_new_snapshot_and_selects_it_for_training(window, qt_app, tmp_path, monkeypatch):
    project, _ = imported_project(tmp_path, 1)
    project.save_caption(1, "A reviewed image caption.")
    window.set_project(project)
    monkeypatch.setattr(window, "request_token_count", lambda: None)
    window.model_field.set_value(tmp_path / "local-model")

    class Counter:
        metadata = {"max_length": 77, "special_tokens_included": True,
                    "tokenizer_sha256": "a" * 64, "tokenizer_2_sha256": "b" * 64}
        def __init__(self, python, model):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def count(self, text):
            return TokenCounts(8, 9)

    monkeypatch.setattr(desktop, "RemoteTokenCounter", Counter)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("sealed-dataset", True))
    window.finalize()
    wait_until(qt_app, lambda: window.task is None)
    output = tmp_path / "sealed-dataset"
    manifest = verify_finalized(output)
    assert manifest["tokenizers"] == Counter.metadata
    assert window.training_controls["dataset_path"].value() == str(output)
    assert not window.test_errors


def fake_token_python(tmp_path, content):
    executable = tmp_path / "token-python"
    executable.write_text(f"#!{sys.executable}\n" + content, encoding="utf-8")
    executable.chmod(0o755)
    return executable


TOKEN_PROTOCOL = '''import json, sys
print(json.dumps({'ready':True,'max_length':77,'metadata':{'tokenizer_sha256':'a'*64,'tokenizer_2_sha256':'b'*64}}),flush=True)
for line in sys.stdin:
    text=json.loads(line)['text']
    print(json.dumps({'clip_l':len(text.split())+2,'clip_g':len(text.split())+3,'max_length':77}),flush=True)
'''


def test_real_local_token_process_updates_editor_counts(window, qt_app, tmp_path):
    project, _ = imported_project(tmp_path, 1)
    project.save_caption(1, "three caption words")
    window.set_project(project)
    executable = fake_token_python(tmp_path, TOKEN_PROTOCOL)
    window.runtime.set_value(executable)
    window.model_field.set_value(tmp_path / "local-model")
    wait_until(qt_app, lambda: "CLIP-L 5 / 77" in window.tokens_label.text())
    assert "CLIP-G 6 / 77" in window.tokens_label.text()
    window.caption.setPlainText("changed caption")
    wait_until(qt_app, lambda: "CLIP-L 4 / 77" in window.tokens_label.text())
    assert "CLIP-G 5 / 77" in window.tokens_label.text()
    window.stop_tokens()
    assert window.token_process is None
    assert not window.test_errors


def test_remote_counter_preserves_metadata_and_real_protocol(tmp_path):
    executable = fake_token_python(tmp_path, TOKEN_PROTOCOL)
    with RemoteTokenCounter(str(executable), "local-model", startup_timeout=2) as counter:
        assert counter.count("a small caption") == TokenCounts(5, 6)
        assert counter.metadata["tokenizer_sha256"] == "a" * 64
        process = counter.process
    assert process.poll() is not None


def test_remote_counter_silent_runtime_times_out_and_does_not_leak_process(tmp_path):
    executable = fake_token_python(tmp_path, "import time\ntime.sleep(30)\n")
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out"):
        RemoteTokenCounter(str(executable), "local-model", startup_timeout=0.1)
    assert time.monotonic() - started < 2


def test_missing_dictionary_does_not_raise_in_editor(qt_app):
    from sdxl_trainer.widgets import CaptionEditor
    class Unavailable:
        available = False
        def misspellings(self, text):
            raise AssertionError("Unavailable dictionaries must not be called by the highlighter.")
    editor = CaptionEditor(Unavailable())
    editor.setPlainText("A caption with an unavailable dictionary.")
    editor.highlighter.rehighlight()
    qt_app.processEvents()
    editor.deleteLater()
