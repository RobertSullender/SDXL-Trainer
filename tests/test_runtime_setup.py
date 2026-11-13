"""Local runtime cloning tests: actual isolated venv, no downloads or ML weights."""
import importlib.util
import os
from pathlib import Path
import sys
import venv

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_local_runtime", ROOT / "scripts/prepare_local_runtime.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


@pytest.fixture
def local_runtime(tmp_path):
    source = tmp_path / "source"
    venv.EnvBuilder(with_pip=False, symlinks=False).create(source)
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    package = source / "lib" / version / "site-packages" / "fixture_dependency"
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 'original'\n")
    (source / "bin" / "fixture-command").write_text(f"#!{source}/bin/python\nprint('fixture')\n")
    (source / "bin" / "fixture-command").chmod(0o755)
    return source, package


def test_independent_copy_has_own_prefix_imports_and_no_source_changes(tmp_path, local_runtime):
    source, package = local_runtime
    before = (package / "__init__.py").read_bytes()
    destination = tmp_path / "managed-runtime"
    setup.prepare_runtime(source, destination, modules=("fixture_dependency",))
    result = setup.validate_runtime(destination, ("fixture_dependency",))
    assert result["prefix"] == str(destination)
    assert Path(result["modules"]["fixture_dependency"]).is_relative_to(destination)
    assert (package / "__init__.py").read_bytes() == before
    copied = Path(result["modules"]["fixture_dependency"])
    assert copied.stat().st_ino != (package / "__init__.py").stat().st_ino
    copied.write_text("VALUE = 'copy changed'\n")
    assert (package / "__init__.py").read_bytes() == before
    assert (destination / "bin/fixture-command").read_text().startswith(f"#!{destination}/bin/python\n")
    assert not list(tmp_path.glob(".runtime-copy-*"))


def test_existing_destination_is_never_replaced(tmp_path, local_runtime):
    source, _ = local_runtime
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "keep"
    sentinel.write_bytes(b"existing user data")
    with pytest.raises(FileExistsError):
        setup.prepare_runtime(source, destination, modules=("fixture_dependency",))
    assert sentinel.read_bytes() == b"existing user data"


def test_outside_symlink_is_rejected_before_copy(tmp_path, local_runtime):
    source, _ = local_runtime
    (source / "outside").symlink_to("/usr/bin/python3")
    destination = tmp_path / "managed-runtime"
    with pytest.raises(ValueError, match="outside symbolic link"):
        setup.prepare_runtime(source, destination, modules=("fixture_dependency",))
    assert not destination.exists()


def test_publication_race_does_not_replace_empty_destination(tmp_path, local_runtime, monkeypatch):
    source, _ = local_runtime
    destination = tmp_path / "managed-runtime"
    validate = setup.validate_runtime

    def concurrent_creation(stage, modules):
        result = validate(stage, modules)
        destination.mkdir()
        return result

    monkeypatch.setattr(setup, "validate_runtime", concurrent_creation)
    with pytest.raises(FileExistsError):
        setup.prepare_runtime(source, destination, modules=("fixture_dependency",))
    assert destination.is_dir() and not any(destination.iterdir())
    assert not list(tmp_path.glob(".runtime-copy-*"))
