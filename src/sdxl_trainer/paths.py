"""Application paths and atomic, private per-user settings."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


TRAINING_DEFAULTS_VERSION = 1


def config_dir() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "sdxl-trainer"


def data_dir() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "sdxl-trainer"


def projects_dir() -> Path:
    """Writable project workspace beside the user's home-folder source checkout."""
    return Path.home() / "SDXL-Trainer" / "projects"


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".save-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load_settings() -> dict:
    try:
        return json.loads((config_dir() / "settings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def migrate_training_defaults(settings: dict) -> bool:
    """Select the fast profile once without touching caption or experiment data."""
    version = settings.get("training_defaults_version", 0)
    if isinstance(version, int) and version >= TRAINING_DEFAULTS_VERSION:
        return False
    # The training config module has no Qt or ML-runtime imports.
    from .training.config import fast_training_settings
    previous = settings.get("training", {})
    training = dict(previous) if isinstance(previous, dict) else {}
    training.update(fast_training_settings(training.get("mode", "lora")))
    settings["training"] = training
    settings["training_defaults_version"] = TRAINING_DEFAULTS_VERSION
    return True


def offline_environment() -> dict[str, str]:
    env = dict(os.environ)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
               DO_NOT_TRACK="1", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    source = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
    return env
