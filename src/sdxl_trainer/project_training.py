"""Project-scoped storage for the active training form and inactive presets."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING

from .paths import atomic_json
from .training.config import TrainingConfig, checkpoint_run_folder, run_checkpoints

if TYPE_CHECKING:
    from .dataset import DatasetProject


TRAINING_SETTINGS_FILE = "training-settings.json"
SHARED_TRAINING_FIELDS = ("model_path", "model_variant", "dataset_path")
_TRAINING_MODES = frozenset(TrainingConfig.__dataclass_fields__["mode"].metadata["choices"])
_PROFILE_FIELDS = TrainingConfig.__dataclass_fields__.keys() - {*SHARED_TRAINING_FIELDS, "mode"}


@dataclass(frozen=True)
class SavedRun:
    path: Path
    mode: str | None
    latest_step: int


def _read_run_metadata(path: Path) -> dict:
    try:
        if not path.is_symlink() and path.is_file():
            values = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(values, dict):
                return values
    except (OSError, ValueError, RecursionError):
        pass
    return {}


def saved_project_runs(project: DatasetProject) -> list[SavedRun]:
    """List project runs with their training method and latest completed step."""
    root = project.root / "runs"
    if root.is_symlink() or not root.is_dir():
        return []
    runs = []
    for run in root.iterdir():
        try:
            if run.is_symlink() or not run.is_dir():
                continue
            checkpoints = run_checkpoints(run)
        except OSError:
            continue
        if not checkpoints:
            continue
        step, checkpoint = checkpoints[0]
        mode = _read_run_metadata(run / "training-config.json").get("mode")
        if not isinstance(mode, str) or mode not in _TRAINING_MODES:
            config = _read_run_metadata(checkpoint / "checkpoint.json").get("config")
            mode = config.get("mode") if isinstance(config, dict) else None
        mode = mode if isinstance(mode, str) and mode in _TRAINING_MODES else None
        runs.append(SavedRun(run, mode, step))
    return sorted(runs, key=lambda run: (
        tuple(int(part) if i % 2 else part.casefold()
              for i, part in enumerate(re.split(r"([0-9]+)", run.path.name))),
        run.path.name,
    ))


def _settings_path(project: DatasetProject) -> Path:
    path = project.root / TRAINING_SETTINGS_FILE
    if path.is_symlink():
        raise ValueError("Project training settings must not be a symbolic link.")
    if path.exists() and not path.is_file():
        raise ValueError(f"Project training settings must be stored in a regular {TRAINING_SETTINGS_FILE} file.")
    return path


def load_project_training_settings(project: DatasetProject) -> TrainingConfig | None:
    """Read the active training form if the project has one."""
    config, _, _ = load_project_training_profiles(project)
    return config


def load_project_training_profiles(
    project: DatasetProject,
) -> tuple[TrainingConfig | None, dict[str, dict], dict[str, str]]:
    """Read the active form, inactive settings, and automatic branch outputs."""
    path = _settings_path(project)
    if not path.exists():
        return None, {}, {}
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Project training settings must be a JSON object.")
    profiles = values.pop("mode_profiles", {})
    branch_outputs = values.pop("suggested_branch_outputs", {})
    config = TrainingConfig.from_dict(values)
    profiles = _validate_profiles(config.mode, profiles)
    for profile in profiles.values():
        # Older projects stored one shared output, even for a checkpoint
        # belonging to the inactive method's run.
        if "output_dir" not in profile:
            run = checkpoint_run_folder(profile.get("resume_from", ""))
            if run is not None:
                profile["output_dir"] = str(run)
    return config, profiles, _saved_branch_outputs(config, profiles, branch_outputs)


def _saved_branch_outputs(config, profiles, suggestions):
    if not isinstance(suggestions, dict):
        return {}
    outputs = {mode: values.get("output_dir") for mode, values in profiles.items()}
    outputs[config.mode] = config.output_dir
    return {mode: output for mode, output in suggestions.items()
            if isinstance(output, str) and output and outputs.get(mode) == output}


def _validate_profiles(active_mode: str, profiles: dict[str, dict]) -> dict[str, dict]:
    if active_mode not in _TRAINING_MODES:
        raise ValueError(f"Unknown training method: {active_mode!r}.")
    if not isinstance(profiles, dict):
        raise ValueError("Training method profiles must be a JSON object.")
    for mode, values in profiles.items():
        if mode not in _TRAINING_MODES or mode == active_mode:
            raise ValueError(f"Invalid inactive training method: {mode!r}.")
        if not isinstance(values, dict) or values.keys() - _PROFILE_FIELDS:
            raise ValueError(f"Invalid controls in the {mode!r} training profile.")
        # Validate types using the recipe loader. Return the supplied values so
        # the app can fill omitted controls from the matching method's preset.
        try:
            TrainingConfig.from_dict(values)
        except OverflowError as exc:
            raise ValueError(f"Numeric value is too large in the {mode!r} training profile.") from exc
    return profiles


def save_project_training_settings(
    project: DatasetProject, config: TrainingConfig, *, mode_profiles: dict[str, dict] | None = None,
    suggested_branch_outputs: dict[str, str] | None = None,
) -> None:
    """Save both methods and their run paths with the shared model and dataset."""
    profiles = _validate_profiles(config.mode, {} if mode_profiles is None else mode_profiles)
    values = config.to_dict()
    if profiles:
        values["mode_profiles"] = profiles
    branches = _saved_branch_outputs(config, profiles, suggested_branch_outputs)
    if branches:
        values["suggested_branch_outputs"] = branches
    atomic_json(_settings_path(project), values)
