"""Project-scoped caption settings stored with the original prompt keys."""
from dataclasses import fields
import json
import math
from pathlib import Path

from .captioning import CaptionConfig
from .paths import atomic_json

PROMPT_KEYS = ("system_prompt", "user_prompt")
PROMPT_FILE = "caption-prompts.json"
_CAPTION_TYPES = {field.name: type(field.default) for field in fields(CaptionConfig)}
_LEGACY_DEFAULTS = {
    "system_prompt": ("You write precise training captions for SDXL. Describe only visible content in the supplied image: "
                      "subject, action, composition, lighting, colors, and medium when evident. Do not invent facts. "
                      "Return only one concise plain-text caption, ideally under 60 words. Do not include analysis, "
                      "headings, quotes or introductory phrases."),
    "user_prompt": "Caption this exact image for an SDXL training dataset.",
}


def default_prompts():
    defaults = CaptionConfig()
    return {name: getattr(defaults, name) for name in PROMPT_KEYS}


def migrate_legacy_prompts(values):
    """Retain custom legacy prompts while replacing the old short-caption defaults."""
    result = default_prompts()
    for name in PROMPT_KEYS:
        value = values.get(name)
        if isinstance(value, str) and value != _LEGACY_DEFAULTS[name]:
            result[name] = value
    return result


def _settings_path(project):
    path = Path(project.root) / PROMPT_FILE
    if path.is_symlink():
        raise ValueError("Project caption settings must not be a symbolic link.")
    if path.exists() and not path.is_file():
        raise ValueError(f"Project caption settings must be stored in a regular {PROMPT_FILE} file.")
    return path


def _valid_setting(value, expected_type):
    value_type = type(value)
    if expected_type is float and value_type in (float, int):
        try:
            return math.isfinite(value)
        except OverflowError:
            return False
    if expected_type is int and value_type is int:
        # QSpinBox.setValue takes a signed 32-bit integer.
        return -(2**31) <= value < 2**31
    return value_type is expected_type


def _caption_values(data, required):
    if not isinstance(data, dict):
        raise ValueError(f"Project caption settings are invalid. Inspect {PROMPT_FILE}.")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"Project caption settings are missing {', '.join(missing)}. Inspect {PROMPT_FILE}.")
    values = {name: data[name] for name in _CAPTION_TYPES if name in data}
    for name, value in values.items():
        if not _valid_setting(value, _CAPTION_TYPES[name]):
            raise ValueError(f"Invalid value for {name} in {PROMPT_FILE}.")
    return values


def load_project_caption_settings(project):
    """Read saved form values, including projects that only store prompts."""
    path = _settings_path(project)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return _caption_values(data, PROMPT_KEYS)


def save_project_caption_settings(project, values):
    """Store the complete caption form in the existing project file."""
    path = _settings_path(project)
    atomic_json(path, _caption_values(values, _CAPTION_TYPES))
