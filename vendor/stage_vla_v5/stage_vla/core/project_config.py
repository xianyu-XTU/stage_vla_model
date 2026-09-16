"""Dependency-free readers for the project's simple YAML config.

The project intentionally keeps ``config/default.yaml`` as the machine-independent
single source for task thresholds. This is not a general YAML parser; it supports
the scalar ``section -> key: value`` structure used by this project.
"""

from __future__ import annotations

from pathlib import Path


def _strip_scalar(text: str) -> str:
    return text.split("#", 1)[0].strip().strip("\"'")


def read_section_scalar(config_file: Path, section: str, key: str) -> str:
    """Read one scalar from a top-level YAML mapping section."""
    if not config_file.exists():
        raise FileNotFoundError(config_file)

    current_section: str | None = None

    for raw in config_file.read_text(encoding="utf-8").splitlines():
        no_comment = raw.split("#", 1)[0].rstrip()
        if not no_comment.strip():
            continue

        if raw[:1].isspace():
            if current_section != section:
                continue
            stripped = no_comment.strip()
            if ":" not in stripped:
                continue
            candidate, value = stripped.split(":", 1)
            if candidate.strip() == key:
                result = _strip_scalar(value)
                if result == "":
                    raise ValueError(f"{section}.{key} is empty in {config_file}")
                return result
        else:
            stripped = no_comment.strip()
            if stripped.endswith(":"):
                current_section = stripped[:-1].strip()
            else:
                current_section = None

    raise KeyError(f"Missing {section}.{key} in {config_file}")


def read_float(config_file: Path, section: str, key: str) -> float:
    return float(read_section_scalar(config_file, section, key))


def read_int(config_file: Path, section: str, key: str) -> int:
    return int(read_section_scalar(config_file, section, key))


def read_str(config_file: Path, section: str, key: str) -> str:
    return read_section_scalar(config_file, section, key)
