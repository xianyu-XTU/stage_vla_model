"""Pure helpers for M9 training-curve discovery/export.

The actual TensorBoard event parsing lives in ``tools/export_m9a_curves.py`` so
this module stays importable in the project's ordinary Python unit-test
environment without requiring TensorBoard.
"""

from __future__ import annotations

import re
from collections.abc import Iterable


CORE_TENSORBOARD_TAGS: tuple[str, ...] = (
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Loss/value_function",
    "Loss/surrogate",
    "Loss/entropy",
    "Loss/learning_rate",
    "Policy/mean_std",
    "Perf/total_fps",
)


def sanitize_tag_filename(tag: str) -> str:
    """Turn a TensorBoard scalar tag into a portable filename stem."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "__", tag.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "metric"


def select_curve_tags(available_tags: Iterable[str]) -> list[str]:
    """Select the scalar tags that should be exported as project curves.

    RSL-RL loss key names can change slightly between versions.  Therefore the
    selector keeps all ``Loss/`` tags rather than assuming only the canonical
    names above.  Isaac Lab RewardManager episode summaries are kept through
    ``Episode_Reward/`` so every active baseline reward component can be
    inspected for reward hacking or saturation.
    """
    available = list(dict.fromkeys(str(tag) for tag in available_tags))
    selected: list[str] = []

    for tag in CORE_TENSORBOARD_TAGS:
        if tag in available:
            selected.append(tag)

    for prefix in ("Loss/", "Episode_Reward/", "M10/", "/M10/"):
        for tag in available:
            if tag.startswith(prefix) and tag not in selected:
                selected.append(tag)

    return selected
