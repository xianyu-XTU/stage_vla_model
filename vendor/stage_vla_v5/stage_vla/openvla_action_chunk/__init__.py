"""Experimental frozen-OpenVLA continuous action-chunk policy.

This package is preparatory work only.  It does not change the frozen v4 stage
status and never treats OpenVLA's native action tokens as supervision.
"""

from .data import build_action_chunks, load_episode_chunk_samples
from .model import ActionChunkHead, action_chunk_loss

__all__ = [
    "ActionChunkHead",
    "action_chunk_loss",
    "build_action_chunks",
    "load_episode_chunk_samples",
]
