"""Safe action helpers for terminal per-environment isolation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


def isolated_reach_raw_action(
    decoder: Callable[[Any], Any],
    action: Any,
    failed_vision: Any,
    previous_action: Any,
) -> Any:
    """Preserve the last gripper command on failed Vision rows."""
    import torch

    raw = decoder(action)
    failed = torch.as_tensor(failed_vision, dtype=torch.bool, device=raw.device)
    previous = torch.as_tensor(
        previous_action, dtype=raw.dtype, device=raw.device
    )
    if failed.shape != (raw.shape[0],) or previous.shape != (raw.shape[0], 5):
        raise ValueError("isolation masks and previous actions must match the batch")
    raw[failed, 6] = previous[failed, 4]
    return raw


@dataclass(frozen=True)
class VisionIsolation:
    enabled: bool
    num_envs: int
    alive: Any
    reach_decoder: Callable[[Any], Any]

    def mask(self, device: Any) -> Any:
        import torch

        if not self.enabled:
            return torch.ones(self.num_envs, dtype=torch.bool, device=device)
        return torch.as_tensor(self.alive, dtype=torch.bool, device=device)

    def reach_raw_action(self, action: Any, previous_action: Any) -> Any:
        return isolated_reach_raw_action(
            self.reach_decoder,
            action,
            ~self.mask(action.device),
            previous_action,
        )


__all__ = ["VisionIsolation", "isolated_reach_raw_action"]
