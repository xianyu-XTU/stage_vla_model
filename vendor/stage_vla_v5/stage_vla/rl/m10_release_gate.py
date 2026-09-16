"""Pure helpers for M10 terminal OPEN-command gating.

M10-8-T1 fixed the causal timing of the direct OPEN command bonus by gating
``a_t`` from the state in which it was selected (``s_t``).  M10-9-H2G keeps
that timing and adds exactly one extra requirement to the command gate:
placement must already be settled in ``s_t``.

This helper is intentionally pure and portable so the gate semantics can be
unit-tested without Isaac Lab.
"""

from __future__ import annotations

import torch
from torch import Tensor


M10_H2_OPEN_GATE_SEMANTICS = "pre_action_release_ready_and_settled"


def settled_open_command_gate(release_ready: Tensor, settled: Tensor) -> Tensor:
    """Return the H2 direct-OPEN gate for a pre-action state.

    Parameters are boolean masks for the *same* pre-action state ``s_t``.
    ``release_ready`` is the existing T1 base gate (PLACE + strict geometry +
    grasp/lift history).  ``settled`` is M7's independently defined low-speed
    condition for red and blue cubes.
    """

    ready = torch.as_tensor(release_ready)
    stable = torch.as_tensor(settled, device=ready.device)
    if ready.shape != stable.shape:
        raise ValueError(
            f"release_ready and settled must share shape; got {tuple(ready.shape)} vs {tuple(stable.shape)}"
        )
    if ready.dtype is not torch.bool or stable.dtype is not torch.bool:
        raise TypeError("release_ready and settled must be torch.bool")
    return ready & stable
