from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import torch

from stage_vla_v7.simulation.environments.align_reward import (
    AlignRewardConfig,
    align_reward_terms,
)
from stage_vla_v7.simulation.environments.transport_handoff import (
    TransportHandoffConfig,
    transport_handoff_ready,
    transport_settle_reward_terms,
)


ROOT = Path(__file__).parents[2]
V5_ROOT = ROOT / "vendor" / "stage_vla_v5"
sys.path.insert(0, str(V5_ROOT))

from stage_vla.rl.align_reward import AlignRewardConfig as V5AlignRewardConfig  # noqa: E402
from stage_vla.rl.align_reward import align_reward_terms as v5_align_reward_terms  # noqa: E402
from stage_vla.rl.transport_handoff import (  # noqa: E402
    TransportHandoffConfig as V5TransportHandoffConfig,
)
from stage_vla.rl.transport_handoff import (  # noqa: E402
    transport_handoff_ready as v5_transport_handoff_ready,
    transport_settle_reward_terms as v5_transport_settle_reward_terms,
)


def _assert_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.isfinite(actual).all()
    if actual.dtype == torch.bool:
        assert torch.equal(actual, expected)
    else:
        assert torch.allclose(actual, expected, atol=0.0, rtol=0.0)


def test_align_reward_terms_match_v5_exactly() -> None:
    generator = torch.Generator().manual_seed(61081)
    count = 128
    vectors = [0.1 * torch.rand(count, generator=generator) for _ in range(11)]
    vectors[-1] += 0.001
    actions = [2.5 * torch.randn(count, 5, generator=generator) for _ in range(3)]
    cfg = AlignRewardConfig()
    legacy_cfg = V5AlignRewardConfig()
    assert asdict(cfg) == asdict(legacy_cfg)
    actual = align_reward_terms(*vectors[:8], *actions, *vectors[8:], cfg=cfg)
    expected = v5_align_reward_terms(
        *vectors[:8], *actions, *vectors[8:], cfg=legacy_cfg
    )
    assert actual.keys() == expected.keys()
    for name in actual:
        _assert_exact(actual[name], expected[name])


def test_transport_handoff_and_reward_terms_match_v5_exactly() -> None:
    generator = torch.Generator().manual_seed(47)
    count = 128
    physical = torch.rand(count, generator=generator) > 0.25
    pressure = torch.rand(count, generator=generator) > 0.25
    xy = 0.1 * torch.rand(count, generator=generator)
    linear = 0.2 * torch.rand(count, generator=generator)
    angular = 3.0 * torch.rand(count, generator=generator)
    action = 2.0 * torch.randn(count, 5, generator=generator)
    cfg = TransportHandoffConfig()
    legacy_cfg = V5TransportHandoffConfig()
    assert asdict(cfg) == asdict(legacy_cfg)
    _assert_exact(
        transport_handoff_ready(
            physical, pressure, xy, linear, angular, cfg=cfg
        ),
        v5_transport_handoff_ready(
            physical, pressure, xy, linear, angular, cfg=legacy_cfg
        ),
    )
    actual = transport_settle_reward_terms(
        xy, linear, angular, action, cfg=cfg
    )
    expected = v5_transport_settle_reward_terms(
        xy, linear, angular, action, cfg=legacy_cfg
    )
    assert actual.keys() == expected.keys()
    for name in actual:
        _assert_exact(actual[name], expected[name])
