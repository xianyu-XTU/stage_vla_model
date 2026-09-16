"""Stable runtime boundary for the frozen v5 skill action models.

The runtime accepts structured task/scene inputs from an upstream adapter and
returns only the selected skill's normalized five-dimensional action.  Vision,
language and task interpretation are deliberately outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Iterable

import torch

from .rl.v5_skill_contracts import ACTION_ORDER, SKILL_SEQUENCE, V5Skill
from .objects import (
    ACTION_CONTEXT_ORDER,
    ACTION_CONTEXT_VERSION,
    ObjectDomain,
    ObjectModule,
    get_object_spec,
    rigid_cube_domain,
)
from .task_dsl.schema import SkillToken


@dataclass(frozen=True)
class SkillModelSpec:
    """One exported TorchScript policy and its observation contract."""

    skill: V5Skill
    policy_path: Path
    observation_dim: int
    action_dim: int = 5
    state_observation_dim: int | None = None
    object_context_version: str | None = None

    def validate(self) -> None:
        if self.action_dim != len(ACTION_ORDER):
            raise ValueError(f"{self.skill}: action_dim must be {len(ACTION_ORDER)}")
        if self.observation_dim < 1:
            raise ValueError(f"{self.skill}: observation_dim must be positive")
        state_dim = (
            self.state_observation_dim
            if self.state_observation_dim is not None
            else self.observation_dim
        )
        if state_dim < 1:
            raise ValueError(f"{self.skill}: state_observation_dim must be positive")
        if self.object_context_version is None:
            if state_dim != self.observation_dim:
                raise ValueError(
                    f"{self.skill}: state_observation_dim requires an object context"
                )
        else:
            if self.object_context_version != ACTION_CONTEXT_VERSION:
                raise ValueError(
                    f"{self.skill}: unsupported object context version "
                    f"{self.object_context_version!r}"
                )
            expected = state_dim + len(ACTION_CONTEXT_ORDER)
            if self.observation_dim != expected:
                raise ValueError(
                    f"{self.skill}: observation_dim must be {expected} for "
                    f"{self.object_context_version}"
                )
        if not self.policy_path.is_file():
            raise FileNotFoundError(self.policy_path)

    @property
    def policy_state_dim(self) -> int:
        return (
            self.state_observation_dim
            if self.state_observation_dim is not None
            else self.observation_dim
        )


@dataclass(frozen=True)
class V5ActionModelManifest:
    """Resolved action-model set; paths are explicit and auditable."""

    models: Mapping[V5Skill, SkillModelSpec]
    action_order: tuple[str, ...] = ACTION_ORDER
    control_hz: float = 20.0
    source_mode: str = "known_size_oracle"
    conditioning: str = "object_support_roles"
    object_domain: ObjectDomain = field(default_factory=rigid_cube_domain)
    validated_object_examples: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.action_order != ACTION_ORDER:
            raise ValueError("action order does not match v5 action contract")
        if self.control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if self.conditioning != "object_support_roles":
            raise ValueError("conditioning must be object_support_roles")
        self.object_domain.validate()
        if len(set(self.validated_object_examples)) != len(self.validated_object_examples):
            raise ValueError("validated_object_examples must not contain duplicates")
        for label in self.validated_object_examples:
            spec = get_object_spec(label)
            if not (
                self.object_domain.supports_object(spec)
                or self.object_domain.supports_support(spec)
            ):
                raise ValueError(f"validated example is outside object domain: {label!r}")
        missing = set(V5Skill) - set(self.models)
        if missing:
            raise ValueError(f"missing skill models: {sorted(skill.value for skill in missing)}")
        for skill in V5Skill:
            spec = self.models[skill]
            if spec.skill != skill:
                raise ValueError(f"model key/spec mismatch for {skill.value}")
            spec.validate()

    def supports_objects(self, objects: Iterable[str]) -> bool:
        """Check whether labels resolve to specs compatible with both roles."""
        specs = [get_object_spec(label) for label in objects]
        return all(
            self.object_domain.supports_object(spec)
            and self.object_domain.supports_support(spec)
            for spec in specs
        )

    def supports_module(self, module: ObjectModule) -> bool:
        """Return whether both roles in an object module are covered."""
        if not isinstance(module, ObjectModule):
            raise TypeError("module must be an ObjectModule")
        return self.object_domain.supports(module)


class V5ActionModelRuntime:
    """Load and run one frozen policy per physical skill."""

    def __init__(self, manifest: V5ActionModelManifest, *, device: str | torch.device = "cpu") -> None:
        manifest.validate()
        self.manifest = manifest
        self.device = torch.device(device)
        self._policies = {
            skill: torch.jit.load(str(spec.policy_path), map_location=self.device).eval()
            for skill, spec in manifest.models.items()
        }

    def action(self, skill: V5Skill | str, observation: torch.Tensor) -> torch.Tensor:
        skill = V5Skill(skill)
        spec = self.manifest.models[skill]
        obs = torch.as_tensor(observation, device=self.device, dtype=torch.float32)
        if obs.ndim != 2 or obs.shape[-1] != spec.observation_dim:
            raise ValueError(
                f"{skill.value} observation must have shape [N,{spec.observation_dim}], got {tuple(obs.shape)}"
            )
        if not torch.isfinite(obs).all():
            raise ValueError("observation contains NaN or Inf")
        with torch.inference_mode():
            output = self._policies[skill](obs)
        action = torch.as_tensor(output, device=self.device, dtype=torch.float32)
        if action.ndim != 2 or action.shape != (obs.shape[0], len(ACTION_ORDER)):
            raise RuntimeError(f"{skill.value} policy returned invalid action shape {tuple(action.shape)}")
        if not torch.isfinite(action).all():
            raise RuntimeError(f"{skill.value} policy returned NaN or Inf")
        return action.clamp(-1.0, 1.0)

    def supports_module(self, module: ObjectModule) -> bool:
        """Expose the bundle's color-independent physical coverage."""
        return self.manifest.supports_module(module)

    def action_for_module(
        self,
        module: ObjectModule,
        skill: V5Skill | str,
        state_observation: torch.Tensor,
    ) -> torch.Tensor:
        """Run a skill for physical object/support roles, never their labels.

        Legacy policies consume their original state vector. A newly trained
        physical-context policy receives the same state followed by the stable
        label-free object feature vector.
        """
        if not isinstance(module, ObjectModule):
            raise TypeError("module must be an ObjectModule")
        if not self.supports_module(module):
            raise ValueError(
                f"no trained action-model coverage for {module.object_label!r} -> "
                f"{module.support_label!r}; required domain: {self.manifest.object_domain}"
            )
        skill = V5Skill(skill)
        spec = self.manifest.models[skill]
        state = torch.as_tensor(
            state_observation, device=self.device, dtype=torch.float32
        )
        if state.ndim != 2 or state.shape[-1] != spec.policy_state_dim:
            raise ValueError(
                f"{skill.value} state observation must have shape "
                f"[N,{spec.policy_state_dim}], got {tuple(state.shape)}"
            )
        policy_observation = state
        if spec.object_context_version is not None:
            context = module.action_features(
                len(state), device=self.device, dtype=state.dtype
            )
            policy_observation = torch.cat((state, context), dim=-1)
        return self.action(skill, policy_observation)

    def action_for_token(self, token: SkillToken, observation: torch.Tensor) -> torch.Tensor:
        """Run the policy selected by a compiled DSL token."""
        try:
            module = ObjectModule(token.object_name, token.target_name)
        except ValueError:
            raise ValueError(
                f"no trained action-model coverage for {token.object_name!r} -> {token.target_name!r}; "
                f"required domain: {self.manifest.object_domain}"
            ) from None
        return self.action_for_module(module, token.skill, observation)


class ActionModelRouter:
    """Select action bundles by physical coverage instead of object identity."""

    def __init__(self, runtimes: Mapping[str, V5ActionModelRuntime]) -> None:
        if not runtimes:
            raise ValueError("at least one action-model runtime is required")
        if any(not name or name.strip() != name for name in runtimes):
            raise ValueError("action-model route names must be non-empty and trimmed")
        self.runtimes = dict(runtimes)

    def route(self, module: ObjectModule) -> tuple[str, V5ActionModelRuntime]:
        matches = [
            (name, runtime) for name, runtime in self.runtimes.items()
            if runtime.supports_module(module)
        ]
        if not matches:
            raise ValueError(
                f"no action-model bundle covers {module.object_label!r} -> "
                f"{module.support_label!r}"
            )
        if len(matches) > 1:
            raise ValueError(
                "ambiguous physical action-model coverage: "
                + ", ".join(name for name, _runtime in matches)
            )
        return matches[0]

    def supports_module(self, module: ObjectModule) -> bool:
        return sum(
            runtime.supports_module(module) for runtime in self.runtimes.values()
        ) == 1

    def action_for_module(
        self,
        module: ObjectModule,
        skill: V5Skill | str,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        _name, runtime = self.route(module)
        return runtime.action_for_module(module, skill, observation)


def default_action_model_manifest(root: Path) -> V5ActionModelManifest:
    """Return the currently selected validated rigid-cube action bundle."""
    return generalized_cube_action_model_manifest(root)


def irregular_cylinder_action_model_manifest(root: Path) -> V5ActionModelManifest:
    """Return the independently validated rigid-cylinder action-model bundle."""
    bundle = Path(root) / "outputs/v5_irregular_cylinder_bc_v3"
    observation_dims = {
        V5Skill.REACH: 52,
        **{skill: 55 for skill in V5Skill if skill != V5Skill.REACH},
    }
    models = {
        skill: SkillModelSpec(
            skill, bundle / skill.value.lower() / "policy.ts", observation_dims[skill]
        )
        for skill in V5Skill
    }
    cylinder_domain = ObjectDomain(
        object_classes=("generic",), support_classes=("generic",),
        object_geometries=("cylinder",), support_geometries=("cylinder",),
        object_grasp_modes=("parallel_jaw",), support_modes=("flat",),
        object_size_min_m=(0.05, 0.05, 0.04),
        object_size_max_m=(0.05, 0.05, 0.04),
        support_size_min_m=(0.05, 0.05, 0.04),
        support_size_max_m=(0.05, 0.05, 0.04),
        object_mass_range_kg=(0.08, 0.08),
        support_mass_range_kg=(0.08, 0.08),
    )
    return V5ActionModelManifest(
        models,
        source_mode="known_size_oracle_rigid_cylinder",
        object_domain=cylinder_domain,
        validated_object_examples=("training_cylinder_object", "training_cylinder_support"),
    )


def generalized_cube_action_model_manifest(root: Path) -> V5ActionModelManifest:
    """Return the role-conditioned rigid-cube bundle validated on a three-stack."""
    bundle = Path(root) / "outputs/v5_generalized_cube_bc_v1"
    observation_dims = {
        V5Skill.REACH: 52,
        **{skill: 55 for skill in V5Skill if skill != V5Skill.REACH},
    }
    models = {
        skill: SkillModelSpec(
            skill, bundle / skill.value.lower() / "policy.ts", observation_dims[skill]
        )
        for skill in V5Skill
    }
    return V5ActionModelManifest(
        models,
        source_mode="known_size_oracle_generalized_cube_v1",
        object_domain=rigid_cube_domain(),
        validated_object_examples=("red_cube", "blue_cube", "green_cube"),
    )
