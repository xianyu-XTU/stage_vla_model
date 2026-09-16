"""M27: only train RELEASE reward; frozen RETREAT and score untouched."""
from dataclasses import asdict

from .stageppo_core import Skill
from .stageppo_scaled_vecenv import ScaledStagePPOVecEnv
from .stageppo_release_quality import ReleaseQuality, release_quality_cost


class ReleaseQualityVecEnv(ScaledStagePPOVecEnv):
    def __init__(self, *args, quality_enabled=True, **kwargs):
        super().__init__(*args, **kwargs)
        if self.initial_skill != Skill.RELEASE_STABILIZE:
            raise ValueError("this training wrapper is release-only")
        self.quality_weights = ReleaseQuality()
        self.quality_weights.validate(self.task_cfg)
        self.quality_enabled = quality_enabled

    def step(self, actions):
        if not self.quality_enabled or self.chain or not self.auto_reset:
            return super().step(actions)
        self.auto_reset = False
        try:
            active = ~self.finished
            _, reward, done, extras = super().step(actions)
            cost = release_quality_cost(self.measured, self.last_success, self.task_cfg, self.quality_weights) * active
            reward = reward - cost
            self.returns -= cost
            self.total_steps += self.num_envs
            ids = done.nonzero(as_tuple=False).flatten()
            self.completed += len(ids)
            self.successes += int(self.last_success.sum())
            self.failures += int(self.last_fail.sum())
            self.timeouts += int(self.last_timeout.sum())
            extras["log"].update({"stageppo/reward": reward.mean(), "stageppo/release_quality_cost": cost.mean()})
            if len(ids):
                extras["log"].update({"stageppo/episode_return": self.returns[ids].mean(),
                                      "stageppo/episode_success": self.last_success[ids].float().mean()})
                self._restore(ids)
                self.measured = self._measure()
            return self._obs(), reward, done, extras
        finally:
            self.auto_reset = True

    def statistics(self):
        return {**super().statistics(), "release_quality_enabled": self.quality_enabled,
                "release_quality_weights": asdict(self.quality_weights)}
