"""Training-only goal cost; physical actions, terminals and evaluation unchanged."""
from dataclasses import asdict

from .stageppo_scaled_vecenv import ScaledStagePPOVecEnv
from .stageppo_retreat_cost import RetreatGoalCost, retreat_goal_cost


class GoalCostStagePPOVecEnv(ScaledStagePPOVecEnv):
    def __init__(self, *args, goal_cost_enabled=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.goal_cost = RetreatGoalCost()
        self.goal_cost.validate(self.task_cfg)
        self.goal_cost_enabled = goal_cost_enabled

    def step(self, actions):
        if not self.goal_cost_enabled or self.chain or not self.auto_reset:
            return super().step(actions)
        # Temporarily defer only the *training* reset until the cost is computed
        # from the true post-action state, never the next episode's snapshot.
        self.auto_reset = False
        try:
            active = ~self.finished
            _, reward, done, extras = super().step(actions)
            cost = retreat_goal_cost(self.measured, self.task_cfg, self.goal_cost) * active
            reward = reward - cost
            self.returns -= cost
            self.total_steps += self.num_envs
            ids = done.nonzero(as_tuple=False).flatten()
            self.completed += len(ids)
            self.successes += int(self.last_success.sum())
            self.failures += int(self.last_fail.sum())
            self.timeouts += int(self.last_timeout.sum())
            extras["log"].update({"stageppo/reward": reward.mean(), "stageppo/goal_cost": cost.mean()})
            if len(ids):
                extras["log"].update({"stageppo/episode_return": self.returns[ids].mean(),
                                      "stageppo/episode_success": self.last_success[ids].float().mean()})
                self._restore(ids)
                self.measured = self._measure()
            return self._obs(), reward, done, extras
        finally:
            self.auto_reset = True

    def statistics(self):
        return {**super().statistics(), "training_goal_cost": asdict(self.goal_cost),
                "training_goal_cost_enabled": self.goal_cost_enabled}
