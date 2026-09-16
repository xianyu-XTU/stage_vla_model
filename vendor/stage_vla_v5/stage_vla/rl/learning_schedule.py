"""Pure helpers for RSL-RL incremental learning budgets.

RSL-RL OnPolicyRunner.learn(num_learning_iterations=N) interprets N as the
number of iterations to execute from the runner's current_learning_iteration.
It internally computes total_it = start_it + N.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LearningSchedule:
    """Resolved incremental learning schedule matching RSL-RL semantics."""

    start_iteration: int
    num_learning_iterations: int
    total_iteration_exclusive: int

    @property
    def update_count(self) -> int:
        return self.total_iteration_exclusive - self.start_iteration

    @property
    def last_iteration(self) -> int:
        """Last iteration index executed by RSL-RL for this schedule."""
        return self.total_iteration_exclusive - 1


def resolve_learning_schedule(
    *,
    current_learning_iteration: int,
    requested_iterations: int,
) -> LearningSchedule:
    """Return the exact incremental count to pass to ``OnPolicyRunner.learn``.

    ``requested_iterations`` is always an *incremental* execution budget.  For a
    fresh runner ``current_learning_iteration`` is normally 0.  For a resumed
    runner it is restored by ``runner.load()``.  RSL-RL itself adds the current
    iteration to ``num_learning_iterations``; callers must therefore NOT add it
    a second time.
    """

    current = int(current_learning_iteration)
    requested = int(requested_iterations)
    if current < 0:
        raise ValueError("current_learning_iteration must be >= 0")
    if requested <= 0:
        raise ValueError("requested_iterations must be > 0")
    return LearningSchedule(
        start_iteration=current,
        num_learning_iterations=requested,
        total_iteration_exclusive=current + requested,
    )
