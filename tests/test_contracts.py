from __future__ import annotations

import pytest

from stage_vla_v7.contracts import ContractError, StackRelation, TaskPlan, expand_skill_tokens


def test_stack_relations_are_ordered_from_bottom_up() -> None:
    plan = TaskPlan(
        (
            StackRelation("red_cube", "blue_cube"),
            StackRelation("blue_cube", "green_cube"),
        )
    )
    assert plan.execution_relations == (
        StackRelation("blue_cube", "green_cube"),
        StackRelation("red_cube", "blue_cube"),
    )
    tokens = expand_skill_tokens(plan)
    assert len(tokens) == 16
    assert tokens[0].object_label == "blue_cube"
    assert tokens[8].object_label == "red_cube"


def test_stack_cycle_is_rejected() -> None:
    with pytest.raises(ContractError, match="cycle"):
        TaskPlan(
            (
                StackRelation("red_cube", "blue_cube"),
                StackRelation("blue_cube", "red_cube"),
            )
        )
