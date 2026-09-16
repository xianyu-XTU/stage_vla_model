from __future__ import annotations

from stage_vla_v7.language import DeterministicLanguageProvider, LanguageRequest, LanguageService


def test_chinese_instruction_is_structured_without_actions() -> None:
    service = LanguageService(DeterministicLanguageProvider())
    result = service.interpret(
        LanguageRequest(
            "把红色方块放到蓝色方块上",
            ("red_cube", "blue_cube"),
        )
    )
    relation = result.plan.execution_relations[0]
    assert (relation.object_label, relation.support_label) == ("red_cube", "blue_cube")
    assert not hasattr(result, "action")


def test_multi_relation_language_orders_dependencies() -> None:
    provider = DeterministicLanguageProvider()
    result = provider.interpret(
        LanguageRequest(
            "先把红方块放到蓝方块上，然后把蓝方块放到绿方块上",
            ("red_cube", "blue_cube", "green_cube"),
        )
    )
    assert [relation.object_label for relation in result.plan.execution_relations] == [
        "blue_cube",
        "red_cube",
    ]


def test_strict_chain_dsl_is_supported() -> None:
    provider = DeterministicLanguageProvider()
    result = provider.interpret(
        LanguageRequest(
            "STACK_CHAIN(cube_1>cube_2,cube_2>cube_3)",
            ("cube_1", "cube_2", "cube_3"),
        )
    )
    assert result.plan.execution_relations[0].object_label == "cube_2"


def test_strict_single_relation_dsl_preserves_argument_comma() -> None:
    provider = DeterministicLanguageProvider()
    result = provider.interpret(
        LanguageRequest(
            "STACK(object=red_cube,target=blue_cube)",
            ("red_cube", "blue_cube"),
        )
    )
    relation = result.plan.execution_relations[0]
    assert (relation.object_label, relation.support_label) == ("red_cube", "blue_cube")
