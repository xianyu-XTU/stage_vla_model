"""Summarize paired oracle/RGB-D multi-cube benchmark results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return centre - radius, centre + radius


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    rows = []
    raw = {}
    for height in (2, 3, 4):
        for mode in ("oracle", "vision"):
            path = root / f"{mode}_{height}cube.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw[(height, mode)] = payload
            successes = int(payload["chain_successes"])
            trials = int(payload["episodes"])
            low, high = wilson_interval(successes, trials)
            passed_seeds = [
                61080 + index
                for index, passed in enumerate(payload["seed_success"])
                if passed
            ]
            rows.append({
                "height": height,
                "mode": mode,
                "successes": successes,
                "trials": trials,
                "success_rate": successes / trials,
                "wilson_95": [low, high],
                "passed_seeds": passed_seeds,
                "failed_seeds": [seed for seed in range(61080, 61100) if seed not in passed_seeds],
                "raw_result": str(path),
            })

    summary = {
        "schema": "stage_vla_v7.stack_success_benchmark.v1",
        "seed_range": [61080, 61099],
        "trials_per_condition": 20,
        "layout_mode": "paired random-safe layout rows",
        "vision_scope": (
            "128x128 RGB-D compact color-depth object positions with temporal hold; "
            "oracle robot proprioception, object orientation, and physical terminal feedback"
        ),
        "action_scope": "same eight learned TorchScript skill policies in all conditions",
        "rows": rows,
        "paired_deltas_percentage_points": {
            str(height): 100.0 * (
                raw[(height, "vision")]["chain_successes"]
                - raw[(height, "oracle")]["chain_successes"]
            ) / 20.0
            for height in (2, 3, 4)
        },
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    table_lines = [
        "| 堆叠高度 | 不用视觉 | 用视觉 | 视觉差值 |",
        "|---:|---:|---:|---:|",
    ]
    for height in (2, 3, 4):
        oracle = int(raw[(height, "oracle")]["chain_successes"])
        vision = int(raw[(height, "vision")]["chain_successes"])
        table_lines.append(
            f"| {height} 方块 | {oracle}/20 ({oracle * 5}%) | "
            f"{vision}/20 ({vision * 5}%) | {(vision - oracle) * 5:+d} 个百分点 |"
        )
    markdown = "\n".join([
        "# Stage VLA V7：2/3/4 方块堆叠 20 种子验证",
        "",
        "## 最终成功率",
        "",
        *table_lines,
        "",
        "最终成功要求每一段 REACH→RETREAT 均完成，并通过无重置的物理稳定堆叠检查。",
        "原始 JSON 中 `status=failed` 表示严格的 20/20 整批门禁未通过；成功率以 `chain_successes` 为准。",
        "",
        "## 口径",
        "",
        "- 测试槽编号：61080–61099，共 20 个并行环境，配对使用同一随机安全布局。",
        "- 不用视觉：策略观察中的目标/支撑方块位置来自 Isaac Lab 仿真真值。",
        "- 用视觉：目标/支撑方块位置来自 128×128 RGB-D 四色几何检测；短时漏检仅保持上一帧，不回退真值。",
        "- 两组动作完全一致：同一套 8 段 TorchScript 策略；机器人本体状态、物体朝向及物理终态判定仍使用仿真真值，因此这是“视觉位置闭环”，不是全视觉状态闭环。",
        "- 2 方块表示两层塔；场景仍保留 Isaac Lab 必需的第三个干扰方块。",
        "",
        "## 逐阶段观察",
        "",
        "- 2 方块不用视觉：GRASP 起 19/20，之后保持到 RETREAT；视觉到 TRANSPORT 为 19/20，ALIGN 降为 16/20，DESCEND 后为 14/20。",
        "- 3 方块不用视觉：第一段 18/20，第二次 GRASP 后 10/20，最终稳定塔 10/20；视觉第一段最终 15/20，第二次 GRASP 后 7/20，最终稳定塔 3/20。",
        "- 4 方块不用视觉：19→9→5 个种子完成三段，最终稳定塔 4/20；视觉第一段在 ALIGN 从 12/20 降到 1/20，该种子最终完成四层塔。",
        "",
        "## 结论",
        "",
        "视觉位置误差对两层任务造成 25 个百分点损失，对三层造成 35 个百分点损失，对四层造成 15 个百分点损失。",
        "主要瓶颈是 ALIGN/DESCEND 的误差放大，以及多层任务中的第二次/第三次抓取；四层视觉还受黄色方块长时间遮挡影响。",
        "",
    ])
    (root / "REPORT.md").write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
