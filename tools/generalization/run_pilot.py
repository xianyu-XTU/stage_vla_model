"""Run the frozen 20-layout Phase 4 pilot through the existing evaluator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

from .analysis import SKILLS, aggregate_cases, extract_batch_cases, write_json
from .manifest import load_phase4_manifest, slice_manifest, write_manifest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = REPOSITORY_ROOT / "tools" / "eval_v7_multicube_chain.py"
CONFIG = REPOSITORY_ROOT / "config" / "evaluation" / "known_size_cube.json"
ARTIFACT_LOCK = REPOSITORY_ROOT / "config" / "artifacts.lock.json"
OBSERVER_CONFIG = REPOSITORY_ROOT / "config" / "simulation" / "observer_camera.json"
POLICY_DIRS = {
    "reach": "reach",
    "grasp": "grasp",
    "lift": "lift",
    "transport": "transport",
    "align": "align",
    "descend": "descend",
    "release": "release_stabilize",
    "retreat": "retreat",
}


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _evaluator_arguments(
    *,
    output: Path,
    layout: Path,
    num_envs: int,
    artifact_root: Path,
    calibration: Path,
    simulator_seed: int,
    video: Path | None = None,
    trace: bool = False,
) -> list[str]:
    arguments = [
        str(EVALUATOR),
        "--config", str(CONFIG),
        "--output", str(output),
        "--artifact_lock", str(ARTIFACT_LOCK),
        "--use_vision",
        "--vision_fail_policy", "strict",
        "--require_v7_chain",
        "--enable_cameras",
        "--camera_calibration", str(calibration),
        "--geometry_bias_m", "-0.0099", "0.00214", "0.0",
        "--num_envs", str(int(num_envs)),
        "--scene_cube_count", "3",
        "--asset_layout_file", str(layout),
        "--seed", str(int(simulator_seed)),
        "--reach_recovery_steps", "0",
        "--grasp_steps", "160",
        "--lift_steps", "500",
        "--transport_steps", "500",
        "--align_steps", "650",
        "--descend_steps", "400",
        "--release_steps", "300",
        "--retreat_steps", "500",
        "--lift_translation_limit_m", "0.005",
        "--transport_translation_limit_m", "0.005",
        "--align_translation_limit_m", "0.005",
        "--descend_translation_limit_m", "0.003",
        "--release_translation_limit_m", "0.003",
        "--retreat_translation_limit_m", "0.005",
        "--headless",
    ]
    for option, directory in POLICY_DIRS.items():
        arguments.extend([
            f"--{option}_checkpoint",
            str(artifact_root / directory / "policy.ts"),
        ])
    if trace:
        arguments.extend(["--trace_env", "0"])
    if video is not None:
        arguments.extend([
            "--video",
            "--video_path", str(video),
            "--video_env", "0",
            "--video_env_only",
            "--observer_camera_config", str(OBSERVER_CONFIG),
            "--recording_strict",
        ])
    return arguments


def _run_evaluator(
    *,
    isaac_python: Path,
    output: Path,
    log: Path,
    arguments: Sequence[str],
    resume: bool,
) -> dict[str, object]:
    if resume and output.is_file():
        return _read_json(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.pop("STAGE_VLA_V5_ROOT", None)
    environment["PYTHONPATH"] = ""
    environment["STAGE_VLA_COMMAND_UTF8_BASE64"] = (
        "5oqK57qi6Imy5pa55Z2X5pS+5Yiw6JOd6Imy5pa55Z2X5LiK"
    )
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            [str(isaac_python), *arguments],
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
            encoding="utf-8",
        )
    if output.is_file():
        return _read_json(output)
    failure = {
        "status": "failed",
        "failure": {
            "type": "EvaluatorProcessError",
            "message": f"evaluator exited {completed.returncode} without a result",
            "scope": "global",
            "taxonomy": "GLOBAL_RUNTIME_ERROR",
        },
        "v7_chain": {"verified": False, "required": True},
        "runtime_purity": {"verified": False},
        "provenance": {},
    }
    write_json(output, failure)
    return failure


def _inspect_video(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "valid": False}
    try:
        import cv2
    except ImportError:
        return {
            "path": str(path),
            "exists": True,
            "valid": path.stat().st_size > 0,
            "decode_check": "opencv_unavailable",
        }
    capture = cv2.VideoCapture(str(path))
    frames = nonblank = 0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames += 1
        if int(frame.max()) > int(frame.min()):
            nonblank += 1
    capture.release()
    return {
        "path": str(path),
        "exists": True,
        "frames": frames,
        "nonblank_frames": nonblank,
        "width": width,
        "height": height,
        "fps": fps,
        "valid": frames > 0 and frames == nonblank and width > 0 and height > 0,
    }


def _run_single(
    *,
    case: Mapping[str, object],
    master: Mapping[str, object],
    manifest_index: int,
    directory: Path,
    isaac_python: Path,
    artifact_root: Path,
    calibration: Path,
    simulator_seed: int,
    resume: bool,
    video_enabled: bool,
) -> tuple[dict[str, object], dict[str, object], Path, Path | None]:
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = write_manifest(
        directory / "layout.json", slice_manifest(master, (manifest_index,))
    )
    result_path = directory / "result.json"
    video_path = directory / "video.mp4" if video_enabled else None
    raw = _run_evaluator(
        isaac_python=isaac_python,
        output=result_path,
        log=directory / "evaluator.log",
        arguments=_evaluator_arguments(
            output=result_path,
            layout=manifest_path,
            num_envs=1,
            artifact_root=artifact_root,
            calibration=calibration,
            simulator_seed=simulator_seed,
            video=video_path,
            trace=True,
        ),
        resume=resume,
    )
    replay_case = extract_batch_cases(
        raw,
        _read_json(manifest_path),
        result_path=str(result_path),
    )[0]
    trace_payload = raw.get("trace", [])
    trace_json = write_json(directory / "trace.json", trace_payload)
    trace_path = trace_json if trace_payload else directory / "evaluator.log"
    return raw, replay_case, trace_path, video_path


def _percentage(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _render_report(
    *,
    aggregate: Mapping[str, object],
    manifest_path: Path,
    manifest: Mapping[str, object],
    batch_size: int,
    replays: Sequence[Mapping[str, object]],
    determinism: Sequence[Mapping[str, object]],
    source_commit: str,
    artifact_lock: object,
    checkpoint_hashes: object,
    infrastructure_pass: bool,
    ready_for_50: bool,
) -> str:
    cases = aggregate["cases"]
    failure_distribution = aggregate["first_failure_distribution"]
    failure_taxonomy = aggregate["failure_taxonomy"]
    funnel = aggregate["skill_funnel"]
    bottleneck_candidates = [
        (skill, values["conditional_success_rate"])
        for skill, values in funnel.items()
        if values["entered"] and values["conditional_success_rate"] is not None
    ]
    bottleneck = min(bottleneck_candidates, key=lambda item: item[1])[0]
    failures = [case for case in cases if not case["stable_success"]]
    replay_passes = sum(replay["single_result"] == "PASS" for replay in replays)
    replay_vision_failures = sum(
        replay["single_first_failure"] == "VISION" for replay in replays
    )
    replay_resolved_successes = int(aggregate["stable_success_count"]) + replay_passes
    lock_hash = (
        artifact_lock.get("sha256")
        if isinstance(artifact_lock, Mapping)
        else artifact_lock
    )
    lines = [
        "# Phase 4-A Random Layout Generalization Pilot",
        "",
        "## A. Experiment Configuration",
        "",
        f"- Source commit: `{source_commit}`",
        f"- Artifact lock SHA-256: `{lock_hash}`",
        f"- Layout manifest: `{manifest_path}`",
        f"- Layout seed: `{manifest['seed']}`",
        f"- Layouts: `{manifest['layout_count']}`",
        f"- Batch size: `{batch_size}`",
        "- Vision: strict RGB-D; oracle fallback disabled",
        "- Chain: all eight learned checkpoints; `--require_v7_chain` enabled",
        "- Reference Skills and recovery: disabled",
        "",
        "Checkpoint SHA-256 values:",
        "",
        "| Skill | SHA-256 |",
        "|---|---|",
    ]
    if isinstance(checkpoint_hashes, Mapping):
        for skill, record in sorted(checkpoint_hashes.items()):
            digest = record.get("sha256") if isinstance(record, Mapping) else record
            lines.append(f"| {skill} | `{digest}` |")
    lines.extend([
        "",
        "## B. Overall Results",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Batch fail-closed physical success | {aggregate['physical_success_count']} / {aggregate['layout_count']} ({_percentage(aggregate['physical_success_rate'])}) |",
        f"| Physical Wilson 95% CI | [{_percentage(aggregate['physical_success_wilson_95'][0])}, {_percentage(aggregate['physical_success_wilson_95'][1])}] |",
        f"| Batch fail-closed stable success | {aggregate['stable_success_count']} / {aggregate['layout_count']} ({_percentage(aggregate['stable_success_rate'])}) |",
        f"| Stable Wilson 95% CI | [{_percentage(aggregate['stable_success_wilson_95'][0])}, {_percentage(aggregate['stable_success_wilson_95'][1])}] |",
        f"| V7-chain verified | {aggregate['v7_chain_valid_count']} / {aggregate['layout_count']} |",
        f"| Vision valid | {aggregate['vision_valid_count']} / {aggregate['layout_count']} |",
        f"| Runtime purity valid | {aggregate['runtime_purity_valid_count']} / {aggregate['layout_count']} |",
        f"| Replay-resolved stable success | {replay_resolved_successes} / {aggregate['layout_count']} |",
        f"| Replay-resolved strict Vision failure | {replay_vision_failures} / {aggregate['layout_count']} |",
        "",
        "The replay-resolved rows replace only original failed cases with their",
        "single-environment outcome. They diagnose batch propagation; they are not",
        "a replacement for the frozen batch success estimate.",
        "",
        "## C. Skill Funnel",
        "",
        "| Skill | Entered | Success | Conditional success |",
        "|---|---:|---:|---:|",
    ])
    for skill in SKILLS:
        values = funnel[skill]
        rate = values["conditional_success_rate"]
        lines.append(
            f"| {skill} | {values['entered']} | {values['success']} | "
            f"{_percentage(rate) if rate is not None else 'n/a'} |"
        )
    lines.extend([
        "",
        "## D. First Failure",
        "",
        "| First failure | Count |",
        "|---|---:|",
    ])
    for name, count in failure_distribution.items():
        lines.append(f"| {name} | {count} |")
    lines.extend([
        "",
        "## E. Failure Taxonomy",
        "",
        "| Failure type | Count |",
        "|---|---:|",
    ])
    for name, count in failure_taxonomy.items():
        lines.append(f"| {name} | {count} |")
    lines.extend([
        "",
        "## F. Layout Table",
        "",
        "| Layout | Object XYZ | Support XYZ | XY distance | Result | First failure | Type |",
        "|---|---|---|---:|---|---|---|",
    ])
    for case in cases:
        geometry = case["geometry"]
        lines.append(
            f"| {case['layout_id']} | {case['object_xyz']} | {case['support_xyz']} | "
            f"{geometry['xy_distance_m']:.4f} | "
            f"{'PASS' if case['stable_success'] else 'FAIL'} | "
            f"{case['first_failure_skill'] or '-'} | {case['failure_type'] or '-'} |"
        )
    lines.extend([
        "",
        "## G. Replay Results",
        "",
        "| Layout | Batch | Single env | First failure | Reproduced | Video | Trace |",
        "|---|---|---|---|---|---|---|",
    ])
    for replay in replays:
        lines.append(
            f"| {replay['layout_id']} | {replay['batch_result']} | "
            f"{replay['single_result']} | {replay['single_first_failure']} | "
            f"{replay['reproduced']} | `{replay['video_path']}` | "
            f"`{replay['trace_path']}` |"
        )
    if not replays:
        lines.append("| - | - | - | - | - | - | - |")
    lines.extend([
        "",
        "Determinism probes:",
        "",
    ])
    for probe in determinism:
        lines.append(
            f"- `{probe['layout_id']}` ({probe['baseline_kind']}): "
            f"consistent={str(probe['consistent']).lower()}, outcomes={probe['outcomes']}"
        )
    lines.extend([
        "",
        "## Position-conditioned summary",
        "",
        "```json",
        json.dumps(aggregate["position_conditioned"], indent=2),
        "```",
        "",
        "## H. Conclusion",
        "",
        "- Overall pilot bottleneck: **strict Vision invalid detections and batch fail-closed propagation**.",
        f"- Lowest observed conditional Skill rate among completed telemetry: **{bottleneck}**.",
        f"- Runtime/Vision infrastructure: **{'PASS' if infrastructure_pass else 'FAIL'}**.",
        f"- Proceed to the frozen 50-case evaluation: **{'yes' if ready_for_50 else 'no'}**.",
        "",
        "The observed batch rate includes peer-triggered fail-closed aborts and is not an",
        "unbiased estimate of the policy-only success rate. The Wilson interval describes",
        "this measured batch outcome, not the model's exact true rate.",
        "No policy, threshold, reward, Vision fallback, or physical task definition was changed.",
        "",
        "## Acceptance Table",
        "",
        "```text",
        f"Source commit                       {source_commit}",
        f"Layout manifest                    {manifest_path}",
        f"Layout count                       {aggregate['layout_count']}",
        f"Batch size                         {batch_size}",
        f"Strict Vision policy               {'PASS' if aggregate['strict_vision_count'] == aggregate['layout_count'] else 'FAIL'}",
        f"Vision-valid cases                 {aggregate['vision_valid_count']} / {aggregate['layout_count']}",
        f"Runtime purity                     {'PASS' if aggregate['runtime_purity_valid_count'] == aggregate['layout_count'] else 'FAIL'}",
        f"Oracle fallback                    {aggregate['oracle_fallback_count']}",
        f"Reference calls                    {aggregate['reference_skill_calls']}",
        f"Recovery calls                     {aggregate['recovery_calls']}",
        f"Evaluated cases                    {aggregate['evaluated_cases']} / {aggregate['layout_count']}",
        f"Physical success                   {aggregate['physical_success_count']} / {aggregate['layout_count']}",
        f"Physical success rate              {_percentage(aggregate['physical_success_rate'])}",
        f"Stable success                     {aggregate['stable_success_count']} / {aggregate['layout_count']}",
        f"Stable success rate                {_percentage(aggregate['stable_success_rate'])}",
        f"95% Wilson CI                      [{_percentage(aggregate['stable_success_wilson_95'][0])}, {_percentage(aggregate['stable_success_wilson_95'][1])}]",
    ])
    for skill in SKILLS:
        values = funnel[skill]
        lines.append(
            f"{skill + ' success':35s} {values['success']} / {values['entered']}"
        )
    lines.extend(["", "First failure:"])
    for name in (*SKILLS, "VISION", "FINAL_STABILITY", "UNKNOWN"):
        lines.append(f"{name:35s} {failure_distribution[name]}")
    lines.append(f"{'RUNTIME_ERROR':35s} {failure_distribution['RUNTIME_ERROR']}")
    reproduced = sum(bool(replay["reproduced"]) for replay in replays)
    replay_failures = sum(replay["single_result"] == "FAIL" for replay in replays)
    valid_videos = sum(bool(replay["video"]["valid"]) for replay in replays)
    lines.extend([
        "",
        f"Failed cases replayed              {len(replays)} / {len(failures)}",
        f"Single-env replay failures         {replay_failures} / {len(failures)}",
        f"Failures reproduced                {reproduced} / {len(failures)}",
        f"Replay videos valid                {valid_videos} / {len(replays)}",
        f"Evaluation infrastructure          {'PASS' if infrastructure_pass else 'FAIL'}",
        f"READY_FOR_PHASE4_50                {str(ready_for_50).lower()}",
        "```",
        "",
    ])
    return "\n".join(lines)


def _render_v2_report(
    *,
    aggregate: Mapping[str, object],
    manifest_path: Path,
    manifest: Mapping[str, object],
    batch_size: int,
    replays: Sequence[Mapping[str, object]],
    source_commit: str,
    source_clean_before_run: bool,
    checkpoint_hashes: object,
    v1_aggregate: Mapping[str, object] | None,
    infrastructure_pass: bool,
    ready_for_50: bool,
) -> str:
    funnel = aggregate["skill_funnel"]
    first = aggregate["first_failure_distribution"]
    outcomes = aggregate["outcome_taxonomy"]
    cases = aggregate["cases"]
    failures = [case for case in cases if not case["stable_success"]]
    matches = sum(bool(row["reproduced"]) for row in replays)

    def rate(value: object) -> str:
        return "n/a" if value is None else _percentage(float(value))

    checkpoint_summary = ", ".join(
        f"{skill}={record.get('sha256')}"
        for skill, record in sorted(checkpoint_hashes.items())
        if isinstance(record, Mapping)
    ) if isinstance(checkpoint_hashes, Mapping) else str(checkpoint_hashes)

    lines = [
        "# Phase 4-A Random Layout Generalization Pilot V2", "",
        "## Configuration", "",
        f"- Source commit: `{source_commit}`",
        f"- Source worktree clean before run: `{str(source_clean_before_run).lower()}`",
        f"- Layout manifest: `{manifest_path}`",
        f"- Manifest seed: `{manifest['seed']}`; layouts: `{manifest['layout_count']}`",
        f"- Batch size: `{batch_size}`",
        "- Strict RGB-D Vision; zero oracle, reference, recovery, and Vision retries",
        "- Eight frozen learned checkpoints; unchanged cameras, physics, thresholds, and task", "",
        f"Checkpoint SHA-256: `{checkpoint_summary}`", "",
        "## Results", "",
        "| Metric | V2 |", "|---|---:|",
        f"| Cases independently accounted | {aggregate['evaluated_cases']} / {aggregate['layout_count']} |",
        f"| Physical success | {aggregate['physical_success_count']} / {aggregate['layout_count']} |",
        f"| Stable success | {aggregate['stable_success_count']} / {aggregate['layout_count']} ({_percentage(aggregate['stable_success_rate'])}) |",
        f"| Stable Wilson 95% CI | [{_percentage(aggregate['stable_success_wilson_95'][0])}, {_percentage(aggregate['stable_success_wilson_95'][1])}] |",
        f"| Vision-valid cases | {aggregate['vision_valid_count']} / {aggregate['layout_count']} |",
        f"| Runtime-pure cases | {aggregate['runtime_purity_valid_count']} / {aggregate['layout_count']} |",
        f"| Peer-aborted cases | {aggregate['peer_aborted_due_to_other_env_failure']} |",
        f"| VISION_FAILURE | {outcomes['VISION_FAILURE']} |",
        f"| RUNTIME_ERROR | {outcomes['RUNTIME_ERROR']} |",
        f"| GLOBAL_RUNTIME_ERROR | {outcomes['GLOBAL_RUNTIME_ERROR']} |", "",
        "## V1 vs V2", "", "| Metric | V1 | V2 |", "|---|---:|---:|",
    ]
    if v1_aggregate is not None:
        v1_cases = v1_aggregate.get("cases", ())
        v1_replays = v1_aggregate.get("replays", ())
        v1_peer = sum(
            isinstance(row, Mapping)
            and row.get("failure_reason") == "batch_aborted_by_peer_strict_vision_failure"
            for row in v1_cases
        )
        v1_matches = sum(
            bool(row.get("reproduced")) for row in v1_replays
            if isinstance(row, Mapping)
        )
        comparison = (
            ("Stable success", v1_aggregate["stable_success_count"], aggregate["stable_success_count"]),
            ("Vision failures", v1_aggregate["first_failure_distribution"]["VISION"], first["VISION"]),
            ("RUNTIME_ERROR", v1_aggregate["first_failure_distribution"]["RUNTIME_ERROR"], first["RUNTIME_ERROR"]),
            ("Peer aborts", v1_peer, aggregate["peer_aborted_due_to_other_env_failure"]),
            ("Batch/replay agreement", f"{v1_matches}/{len(v1_replays)}", f"{matches}/{len(replays)}"),
            ("ALIGN conditional success", rate(v1_aggregate["skill_funnel"]["ALIGN"]["conditional_success_rate"]), rate(funnel["ALIGN"]["conditional_success_rate"])),
            ("DESCEND conditional success", rate(v1_aggregate["skill_funnel"]["DESCEND"]["conditional_success_rate"]), rate(funnel["DESCEND"]["conditional_success_rate"])),
        )
        lines.extend(f"| {name} | {old} | {new} |" for name, old, new in comparison)
    else:
        lines.append("| V1 aggregate unavailable | n/a | n/a |")
    lines.extend(["", "## Skill Funnel", "", "| Skill | Entered | Success | Conditional success |", "|---|---:|---:|---:|"])
    for skill in SKILLS:
        row = funnel[skill]
        lines.append(f"| {skill} | {row['entered']} | {row['success']} | {rate(row['conditional_success_rate'])} |")
    lines.extend(["", "## Failures", "", "| First failure | Count |", "|---|---:|"])
    lines.extend(f"| {name} | {count} |" for name, count in first.items())
    lines.extend(["", "| Outcome | Count |", "|---|---:|"])
    lines.extend(f"| {name} | {count} |" for name, count in outcomes.items())
    lines.extend(["", "## Layout Outcomes", "", "| Layout | Result | First failure | Taxonomy |", "|---|---|---|---|"])
    for case in cases:
        lines.append(f"| {case['layout_id']} | {'PASS' if case['stable_success'] else 'FAIL'} | {case['first_failure_skill'] or '-'} | {case['outcome']} |")
    lines.extend(["", "## Replays", "", "| Layout | Batch | Replay | Batch first | Replay first | Match | Video valid |", "|---|---|---|---|---|---|---|"])
    for row in replays:
        lines.append(f"| {row['layout_id']} | {row['batch_result']} | {row['single_result']} | {row['batch_first_failure']} | {row['single_first_failure']} | {row['reproduced']} | {row['video']['valid']} |")
    lines.extend([
        "", "## Acceptance", "", "```text",
        f"Source commit                      {source_commit}",
        f"Source clean before Pilot          {str(source_clean_before_run).lower()}",
        f"Phase 3 runtime purity             {'PASS' if aggregate['runtime_purity_valid_count'] == aggregate['layout_count'] else 'FAIL'}",
        f"Strict Vision                      {'PASS' if aggregate['strict_vision_count'] == aggregate['layout_count'] else 'FAIL'}",
        f"Loaded V5 modules                 {max(int(case['loaded_v5_module_count'] or 0) for case in cases)}",
        f"Vendor path exposed               {any(case['vendor_path_exposed'] is True for case in cases)}",
        f"Per-env Vision isolation          {'PASS' if aggregate['peer_aborted_due_to_other_env_failure'] == 0 else 'FAIL'}",
        f"Peer-abort count                  {aggregate['peer_aborted_due_to_other_env_failure']}",
        f"Oracle fallback                   {aggregate['oracle_fallback_count']}",
        f"Reference calls                   {aggregate['reference_skill_calls']}",
        f"Recovery calls                    {aggregate['recovery_calls']}",
        f"Evaluated cases                   {aggregate['evaluated_cases']} / {aggregate['layout_count']}",
        f"Failed cases replayed             {len(replays)} / {len(failures)}",
        f"Batch/replay matched              {matches} / {len(replays)}",
        f"Evaluation infrastructure         {'PASS' if infrastructure_pass else 'FAIL'}",
        f"READY_FOR_PHASE4_50               {str(ready_for_50).lower()}", "```", "",
    ])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    master_path = args.manifest.resolve()
    master = load_phase4_manifest(master_path)
    if int(master["layout_count"]) != 20:
        raise ValueError("Phase 4-A requires exactly 20 frozen layouts")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    isaac_python = args.isaac_python.resolve()
    artifact_root = args.artifact_root.resolve()
    calibration = args.calibration.resolve()
    required = [isaac_python, calibration, ARTIFACT_LOCK, CONFIG, EVALUATOR]
    required.extend(artifact_root / directory / "policy.ts" for directory in POLICY_DIRS.values())
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Phase 4 dependencies: {missing}")

    all_cases: list[dict[str, object]] = []
    batch_records: list[dict[str, object]] = []
    count = int(master["layout_count"])
    for batch_number, start in enumerate(range(0, count, args.batch_size), start=1):
        indices = tuple(range(start, min(start + args.batch_size, count)))
        batch_dir = output_root / "batches" / f"batch_{batch_number:03d}"
        batch_manifest = slice_manifest(master, indices)
        manifest_path = write_manifest(batch_dir / "layout.json", batch_manifest)
        result_path = batch_dir / "result.json"
        result = _run_evaluator(
            isaac_python=isaac_python,
            output=result_path,
            log=batch_dir / "evaluator.log",
            arguments=_evaluator_arguments(
                output=result_path,
                layout=manifest_path,
                num_envs=len(indices),
                artifact_root=artifact_root,
                calibration=calibration,
                simulator_seed=args.simulator_seed,
            ),
            resume=args.resume,
        )
        cases = extract_batch_cases(
            result,
            batch_manifest,
            result_path=str(result_path),
        )
        all_cases.extend(cases)
        batch_records.append({
            "batch": batch_number,
            "layout_ids": batch_manifest["layout_ids"],
            "manifest": str(manifest_path),
            "result": str(result_path),
            "status": result.get("status"),
            "source_commit": result.get("provenance", {}).get("source_commit"),
            "source_clean_before_run": result.get("provenance", {}).get(
                "source_worktree_clean_before_run"
            ),
        })
        print(f"batch {batch_number}: {sum(case['stable_success'] for case in cases)}/{len(cases)}")

    aggregate = aggregate_cases(all_cases)
    aggregate["manifest"] = str(master_path)
    aggregate["manifest_seed"] = master["seed"]
    aggregate["batch_size"] = args.batch_size
    aggregate["batches"] = batch_records
    write_json(output_root / "phase4_pilot_v2_aggregate.json", aggregate)
    failures = [case for case in all_cases if not case["stable_success"]]
    write_json(output_root / "phase4_pilot_v2_failures.json", failures)

    index_by_id = {
        str(layout_id): index for index, layout_id in enumerate(master["layout_ids"])
    }
    replay_records: list[dict[str, object]] = []
    replay_cases: dict[str, dict[str, object]] = {}
    for case in failures:
        layout_id = str(case["layout_id"])
        directory = output_root / "replays" / layout_id
        _raw, replay_case, trace_path, video_path = _run_single(
            case=case,
            master=master,
            manifest_index=index_by_id[layout_id],
            directory=directory,
            isaac_python=isaac_python,
            artifact_root=artifact_root,
            calibration=calibration,
            simulator_seed=args.simulator_seed,
            resume=args.resume,
            video_enabled=True,
        )
        replay_cases[layout_id] = replay_case
        reproduced = (
            not bool(replay_case["stable_success"])
            and replay_case["first_failure_skill"] == case["first_failure_skill"]
        )
        video_info = _inspect_video(video_path) if video_path is not None else None
        replay_records.append({
            "layout_id": layout_id,
            "batch_result": "PASS" if case["stable_success"] else "FAIL",
            "batch_first_failure": case["first_failure_skill"],
            "single_result": "PASS" if replay_case["stable_success"] else "FAIL",
            "single_first_failure": replay_case["first_failure_skill"],
            "reproduced": reproduced,
            "classification": (
                "REPRODUCED" if reproduced else "NONDETERMINISTIC_OR_BATCH_EFFECT"
            ),
            "result_path": replay_case["evaluation_result_path"],
            "video_path": str(video_path),
            "video": video_info,
            "trace_path": str(trace_path),
        })
        print(f"replay {layout_id}: reproduced={reproduced}")
    write_json(output_root / "phase4_pilot_v2_replays.json", replay_records)

    determinism_records: list[dict[str, object]] = []
    success_case = next((case for case in all_cases if case["stable_success"]), None)
    reproduced_failure_ids = {
        str(record["layout_id"])
        for record in replay_records
        if record["reproduced"]
    }
    failure_case = next(
        (
            case for case in failures
            if str(case["layout_id"]) in reproduced_failure_ids
        ),
        failures[0] if failures else None,
    )
    for baseline_kind, case in (("success", success_case), ("failure", failure_case)):
        if case is None:
            continue
        layout_id = str(case["layout_id"])
        baseline = (bool(case["stable_success"]), case["first_failure_skill"])
        outcomes = [{"source": "batch", "stable_success": baseline[0], "first_failure": baseline[1]}]
        if baseline_kind == "failure" and layout_id in replay_cases:
            replay = replay_cases[layout_id]
            outcomes.append({
                "source": "failure_replay",
                "stable_success": bool(replay["stable_success"]),
                "first_failure": replay["first_failure_skill"],
            })
            repeat_start = 2
        else:
            repeat_start = 1
        for repeat in range(repeat_start, 3):
            directory = output_root / "determinism" / layout_id / f"repeat_{repeat:03d}"
            _raw, repeated_case, _trace, _video = _run_single(
                case=case,
                master=master,
                manifest_index=index_by_id[layout_id],
                directory=directory,
                isaac_python=isaac_python,
                artifact_root=artifact_root,
                calibration=calibration,
                simulator_seed=args.simulator_seed,
                resume=args.resume,
                video_enabled=False,
            )
            outcomes.append({
                "source": f"repeat_{repeat:03d}",
                "stable_success": bool(repeated_case["stable_success"]),
                "first_failure": repeated_case["first_failure_skill"],
            })
        consistent = all(
            (item["stable_success"], item["first_failure"]) == baseline
            for item in outcomes[1:]
        )
        determinism_records.append({
            "layout_id": layout_id,
            "baseline_kind": baseline_kind,
            "consistent": consistent,
            "outcomes": outcomes,
        })
        print(f"determinism {layout_id}: consistent={consistent}")
    write_json(output_root / "phase4_pilot_v2_determinism.json", determinism_records)

    all_strict = all(bool(case["strict_vision"]) for case in all_cases)
    all_pure = int(aggregate["runtime_purity_valid_count"]) == count
    all_source_clean = all(
        record["source_clean_before_run"] is True for record in batch_records
    )
    one_source_commit = len({record["source_commit"] for record in batch_records}) == 1
    no_forbidden = all(
        int(aggregate[name]) == 0
        for name in ("oracle_fallback_count", "reference_skill_calls", "recovery_calls")
    )
    all_replayed = len(replay_records) == len(failures)
    all_reproduced = all(bool(record["reproduced"]) for record in replay_records)
    deterministic = all(bool(record["consistent"]) for record in determinism_records)
    videos_valid = all(bool(record["video"]["valid"]) for record in replay_records)
    peer_abort_free = int(
        aggregate["peer_aborted_due_to_other_env_failure"]
    ) == 0
    no_global_errors = int(
        aggregate["outcome_taxonomy"]["GLOBAL_RUNTIME_ERROR"]
    ) == 0
    no_runtime_errors = int(aggregate["outcome_taxonomy"]["RUNTIME_ERROR"]) == 0
    infrastructure_pass = all(
        (
            len(all_cases) == count,
            all_strict,
            all_pure,
            all_source_clean,
            one_source_commit,
            no_forbidden,
            peer_abort_free,
            no_global_errors,
            no_runtime_errors,
        )
    )
    ready_for_50 = all(
        (
            infrastructure_pass,
            all_replayed,
            all_reproduced,
            deterministic,
            videos_valid,
        )
    )
    aggregate["replays"] = replay_records
    aggregate["replay_resolved_stable_success_count"] = (
        int(aggregate["stable_success_count"])
        + sum(record["single_result"] == "PASS" for record in replay_records)
    )
    aggregate["replay_resolved_vision_failure_count"] = sum(
        record["single_first_failure"] == "VISION" for record in replay_records
    )
    aggregate["determinism"] = determinism_records
    aggregate["evaluation_infrastructure_pass"] = infrastructure_pass
    aggregate["ready_for_phase4_50"] = ready_for_50
    aggregate["source_clean_before_run"] = all_source_clean
    aggregate["source_commit_consistent"] = one_source_commit
    write_json(output_root / "phase4_pilot_v2_aggregate.json", aggregate)

    first_result = _read_json(Path(batch_records[0]["result"]))
    provenance = first_result.get("provenance", {})
    source_commit = str(provenance.get("source_commit", "UNKNOWN"))
    source_clean_before_run = provenance.get("source_worktree_clean_before_run") is True
    v1_aggregate = (
        _read_json(args.v1_aggregate.resolve())
        if args.v1_aggregate is not None else None
    )
    report = _render_v2_report(
        aggregate=aggregate,
        manifest_path=master_path,
        manifest=master,
        batch_size=args.batch_size,
        replays=replay_records,
        source_commit=source_commit,
        source_clean_before_run=source_clean_before_run,
        checkpoint_hashes=provenance.get("checkpoint_hashes"),
        v1_aggregate=v1_aggregate,
        infrastructure_pass=infrastructure_pass,
        ready_for_50=ready_for_50,
    )
    (output_root / "PHASE4_PILOT_V2_REPORT.md").write_text(report, encoding="utf-8")
    print(output_root / "PHASE4_PILOT_V2_REPORT.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--isaac-python",
        type=Path,
        default=Path(r"E:\work\IsaacLab\_isaac_sim\python.bat"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(r"E:\stage_vla_v5\outputs\v5_generalized_cube_bc_v1"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path(
            r"E:\stage_vla_v5\outputs\vision_rgbd_mapping_calibration_train_20260910.json"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--simulator-seed", type=int, default=61081)
    parser.add_argument("--v1-aggregate", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size != 4:
        parser.error("Phase 4 Pilot V2 requires batch size 4")
    run(args)


if __name__ == "__main__":
    main()
