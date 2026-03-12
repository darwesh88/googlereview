from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINES = ROOT / "loopy" / "experiment_baselines.json"
DEFAULT_LEDGER = ROOT / "loopy" / "experiment_ledger.jsonl"
DEFAULT_BATCH_ROOT = ROOT / "loopy" / "experiment_batches"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "batch"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def deep_format(value: Any, context: dict[str, str]) -> Any:
    if isinstance(value, str):
        return value.format_map(context)
    if isinstance(value, list):
        return [deep_format(item, context) for item in value]
    if isinstance(value, dict):
        return {key: deep_format(item, context) for key, item in value.items()}
    return value


def command_args(step: dict[str, Any], python_bin: str) -> list[str]:
    args = [python_bin, "-m", step["module"]]
    for key, value in step.get("params", {}).items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        if isinstance(value, list):
            for item in value:
                args.extend([flag, str(item)])
            continue
        args.extend([flag, str(value)])
    return args


def printable_command(args: list[str]) -> str:
    rendered: list[str] = []
    for arg in args:
        if " " in arg or '"' in arg:
            rendered.append(json.dumps(arg))
        else:
            rendered.append(arg)
    return " ".join(rendered)


def resolve_result_dir(step: dict[str, Any]) -> Path | None:
    output_dir = step.get("params", {}).get("output-dir")
    if not output_dir:
        return None
    return ROOT / str(output_dir)


def resolve_result_file(step: dict[str, Any]) -> Path | None:
    result_dir = resolve_result_dir(step)
    if result_dir is None:
        return None
    return result_dir / step.get("result_file", "best_metrics.json")


def compare_metric(actual: float, baseline: float, direction: str) -> tuple[str, float]:
    if direction == "lower":
        if actual < baseline:
            return "better", baseline - actual
        if actual > baseline:
            return "worse", actual - baseline
        return "tie", 0.0
    if actual > baseline:
        return "better", actual - baseline
    if actual < baseline:
        return "worse", baseline - actual
    return "tie", 0.0


def build_experiment(plan: dict[str, Any], experiment: dict[str, Any]) -> dict[str, Any]:
    templates = plan.get("step_templates", {})
    steps = experiment.get("steps")
    if steps is None:
        order = experiment.get("step_order") or list(templates.keys())
        steps = []
        for name in order:
            step = deepcopy(templates[name])
            step["name"] = name
            override = experiment.get("step_overrides", {}).get(name, {})
            step = merge_dicts(step, override)
            steps.append(step)

    context = {"experiment_id": experiment["id"]}
    for key, value in experiment.get("vars", {}).items():
        context[key] = str(value)

    expanded_steps = [deep_format(step, context) for step in steps]
    expanded_comparisons = deep_format(
        experiment.get("comparisons", plan.get("comparisons", [])),
        context,
    )

    return {
        "id": experiment["id"],
        "description": experiment.get("description", ""),
        "vars": experiment.get("vars", {}),
        "steps": expanded_steps,
        "comparisons": expanded_comparisons,
    }


def prepare_batch(plan_file: Path, batch_dir: Path | None) -> Path:
    plan = load_json(plan_file)
    batch_name = slugify(plan.get("name", plan_file.stem))
    if batch_dir is None:
        batch_dir = DEFAULT_BATCH_ROOT / f"{batch_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir = batch_dir / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    cwd = Path(plan.get("cwd", str(ROOT)))
    experiments: list[dict[str, Any]] = []
    command_lines = [f"# Batch: {plan.get('name', batch_name)}", "", f"- Created: `{utc_now()}`", f"- CWD: `{cwd}`", ""]

    for raw_experiment in plan.get("experiments", []):
        experiment = build_experiment(plan, raw_experiment)
        for step in experiment["steps"]:
            exec_args = command_args(step, sys.executable)
            local_args = command_args(step, "python")
            colab_args = command_args(step, "python")
            step["command_args"] = exec_args
            step["local_command"] = printable_command(local_args)
            step["colab_command"] = "!" + printable_command(colab_args)
        experiments.append(experiment)
        dump_json(experiments_dir / f"{experiment['id']}.json", experiment)

        command_lines.append(f"## {experiment['id']}")
        if experiment["description"]:
            command_lines.append(experiment["description"])
            command_lines.append("")
        for step in experiment["steps"]:
            command_lines.append(f"### {step['name']}")
            command_lines.append("Local:")
            command_lines.append("```bash")
            command_lines.append(step["local_command"])
            command_lines.append("```")
            command_lines.append("")
            command_lines.append("Colab:")
            command_lines.append("```python")
            command_lines.append(step["colab_command"])
            command_lines.append("```")
            command_lines.append("")

    batch = {
        "schema_version": 1,
        "created_at": utc_now(),
        "name": plan.get("name", batch_name),
        "description": plan.get("description", ""),
        "cwd": str(cwd),
        "plan_file": str(plan_file),
        "experiments": experiments,
    }
    dump_json(batch_dir / "batch.json", batch)
    (batch_dir / "commands.md").write_text("\n".join(command_lines) + "\n", encoding="utf-8")
    return batch_dir


def select_experiments(batch: dict[str, Any], pattern: str | None, limit: int | None) -> list[dict[str, Any]]:
    experiments = batch["experiments"]
    if pattern:
        regex = re.compile(pattern)
        experiments = [item for item in experiments if regex.search(item["id"])]
    if limit is not None:
        experiments = experiments[:limit]
    return experiments


def run_batch(batch_dir: Path, pattern: str | None, limit: int | None, continue_on_error: bool) -> int:
    batch = load_json(batch_dir / "batch.json")
    cwd = Path(batch["cwd"])
    failures = 0
    for experiment in select_experiments(batch, pattern, limit):
        for step in experiment["steps"]:
            args = step.get("command_args") or command_args(step, sys.executable)
            print(f"[{experiment['id']}] {step['name']}: {printable_command(args)}")
            completed = subprocess.run(args, cwd=cwd)
            if completed.returncode != 0:
                failures += 1
                if not continue_on_error:
                    return failures
                break
    return failures


def status_batch(batch_dir: Path) -> dict[str, Any]:
    batch = load_json(batch_dir / "batch.json")
    experiments = []
    for experiment in batch["experiments"]:
        steps = []
        for step in experiment["steps"]:
            result_file = resolve_result_file(step)
            steps.append(
                {
                    "name": step["name"],
                    "result_file": str(result_file) if result_file else "",
                    "complete": bool(result_file and result_file.exists()),
                }
            )
        experiments.append(
            {
                "id": experiment["id"],
                "complete_steps": sum(1 for step in steps if step["complete"]),
                "total_steps": len(steps),
                "steps": steps,
            }
        )
    return {"batch_dir": str(batch_dir), "name": batch["name"], "experiments": experiments}


def ingest_batch(batch_dir: Path, baselines_path: Path, ledger_path: Path) -> dict[str, Any]:
    batch = load_json(batch_dir / "batch.json")
    baselines = load_json(baselines_path)
    results = []
    for experiment in batch["experiments"]:
        step_metrics: dict[str, dict[str, Any]] = {}
        for step in experiment["steps"]:
            result_file = resolve_result_file(step)
            if result_file and result_file.exists():
                step_metrics[step["name"]] = load_json(result_file)

        comparisons = []
        for comparison in experiment.get("comparisons", []):
            step_name = comparison["step"]
            metric_name = comparison["metric"]
            baseline_key = comparison["baseline"]
            if step_name not in step_metrics or baseline_key not in baselines:
                continue
            actual = step_metrics[step_name].get(metric_name)
            baseline = baselines[baseline_key]["metrics"].get(metric_name)
            if actual is None or baseline is None:
                continue
            verdict, margin = compare_metric(float(actual), float(baseline), comparison["direction"])
            comparisons.append(
                {
                    "label": comparison["label"],
                    "metric": metric_name,
                    "actual": actual,
                    "baseline": baseline,
                    "baseline_key": baseline_key,
                    "direction": comparison["direction"],
                    "verdict": verdict,
                    "margin": margin,
                }
            )

        result = {
            "id": experiment["id"],
            "description": experiment.get("description", ""),
            "metrics": step_metrics,
            "comparisons": comparisons,
        }
        results.append(result)

    payload = {"batch_dir": str(batch_dir), "ingested_at": utc_now(), "results": results}
    dump_json(batch_dir / "results.json", payload)

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        for result in results:
            handle.write(
                json.dumps(
                    {
                        "ingested_at": payload["ingested_at"],
                        "batch_dir": str(batch_dir),
                        "experiment_id": result["id"],
                        "comparisons": result["comparisons"],
                    }
                )
                + "\n"
            )

    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare, run, and ingest Loopy experiment batches.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--plan-file", required=True)
    prepare_parser.add_argument("--batch-dir")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--batch-dir", required=True)
    run_parser.add_argument("--match")
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--continue-on-error", action="store_true")

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--batch-dir", required=True)

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--batch-dir", required=True)
    ingest_parser.add_argument("--baselines-file", default=str(DEFAULT_BASELINES))
    ingest_parser.add_argument("--ledger-file", default=str(DEFAULT_LEDGER))

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        batch_dir = prepare_batch(Path(args.plan_file), Path(args.batch_dir) if args.batch_dir else None)
        print(json.dumps({"batch_dir": str(batch_dir), "commands_file": str(batch_dir / "commands.md")}, indent=2))
        return
    if args.command == "run":
        failures = run_batch(Path(args.batch_dir), args.match, args.limit, args.continue_on_error)
        print(json.dumps({"batch_dir": args.batch_dir, "failures": failures}, indent=2))
        return
    if args.command == "status":
        print(json.dumps(status_batch(Path(args.batch_dir)), indent=2))
        return
    if args.command == "ingest":
        payload = ingest_batch(Path(args.batch_dir), Path(args.baselines_file), Path(args.ledger_file))
        print(json.dumps(payload, indent=2))
        return


if __name__ == "__main__":
    main()
