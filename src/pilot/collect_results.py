"""Collect pilot diagnostics without choosing an architecture/LR or claiming final results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import fingerprint
from .runtime import is_strict_p0_evidence, write_json
from ..preprocessing.geoguessr_v1 import sha256_file


def collect(runs_dir):
    rows = []
    for path in sorted(Path(runs_dir).rglob("run_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("purpose") != "pilot" or summary.get("stage") not in {"p0", "p1"}:
            raise ValueError(f"Not a pilot summary: {path}")
        folder = path.parent
        cfg_path = folder / "config_resolved.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        env_path = folder / "environment.json"
        env = json.loads(env_path.read_text(encoding="utf-8")) if env_path.exists() else None
        rows.append({"run_dir": str(folder.resolve()), "summary_sha256": sha256_file(path),
                     "config_sha256": fingerprint(cfg), "config": cfg, "environment": env,
                     "summary": summary})
    return rows


def export_reviewed_recipe(source, recipe_id, rows, output_root):
    """Explicit user-supplied numeric contract only; never infer values from scores."""
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]+", recipe_id):
        raise ValueError("Unsafe recipe ID")
    recipe = json.loads(Path(source).read_text(encoding="utf-8"))
    required = {"source_runs", "seeds", "lr_candidates", "runtime", "optimizer", "scheduler",
                "batch", "augmentation", "initialization", "e1hr_resolution_wh", "checkpoint",
                "numeric_value_rationales"}
    if required - recipe.keys():
        raise ValueError(f"Reviewed training recipe missing keys: {sorted(required - recipe.keys())}")
    def check(value):
        if value is None:
            raise ValueError("Reviewed recipe cannot contain unresolved null values")
        if isinstance(value, dict):
            for v in value.values():
                check(v)
        elif isinstance(value, list):
            for v in value:
                check(v)
    check(recipe)
    for field in required - {"source_runs", "lr_candidates", "e1hr_resolution_wh"}:
        if not isinstance(recipe[field], dict) or not recipe[field]:
            raise ValueError(f"Nonempty reviewed object required: {field}")
    grid = recipe["lr_candidates"]
    if len(grid) != 4 or len(set(grid)) != 4 or any(type(v) not in {int, float} or v <= 0 for v in grid):
        raise ValueError("Exactly four reviewed positive base LR values required")
    seeds = recipe["seeds"]
    if type(seeds.get("tuning")) is not int or len(seeds.get("final", [])) != 3:
        raise ValueError("One tuning seed and three final seed IDs required")
    if len(set(seeds["final"])) != 3 or any(type(v) is not int or v < 0 for v in [seeds["tuning"], *seeds["final"]]):
        raise ValueError("Invalid seed IDs")
    selected = []
    for run_dir in recipe["source_runs"]:
        row = next((r for r in rows if Path(r["run_dir"]) == Path(run_dir).resolve()), None)
        if row is None or row["summary"]["status"] not in {"pass", "complete"}:
            raise ValueError("Recipe must reference successful collected pilot runs")
        if row["summary"]["stage"] == "p0":
            if (not is_strict_p0_evidence(row["summary"]) or row["summary"]["status"] != "pass"
                    or row["summary"].get("full_epoch_measured") is not True
                    or row["config"].get("feasibility") != row["summary"].get("feasibility")
                    or row["config"].get("budget", {}).get("full_epoch") is not True):
                raise ValueError("Recipe P0 evidence must be a strict criteria full-epoch pass, never measurement-only")
        selected.append(row)
    profiles = {(r["summary"]["stage"], r["summary"]["experiment"]) for r in selected}
    if profiles != {(stage, exp) for stage in ("p0", "p1") for exp in ("e1", "e1hr", "e2")}:
        raise ValueError("Require successful P0 and completed bounded P1 for all three profiles")
    measured_hr = next(r for r in selected if r["summary"]["stage"] == "p0" and r["summary"]["experiment"] == "e1hr")
    if recipe["e1hr_resolution_wh"] != measured_hr["config"]["global_wh"]:
        raise ValueError("Reviewed HR resolution must have passing P0 evidence")
    if any(recipe["runtime"] != r["config"]["runtime"] for r in selected):
        raise ValueError("Reviewed runtime must match comparable pilot evidence")
    recipe.update(recipe_id=recipe_id, status="frozen_user_supplied_contract",
                  scope="pilot-derived fine-tuning recipe only; does not authorize official experiments or E3",
                  source_evidence=[{"run_dir": r["run_dir"], "summary_sha256": r["summary_sha256"],
                                    "config_sha256": r["config_sha256"]} for r in selected])
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{recipe_id}.json"
    checksum = root / f"{recipe_id}.sha256"
    if target.exists() or checksum.exists():
        raise FileExistsError("Never overwrite a frozen training recipe; choose a new reviewed version")
    write_json(target, recipe)
    checksum.write_text(f"{sha256_file(target)}  {target.name}\n", encoding="utf-8")
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", required=True, type=Path)
    p.add_argument("--output", type=Path, default=Path("reports/pilot/pilot_results.md"))
    p.add_argument("--reviewed-recipe", type=Path)
    p.add_argument("--recipe-id")
    p.add_argument("--recipes-dir", type=Path, default=Path("configs/training"))
    args = p.parse_args()
    rows = collect(args.runs_dir)
    if not rows:
        p.error("No pilot summaries found; do not create fabricated pilot results")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# UIT pilot results — diagnostic only", "", "No official experiment, test evaluation, or automatic LR/resolution ranking.",
             "Measured runs are diagnostics, not feasibility passes or recipe evidence.",
             "", "| Run | Stage/profile | Mode | Status | Seed |", "|---|---|---|---|---|"]
    for r in rows:
        s = r["summary"]
        mode = s.get("feasibility", {}).get("mode", "n/a")
        lines.append(f"| {Path(r['run_dir']).name} | {s['stage']}/{s['experiment']} | {mode} | {s['status']} | {r['config']['seed']} |")
    lines += ["", "## Environment, measurements, failures and declared policy", "",
              "Full evidence below includes resolved numeric values, environment, pass/fail and failure reasons. "
              "A numeric value is not frozen merely because it appears in a pilot config. "
              "User-reviewed rationale is required for the separate training recipe.", "", "```json",
              json.dumps(rows, indent=2), "```", ""]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    write_json(args.output.with_suffix(".json"), {"purpose": "pilot", "runs": rows})
    if bool(args.reviewed_recipe) != bool(args.recipe_id):
        p.error("--reviewed-recipe and --recipe-id must be supplied together")
    if args.reviewed_recipe:
        print(export_reviewed_recipe(args.reviewed_recipe, args.recipe_id, rows, args.recipes_dir))
    print(args.output)


if __name__ == "__main__":
    main()
