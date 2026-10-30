import json
import copy

import pytest

from src.pilot.collect_results import collect, export_reviewed_recipe
from src.pilot.config import fingerprint
from test_config import complete_config


def test_no_fabricated_results_or_nonpilot_runs(tmp_path):
    assert collect(tmp_path) == []
    folder = tmp_path / "bad"
    folder.mkdir()
    (folder / "run_summary.json").write_text(json.dumps({"purpose": "official"}))
    with pytest.raises(ValueError, match="Not a pilot"):
        collect(tmp_path)


def test_export_requires_explicit_complete_reviewed_values(tmp_path):
    source = tmp_path / "review.json"
    source.write_text(json.dumps({"lr_candidates": [1e-5]}))
    with pytest.raises(ValueError, match="missing keys"):
        export_reviewed_recipe(source, "training_recipe_v1", [], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_collector_preserves_all_diagnostic_statuses(tmp_path):
    expected = ["pass", "fail", "measured", "complete", "interrupted_at_epoch_boundary"]
    for i, status in enumerate(expected):
        run = tmp_path / str(i)
        run.mkdir()
        stage = "p0" if i < 3 else "p1"
        (run / "run_summary.json").write_text(json.dumps({"purpose": "pilot", "stage": stage, "status": status}))
        (run / "config_resolved.json").write_text(json.dumps({"seed": 17, "purpose": "pilot", "stage": stage}))
    assert [r["summary"]["status"] for r in collect(tmp_path)] == expected


@pytest.mark.parametrize("mutation", [None, "measured", "promoted_status", "promoted_mode",
                                     "missing_mode", "short_window"])
def test_recipe_only_accepts_strict_full_epoch_p0_evidence(tmp_path, mutation):
    # Synthetic rows/recipe under pytest temp; never a real training recipe.
    cfg = complete_config()
    runtime = copy.deepcopy(cfg["runtime"])
    runtime.pop("prefetch_factor")
    runtime["scaler"] = {"enabled": False}  # valid omission of inactive null options
    rows = []
    for stage in ("p0", "p1"):
        for exp in ("e1", "e1hr", "e2"):
            c = complete_config(stage, exp)
            c["runtime"] = copy.deepcopy(runtime)
            summary = {"purpose": "pilot", "stage": stage, "experiment": exp,
                       "status": "pass" if stage == "p0" else "complete"}
            if stage == "p0":
                context = {"version": 2, "feasibility_mode": "criteria", "feasibility": copy.deepcopy(c["feasibility"]),
                           "budget": copy.deepcopy(c["budget"])}
                summary.update(feasibility=copy.deepcopy(c["feasibility"]), feasibility_context=context,
                               feasibility_fingerprint=fingerprint(context), full_epoch_measured=True)
            rows.append({"run_dir": str((tmp_path / f"{stage}-{exp}").resolve()), "summary": summary,
                         "config": c, "summary_sha256": "synthetic-only", "config_sha256": fingerprint(c)})
    recipe = {"source_runs": [r["run_dir"] for r in rows], "seeds": {"tuning": 1, "final": [2, 3, 4]},
              "lr_candidates": [.001, .002, .003, .004], "runtime": runtime, "optimizer": {"name": "AdamW"},
              "scheduler": {"max_epochs": 2}, "batch": {"effective": 2}, "augmentation": {"enabled": False},
              "initialization": {"method": "linear_default"}, "e1hr_resolution_wh": [1344, 576],
              "checkpoint": {"tie_break": "earlier"}, "numeric_value_rationales": {"all": "synthetic test only"}}
    path = tmp_path / "review.json"
    path.write_text(json.dumps(recipe))
    if mutation is None:
        output = export_reviewed_recipe(path, "synthetic_recipe", rows, tmp_path / "out")
        assert output.is_file() and output.with_suffix(".sha256").is_file()
        return
    s = rows[0]["summary"]
    if mutation in {"measured", "promoted_status", "promoted_mode"}:
        s["status"] = "measured" if mutation == "measured" else "pass"
        s["feasibility"]["mode"] = "measurement_only"
        s["feasibility_context"]["feasibility_mode"] = "measurement_only"
        s["feasibility_context"]["feasibility"]["mode"] = "measurement_only"
        if mutation == "promoted_mode":
            s["feasibility"]["mode"] = "criteria"
        s["feasibility_fingerprint"] = fingerprint(s["feasibility_context"])
    elif mutation == "missing_mode":
        del s["feasibility"]["mode"]
    else:
        s["full_epoch_measured"] = False
    with pytest.raises(ValueError):
        export_reviewed_recipe(path, "synthetic_recipe", rows, tmp_path / "out")
    assert not (tmp_path / "out").exists()
