import copy
from pathlib import Path

import pytest


def install_trial_edge(monkeypatch, recipe, tmp_path, scores=None):
    import src.e1.campaign as module
    from src.e1.config import resolve_run, sha256_file
    from src.e1.checkpoint import atomic_json
    paths = {"data_root": str(tmp_path / "data"), "weights": str(tmp_path / "weights"), "runs_dir": str(tmp_path / "runs")}
    records = []
    values = scores or [.5000000, .5000006, .5000012, .49]
    mapped = {}
    for lr, score in zip(recipe["campaign"]["lr_candidates"], values):
        cfg = resolve_run(recipe, stage="tuning", lr=lr, seed=1001, paths=paths)
        folder = tmp_path / cfg["run_id"]
        folder.mkdir()
        atomic_json(folder / "run_summary.json", {"lr": lr, "score": score})
        mapped[str(folder.resolve())] = {"config": cfg, "summary": {"lr": lr, "seed": 1001, "stage": "tuning", "identity": {"source_sha256": "a"*64}, "selected": {"epoch": 1, "macro_f1_supported": score}},
                                       "run_dir": str(folder.resolve()), "summary_sha256": sha256_file(folder / "run_summary.json"),
                                       "best_sha256": str(len(records))*64, "events_sha256": "e"*64}
        records.append({"run_dir": str(folder)})
    def validated(folder):
        return copy.deepcopy(mapped[str(Path(folder).resolve())])
    monkeypatch.setattr(module, "validate_completed_run", validated)
    return module, paths, records, mapped


def test_tuning_declarations_are_exactly_four_and_create_no_jobs(recipe, tmp_path):
    from src.e1.campaign import plan_tuning, write_declarations
    paths = {"data_root": str(tmp_path), "weights": str(tmp_path / "weights"), "runs_dir": str(tmp_path / "runs")}
    declarations = plan_tuning(recipe, paths)
    assert [d["lr"] for d in declarations] == [2.5e-5, 5e-5, 1e-4, 2e-4]
    assert {d["seed"] for d in declarations} == {1001}
    assert not (tmp_path / "runs").exists()
    output = tmp_path / "declarations"
    write_declarations(output, declarations)
    assert len(list(output.glob("*-seed1001.json"))) == 4
    with pytest.raises(FileExistsError):
        write_declarations(output, declarations)


def test_global_maximum_tie_is_order_independent_and_final_seeds_fresh(recipe, tmp_path, monkeypatch):
    module, paths, records, mapped = install_trial_edge(monkeypatch, recipe, tmp_path)
    selection = module.select_lr(recipe, records)
    assert selection["lr"] == 5e-5
    assert module.select_lr(recipe, list(reversed(records))) == selection
    path = tmp_path / "selection.json"
    module.write_selection(path, recipe, selection)
    final = module.plan_final(recipe, selection, {**paths, "selection": str(path)})
    assert [c["seed"] for c in final] == [2001, 3001, 4001]
    assert {c["lr"] for c in final} == {5e-5}
    assert all(c["recipe"]["model"]["pretrained"]["starting_state"].startswith("Fresh pinned ImageNet") for c in final)
    with pytest.raises(FileExistsError):
        module.write_selection(path, recipe, selection)
    changed = copy.deepcopy(selection)
    changed["lr"] = 2e-4
    with pytest.raises(ValueError):
        module.plan_final(recipe, changed, {**paths, "selection": str(path)})


def test_missing_duplicate_or_extra_trials_stop_selection(recipe, tmp_path, monkeypatch):
    module, paths, records, mapped = install_trial_edge(monkeypatch, recipe, tmp_path)
    for bad in [records[:3], records+[records[0]], [records[0]]*4]:
        with pytest.raises(ValueError):
            module.select_lr(recipe, bad)


def test_synthetic_and_incomplete_evidence_is_not_promoted(tmp_path):
    from src.e1.campaign import validate_completed_run
    from src.e1.checkpoint import atomic_json
    for summary in [{"status": "complete", "evidence_kind": "synthetic_cpu_fixture"}, {"status": "interrupted"}]:
        atomic_json(tmp_path / "run_summary.json", summary)
        with pytest.raises(ValueError):
            validate_completed_run(tmp_path)


@pytest.mark.parametrize("field,value", [("completed_epochs", 7), ("validations", 7), ("optimizer_updates", 52497), ("seed", 2001)])
def test_forged_complete_prefix_or_wrong_seed_cannot_reach_selection(run_config, tmp_path, field, value):
    from src.e1.campaign import validate_completed_run
    from src.e1.checkpoint import atomic_json
    from src.e1.config import resolve_run, CONTRACT_SHA256
    c = resolve_run(run_config["recipe"], stage="tuning", lr=1e-4, seed=1001,
                    paths={**run_config["paths"], "runs_dir": str(tmp_path)})
    folder = tmp_path / c["run_id"]
    folder.mkdir()
    atomic_json(folder / "config_resolved.json", c)
    summary = {"kind": "official_e1_run_summary_v1", "status": "complete", "evidence_kind": "official_uit_slurm",
               "test_images_opened": 0, "contract_sha256": CONTRACT_SHA256, "identity": {},
               **{k: c[k] for k in ("run_id", "stage", "seed", "lr", "config_sha256")},
               "planned_epochs": 8, "completed_epochs": 8, "validations": 8, "optimizer_updates": 139992,
               "planned_optimizer_updates": 139992, "termination_reason": "eight_epoch_budget_complete"}
    summary[field] = value
    atomic_json(folder / "run_summary.json", summary)
    with pytest.raises(ValueError, match="identity|budget"):
        validate_completed_run(folder)
