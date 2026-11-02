import copy
from pathlib import Path

import pytest
import torch

from .test_train import install_cpu_edges


def test_selected_checkpoint_analysis_uses_earlier_weights_and_exports(run_config, monkeypatch):
    import src.e1.report as report
    from src.e1.config import read_json, sha256_file
    module = install_cpu_edges(monkeypatch)
    setup = module._setup
    def stable(config, checked):
        ctx = setup(config, checked)
        with torch.no_grad():
            ctx["model"].net[-1].bias.zero_()
            ctx["model"].net[-1].bias[0] = 3.
        return ctx
    monkeypatch.setattr(module, "_setup", stable)
    folder = Path(run_config["paths"]["runs_dir"]) / run_config["run_id"]
    summary = module.run(run_config, folder)
    assert summary["selected"]["epoch"] == 1
    state = torch.load(folder / "last.pt", weights_only=False)
    row = {"config": state["config"], "summary": summary, "run_dir": str(folder), "best_sha256": sha256_file(folder / "best.pt")}
    monkeypatch.setattr(report, "validate_completed_run", lambda _: row)
    monkeypatch.setattr(report, "_evaluation_context", lambda c: stable(c, summary["identity"]))
    result = report.evaluate_selected(state["config"], folder)
    history = read_json(folder / "history.json")
    assert result["metrics"]["nll"] == pytest.approx(history[0]["validation"]["nll"])
    assert result["selected_epoch"] == 1 and len(result["metrics"]["predictions"]) == 5
    assert result["evidence_kind"] == "synthetic_cpu_fixture"
    for name in ["analysis.json", "overall.csv", "per_class.csv", "buckets.csv", "confusion.csv", "predictions.csv", "most_confused.csv", "learning_curves.csv"]:
        assert (folder / "validation_analysis" / name).is_file()


def aggregate_fixture(recipe, tmp_path, monkeypatch):
    import src.e1.report as report
    from src.e1.checkpoint import atomic_json
    from src.e1.config import CONTRACT_SHA256, sha256_file
    from src.e1.metrics import Metrics124
    selection = {"lr": 1e-4, "selection_fingerprint": "fixture-only"}
    path = tmp_path / "selection.json"
    atomic_json(path, selection)
    monkeypatch.setattr(report, "_verify_selection", lambda r, s: None)
    monkeypatch.setattr(report, "read_selection", lambda p, r: selection)
    mapped = {}
    folders = []
    metric = Metrics124()
    logits = torch.zeros(1, 124)
    metric.update(logits, torch.tensor([0]), 1.)
    template = metric.compute({i: 1 for i in range(124)}, [str(i) for i in range(124)])
    for seed, value in zip([2001, 3001, 4001], [.1, .2, .3]):
        folder = tmp_path / str(seed)
        folder.mkdir()
        folders.append(folder)
        metrics = copy.deepcopy(template)
        for k in ["top1", "top3", "top5", "macro_f1_supported", "weighted_f1"]:
            metrics[k] = value
        config = {"stage": "final", "seed": seed, "lr": 1e-4, "contract_sha256": CONTRACT_SHA256,
                  "config_sha256": "c"*64, "run_id": str(seed), "recipe": recipe,
                  "selection_sha256": sha256_file(path), "paths": {"selection": str(path)}}
        summary = {"selected": {"epoch": 1}, "seed": seed, "stage": "final", "lr": 1e-4,
                   "identity": {"evidence_kind": "synthetic_cpu_fixture"}}
        row = {"run_dir": str(folder.resolve()), "config": config, "summary": summary, "best_sha256": "b"*64}
        mapped[str(folder.resolve())] = row
        atomic_json(folder / "history.json", [{"epoch": 1, "validation": metrics}])
        atomic_json(folder / "validation_analysis/analysis.json", {"contract_sha256": CONTRACT_SHA256,
                    "config_sha256": "c"*64, "run_id": str(seed), "best_sha256": "b"*64, "selected_epoch": 1,
                    "evidence_kind": "synthetic_cpu_fixture", "metrics": metrics})
    monkeypatch.setattr(report, "validate_completed_run", lambda p: mapped[str(Path(p).resolve())])
    return report, folders, selection, mapped


def test_three_seed_sample_std_without_rounding_or_ensembling(recipe, tmp_path, monkeypatch):
    report, folders, selection, mapped = aggregate_fixture(recipe, tmp_path, monkeypatch)
    r = report.aggregate_final(folders, recipe, selection)
    assert r["overall"]["top1"]["mean"] == .2
    assert r["overall"]["top1"]["std"] == pytest.approx(.1)
    assert r["ddof"] == 1
    assert "20.0000% +/- 10.0000%" in report.render_aggregate(r)
    assert [s["seed"] for s in r["per_seed"]] == [2001, 3001, 4001]
    with pytest.raises(ValueError):
        report.aggregate_final([folders[0]]*3, recipe, selection)
    mapped[str(folders[-1].resolve())]["config"]["seed"] = 1001
    with pytest.raises(ValueError):
        report.aggregate_final(folders, recipe, selection)


def test_analysis_with_wrong_checkpoint_lineage_is_rejected(recipe, tmp_path, monkeypatch):
    report, folders, selection, mapped = aggregate_fixture(recipe, tmp_path, monkeypatch)
    from src.e1.checkpoint import atomic_json
    from src.e1.config import read_json
    p = folders[0] / "validation_analysis/analysis.json"
    analysis = read_json(p)
    analysis["best_sha256"] = "wrong"
    atomic_json(p, analysis)
    with pytest.raises(ValueError):
        report.aggregate_final(folders, recipe, selection)
