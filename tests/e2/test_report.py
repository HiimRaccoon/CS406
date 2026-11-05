import copy
from pathlib import Path

import pytest
import torch

from .test_train import install_cpu_edges


def test_selected_checkpoint_analysis_uses_earlier_weights_and_exports(run_config, monkeypatch):
    import src.e2.report as report
    from src.e2.config import read_json, sha256_file
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
    from .helpers import read_fixture
    monkeypatch.setattr(report,"read_checkpoint",read_fixture)
    result = report.evaluate_selected(state["config"], folder)
    history = read_json(folder / "history.json")
    assert result["metrics"]["nll"] == pytest.approx(history[0]["validation"]["nll"])
    assert result["selected_epoch"] == 1 and len(result["metrics"]["predictions"]) == 5
    assert result["evidence_kind"] == "synthetic_cpu_fixture"
    for name in ["analysis.json", "overall.csv", "per_class.csv", "buckets.csv", "confusion.csv", "predictions.csv", "most_confused.csv", "learning_curves.csv"]:
        assert (folder / "validation_analysis" / name).is_file()
    report._selected_analysis(row)
    (folder/"validation_analysis/predictions.csv").write_text("altered")
    with pytest.raises(ValueError):report._selected_analysis(row)


def aggregate_fixture(recipe, tmp_path, monkeypatch):
    import src.e2.report as report
    from src.e2.checkpoint import atomic_json
    from src.e2.config import CONTRACT_SHA256, sha256_file
    from src.e2.metrics import Metrics124
    selection = {"lr": 1e-4, "selection_fingerprint": "fixture-only"}
    path = tmp_path / "selection.json"
    atomic_json(path, selection)
    monkeypatch.setattr(report, "_verify_selection", lambda r, s: None)
    monkeypatch.setattr(report, "read_selection", lambda p, r: selection)
    mapped = {}
    folders = []
    from src.e2.data import FixedManifest
    manifest=FixedManifest()
    targets=torch.tensor([r.label_id for r in manifest.records('val')])
    metric = Metrics124()
    metric.update(torch.zeros(len(targets),124),targets,float(len(targets)))
    template = metric.compute(manifest.n_train_by_label,manifest.labels)
    for seed, value in zip([2001, 3001, 4001], [.1, .2, .3]):
        folder = tmp_path / str(seed)
        folder.mkdir()
        folders.append(folder)
        metrics = copy.deepcopy(template)
        for k in ["top1", "top3", "top5", "macro_f1_supported", "weighted_f1"]:
            metrics[k] = value
        config = {"stage": "final", "seed": seed, "lr": 1e-4, "contract_sha256": CONTRACT_SHA256,
                  "config_sha256": "c"*64, "run_id": str(seed), "recipe": recipe, "experiment":"E2", "views":recipe["views"],"representation_dimension":1536,"fusion":"global_then_mean_local","view_execution":"global_then_flattened_locals_batch_major", "budget":recipe["budget"],
                  "selection_sha256": sha256_file(path), "paths": {"selection": str(path)}}
        summary = {"selected": {"epoch": 1}, "seed": seed, "stage": "final", "lr": 1e-4,
                   "identity": {"scientific_runtime": {"evidence_kind": "synthetic_cpu_fixture"}, "audit": {"clean_git_commit":"a"*40}}}
        row = {"run_dir": str(folder.resolve()), "config": config, "summary": summary, "best_sha256": "b"*64}
        mapped[str(folder.resolve())] = row
        atomic_json(folder / "history.json", [{"epoch": 1, "validation": metrics}])
        output=folder/"validation_analysis";output.mkdir()
        exports={}
        for name in ("overall","per_class","buckets","predictions","confusion","most_confused","learning_curves"):
            f=output/(name+".csv");f.write_text("fixture export")
            exports[f.name]=sha256_file(f)
        atomic_json(folder / "validation_analysis/analysis.json", {"exports_sha256":exports,"contract_sha256": CONTRACT_SHA256,
                    "config_sha256": "c"*64, "run_id": str(seed), "best_sha256": "b"*64, "selected_epoch": 1,
                    "kind":"e2_selected_validation_analysis_v1", "experiment":"E2", "views":recipe["views"],"representation_dimension":1536,"fusion":"global_then_mean_local","view_execution":"global_then_flattened_locals_batch_major", "stage":"final", "seed":seed, "lr":1e-4, "budget":recipe["budget"], "identity":summary["identity"], "history_sha256":sha256_file(folder/"history.json"), "test_images_opened":0, "evidence_kind": "synthetic_cpu_fixture", "metrics": metrics})
    monkeypatch.setattr(report, "validate_completed_run", lambda p: mapped[str(Path(p).resolve())])
    return report, folders, selection, mapped


def test_three_seed_sample_std_without_rounding_or_ensembling(recipe, tmp_path, monkeypatch):
    report, folders, selection, mapped = aggregate_fixture(recipe, tmp_path, monkeypatch)
    r = report.aggregate_final(folders, recipe, selection)
    assert r["overall"]["top1"]["mean"] == .2
    assert r["overall"]["top1"]["std"] == pytest.approx(.1)
    assert r["ddof"] == 1
    assert r['kind']=='e2_three_seed_validation_report_v1'
    assert r['experiment']=='E2' and r['views']==recipe['views']
    assert "20.0000% +/- 10.0000%" in report.render_aggregate(r)
    assert [s["seed"] for s in r["per_seed"]] == [2001, 3001, 4001]
    with pytest.raises(ValueError):
        report.aggregate_final([folders[0]]*3, recipe, selection)
    mapped[str(folders[-1].resolve())]["config"]["seed"] = 1001
    with pytest.raises(ValueError):
        report.aggregate_final(folders, recipe, selection)


def test_analysis_with_wrong_checkpoint_lineage_is_rejected(recipe, tmp_path, monkeypatch):
    report, folders, selection, mapped = aggregate_fixture(recipe, tmp_path, monkeypatch)
    from src.e2.checkpoint import atomic_json
    from src.e2.config import read_json
    p = folders[0] / "validation_analysis/analysis.json"
    analysis = read_json(p)
    analysis["best_sha256"] = "wrong"
    atomic_json(p, analysis)
    with pytest.raises(ValueError):
        report.aggregate_final(folders, recipe, selection)


def test_aggregate_accepts_audit_only_commit_change_but_rejects_runtime_change(recipe,tmp_path,monkeypatch):
    report,folders,selection,mapped=aggregate_fixture(recipe,tmp_path,monkeypatch)
    row=mapped[str(folders[-1].resolve())]
    row['summary']['identity']['audit']['clean_git_commit']='b'*40
    assert report.aggregate_final(folders,recipe,selection)['ddof']==1
    row['summary']['identity']['scientific_runtime']['source']='changed'
    with pytest.raises(ValueError):report.aggregate_final(folders,recipe,selection)


def test_comparator_artifacts_are_explicit_hash_bound_inputs(recipe,tmp_path,monkeypatch):
    from src.e2.config import ROOT,sha256_file
    report,folders,selection,mapped=aggregate_fixture(recipe,tmp_path,monkeypatch)
    e2=report.aggregate_final(folders,recipe,selection)
    # These are supplied fixture paths only; production has no path defaults.
    e1=ROOT/'src/e1/e1_three_seed_validation_report_v1.json'
    hr=ROOT/'src/e1hr/final-report-v1/e1hr_three_seed_validation_report_v1.json'
    before=[sha256_file(p) for p in (e1,hr)]
    result=report.compare_baselines(e2,e1,hr)
    assert result['comparators']['E1']['role']=='primary_architectural'
    assert result['comparators']['E1-HR']['role']=='contextual_complementary'
    assert result['comparators']['E1']['overall']['top1']['mean_delta_percentage_points']==pytest.approx((.2-.7550575785869904)*100)
    assert [sha256_file(p) for p in (e1,hr)]==before
    with pytest.raises(ValueError):report.compare_baselines(e2,hr,e1)
    bad=tmp_path/'pilot.json';bad.write_text('{"kind":"pilot"}')
    with pytest.raises(ValueError):report.compare_baselines(e2,bad,hr)


def test_report_labels_primary_and_contextual_comparisons(recipe,tmp_path,monkeypatch):
    report,folders,selection,mapped=aggregate_fixture(recipe,tmp_path,monkeypatch)
    from src.e2.config import ROOT
    result=report.compare_baselines(report.aggregate_final(folders,recipe,selection),
        ROOT/'src/e1/e1_three_seed_validation_report_v1.json',
        ROOT/'src/e1hr/final-report-v1/e1hr_three_seed_validation_report_v1.json')
    assert result['comparators']['E1']['sha256']==recipe['reporting']['comparison']['comparators']['E1']['required_aggregate_sha256']
    assert 'does not isolate local views or resolution' in ' '.join(result['limitations'])
