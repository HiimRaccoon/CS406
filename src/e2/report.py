"""Selected-checkpoint validation exports and three-final-seed summaries."""
from __future__ import annotations

import argparse
import csv
import io
import math
import statistics
from pathlib import Path

from .campaign import _verify_selection, read_selection, validate_completed_run
from .checkpoint import atomic_json, atomic_text, read_checkpoint
from .config import CONTRACT_SHA256, ROOT, fingerprint, load_recipe, read_json, sha256_file, validate_recipe, scientific_identity, topology
from .engine import evaluate
from .train import _setup, preflight, run_lock

SCORES = ("top1", "top3", "top5", "macro_f1_supported", "weighted_f1")


def _reporting_inventory():
    path=ROOT/'src/e2/report.py'
    return {'path':'src/e2/report.py','bytes':path.stat().st_size,'sha256':sha256_file(path),
            'role':'reporting_only_tool; excluded from training scientific/runtime fingerprint'}


def _same_metrics(actual, expected):
    for key in (*SCORES, "samples", "supported_classes", "output_classes", "per_class", "buckets", "confusion"):
        if actual[key] != expected[key]:
            raise ValueError(f"Selected-checkpoint validation differs from committed history: {key}")
    if not math.isclose(actual["nll"], expected["nll"], rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("Selected-checkpoint NLL differs from committed history")


def _csv(path, rows, fields):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, output.getvalue())


def _evaluation_context(config):
    # Actual validation forward passes still require accepted UIT Slurm runtime and stage authorization.
    return _setup(config, preflight(config))


def evaluate_selected(config: dict, run_dir: Path) -> dict:
    folder = Path(run_dir).resolve()
    with run_lock(folder):
        row = validate_completed_run(folder)
        if fingerprint(config) != fingerprint(row["config"]):
            raise ValueError("Report configuration differs from completed run")
        summary = row["summary"]
        state = read_checkpoint(folder / "best.pt", config=config, identity=summary["identity"])
        if state["role"] != "best":
            raise ValueError("Validation analysis requires the selected checkpoint")
        context = _evaluation_context(config)
        context["model"].load_state_dict(state["model"], strict=True)
        metrics = evaluate(context["model"], context["loaders"]["val"], device=context["device"], runtime=config["recipe"]["runtime"],
                           n_train=context["n_train"], labels=context["labels"], collect_predictions=True)
        epoch = summary["selected"]["epoch"]
        history = read_json(folder / "history.json")
        _same_metrics(metrics, history[epoch - 1]["validation"])
        analysis = {"kind": "e2_selected_validation_analysis_v1", **topology(config["recipe"]),
                    "stage": config["stage"], "seed": config["seed"], "lr": config["lr"], "budget": config["budget"],
                    "run_id": config["run_id"], "contract_sha256": CONTRACT_SHA256,
                    "config_sha256": config["config_sha256"], "best_sha256": row["best_sha256"], "selected_epoch": epoch,
                    "evidence_kind": summary["identity"]["scientific_runtime"]["evidence_kind"], "identity": summary["identity"],
                    "history_sha256": sha256_file(folder / "history.json"), "metrics": metrics, "test_images_opened": 0,
                    "reporting_tool_inventory": _reporting_inventory(), "reporting_attempt_audit": context["identity"]["audit"]}
        output = folder / "validation_analysis"
        output.mkdir(exist_ok=True)
        if (output / "analysis.json").exists() and read_json(output / "analysis.json")["best_sha256"] != row["best_sha256"]:
            raise ValueError("Existing analysis belongs to another selected checkpoint")
        _csv(output / "overall.csv", [metrics], [*SCORES, "nll", "samples", "supported_classes", "output_classes"])
        _csv(output / "per_class.csv", metrics["per_class"], ["label_id", "label", "n_train", "bucket", "support", "predicted_count", "precision", "recall", "f1"])
        _csv(output / "buckets.csv", [{"bucket": k, **v} for k, v in metrics["buckets"].items()],
             ["bucket", "total_classes", "supported_classes", "samples", "macro_f1_supported", "top1", "top3", "top5"])
        predictions = [{"relative_path": r["relative_path"], "label_id": r["label_id"], "label": context["labels"][r["label_id"]],
                        **{f"top{k+1}_label_id": i for k, i in enumerate(r["top5"])}} for r in metrics["predictions"]]
        _csv(output / "predictions.csv", predictions, ["relative_path", "label_id", "label", *[f"top{k}_label_id" for k in (1, 2, 3, 4, 5)]])
        _csv(output / "confusion.csv", [{"true_label_id": i, **{str(j): n for j, n in enumerate(counts)}} for i, counts in enumerate(metrics["confusion"])],
             ["true_label_id", *map(str, range(124))])
        pairs = [{"true_label_id": i, "predicted_label_id": j, "count": n} for i, counts in enumerate(metrics["confusion"]) for j, n in enumerate(counts) if i != j and n]
        pairs.sort(key=lambda r: (-r["count"], r["true_label_id"], r["predicted_label_id"]))
        _csv(output / "most_confused.csv", pairs, ["true_label_id", "predicted_label_id", "count"])
        curves = [{"epoch": r["epoch"], "train_loss": r["train"]["loss"], "validation_nll": r["validation"]["nll"],
                   **{k: r["validation"][k] for k in SCORES}, "lr_first": r["train"]["lr_first"], "lr_last": r["train"]["lr_last"],
                   "epoch_seconds": r["epoch_seconds"]} for r in history]
        _csv(output / "learning_curves.csv", curves, list(curves[0]))
        # Write the lineage-bearing index last, after every derived export succeeds.
        analysis['exports_sha256']={p.name:sha256_file(p) for p in output.iterdir() if p.suffix=='.csv'}
        atomic_json(output / "analysis.json", analysis)
        return analysis


def _selected_analysis(row):
    folder = Path(row["run_dir"])
    analysis = read_json(folder / "validation_analysis/analysis.json")
    config, summary = row["config"], row["summary"]
    if (analysis.get("kind") != "e2_selected_validation_analysis_v1" or analysis.get("experiment") != "E2"
            or any(analysis.get(k) != v for k,v in topology(config["recipe"]).items()) or analysis.get("test_images_opened") != 0
            or any(analysis[k] != config[k] for k in ("stage","seed","lr","budget"))
            or scientific_identity(analysis["identity"]) != scientific_identity(summary["identity"])
            or analysis["history_sha256"] != sha256_file(folder / "history.json")
            or analysis["contract_sha256"] != CONTRACT_SHA256 or analysis["config_sha256"] != config["config_sha256"]
            or analysis["run_id"] != config["run_id"] or analysis["best_sha256"] != row["best_sha256"]
            or analysis["selected_epoch"] != summary["selected"]["epoch"]):
        raise ValueError("Analysis has wrong selected-checkpoint lineage")
    expected={name+'.csv' for name in ('overall','per_class','buckets','predictions','confusion','most_confused','learning_curves')}
    inventory=analysis.get('exports_sha256',{})
    if set(inventory)!=expected or any(sha256_file(folder/'validation_analysis'/p)!=digest for p,digest in inventory.items()):
        raise ValueError('Selected validation export inventory differs')
    history = read_json(folder / "history.json")
    _same_metrics(analysis["metrics"], history[analysis["selected_epoch"] - 1]["validation"])
    return analysis


def _stats(values):
    if all(v is None for v in values):
        return {"mean": None, "std": None}
    if any(v is None or not math.isfinite(v) for v in values):
        raise ValueError("Inconsistent supported coverage or nonfinite aggregate")
    return {"mean": statistics.mean(values), "std": statistics.stdev(values)}


def aggregate_final(run_dirs: list[Path], recipe: dict, selection: dict) -> dict:
    recipe = validate_recipe(recipe)
    _verify_selection(recipe, selection)
    if len(run_dirs) != 3 or len({str(Path(p).resolve()) for p in run_dirs}) != 3:
        raise ValueError("Exactly three distinct complete final runs required")
    rows = [validate_completed_run(p) for p in run_dirs]
    rows.sort(key=lambda r: r["config"]["seed"])
    if {r["config"]["seed"] for r in rows} != {2001, 3001, 4001}:
        raise ValueError("Final seeds must be distinct and exclude tuning")
    for row in rows:
        c = row["config"]
        if (c["stage"] != "final" or c["lr"] != selection["lr"] or c["contract_sha256"] != CONTRACT_SHA256
                or sha256_file(Path(c["paths"]["selection"])) != c["selection_sha256"]
                or read_selection(Path(c["paths"]["selection"]), recipe) != selection
                or scientific_identity(row["summary"]["identity"]) != scientific_identity(rows[0]["summary"]["identity"])):
            raise ValueError("Final records do not share the frozen selection/recipe/accepted identity")
    analyses = [_selected_analysis(r) for r in rows]
    metrics = [a["metrics"] for a in analyses]
    if any((m["samples"], m["supported_classes"], m["output_classes"]) !=
           (metrics[0]["samples"], metrics[0]["supported_classes"], metrics[0]["output_classes"]) for m in metrics):
        raise ValueError("Final split coverage differs")
    buckets = {}
    for key, first in metrics[0]["buckets"].items():
        coverage = {k: first[k] for k in ("total_classes", "supported_classes", "samples")}
        if any(any(m["buckets"][key][k] != v for k, v in coverage.items()) for m in metrics):
            raise ValueError("Final bucket coverage differs")
        buckets[key] = {**coverage, "metrics": {k: _stats([m["buckets"][key][k] for m in metrics])
                                              for k in ("macro_f1_supported", "top1", "top3", "top5")}}
    classes = []
    for i, first in enumerate(metrics[0]["per_class"]):
        metadata = {k: first[k] for k in ("label_id", "label", "n_train", "bucket", "support")}
        if any(any(m["per_class"][i][k] != v for k, v in metadata.items()) for m in metrics):
            raise ValueError("Final per-class coverage/mapping differs")
        classes.append({**metadata, **{k: _stats([m["per_class"][i][k] for m in metrics]) for k in ("precision", "recall", "f1")}})
    return {"kind": "e2_three_seed_validation_report_v1", **topology(recipe),
            "stage": "final", "seeds": [2001,3001,4001], "budget": recipe["budget"], "identity": rows[0]["summary"]["identity"],
            "contract_sha256": CONTRACT_SHA256, "reporting_tool_inventory": _reporting_inventory(),
            "selection_fingerprint": selection["selection_fingerprint"], "lr": selection["lr"], "ddof": 1,
            "per_seed": [{"seed": r["config"]["seed"], "stage": "final", "lr": r["config"]["lr"],
                          "config_sha256": r["config"]["config_sha256"], "history_sha256": a["history_sha256"],
                          "identity": r["summary"]["identity"], "run_id": r["config"]["run_id"], "best_sha256": r["best_sha256"],
                          "selected_epoch": a["selected_epoch"], "views": r["config"]["views"], "budget": r["config"]["budget"], **{k: a["metrics"][k] for k in (*SCORES, "nll")}}
                         for r, a in zip(rows, analyses)],
            "overall": {k: _stats([m[k] for m in metrics]) for k in (*SCORES, "nll")},
            "buckets": buckets, "per_class": classes, "test_images_opened": 0,
            "limitations": ["Exact fixed split does not establish geographic/panorama independence",
                            "Extreme-tail support is small; seed variation does not remove sampling uncertainty",
                            "Validation analysis only; test evaluation requires separate authorization"]}


def render_aggregate(report):
    def percent(pair):
        return "not evaluated" if pair["mean"] is None else f'{pair["mean"]*100:.4f}% +/- {pair["std"]*100:.4f}%'
    lines = ["# E2 three-seed validation report", "", "Arithmetic mean and sample standard deviation (ddof=1); unrounded per-seed values used.", "",
             "| Metric | Mean +/- std |", "|---|---|"]
    lines += [f"| {k} | {percent(report['overall'][k])} |" for k in SCORES]
    lines += ["", "| Seed | Selected epoch | " + " | ".join(SCORES) + " |", "|---|---|" + "---|"*len(SCORES)]
    for row in report["per_seed"]:
        lines.append(f"| {row['seed']} | {row['selected_epoch']} | " + " | ".join(f"{row[k]*100:.4f}%" for k in SCORES) + " |")
    lines += ["", *["- " + s for s in report["limitations"]], ""]
    return "\n".join(lines)


def compare_baselines(e2_report: dict, e1_report_path: Path, e1hr_report_path: Path) -> dict:
    """Both comparator paths are explicit execution-environment inputs."""
    recipe=load_recipe()
    if (e2_report.get('kind')!='e2_three_seed_validation_report_v1'
            or any(e2_report.get(k)!=v for k,v in topology(recipe).items())
            or e2_report.get('contract_sha256')!=CONTRACT_SHA256 or e2_report.get('ddof')!=1
            or e2_report.get('test_images_opened')!=0 or len(e2_report['per_seed'])!=3
            or {r['seed'] for r in e2_report['per_seed']}!={2001,3001,4001}
            or len(e2_report['per_class'])!=124):
        raise ValueError('Complete E2 aggregate and frozen reporting conventions required')
    comparators={}
    contracts={'E1':'docs/E1/e1_numeric_contract_v1.json','E1-HR':'docs/E1-HR/e1hr_numeric_contract_v1.json'}
    for name,provided in [('E1',e1_report_path),('E1-HR',e1hr_report_path)]:
        binding=recipe['reporting']['comparison']['comparators'][name]
        path=Path(provided).resolve()
        if sha256_file(path)!=binding['required_aggregate_sha256']:
            raise ValueError('Exact official '+name+' comparator aggregate SHA-256 required')
        baseline=read_json(path)
        contract_path=ROOT/contracts[name]
        if sha256_file(contract_path)!=binding['required_contract_sha256']:
            raise ValueError('Exact comparator frozen contract required')
        contract=read_json(contract_path)
        semantics=('topk','ranking','rank_outputs','macro_f1','unsupported_f1','weighted_f1','nll',
                   'frequency_buckets_train_images','extreme_tail_subset_train_images','bucket_f1','bucket_topk','aggregation','machine_precision')
        if (baseline.get('kind')!=binding['aggregate_kind'] or baseline.get('contract_sha256')!=binding['required_contract_sha256']
                or baseline.get('ddof')!=1 or baseline.get('test_images_opened')!=0
                or len(baseline['per_seed'])!=3 or {r['seed'] for r in baseline['per_seed']}!={2001,3001,4001}
                or contract['data']['artifacts']!=recipe['data']['artifacts']
                or any(contract['reporting'][k]!=recipe['reporting'][k] for k in semantics)
                or len(baseline['per_class'])!=124):
            raise ValueError('Incompatible official comparator benchmark/metric schema')
        for a,b in zip(e2_report['per_class'],baseline['per_class']):
            if any(a[k]!=b[k] for k in ('label_id','label','n_train','bucket','support')):
                raise ValueError('Fixed comparator class coverage differs')
        if set(e2_report['buckets'])!=set(baseline['buckets']):raise ValueError('Wrong frequency buckets')
        for bucket in baseline['buckets']:
            if any(e2_report['buckets'][bucket][k]!=baseline['buckets'][bucket][k]
                   for k in ('samples','supported_classes','total_classes')):
                raise ValueError('Fixed comparator bucket coverage differs')
        by_seed={r['seed']:r for r in baseline['per_seed']}
        comparators[name]={'role':binding['role'],'role_description':binding['role_description'],
            'actual_path':str(path),'sha256':binding['required_aggregate_sha256'],
            'contract_sha256':binding['required_contract_sha256'],
            'overall':{k:{'e2':e2_report['overall'][k],'comparator':baseline['overall'][k],
                         'mean_delta_percentage_points':100*(e2_report['overall'][k]['mean']-baseline['overall'][k]['mean'])} for k in SCORES},
            'buckets':baseline['buckets'],
            'paired_seed_deltas_percentage_points':[{'seed':r['seed'],**{k:100*(r[k]-by_seed[r['seed']][k]) for k in SCORES}} for r in e2_report['per_seed']]}
    return {'kind':'e2_vs_e1_e1hr_validation_comparison_v1',**topology(recipe),
            'contract_sha256':CONTRACT_SHA256,'e2_report_fingerprint':fingerprint(e2_report),
            'reporting_tool_inventory':_reporting_inventory(),'comparators':comparators,'ddof':1,'test_images_opened':0,
            'limitations':recipe['reporting']['limitations']+[
                'E1 is the primary architectural comparator with the same 896x384 global input',
                'E1-HR is contextual/complementary; this comparison does not isolate local views or resolution',
                'Gains cannot be attributed exclusively to local detail, head capacity or compute',
                'Three paired seeds do not establish significance; compute matching is not required']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    selected = sub.add_parser("evaluate-selected")
    selected.add_argument("--run-dir", required=True, type=Path)
    aggregate = sub.add_parser("aggregate-final")
    aggregate.add_argument("--run-dirs", nargs=3, required=True, type=Path)
    aggregate.add_argument("--selection", required=True, type=Path)
    aggregate.add_argument("--output", required=True, type=Path)
    aggregate.add_argument("--e1-report", required=True, type=Path)
    aggregate.add_argument("--e1hr-report", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "evaluate-selected":
        evaluate_selected(read_json(args.run_dir / "config_resolved.json"), args.run_dir)
    else:
        recipe = load_recipe()
        report = aggregate_final(args.run_dirs, recipe, read_selection(args.selection, recipe))
        comparison = compare_baselines(report, args.e1_report, args.e1hr_report)
        atomic_json(args.output, report)
        atomic_text(args.output.with_suffix(".md"), render_aggregate(report))
        atomic_json(args.output.with_name(args.output.stem + "_vs_e1_e1hr.json"), comparison)


if __name__ == "__main__":
    main()
