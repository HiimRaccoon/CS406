"""Selected validation exports and seed-paired six-arm comparisons."""
from __future__ import annotations

import csv
import io
import math
from pathlib import Path
import statistics

import torch
from torch.utils.data import DataLoader

from .campaign import validate_completed_head_run,validate_final_declarations
from .config import (ROOT,SOURCE_SHA256,SEEDS,DIMENSIONS,contained,fingerprint,payload,policy,
                     read_json,sha256_file,validate_common_protocol,validate_record,verify_payload)
from .data import FixedManifest
from .io import atomic_json,atomic_write,writer_lock
from .metrics import Metrics124
from .model import tensor_state_sha256

SCORES=("top1","top3","top5","macro_f1_supported","weighted_f1")
ARMS=("E1-original","E3-E1-N","E3-E1-cRT","E2-original","E3-E2-N","E3-E2-cRT")


def _stats(values):
    if all(v is None for v in values): return {"mean":None,"std":None,"values":values}
    if len(values)!=3 or any(v is None or not math.isfinite(v) for v in values):
        raise ValueError("Exactly three finite matching-seed values required")
    return {"mean":statistics.mean(values),"std":statistics.stdev(values),"values":values}


def _summarize_study_rows(rows):
    keys=[(r["arm"],r["seed"]) for r in rows]
    if len(keys)!=18 or len(set(keys))!=18 or set(keys)!={(a,s) for a in ARMS for s in SEEDS}:
        raise ValueError("Complete 12-head/6-original matching-seed coverage required")
    indexed={(r["arm"],r["seed"]):r["metrics"] for r in rows}
    arms={a:{metric:_stats([indexed[a,s][metric] for s in SEEDS]) for metric in (*SCORES,"nll")} for a in ARMS}
    formulas={**{f"PRIMARY:{f}":[(f"E3-{f}-cRT",1),(f"E3-{f}-N",-1)] for f in DIMENSIONS},
              **{f"SECONDARY:{f}":[(f"E3-{f}-N",1),(f"{f}-original",-1)] for f in DIMENSIONS},
              "EXPLORATORY:E1-E2":[("E3-E1-cRT",1),("E3-E1-N",-1),("E3-E2-cRT",-1),("E3-E2-N",1)]}
    contrasts={name:{metric:_stats([sum(sign*indexed[a,s][metric] for a,sign in terms) for s in SEEDS])
                     for metric in (*SCORES,"nll")} for name,terms in formulas.items()}
    if all("buckets" in row["metrics"] for row in rows):
        for arm in ARMS:
            arms[arm]["buckets"]={b:{m:_stats([indexed[arm,s]["buckets"][b][m] for s in SEEDS])
                for m in ("macro_f1_supported","top1","top3","top5")} for b in indexed[arm,2001]["buckets"]}
        for name,terms in formulas.items():
            contrasts[name]["buckets"]={b:{m:_stats([None if all(indexed[a,s]["buckets"][b][m] is None for a,_ in terms)
                else sum(sign*indexed[a,s]["buckets"][b][m] for a,sign in terms) for s in SEEDS])
                for m in ("macro_f1_supported","top1","top3","top5")} for b in indexed["E1-original",2001]["buckets"]}
    return {"kind":"e3_study_metrics_payload_v1","seeds":list(SEEDS),"ddof":1,"rows":rows,"arms":arms,"contrasts":contrasts,
            "test_access":"LOCKED / NOT AUTHORIZED","limitations":["Fixed split does not establish geographic/panorama independence",
            "The 1-2 diagnostic bucket has only two validation-supported labels and two validation images",
            "Three paired seeds do not establish significance","Validation only; test locked",
            "E1/E2 effect comparison is exploratory and does not isolate architecture, dimension or local detail"]}


def _validate_predictions(predictions,records):
    if len(predictions)!=len(records): raise ValueError("Truncated prediction coverage")
    for row,expected in zip(predictions,records):
        ranks=row.get("top5")
        if (row.get("relative_path")!=expected["relative_path"] or row.get("label_id")!=expected["label_id"]
                or not isinstance(ranks,list) or len(ranks)!=5 or len(set(ranks))!=5
                or any(type(i) is not int or not 0<=i<124 for i in ranks)
                or row.get("prediction")!=ranks[0]):
            raise ValueError("Prediction order/GT/ranking/label IDs differ from fixed manifest")


def _metrics_from_predictions(predictions,n_train,labels,*,nll):
    if not math.isfinite(nll) or nll<0: raise ValueError("Authoritative finite sample-mean CE required")
    target=torch.tensor([r["label_id"] for r in predictions],dtype=torch.int64)
    ranking=torch.tensor([r["top5"] for r in predictions],dtype=torch.int64)
    metric=Metrics124(); metric.n=len(predictions); metric.loss_sum=nll*len(predictions)
    metric.confusion=torch.bincount(target*124+ranking[:,0],minlength=124**2).reshape(124,124)
    for k in (1,3,5):
        hits=(ranking[:,:k]==target[:,None]).any(1)
        metric.correct_by_label[k]=torch.bincount(target[hits],minlength=124)
    result=metric.compute(n_train,labels)
    # NLL is bound to actual CE evidence, never reconstructed from Top-5 IDs.
    result["nll"]=nll
    return result


def _same_metrics(actual,expected):
    for k in (*SCORES,"nll","samples","supported_classes","output_classes","per_class","buckets","confusion"):
        if actual[k]!=expected[k]: raise ValueError("Selected metric/history/discrete content mismatch: "+k)


def _csv_bytes(rows,fields):
    stream=io.StringIO(newline="")
    writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore",lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _tables(metrics,predictions,history,labels):
    pairs=[{"true_label_id":i,"predicted_label_id":j,"count":n} for i,row in enumerate(metrics["confusion"]) for j,n in enumerate(row) if i!=j and n]
    pairs.sort(key=lambda r:(-r["count"],r["true_label_id"],r["predicted_label_id"]))
    curves=[{"window":r["window"],"presentations":r["presentations"],"updates":r["updates"],"train_ce":r["train"]["ce"],
             "validation_nll":r["validation"]["nll"],**{k:r["validation"][k] for k in SCORES},
             "lr_first":r["train"]["lr_first"],"lr_last":r["train"]["lr_last"]} for r in history]
    prediction_rows=[{"relative_path":r["relative_path"],"label_id":r["label_id"],"label":labels[r["label_id"]],
                      "predicted_label_id":r["prediction"],**{f"top{k+1}_label_id":v for k,v in enumerate(r["top5"])},"nll":r["nll"]} for r in predictions]
    return {"overall.csv":([metrics],[*SCORES,"nll","samples","supported_classes","output_classes"]),
            "per_class.csv":(metrics["per_class"],["label_id","label","n_train","bucket","support","predicted_count","precision","recall","f1"]),
            "buckets.csv":([{"bucket":k,**v} for k,v in metrics["buckets"].items()],["bucket","total_classes","supported_classes","samples","macro_f1_supported","top1","top3","top5"]),
            "predictions.csv":(prediction_rows,["relative_path","label_id","label","predicted_label_id",*[f"top{k}_label_id" for k in range(1,6)],"nll"]),
            "confusion.csv":([{"true_label_id":i,**{str(j):n for j,n in enumerate(row)}} for i,row in enumerate(metrics["confusion"])],["true_label_id",*map(str,range(124))]),
            "most_confused.csv":(pairs,["true_label_id","predicted_label_id","count"]),
            "learning_curves.csv":(curves,list(curves[0]))}


def _validate_tables(folder,tables):
    for name,(rows,fields) in tables.items():
        if (Path(folder)/name).read_bytes()!=_csv_bytes(rows,fields):
            raise ValueError("Derived CSV headers/rows/content differ: "+name)


def load_source_reference(path: Path, *, source: dict) -> dict:
    path=Path(path).resolve(); analysis=read_json(path)
    family=source["family"]
    if (analysis.get("kind")!=f"{family.lower()}_selected_validation_analysis_v1"
            or analysis.get("contract_sha256")!=source["source_contract_sha256"]
            or analysis.get("best_sha256")!=source["binary_sha256"]
            or analysis.get("selected_epoch")!=source["selected_epoch"] or analysis.get("test_images_opened")!=0):
        raise ValueError("Original selected analysis family/contract/checkpoint/epoch mismatch")
    if family=="E2" and (analysis.get("experiment")!="E2" or analysis.get("representation_dimension")!=1536):
        raise ValueError("E2 historical feature topology mismatch")
    report_path=Path(source["actual_report_path"])
    if sha256_file(report_path)!=source["report_sha256"]: raise ValueError("Original aggregate bytes differ")
    report=read_json(report_path)
    if report.get("kind")!=f"{family.lower()}_three_seed_validation_report_v1" or report["contract_sha256"]!=source["source_contract_sha256"]:
        raise ValueError("Historical aggregate schema mismatch")
    selected=next((r for r in report["per_seed"] if r["seed"]==source["source_seed"]),None)
    if not selected or selected["best_sha256"]!=source["binary_sha256"] or selected["selected_epoch"]!=source["selected_epoch"]:
        raise ValueError("Original source seed/selection mismatch")
    metrics=analysis["metrics"]
    if any(metrics[k]!=selected[k] for k in (*SCORES,"nll")):
        raise ValueError("Original reference differs from immutable aggregate")
    run=path.parent.parent
    if path.name!="analysis.json" or path.parent.name!="validation_analysis": raise ValueError("Explicit historical selected-analysis role required")
    history=read_json(run/"history.json")
    if sha256_file(run/"history.json")!=analysis["history_sha256"]: raise ValueError("Original history bytes differ")
    _same_metrics(metrics,history[source["selected_epoch"]-1]["validation"])
    # This narrow completed helper checks the common CSV/prediction wire format.
    # E1 keeps its own historical envelope; no E2 top-level metadata is imposed on it.
    from ..e2.report import _validate_prediction_exports
    _validate_prediction_exports(run,metrics,history)
    predictions=[{"relative_path":r["relative_path"],"label_id":r["label_id"],"prediction":r["top5"][0],"top5":r["top5"]} for r in metrics["predictions"]]
    return {"source":source,"metrics":metrics,"predictions":predictions,"analysis_path":str(path)}


def evaluate_selected(config, run_dir: Path) -> dict:
    from .runtime import preflight,configure_runtime,ExecutionContext
    from .cache import CacheDataset
    from .engine import evaluate_head
    checked=preflight(config,operation=config.request["operation"] if isinstance(config,ExecutionContext) else config["operation"])
    row=validate_completed_head_run(run_dir); declaration=row["declaration"]
    if checked["declaration"]!=declaration: raise ValueError("Selected evaluation declaration differs")
    root=Path(run_dir).resolve(); output=root/"validation_analysis"
    with writer_lock(root):
        configure_runtime(root_seed=declaration["root_seed"])
        dataset=CacheDataset(contained(ROOT,declaration["cache"]["validation_index_path"]),expected={"family":declaration["family"],
            "source_seed":declaration["source_seed"],"split":"val","source":declaration["source"],"runtime":declaration["runtime"],
            "cache_acceptance_receipt":checked["artifact"](declaration["cache"]["cache_acceptance_sha256"])})
        head=torch.nn.Linear(DIMENSIONS[declaration["family"]],124,bias=True)
        head.load_state_dict(row["selected_weights"],strict=True); head.to("cuda:0")
        result=evaluate_head(head,DataLoader(dataset,batch_size=512,shuffle=False,num_workers=0,drop_last=False),
                             manifest=FixedManifest(),device="cuda:0",collect_predictions=True)
        _same_metrics(result["metrics"],row["selected_metrics"])
        state_hash=tensor_state_sha256(row["selected_weights"])
        content={"kind":"e3_selected_head_analysis_payload_v1","run_id":declaration["run_id"],"selected":row["progress"]["selected"],
                 "metrics":result["metrics"],"predictions":result["predictions"],"history_sha256":sha256_file(root/"history.json"),
                 "nll_evidence":{"selected_head_sha256":state_hash,"ordered_row_ce_sha256":fingerprint([r["nll"] for r in result["predictions"]]),
                                 "mean":math.fsum(r["nll"] for r in result["predictions"])/7497}}
        tables=_tables(result["metrics"],result["predictions"],row["progress"]["history"],FixedManifest().labels)
        for name,(rows,fields) in tables.items(): atomic_write(output/name,lambda stream,r=rows,f=fields:stream.write(_csv_bytes(r,f)))
        atomic_json(output/"analysis.json",content)
        index={"kind":"e3_selected_validation_analysis_v1","run_id":declaration["run_id"],"split":"val","rows":7497,
               "selected_head_sha256":state_hash,"selected_update":row["progress"]["selected"]["updates"],
               "declaration_sha256":sha256_file(root/"declaration.json"),**{k:declaration[k] for k in ("source","cache","profile")},
               "exports":[payload(output/name,output) for name in ["analysis.json",*tables]],"content_verification_sha256":fingerprint(content)}
        validate_record(index,"selected_analysis")
        atomic_json(output/"analysis_index.json",index)
    return validate_analysis(root)


def validate_analysis(run_dir: Path) -> dict:
    row=validate_completed_head_run(run_dir); d=row["declaration"]; root=Path(run_dir).resolve(); folder=root/"validation_analysis"
    index=validate_record(read_json(folder/"analysis_index.json"),"selected_analysis")
    if (index["run_id"]!=d["run_id"] or index["selected_head_sha256"]!=tensor_state_sha256(row["selected_weights"])
            or index["selected_update"]!=row["progress"]["selected"]["updates"]
            or index["declaration_sha256"]!=sha256_file(root/"declaration.json")
            or any(index[k]!=d[k] for k in ("source","cache","profile"))):
        raise ValueError("Analysis selected state/declaration/lineage mismatch")
    expected={"analysis.json","overall.csv","per_class.csv","buckets.csv","predictions.csv","confusion.csv","most_confused.csv","learning_curves.csv"}
    if len(index["exports"])!=8 or {p["relative_path"] for p in index["exports"]}!=expected:
        raise ValueError("Exactly all eight selected exports required")
    for p in index["exports"]: verify_payload(p,folder)
    content=read_json(folder/"analysis.json")
    if (fingerprint(content)!=index["content_verification_sha256"] or content["selected"]!=row["progress"]["selected"]
            or content["history_sha256"]!=sha256_file(root/"history.json")):
        raise ValueError("Analysis content/history binding differs")
    manifest=FixedManifest(); predictions=content["predictions"]
    _validate_predictions(predictions,manifest.records("val"))
    nll=math.fsum(r["nll"] for r in predictions)/7497
    if (not all(math.isfinite(r["nll"]) and r["nll"]>=0 for r in predictions)
            or content["nll_evidence"]!={"selected_head_sha256":index["selected_head_sha256"],
                "ordered_row_ce_sha256":fingerprint([r["nll"] for r in predictions]),"mean":nll}):
        raise ValueError("NLL requires complete selected-head per-row CE evidence")
    rebuilt=_metrics_from_predictions(predictions,manifest.n_train,manifest.labels,nll=nll)
    _same_metrics(rebuilt,content["metrics"]); _same_metrics(rebuilt,row["selected_metrics"])
    _validate_tables(folder,_tables(rebuilt,predictions,row["progress"]["history"],manifest.labels))
    return {"index":index,"metrics":rebuilt,"predictions":predictions,"run":row}


def aggregate_study(head_run_dirs: list[Path], source_manifest: dict, cache_indices: list[dict], protocol: dict) -> dict:
    validate_common_protocol(protocol)
    if len(head_run_dirs)!=12 or len(set(map(lambda p:str(Path(p).resolve()),head_run_dirs)))!=12:
        raise ValueError("Twelve distinct complete final head runs required")
    heads=[validate_analysis(p) for p in head_run_dirs]
    declarations=validate_final_declarations([r["run"]["declaration"] for r in heads])
    if any(d["profile"]["profile_sha256"]!=protocol["candidate_profile_sha256"] for d in declarations):
        raise ValueError("Study profile differs from final common profile")
    if source_manifest["registry_sha256"]!=SOURCE_SHA256 or len(source_manifest["sources"])!=6 or len(source_manifest["original_reference_indices"])!=6:
        raise ValueError("Complete fixed source registry/original reference package required")
    sources={(s["family"],s["source_seed"]):s for s in source_manifest["sources"]}
    if set(sources)!={(f,s) for f in DIMENSIONS for s in SEEDS}: raise ValueError("Missing original source seeds")
    references=[]
    for source,p in zip(source_manifest["sources"],source_manifest["original_reference_indices"]):
        references.append(load_source_reference(verify_payload(p),source=source))
    from .cache import validate_cache
    caches={}
    for descriptor in cache_indices:
        path=Path(descriptor["index_path"]); index=read_json(path)
        validate_cache(path,expected={k:index[k] for k in ("family","source_seed","split")})
        key=(index["family"],index["source_seed"],index["split"])
        if key in caches: raise ValueError("Duplicate study cache")
        caches[key]=sha256_file(path)
    if len(caches)!=12: raise ValueError("Complete twelve split-cache indices required")
    rows=[]
    for item in heads:
        d=item["run"]["declaration"]
        if d["source"]!=sources[d["family"],d["source_seed"]]: raise ValueError("Reference/retrained source differs")
        if (d["cache"]["train_index_sha256"]!=caches[d["family"],d["source_seed"],"train"]
                or d["cache"]["validation_index_sha256"]!=caches[d["family"],d["source_seed"],"val"]):
            raise ValueError("Study cache lineage differs")
        rows.append({"arm":f"E3-{d['family']}-{'N' if d['arm']=='natural' else 'cRT'}","seed":d["source_seed"],"metrics":item["metrics"]})
    rows.extend({"arm":r["source"]["family"]+"-original","seed":r["source"]["source_seed"],"metrics":r["metrics"]} for r in references)
    report=_summarize_study_rows(rows)
    destination=contained(ROOT,policy()["planned_output_roots"]["paths"]["reports"])
    with writer_lock(destination):
        atomic_json(destination/"study_metrics.json",report)
        atomic_write(destination/"study_report.md",lambda stream:stream.write(render_study(report).encode("utf-8")))
        index={"kind":"e3_study_validation_report_v1","split":"val","seeds":list(SEEDS),"arms":list(ARMS),
               "source_manifest_sha256":SOURCE_SHA256,"common_profile_sha256":protocol["candidate_profile_sha256"],
               "completed_new_head_runs":12,"verified_original_references":6,"ddof":1,
               "analysis_indices":[payload(Path(r["run"]["run_dir"])/"validation_analysis/analysis_index.json") for r in heads],
               "original_reference_indices":source_manifest["original_reference_indices"],
               "exports":[payload(destination/n,destination) for n in ("study_metrics.json","study_report.md")],
               "comparison_levels":["PRIMARY","SECONDARY","EXPLORATORY"],"test_access":"LOCKED / NOT AUTHORIZED"}
        validate_record(index,"study_report"); atomic_json(destination/"study_index.json",index)
    return index


def render_study(report: dict) -> str:
    lines=["# E3 validation comparison","","Classification: mean +/- sample std (%), ddof=1; differences: percentage points.",
           "NLL is diagnostic and is not a percentage.","","| Arm | Top-1 | Top-3 | Top-5 | Supported Macro-F1 | Weighted-F1 | NLL |",
           "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        cells=[]
        for metric in (*SCORES,"nll"):
            s=report["arms"][arm][metric]; scale=1 if metric=="nll" else 100
            cells.append(f"{scale*s['mean']:.4f} +/- {scale*s['std']:.4f}")
        lines.append("| "+" | ".join([arm,*cells])+" |")
    lines.extend(["","| Contrast | Supported Macro-F1 delta (pp) |","|---|---|"])
    for name,values in report["contrasts"].items():
        s=values["macro_f1_supported"]; lines.append(f"| {name} | {100*s['mean']:.4f} +/- {100*s['std']:.4f} |")
    lines.extend(["",*report["limitations"],"",report["test_access"],""])
    return "\n".join(lines)
