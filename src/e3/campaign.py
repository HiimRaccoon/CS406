"""Complete-run validation, finite common-profile selection and twelve declarations."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import statistics

import torch

from .checkpoint import read_head_checkpoint
from .config import (ROOT, SEEDS, DIMENSIONS, fingerprint, payload, policy, read_json,
                     require_approval, resolve_head_run, sha256_file, validate_common_protocol,
                     validate_record, validate_source_binding, verify_payload, contained)
from .io import atomic_json,atomic_write,writer_lock
from .model import make_paired_heads,tensor_state_sha256
from .runtime import ExecutionContext,_validate_declaration,preflight


def validate_completed_head_run(run_dir: Path) -> dict:
    root=Path(run_dir).resolve()
    try:
        complete=read_json(root/"complete.json")
        if complete["kind"]!="e3_completed_head_run_v1": raise ValueError("Wrong completed-run kind")
        declaration=read_json(verify_payload(complete["declaration"],root))
        operation=_validate_declaration(declaration)
        wrapper=read_json(verify_payload(complete["execution_request"],root))
        inventory_path=verify_payload(complete["input_inventory"],root)
        checked=preflight(ExecutionContext(wrapper,inventory_path),operation=operation)
        if checked["declaration"]!=declaration or complete["stage"]!=declaration["stage"] or complete["run_id"]!=declaration["run_id"]:
            raise ValueError("Completed-run declaration/stage mismatch")
        header=validate_record(read_json(verify_payload(complete["checkpoint"],root)),"head_checkpoint")
        last=verify_payload(header["full_state_payload"],root)
        if last!=root/"last.pt": raise ValueError("Authoritative last role required")
        identity={k:declaration[k] for k in ("source","cache","initialization","profile","runtime")}
        state=read_head_checkpoint(last,config=declaration,identity=identity)
        progress=state["progress"]; budget=declaration["original_budget"]
        if (progress["window"]!=budget["windows"] or header["committed_window"]!=progress["window"]
                or header["committed_updates"]!=progress["updates"] or header["committed_presentations"]!=progress["presentations"]
                or header["declaration_sha256"]!=complete["declaration"]["sha256"]
                or any(header[k]!=declaration[k] for k in identity)):
            raise ValueError("Run did not complete its exact original budget/lineage")
        if read_json(verify_payload(header["history_payload"],root))!=progress["history"]:
            raise ValueError("History publication differs from committed state")
        best=torch.load(verify_payload(header["selected_head_state"]["head_payload"],root),map_location="cpu",weights_only=True)
        if (best["run_id"]!=declaration["run_id"] or best["selected"]!=progress["selected"]
                or best["tensor_state_sha256"]!=tensor_state_sha256(progress["selected_weights"])
                or tensor_state_sha256(best["state"])!=best["tensor_state_sha256"]):
            raise ValueError("Selected state does not match authoritative history")
        attempts=[read_json(verify_payload(p,root)) for p in complete["attempts"]]
        if (not 1<=len(attempts)<=2 or [a["attempt"] for a in attempts]!=list(range(1,len(attempts)+1))
                or any(a["run_id"]!=declaration["run_id"] or a["declaration_sha256"]!=complete["declaration"]["sha256"] for a in attempts)
                or attempts[-1]["status"]!="COMPLETE"):
            raise ValueError("Attempt inventory incomplete or outside retry cap")
        selected_metrics=progress["history"][progress["selected"]["window"]-1]["validation"]
        return {"run_dir":str(root),"declaration":declaration,"progress":progress,"selected_weights":progress["selected_weights"],
                "selected_metrics":selected_metrics,"attempt_records":attempts,"complete":complete}
    except (KeyError,TypeError,OSError,RuntimeError) as exc:
        raise ValueError("Incomplete/corrupt completed head run") from exc


def _choose_protocol(trials: list[dict], envelope: dict) -> dict:
    if envelope!=policy()["pilot"]: raise ValueError("Unreviewed pilot envelope")
    profiles={p["candidate_id"]:p for p in policy()["candidate_profiles"]}
    required={(p,f,a) for p in profiles for f in DIMENSIONS for a in ("natural","crt")}
    keys=[(r["candidate_id"],r["family"],r["arm"]) for r in trials]
    if len(trials)!=8 or set(keys)!=required or len(set(keys))!=len(keys):
        raise ValueError("Both finite protocols must complete all four arms")
    if any(r["stage"]!="pilot" or r["source_seed"]!=2001 or r["root_seed"]!=1001
           or not 1<=r["attempts"]<=2 or len(r["history"])!=20 for r in trials):
        raise ValueError("Wrong pilot source/root/stage/budget/attempt scope")
    rule=envelope["common_protocol_selection"]; adequacy=rule["adequacy_per_arm"]
    diagnostics={}; scores={}
    for candidate in profiles:
        arms=[r for r in trials if r["candidate_id"]==candidate]
        eligible=True; checks=[]
        for trial in arms:
            history=trial["history"]
            if [r["window"] for r in history]!=list(range(1,21)): raise ValueError("Truncated/reordered pilot history")
            previous=statistics.mean(r["train"]["ce"] for r in history[10:15])
            late=statistics.mean(r["train"]["ce"] for r in history[15:20])
            relative=abs(late-previous)/max(abs(previous),1e-12)
            late_scores=[r["validation"]["macro_f1_supported"] for r in history[15:20]]
            earlier=[r["validation"]["macro_f1_supported"] for r in history[:15]]
            numbers=[previous,late,relative,*late_scores,*earlier,trial["selected_metric"]]
            if not all(math.isfinite(v) for v in numbers): raise ValueError("Nonfinite pilot history/score")
            passed=(relative<=adequacy["relative_training_ce_mean_change_max"]
                    and max(late_scores)-min(late_scores)<=adequacy["late_macro_f1_range_max"]
                    and max(late_scores)-max(earlier)<=adequacy["late_best_minus_earlier_best_max"])
            checks.append({"family":trial["family"],"arm":trial["arm"],"relative_training_ce_mean_change":relative,
                           "late_macro_f1_range":max(late_scores)-min(late_scores),
                           "late_best_minus_earlier_best":max(late_scores)-max(earlier),"adequate":passed})
            eligible &= passed
        score=statistics.mean(r["selected_metric"] for r in arms)
        diagnostics[candidate]={"eligible":eligible,"score":score,"arms":checks}
        if eligible: scores[candidate]=score
    if not scores: raise ValueError("No eligible common protocol; STOP for review")
    maximum=max(scores.values())
    tied=[p for p,score in scores.items() if maximum-score<=rule["tie_tolerance"]]
    winner=min(tied,key=lambda p:(profiles[p]["base_lr"],p))
    return {"kind":"e3_pilot_outcome_candidate_v1","status":"REVIEW_REQUIRED","winner_candidate_id":winner,
            "profile":copy.deepcopy(profiles[winner]),"diagnostics":diagnostics,"official_final_authorized":False,
            "attempts":sum(r["attempts"] for r in trials),"caps":envelope["caps"]}


def select_common_protocol(pilot_records: list[dict], envelope: dict) -> dict:
    parsed=[]
    for row in pilot_records:
        if not isinstance(row,dict) or not row.get("run_dir"):
            raise ValueError("Pilot selection requires fully validated completed run artifacts")
        parsed.append(validate_completed_head_run(Path(row["run_dir"])))
    trials=[]
    for row in parsed:
        d=row["declaration"]
        trials.append({"candidate_id":d["profile"]["candidate_id"],"family":d["family"],"arm":d["arm"],
                       "source_seed":d["source_seed"],"root_seed":d["root_seed"],"stage":d["stage"],
                       "selected_metric":row["selected_metrics"]["macro_f1_supported"],"history":row["progress"]["history"],
                       "attempts":len(row["attempt_records"])})
    result=_choose_protocol(trials,envelope)
    result["run_indices"]=[payload(Path(row["run_dir"])/"complete.json") for row in parsed]
    return result


def validate_final_declarations(declarations: list[dict]) -> list[dict]:
    if not isinstance(declarations,list) or len(declarations)!=12:
        raise ValueError("Exactly twelve final declarations required")
    keys=[]; profiles=set(); freezes=set()
    try:
        for d in declarations:
            if _validate_declaration(d)!="final_head": raise ValueError("Pilot/acceptance cannot substitute for final")
            keys.append((d["family"],d["source_seed"],d["arm"]))
            profiles.add(fingerprint(d["profile"])); freezes.add(d["head_freeze_receipt_sha256"])
        if (set(keys)!={(f,s,a) for f in DIMENSIONS for s in SEEDS for a in ("natural","crt")}
                or len(set(keys))!=12 or len(profiles)!=1 or len(freezes)!=1):
            raise ValueError("Missing/extra final IDs or independently tuned protocols")
        for family in DIMENSIONS:
            for seed in SEEDS:
                pair=[d for d in declarations if (d["family"],d["source_seed"])==(family,seed)]
                if any(pair[0][k]!=pair[1][k] for k in ("source","cache","initialization","profile","runtime","original_budget")):
                    raise ValueError("Final pair does not share source/cache/actual initial state/common protocol")
    except (KeyError,TypeError) as exc:
        raise ValueError("Incomplete final campaign") from exc
    return declarations


def plan_final(protocol: dict, source_manifest: dict, cache_indices: list[dict], paths: dict) -> list[dict]:
    validate_common_protocol(protocol)
    freeze=Path(paths["head_freeze_receipt"])
    require_approval(freeze,gate="E3-HEAD-FREEZE",expected={"subjects":{"common_profile":protocol["candidate_profile_sha256"]}})
    sources=source_manifest["sources"]
    if len(sources)!=6 or {(s["family"],s["source_seed"]) for s in sources}!={(f,s) for f in DIMENSIONS for s in SEEDS}:
        raise ValueError("Complete six-source handoff required")
    from .cache import validate_cache
    indices={}
    receipt=Path(paths["cache_acceptance_receipt"])
    for descriptor in cache_indices:
        file=Path(descriptor["index_path"])
        raw=read_json(file)
        index=validate_cache(file,expected={k:raw[k] for k in ("family","source_seed","split")})
        key=(index["family"],index["source_seed"],index["split"])
        if key in indices: raise ValueError("Duplicate cache input")
        indices[key]=(file,index)
    if set(indices)!={(f,s,t) for f in DIMENSIONS for s in SEEDS for t in ("train","val")}:
        raise ValueError("All twelve accepted cache indices required")
    require_approval(receipt,gate="E3-CACHE-ACCEPT",expected={"subjects":{
        f"cache:{f}:{s}:{t}":sha256_file(file) for (f,s,t),(file,index) in indices.items()}})
    declarations=[]
    for source in sources:
        validate_source_binding(source)
        family,seed=source["family"],source["source_seed"]
        train_file,train=indices[(family,seed,"train")]; val_file,val=indices[(family,seed,"val")]
        if train["source"]!=source or val["source"]!=source or train["runtime"]!=val["runtime"]:
            raise ValueError("Source/cache/runtime mismatch")
        first,_,initial=make_paired_heads(family=family,seed=seed,protocol=protocol,stage="final",root_seed=seed)
        init_root=contained(ROOT,paths["initialization_root"])
        init_path=init_root/f"{family}_seed{seed}_{initial['tensor_state_sha256']}.pt"
        if init_path.exists():
            if tensor_state_sha256(torch.load(init_path,map_location="cpu",weights_only=True)["state"])!=initial["tensor_state_sha256"]:
                raise ValueError("Existing paired initial state differs")
        else:
            atomic_write(init_path,lambda stream:torch.save({"kind":"e3_initial_head_tensor_v1","state":first.state_dict(),**initial},stream))
        initial["tensor_payload_path"]=init_path.relative_to(ROOT).as_posix()
        bindings={"source":source,"runtime":train["runtime"],"initialization":initial,"head_freeze_receipt":freeze,
                  "cache":{"train_index_path":train_file.resolve().relative_to(ROOT).as_posix(),"train_index_sha256":sha256_file(train_file),
                           "validation_index_path":val_file.resolve().relative_to(ROOT).as_posix(),"validation_index_sha256":sha256_file(val_file),
                           "cache_acceptance_sha256":sha256_file(receipt)}}
        for arm in ("natural","crt"):
            declarations.append(resolve_head_run(policy(),protocol,stage="final",family=family,seed=seed,arm=arm,bindings=bindings))
    return validate_final_declarations(declarations)


def write_declarations(output: Path, declarations: list[dict]) -> None:
    validate_final_declarations(declarations)
    with writer_lock(output):
        for d in declarations:
            path=Path(output)/(d["run_id"]+".json")
            if path.exists() and read_json(path)!=d: raise ValueError("Cannot overwrite immutable declaration")
            if not path.exists(): atomic_json(path,d)
        manifest=Path(output)/"declarations.json"
        if manifest.exists() and read_json(manifest)!=declarations: raise ValueError("Campaign manifest is immutable")
        if not manifest.exists(): atomic_json(manifest,declarations)
