"""One immutable standalone-head run; no source encoder is loaded here."""
from __future__ import annotations

import copy
from pathlib import Path
import traceback

import torch
from torch.utils.data import DataLoader

from .config import ROOT, contained, fingerprint, payload, policy, read_json, sha256_file
from .checkpoint import capture_head_state,restore_head_state,save_head_checkpoint,read_head_checkpoint,publish_derived
from .engine import train_window,evaluate_head,make_head_scheduler,select_head
from .io import atomic_json,writer_lock,publish_exact_bytes
from .model import make_paired_heads,build_head_optimizer,tensor_state_sha256
from .runtime import ExecutionContext,preflight,configure_runtime
from ..e1.data import derived_seed


def _execute_windows(*, budget, progress, train, evaluate, weights, commit, stop_requested):
    progress=copy.deepcopy(progress)
    while progress["window"] < budget["windows"] and not stop_requested():
        trained=train()
        if trained["presentations"] != budget["presentations_per_window"] or trained["updates"] != budget["updates_per_window"]:
            raise ValueError("Uncommitted incomplete/excessive window")
        validation=evaluate()
        progress["window"]+=1
        progress["presentations"]+=trained["presentations"]; progress["updates"]+=trained["updates"]
        progress["history"].append({"window":progress["window"],"presentations":progress["presentations"],
                                    "updates":progress["updates"],"train":trained,"validation":validation})
        candidate={"window":progress["window"],"updates":progress["updates"],"macro_f1_supported":validation["macro_f1_supported"]}
        if select_head(candidate,progress["selected"]):
            progress["selected"]=candidate; progress["selected_weights"]=weights()
        commit(copy.deepcopy(progress))
    return progress


def _attempt_accounting(*, start, end, budget, interrupted):
    if type(start) is not int or type(end) is not int or not 0<=start<=end<=budget["windows"]:
        raise ValueError("Invalid attempt start/end boundary")
    windows=end-start+int(interrupted and end<budget["windows"])
    return {"train_presentations":windows*budget["presentations_per_window"],
            "optimizer_updates":windows*budget["updates_per_window"],"validation_passes":windows}


def run_head(config, run_dir: Path, *, resume: Path | None = None, stop_requested=lambda:False) -> dict:
    operation=config.request["operation"] if isinstance(config,ExecutionContext) else config["operation"]
    if operation not in {"pilot_head","final_head"}: raise ValueError("Head run requires head operation wrapper")
    checked=preflight(config,operation=operation); declaration=checked["declaration"]
    root=Path(run_dir).resolve()
    expected=contained(ROOT,policy()["planned_output_roots"]["paths"][declaration["stage"]]+"/"+declaration["run_id"])
    if root!=expected: raise ValueError("Run destination differs from exact planned identity")
    if resume is not None and Path(resume).resolve()!=root/"last.pt": raise ValueError("Same-run last.pt resume only")
    if (root/"last.pt").exists() and resume is None: raise ValueError("Existing run requires explicit same-run resume")
    protocol=next(p for p in policy()["candidate_profiles"] if p["candidate_id"]==declaration["profile"]["candidate_id"])
    identity={k:declaration[k] for k in ("source","cache","initialization","profile","runtime")}
    with writer_lock(root):
        publish_exact_bytes(root/"declaration.json",verify_declaration_path(checked).read_bytes())
        saved_resume=read_head_checkpoint(resume,config=declaration,identity=identity) if resume is not None else None
        started_window=saved_resume["progress"]["window"] if saved_resume else 0
        attempts=root/"attempts"; attempts.mkdir(exist_ok=True)
        previous=sorted(attempts.glob("attempt_*.json"))
        if len(previous)>=2: raise ValueError("At most initial plus one same-identity retry")
        for previous_path in previous:
            old=read_json(previous_path)
            if old["run_id"]!=declaration["run_id"] or old["declaration_sha256"]!=sha256_file(root/"declaration.json"):
                raise ValueError("Foreign attempt inventory")
            if old["status"]=="STARTED":
                old.update(status="INTERRUPTED",ended_window=started_window,
                    operational_charge=_attempt_accounting(start=old.get("started_window",0),end=started_window,
                        budget=declaration["original_budget"],interrupted=True))
                atomic_json(previous_path,old)
        attempt=attempts/f"attempt_{len(previous)+1:04d}.json"
        record={"kind":"e3_head_attempt_v1","run_id":declaration["run_id"],"attempt":len(previous)+1,
                "declaration_sha256":sha256_file(root/"declaration.json"),"status":"STARTED","started_window":started_window}
        atomic_json(attempt,record)
        try:
            environment=configure_runtime(root_seed=declaration["root_seed"])
            from .runtime import verify_current_environment
            verify_current_environment(checked,environment)
            atomic_json(root/"execution_request.json",checked["context"].request)
            from .io import atomic_write
            inventory_bytes=Path(checked["context"].inventory_path).read_bytes()
            atomic_write(root/"input_inventory.json",lambda stream:stream.write(inventory_bytes))
            from .cache import CacheDataset
            from .data import FixedManifest
            from .sampling import make_sampler
            acceptance=checked["artifact"](declaration["cache"]["cache_acceptance_sha256"])
            datasets={}
            for split,key in [("train","train"),("val","validation")]:
                datasets[split]=CacheDataset(contained(ROOT,declaration["cache"][key+"_index_path"]),expected={
                    "family":declaration["family"],"source_seed":declaration["source_seed"],"split":split,
                    "source":declaration["source"],"runtime":declaration["runtime"],"cache_acceptance_receipt":acceptance})
            sampler=make_sampler(datasets["train"].index,policy=declaration["arm"],seed=declaration["root_seed"],
                rng_contract={"policy":policy()["rng"],"stage":declaration["stage"],"root_seed":declaration["root_seed"],"source_seed":declaration["source_seed"]},cache_root=datasets["train"].root)
            loaders={}
            for split in ("train","val"):
                generator=torch.Generator(device="cpu").manual_seed(derived_seed(declaration["root_seed"],"E3-v1",declaration["stage"],declaration["source_seed"],"loader",split))
                loaders[split]=DataLoader(datasets[split],batch_size=512,sampler=sampler if split=="train" else None,
                                          shuffle=False,num_workers=0,pin_memory=False,drop_last=False,generator=generator)
            first,second,initial=make_paired_heads(family=declaration["family"],seed=declaration["source_seed"],protocol=protocol,
                                                   stage=declaration["stage"],root_seed=declaration["root_seed"])
            initial_path=contained(ROOT,declaration["initialization"]["tensor_payload_path"])
            inventory=checked["inventory"]["artifacts"]
            actual=next((p for p in inventory.values() if p["relative_path"]==initial_path.relative_to(ROOT).as_posix()),None)
            if actual is None: raise ValueError("Initialization payload absent from explicit inventory")
            from .config import verify_payload
            verify_payload(actual)
            state=torch.load(initial_path,map_location="cpu",weights_only=True)
            initial_weights=state["state"]
            if (tensor_state_sha256(initial_weights)!=declaration["initialization"]["tensor_state_sha256"]
                    or initial["tensor_state_sha256"]!=declaration["initialization"]["tensor_state_sha256"]):
                raise ValueError("Paired initial tensor state differs from approved declaration")
            head=first if declaration["arm"]=="natural" else second
            head.load_state_dict(initial_weights,strict=True); head.to("cuda:0")
            optimizer,_=build_head_optimizer(head,protocol=protocol); scheduler=make_head_scheduler(optimizer,protocol=protocol)
            progress={"window":0,"presentations":0,"updates":0,"history":[],"selected":None,"selected_weights":None}
            def commit(progress):
                captured=capture_head_state(head=head,optimizer=optimizer,scheduler=scheduler,loaders=loaders,sampler=sampler,progress=progress)
                save_head_checkpoint(root/"last.pt",state=captured,config=declaration,identity=identity,role="last")
            if resume is not None:
                saved=saved_resume
                progress=restore_head_state(saved,head=head,optimizer=optimizer,scheduler=scheduler,loaders=loaders,sampler=sampler)
            else:
                commit(progress)
            manifest=FixedManifest()
            record["started_window"]=progress["window"]
            atomic_json(attempt,record)
            next_window=progress["window"]
            def train():
                nonlocal next_window
                sampler.set_window(next_window)
                result=train_window(head,loaders["train"],optimizer,scheduler,protocol=protocol,device="cuda:0")
                next_window+=1
                return result
            final=_execute_windows(budget=protocol["budget"],progress=progress,train=train,
                evaluate=lambda:evaluate_head(head,loaders["val"],manifest=manifest,device="cuda:0",collect_predictions=False)["metrics"],
                weights=lambda:{k:v.detach().cpu().clone() for k,v in head.state_dict().items()},commit=commit,stop_requested=stop_requested)
            record.update(status="COMPLETE" if final["window"]==20 else "STOPPED",committed_window=final["window"],
                          committed_presentations=final["presentations"],committed_updates=final["updates"],environment=environment,
                          ended_window=final["window"],operational_charge=_attempt_accounting(start=started_window,end=final["window"],
                            budget=protocol["budget"],interrupted=False))
            atomic_json(attempt,record)
            if final["window"]==20:
                saved=read_head_checkpoint(root/"last.pt",config=declaration,identity=identity)
                header=publish_derived(root/"last.pt",saved)
                completed={"kind":"e3_completed_head_run_v1","run_id":declaration["run_id"],"stage":declaration["stage"],
                           "declaration":payload(root/"declaration.json",root),"checkpoint":payload(root/"last.json",root),
                           "execution_request":payload(root/"execution_request.json",root),"input_inventory":payload(root/"input_inventory.json",root),
                           "attempts":[payload(p,root) for p in sorted(attempts.glob("attempt_*.json"))]}
                atomic_json(root/"complete.json",completed)
            return {"run_id":declaration["run_id"],"status":record["status"],"progress":final}
        except BaseException as exc:
            end=started_window
            if (root/"last.pt").exists():
                try: end=read_head_checkpoint(root/"last.pt",config=declaration,identity=identity)["progress"]["window"]
                except (ValueError,OSError,RuntimeError):
                    end=protocol["budget"]["windows"]
                    record["unknown_work"]="Unreadable last state; full remaining attempt charged; review required"
            record.update(status="FAILED",exception=type(exc).__name__,message=str(exc),ended_window=end,
                          operational_charge=_attempt_accounting(start=started_window,end=end,budget=protocol["budget"],interrupted=True))
            atomic_json(attempt,record)
            raise


def verify_declaration_path(checked):
    from .config import verify_payload
    return verify_payload(checked["context"].request["scientific_declaration"])
