"""Read-only inspection, exact operation receipts and accepted input closures."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import os
from pathlib import Path
import platform

from .config import (ROOT, OP_GATES, CACHE_SHA256, NUMERIC_SHA256, SCHEMA_SHA256,
                     DIMENSIONS, contained, fingerprint, g1_receipt, load_contract, numeric,
                     policy, read_json, reject_unresolved, require_approval, sha256_file,
                     validate_record, validate_source_binding, verify_payload)


@dataclass(frozen=True)
class ExecutionContext:
    """Operational lookup inventory is separately byte-bound by the operation receipt."""
    request: dict
    inventory_path: Path


def code_inventory() -> dict:
    pending = list((ROOT/"src/e3").glob("*.py"))
    files = {}
    while pending:
        path = pending.pop()
        relative = path.relative_to(ROOT).as_posix()
        if relative in files: continue
        files[relative] = sha256_file(path)
        package = relative[:-3].split("/")[:-1]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = []
            if isinstance(node,ast.ImportFrom):
                base = package[:len(package)-node.level+1] if node.level else []
                module = base + (node.module.split(".") if node.module else [])
                modules = [module] + [module+alias.name.split(".") for alias in node.names]
            elif isinstance(node,ast.Import):
                modules = [alias.name.split(".") for alias in node.names]
            for module in modules:
                if module and module[0] == "src":
                    dependency = ROOT.joinpath(*module).with_suffix(".py")
                    if dependency.is_file(): pending.append(dependency)
    return {"kind":"e3_consumed_source_inventory_v1","files":dict(sorted(files.items()))}


def verify_code_inventory(value: dict):
    if value.get("kind") != "e3_consumed_source_inventory_v1" or value.get("files") != code_inventory()["files"]:
        raise ValueError("Consumed scientific source closure changed; renewed acceptance required")


def _operation(declaration):
    return {"e3_acceptance_request_v1":"bounded_acceptance","e3_cache_request_v1":"bulk_cache",
            "e3_pilot_run_v1":"pilot_head","e3_final_head_run_v1":"final_head"}.get(declaration.get("kind"))


def declaration_role(declaration):
    operation = _operation(declaration)
    if operation in {"pilot_head","final_head"}: return "declaration:"+declaration["run_id"]
    if operation == "bulk_cache": return f"declaration:{declaration['family']}:{declaration['source_seed']}:{declaration['split']}"
    return "scientific_declaration"


def _validate_declaration(d):
    validate_record(d); reject_unresolved(d)
    operation = _operation(d)
    if operation is None: raise ValueError("Executable stage declaration required")
    if d["required_authorization_gate"] != OP_GATES[operation]: raise ValueError("Wrong declared gate")
    fixed = {k:v["sha256"] for k,v in numeric()["data"]["artifacts"].items()}
    if operation == "bounded_acceptance":
        if (d["scope"] != policy()["bounded_acceptance"] or d["cache_pilot_contract_sha256"] != CACHE_SHA256
                or d["runtime"]["requested_runtime_policy_sha256"] != fingerprint(policy()["runtime"])
                or d["runtime"]["allocation_policy_sha256"] != fingerprint(policy()["runtime"]["allocation"])
                or d["runtime"]["fixed_metadata_sha256"] != fixed):
            raise ValueError("Bounded request differs from approved finite policy")
        keys=set()
        for source in d["sources"]:
            validate_source_binding(source,accepted=False); keys.add((source["family"],source["source_seed"]))
        if keys != {(f,s) for f in DIMENSIONS for s in (2001,3001,4001)} or len(d["sources"]) != 6:
            raise ValueError("Acceptance requires all six fixed source lineages")
    else:
        validate_source_binding(d["source"])
        if (d["source"]["family"],d["source"]["source_seed"]) != (d["family"],d["source_seed"]):
            raise ValueError("Source/run family or seed mismatch")
        if operation == "bulk_cache":
            if (d["fixed_artifact_sha256"] != fixed or d["cache_policy_sha256"] != fingerprint(policy()["storage"])
                    or d["preprocessing_sha256"] != numeric()["preprocessing"]["config"]["sha256"]):
                raise ValueError("Unapproved fixed/cache/preprocessing identity")
        else:
            profile = next((p for p in policy()["candidate_profiles"] if p["candidate_id"]==d["profile"]["candidate_id"]),None)
            expected = {"candidate_id":profile["candidate_id"],"profile_sha256":profile["candidate_profile_sha256"],
                        "numeric_contract_sha256":NUMERIC_SHA256,"cache_pilot_contract_sha256":CACHE_SHA256,"schema_bundle_sha256":SCHEMA_SHA256}
            if d["profile"] != expected or d["original_budget"] != profile["budget"]:
                raise ValueError("Profile/budget differs from common approved candidate")
            root = 1001 if d["stage"]=="pilot" else d["source_seed"]
            from ..e1.data import derived_seed
            init_seed=derived_seed(root,"E3-v1",d["stage"],d["family"],d["source_seed"],"head-init")
            if (d["root_seed"] != root or d["initialization"]["fan_in"] != DIMENSIONS[d["family"]]
                    or d["initialization"]["initialization_seed"] != init_seed
                    or d["initialization"]["initializer_policy_sha256"] != fingerprint(policy()["initializer"])):
                raise ValueError("Wrong root/init stream/shape/initializer binding")
            canonical={k:v for k,v in d.items() if k!="run_id"}
            if d["run_id"] != f"e3-{d['stage']}-{fingerprint(canonical)[:24]}":
                raise ValueError("Run ID does not identify immutable declaration")
    return operation


def inspect_inputs(request: dict) -> dict:
    if request.get("kind") == "e3_numeric_contract_v1":
        return {"kind":request["kind"],"executable":False,"ready_for_execution":False}
    if request.get("kind") == "e3_stage_execution_request_v1":
        validate_record(request,"stage_execution_request")
        declaration = read_json(verify_payload(request["scientific_declaration"]))
    else: declaration=request
    operation=_validate_declaration(declaration)
    return {"kind":declaration["kind"],"operation":operation,"ready_for_execution":False,
            "required_gate":OP_GATES[operation],"scope":"schema/metadata inspection only; no model/GPU/image/feature allocation"}


def preflight(request, *, operation: str) -> dict:
    context=request if isinstance(request,ExecutionContext) else None
    request=context.request if context else request
    g1_receipt()
    validate_record(request,"stage_execution_request")
    if operation not in OP_GATES or request["operation"] != operation:
        raise ValueError("Operation wrapper/stage mismatch")
    d=read_json(verify_payload(request["scientific_declaration"]))
    if _validate_declaration(d) != operation: raise ValueError("Wrong declaration stage")
    binding=request["authorization"]
    receipt_path=contained(ROOT,binding["receipt_path"])
    if (sha256_file(receipt_path) != binding["receipt_sha256"]
            or binding["approved_scope_sha256"] != request["scientific_declaration"]["sha256"]):
        raise ValueError("Execution receipt/declaration bytes mismatch")
    if context is None: raise ValueError("Explicit input inventory path is required")
    expected={"operation":operation,"subjects":{
        declaration_role(d):request["scientific_declaration"]["sha256"],"input_inventory":sha256_file(context.inventory_path)}}
    splits=["train"] if operation=="bounded_acceptance" else [d["split"]] if operation=="bulk_cache" else ["train","val"]
    for split in splits:
        require_approval(receipt_path,gate=OP_GATES[operation],expected=dict(expected,split=split))
    inventory=read_json(context.inventory_path)
    if inventory.get("kind") != "e3_input_inventory_v1" or not isinstance(inventory.get("artifacts"),dict):
        raise ValueError("Invalid explicit input inventory")
    def artifact(digest):
        row=inventory["artifacts"].get(digest)
        if not row or row["sha256"] != digest: raise ValueError("Missing exact input payload: "+digest)
        return verify_payload(row)
    def role(name):
        if name not in inventory.get("roles",{}): raise ValueError("Missing explicit inventory role: "+name)
        return artifact(inventory["roles"][name])
    checked={"declaration":d,"inventory":inventory,"artifact":artifact,"role":role,"context":context}
    if operation=="bounded_acceptance":
        verify_code_inventory(read_json(artifact(d["runtime"]["declared_code_inventory_sha256"])))
        checked["data_root"]=Path(d["runtime"]["actual_data_root"]).resolve()
        return checked
    runtime=d["runtime"]
    verify_code_inventory(read_json(artifact(runtime["consumed_source_sha256"])))
    audit=read_json(artifact(runtime["current_data_audit_sha256"]))
    if (audit.get("kind") != "e3_current_data_audit_v1" or audit.get("test_images_opened") != 0
            or audit.get("fixed_metadata") != {k:v["sha256"] for k,v in numeric()["data"]["artifacts"].items()}):
        raise ValueError("Current root/data audit is not bound to fixed inputs")
    checked["data_root"]=Path(audit["actual_root"]).resolve()
    actual_runtime=read_json(artifact(runtime["accepted_runtime_sha256"]))
    environment=read_json(artifact(runtime["actual_environment_evidence_sha256"]))
    if (actual_runtime.get("kind") != "e3_target_runtime_observation_v1"
            or actual_runtime.get("effective_policy") != policy()["runtime"]
            or actual_runtime.get("environment_sha256") != runtime["actual_environment_evidence_sha256"]
            or actual_runtime.get("consumed_source_sha256") != runtime["consumed_source_sha256"]
            or actual_runtime.get("data_audit_sha256") != runtime["current_data_audit_sha256"]
            or environment.get("software") != policy()["runtime"]["software_lock"]):
        raise ValueError("Runtime/device/software/code/data acceptance binding mismatch")
    acceptance_path=artifact(d["source"]["source_acceptance_sha256"])
    evidence=validate_record(read_json(acceptance_path),"source_acceptance_evidence")
    source={k:v for k,v in d["source"].items() if k!="source_acceptance_sha256"}
    if source not in evidence["sources"]: raise ValueError("Source not covered by accepted evidence")
    require_approval(role("g2_receipt"),gate="E3-G2",expected={"subjects":{
        "source_acceptance":sha256_file(acceptance_path),"runtime_observation":runtime["accepted_runtime_sha256"]}})
    if operation in {"pilot_head","final_head"}:
        cache=d["cache"]
        acceptance=artifact(cache["cache_acceptance_sha256"])
        subjects={}
        for split,key in [("train","train"),("val","validation")]:
            file=contained(ROOT,cache[key+"_index_path"])
            if sha256_file(file) != cache[key+"_index_sha256"]: raise ValueError("Cache index bytes changed")
            subjects[f"cache:{d['family']}:{d['source_seed']}:{split}"]=sha256_file(file)
        require_approval(acceptance,gate="E3-CACHE-ACCEPT",expected={"subjects":subjects})
        if operation=="final_head":
            require_approval(artifact(d["head_freeze_receipt_sha256"]),gate="E3-HEAD-FREEZE",
                             expected={"subjects":{"common_profile":d["profile"]["profile_sha256"]}})
            from .campaign import validate_final_declarations
            campaign_path=role("final_declarations")
            campaign=validate_final_declarations(read_json(campaign_path))
            if d not in campaign: raise ValueError("Final declaration outside twelve-run campaign")
            require_approval(receipt_path,gate="E3-FINAL-AUTH",expected={"subjects":{"final_declarations":sha256_file(campaign_path)}})
    else:
        checked["replay_evaluation"]=lambda index,model:_replay_evaluation(index,model,checked)
    return checked


def configure_runtime(*, root_seed: int) -> dict:
    """Called only after all public execution guards; never from check-config."""
    import importlib.metadata
    import random
    import numpy as np
    import torch
    p=policy()["runtime"]
    if not os.environ.get("SLURM_JOB_ID"): raise ValueError("Explicit Slurm GPU allocation required")
    for key,value in {**p["environment"],"PYTHONHASHSEED":str(root_seed)}.items():
        if os.environ.get(key)!=value: raise ValueError("Process environment differs from reviewed runtime: "+key)
    software={"python":platform.python_version(),"torch":str(torch.__version__),
              **{key:importlib.metadata.version(package) for key,package in
                 [("torchvision","torchvision"),("Pillow","Pillow"),("numpy","numpy"),("pandas","pandas"),("scikit-learn","scikit-learn")]},
              "cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version()}
    if software!=p["software_lock"]: raise ValueError("Actual software differs from reviewed target lock")
    if not torch.cuda.is_available(): raise ValueError("Reviewed CUDA device required")
    props=torch.cuda.get_device_properties(0)
    if "H200" not in props.name or not 30*1024**3 <= props.total_memory <= 40*1024**3:
        raise ValueError("Reviewed H200 MIG 2g.35gb class required")
    torch.set_num_threads(p["cpu_threads"]); torch.set_num_interop_threads(p["interop_threads"])
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=False)
    random.seed(root_seed); np.random.seed(root_seed); torch.manual_seed(root_seed); torch.cuda.manual_seed_all(root_seed)
    return {"software":software,"device_name":props.name,"device_memory":props.total_memory,
            "slurm_job_id":os.environ["SLURM_JOB_ID"],"effective_policy":p}


def _replay_evaluation(index, model, checked):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from .cache import compare_replay
    from .data import FixedManifest,make_source_loader
    from .metrics import rank_logits
    d=checked["declaration"]; root=contained(ROOT,d["output_root"])
    arrays=[np.load(verify_payload(c["payload"],root),mmap_mode="r",allow_pickle=False) for c in index["chunks"]]
    features=np.concatenate(arrays,axis=0)
    loader=make_source_loader(FixedManifest(),source=d["source"],request={"split":d["split"],"stage":"final",
        "root_seed":d["source_seed"],"data_root":checked["data_root"]})
    actual={"keys":[],"labels":[],"top5":[],"nll":0.}; direct={"keys":[],"labels":[],"top5":[],"nll":0.}
    cursor=0; device=next(model.parameters()).device
    with torch.no_grad():
        for views,labels,keys in loader:
            views={k:v.to(device) for k,v in views.items()}; labels=labels.to(device)
            reference=model(views)
            cached=model.original_head(torch.from_numpy(features[cursor:cursor+len(keys)].copy()).to(device))
            compare_replay(reference,cached,labels,policy()["replay"])
            for output,logits in [(actual,cached),(direct,reference)]:
                output["keys"].extend(keys); output["labels"].extend(labels.cpu().tolist())
                output["top5"].extend(rank_logits(logits)[:,:5].cpu().tolist())
                output["nll"]+=float(F.cross_entropy(logits,labels,reduction="sum"))
            cursor+=len(keys)
    for output in (actual,direct): output["nll"]/=cursor
    reference_digest=fingerprint(direct)
    if d["split"]=="val":
        from .report import load_source_reference
        reference_path=checked["role"](f"selected_source_analysis:{d['family']}:{d['source_seed']}")
        reference=load_source_reference(reference_path,source=d["source"])
        direct={"keys":[r["relative_path"] for r in reference["predictions"]],
                "labels":[r["label_id"] for r in reference["predictions"]],
                "top5":[r["top5"] for r in reference["predictions"]],"nll":reference["metrics"]["nll"]}
        reference_digest=sha256_file(reference_path)
    return {"source":d["source"],"actual":actual,"reference":direct,"reference_evidence_sha256":reference_digest}
