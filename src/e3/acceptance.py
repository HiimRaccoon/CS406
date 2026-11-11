"""Opt-in, train-only bounded acceptance harness. It emits evidence, never G2 approval."""
from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader,TensorDataset,Subset

from .cache import _build_payloads,compare_replay
from .checkpoint import capture_head_state,restore_head_state
from .config import ROOT,DIMENSIONS,contained,fingerprint,payload,policy,read_json,sha256_file,validate_record
from .data import FixedManifest,ImageRows,seed_worker
from .engine import _train_batches,make_head_scheduler
from .io import atomic_json,atomic_write,writer_lock
from .model import make_paired_heads,build_head_optimizer,tensor_state_sha256
from .runtime import preflight,configure_runtime
from .sampling import DrawSampler
from .sources import load_frozen_source
from ..e1.data import derived_seed


def _equal(a,b):
    if type(a)!=type(b): return False
    if isinstance(a,torch.Tensor): return torch.equal(a,b)
    if isinstance(a,dict): return set(a)==set(b) and all(_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(_equal(x,y) for x,y in zip(a,b))
    return a==b


def _head_recovery_fixture(*, device):
    p=policy(); protocol=p["candidate_profiles"][0]; results=[]; updates=presentations=0
    for family,dimension in DIMENSIONS.items():
        first,second,_=make_paired_heads(family=family,seed=2001,protocol=protocol,stage="acceptance_fixture",root_seed=1001)
        generator=torch.Generator().manual_seed(derived_seed(1001,"E3-v1","acceptance_fixture",family,"synthetic-features"))
        features=torch.randn(1717,dimension,generator=generator,dtype=torch.float32)
        labels=torch.arange(1717)%124
        rows=[{"relative_path":f"nonofficial/{i}","label_id":int(labels[i])} for i in range(1717)]
        for arm,initial in [("natural",first),("crt",second)]:
            def context():
                head=copy.deepcopy(initial).to(device); optimizer,_=build_head_optimizer(head,protocol=protocol)
                scheduler=make_head_scheduler(optimizer,protocol=protocol)
                sampler=DrawSampler(rows,policy=arm,root_seed=1001,source_seed=2001,stage="acceptance_fixture")
                loader=DataLoader(TensorDataset(features,labels),batch_size=512,sampler=sampler,num_workers=0,drop_last=False,
                                  generator=torch.Generator().manual_seed(derived_seed(1001,"E3-v1","acceptance_fixture",2001,"loader","train")))
                return head,optimizer,scheduler,sampler,loader
            def train(ctx,batches):
                head,opt,scheduler,sampler,loader=ctx
                clip=protocol["optimizer"]["clipping"]
                return _train_batches(head,batches,opt,scheduler,accumulation=1,
                    clipping={k:clip[k] for k in ("max_norm","norm_type","error_if_nonfinite","foreach")},device=device)
            def capture(ctx,n):
                h,o,s,sa,lo=ctx
                return capture_head_state(head=h,optimizer=o,scheduler=s,loaders={"train":lo},sampler=sa,progress={"fixture_updates":n})
            reference=context(); ref_result=train(reference,reference[-1]); reference_state=capture(reference,4)
            interrupted=context(); iterator=iter(interrupted[-1]); prefix=[next(iterator),next(iterator)]
            prefix_result=train(interrupted,prefix); committed=capture(interrupted,2)
            # Consume uncommitted loader work, discard it, then reconstruct every object.
            next(iterator)
            restored=context(); h,o,s,sa,lo=restored
            restore_head_state(committed,head=h,optimizer=o,scheduler=s,loaders={"train":lo},sampler=sa)
            loader_state=lo.generator.get_state(); suffix=iter(lo)
            # No head workers exist. Iterator construction's incidental base-seed draw
            # is not a scientific sample draw and must not advance the restored stream.
            lo.generator.set_state(loader_state)
            suffix_result=train(restored,suffix); recovered_state=capture(restored,4)
            if not _equal(reference_state,recovered_state): raise ValueError("Head fixture logical state differs after reconstruct/replay")
            trial_updates=ref_result["updates"]+prefix_result["updates"]+suffix_result["updates"]
            trial_presentations=ref_result["presentations"]+prefix_result["presentations"]+suffix_result["presentations"]
            updates+=trial_updates; presentations+=trial_presentations
            results.append({"family":family,"arm":arm,"logical_state_equal":True,"updates":trial_updates,
                            "feature_presentations":trial_presentations,"initial_tensor_sha256":tensor_state_sha256(initial.state_dict())})
    caps=p["bounded_acceptance"]["head_fixtures"]
    if updates>caps["maximum_updates_including_replay"] or presentations>caps["maximum_feature_presentations_including_replay"]:
        raise ValueError("Head fixture envelope exhausted")
    return {"kind":"nonofficial_e3_head_recovery_evidence_v1","optimizer_updates":updates,
            "feature_presentations":presentations,"trials":results,"official_run":False}


def run_acceptance(request, *, output: Path, gpu: bool=False) -> dict:
    if not gpu:
        return {"kind":"e3_unexecuted_target_check_v1","executed":False,"g2_accepted":False,"reason":"Explicit GPU opt-in absent"}
    checked=preflight(request,operation="bounded_acceptance"); declaration=checked["declaration"]
    destination=contained(ROOT,declaration["output_root"])
    if Path(output).resolve()!=destination: raise ValueError("Output differs from approved bounded declaration")
    audit_path=checked["role"]("current_data_audit"); audit=read_json(audit_path)
    if (Path(audit["actual_root"]).resolve()!=checked["data_root"] or audit.get("test_images_opened")!=0
            or audit.get("fixed_metadata")!=declaration["runtime"]["fixed_metadata_sha256"]):
        raise ValueError("Actual reusable root/content/software audit must be supplied before target acceptance")
    manifest=FixedManifest(); records=manifest.records("train"); scope=policy()["bounded_acceptance"]
    members=[]
    for member in scope["fixture_members"]:
        row=next((r for r in records if r["relative_path"]==member["relative_path"]),None)
        if (row is None or row["label_id"]!=member["label_id"] or [row["width"],row["height"]]!=member["raw_wh"]
                or row["sha256"]!=member["expected_image_sha256"]):
            raise ValueError("Bounded fixture membership differs from frozen train metadata")
        members.append(row)
    environment=configure_runtime(root_seed=1001)
    if audit.get("software")!=environment["software"]:
        raise ValueError("Reusable current-data audit software binding differs from target")
    from .report import load_source_reference
    source_presentations=0; strict_evidence=[]
    with writer_lock(destination):
        if (destination/"source_acceptance.json").exists(): raise ValueError("Target evidence cannot be overwritten")
        for source in declaration["sources"]:
            reference_path=checked["role"](f"selected_source_analysis:{source['family']}:{source['source_seed']}")
            load_source_reference(reference_path,source=source)
            model,verification=load_frozen_source(source,contract=request,device="cuda:0")
            loader_generator=torch.Generator().manual_seed(derived_seed(source["source_seed"],"E3-v1","acceptance_fixture",source["source_seed"],"loader","train"))
            data=ImageRows(members,checked["data_root"],source["family"])
            fixture=DataLoader(Subset(data,[0,1]*8),batch_size=16,num_workers=2,pin_memory=True,prefetch_factor=2,
                               persistent_workers=True,shuffle=False,drop_last=False,in_order=True,worker_init_fn=seed_worker,generator=loader_generator)
            views,labels,keys=next(iter(fixture)); views={k:v.to("cuda:0") for k,v in views.items()}; labels=labels.to("cuda:0")
            before=tensor_state_sha256(model.source.state_dict())
            with torch.no_grad():
                direct=model(views); source_presentations+=16
                features=model.encode(views); source_presentations+=16
                comparison=compare_replay(direct,model.original_head(features),labels,policy()["replay"])
                again=model.encode(views); source_presentations+=16
            if not torch.equal(features,again) or before!=tensor_state_sha256(model.source.state_dict()):
                raise ValueError("Frozen source values/buffers changed")
            folder=destination/f"{source['family']}_seed{source['source_seed']}"
            private_rows=[{"relative_path":k,"label_id":int(label)} for k,label in zip(keys,labels.cpu().tolist())]
            build=_build_payloads(folder/"private_chunks",private_rows,[(features.cpu(),labels.cpu(),keys)],
                                  dimension=DIMENSIONS[source["family"]],identity={"source":source,"kind":"nonofficial_fixture"})
            orphan=folder/"private_chunks/chunks/features.tmp.interrupted"
            orphan.write_bytes(b"uncommitted fixture bytes")
            del model,fixture
            model,_=load_frozen_source(source,contract=request,device="cuda:0")
            reconstructed=model.encode(views); source_presentations+=16
            if not torch.equal(features,reconstructed) or before!=tensor_state_sha256(model.source.state_dict()):
                raise ValueError("Source reconstruction is not bitwise equal under accepted runtime")
            reused=_build_payloads(folder/"private_chunks",private_rows,[],dimension=DIMENSIONS[source["family"]],identity={"source":source,"kind":"nonofficial_fixture"})
            if reused!=build or orphan.exists(): raise ValueError("Committed chunk recovery failed")
            evidence={"kind":"e3_source_strict_load_evidence_v1","source":source,"verification":verification,
                      "source_tensor_sha256":before,"selected_analysis":payload(reference_path),"original_head_replay":comparison,
                      "feature_shape":list(features.shape),"representation_trainable_parameters":0,"source_presentations":64,
                      "reconstruction_equal":True,"test_images_opened":0}
            atomic_json(folder/"strict_load.json",evidence); strict_evidence.append(payload(folder/"strict_load.json"))
            del model,features,again,reconstructed,views
        if source_presentations>scope["maximum_source_original_image_presentations"]: raise ValueError("Source fixture envelope exhausted")
        recovery=_head_recovery_fixture(device="cuda:0")
        atomic_json(destination/"head_recovery.json",recovery)
        atomic_json(destination/"environment.json",environment)
        runtime={"kind":"e3_target_runtime_observation_v1","effective_policy":policy()["runtime"],
                 "environment_sha256":sha256_file(destination/"environment.json"),
                 "consumed_source_sha256":declaration["runtime"]["declared_code_inventory_sha256"],
                 "data_audit_sha256":sha256_file(audit_path)}
        atomic_json(destination/"runtime.json",runtime)
        result={"kind":"e3_source_acceptance_evidence_v1","scope":"bounded_target_acceptance","sources":declaration["sources"],
                "source_strict_load_and_freeze_evidence":strict_evidence,"runtime_observations":payload(destination/"runtime.json"),
                "recovery_evidence":payload(destination/"head_recovery.json"),"source_presentations":source_presentations,
                "head_fixture_updates":recovery["optimizer_updates"],"head_fixture_presentations":recovery["feature_presentations"],
                "all_required_checks_passed":True,"test_images_opened":0}
        validate_record(result,"source_acceptance_evidence"); atomic_json(destination/"source_acceptance.json",result)
        return {"executed":True,"g2_accepted":False,"evidence":payload(destination/"source_acceptance.json"),"next_gate":"E3-G2 REVIEW REQUIRED"}
