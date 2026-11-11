import copy
import pytest


def test_wrong_stage_receipt_fails_before_allocation_or_payload_open(monkeypatch):
    from src.e3.runtime import preflight
    from src.e3 import config
    monkeypatch.setattr(config,"verify_payload",lambda *a,**k:pytest.fail("must reject stage before opening declaration"))
    request={"kind":"e3_stage_execution_request_v1","operation":"final_head",
             "scientific_declaration":{"relative_path":"nonofficial/no.json","bytes":1,"sha256":"a"*64},
             "authorization":{"gate":"E3-PILOT-AUTH","receipt_path":"none.json","receipt_sha256":"b"*64,"approved_scope_sha256":"a"*64}}
    with pytest.raises(ValueError): preflight(request,operation="final_head")


def test_pre_target_inspection_needs_no_future_g2_receipt():
    from src.e3.runtime import inspect_inputs
    from src.e3.config import policy,registry,numeric,fingerprint,SOURCE_SHA256
    sources=[]
    for r in registry()["selected_sources"]:
        sources.append({"family":r["family"],"source_seed":r["source_seed"],"selected_epoch":r["selected_epoch"],
        "actual_binary_path":"nonofficial/best.pt","binary_sha256":r["required_binary_sha256"],
        "actual_report_path":"nonofficial/report.json","report_sha256":r["required_report_sha256"],
        "source_contract_sha256":r["required_source_contract_sha256"],"source_manifest_sha256":SOURCE_SHA256})
    request={"kind":"e3_acceptance_request_v1","scope":policy()["bounded_acceptance"],"sources":sources,
        "runtime":{"requested_runtime_policy_sha256":fingerprint(policy()["runtime"]),"actual_data_root":"nonofficial/data",
        "fixed_metadata_sha256":{k:v["sha256"] for k,v in numeric()["data"]["artifacts"].items()},
        "allocation_policy_sha256":fingerprint(policy()["runtime"]["allocation"]),"declared_code_inventory_sha256":"c"*64},
        "cache_pilot_contract_sha256":"1626caa8bf87900ebcbd6962bcfed846d6a8c301a83d922162d07be7c9cf33bc",
        "output_root":"artifacts/E3/acceptance/v1","required_authorization_gate":"E3-TARGET-AUTH"}
    result=inspect_inputs(request)
    assert result["operation"]=="bounded_acceptance" and not result["ready_for_execution"]


def test_scientific_closure_excludes_operational_git_commit_and_rejects_code_change():
    from src.e3.runtime import code_inventory,verify_code_inventory
    inventory=code_inventory()
    verify_code_inventory(dict(inventory,audit_git_commit="another-docs-only-commit"))
    bad=copy.deepcopy(inventory); bad["files"]["src/e3/engine.py"]="0"*64
    with pytest.raises(ValueError): verify_code_inventory(bad)


def test_consumed_closure_binds_imported_package_initializers_and_launchers(monkeypatch):
    from src.e3 import runtime
    from src.e3.config import ROOT
    inventory=runtime.code_inventory()
    required={"src/__init__.py","src/e3/__init__.py","src/e1/__init__.py",
              "src/e2/__init__.py","src/preprocessing/__init__.py",
              "scripts/slurm/e3_extract.sbatch","scripts/slurm/e3_head.sbatch"}
    assert required <= inventory["files"].keys()
    original=runtime.sha256_file
    for relative in required:
        monkeypatch.setattr(runtime,"sha256_file",lambda path,relative=relative:
            "0"*64 if path==ROOT/relative else original(path))
        with pytest.raises(ValueError,match="closure changed"):
            runtime.verify_code_inventory(inventory)
