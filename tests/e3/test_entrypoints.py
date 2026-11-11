import json
from pathlib import Path
import subprocess
import sys

import pytest


def test_check_config_reports_non_executable_candidate_without_allocation():
    result=subprocess.run([sys.executable,"-m","src.e3","check-config","--input","docs/E3/e3_numeric_contract_v1.json"],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)["ready_for_execution"] is False


def test_cli_rejects_scientific_override_and_test_flags():
    result=subprocess.run([sys.executable,"-m","src.e3","run-head","--lr",".1","--test"],capture_output=True,text=True)
    assert result.returncode!=0


def test_no_gpu_opt_in_is_unexecuted_not_acceptance():
    from src.e3.acceptance import run_acceptance
    from src.e3.config import ROOT,read_json
    from src.e3.runtime import ExecutionContext
    # No actual source or receipt is opened without explicit target execution.
    result=run_acceptance(ExecutionContext({},ROOT/"nonexistent_inventory.json"),output=ROOT/"nonexistent_output",gpu=False)
    assert result["executed"] is False and result["g2_accepted"] is False


def test_gpu_opt_in_with_missing_permission_is_failure(tmp_path):
    from src.e3.acceptance import run_acceptance
    with pytest.raises(ValueError): run_acceptance({},output=tmp_path,gpu=True)


def test_incompatible_approved_evidence_cap_blocks_before_target_work(tmp_path, monkeypatch):
    from src.e3 import acceptance
    from src.e3.config import policy, schemas
    cap=policy()["bounded_acceptance"]["maximum_source_original_image_presentations"]
    evidence_cap=schemas()["$defs"]["source_acceptance_evidence"]["properties"]["source_presentations"]["maximum"]
    assert cap==384 and evidence_cap==48
    def forbidden(*args, **kwargs):
        pytest.fail("Target preflight/allocation must not run with incompatible approved evidence schema")
    monkeypatch.setattr(acceptance,"preflight",forbidden)
    with pytest.raises(ValueError, match="approved schema amendment"):
        acceptance.run_acceptance({},output=tmp_path,gpu=True)


def test_wrapper_syntax_and_rejects_unreviewed_seed_before_python(tmp_path):
    import os
    bash=Path("C:/Program Files/Git/bin/bash.exe")
    assert bash.is_file(),"Git Bash required for actual wrapper syntax verification"
    for name in ("e3_extract.sbatch","e3_head.sbatch"):
        script=Path("scripts/slurm")/name
        result=subprocess.run([str(bash),"-n",script.as_posix()],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        env=dict(os.environ,CS406_REPO=str(Path.cwd()),CS406_PYTHON="must-not-run",CS406_E3_REQUEST="missing.json",
                 CS406_E3_INVENTORY="missing.json",CS406_E3_OUTPUT="missing",CS406_E3_ROOT_SEED="406",CS406_E3_OPERATION="extract-cache")
        result=subprocess.run([str(bash),script.as_posix()],env=env,capture_output=True,text=True)
        assert result.returncode==2 and "seed" in result.stderr.lower()


def test_cpu_head_fixture_reconstructs_both_samplers_within_reviewed_caps():
    from src.e3.acceptance import _head_recovery_fixture
    result=_head_recovery_fixture(device="cpu")
    assert result["kind"]=="nonofficial_e3_head_recovery_evidence_v1"
    assert result["optimizer_updates"]==32 and result["feature_presentations"]==13736
    assert len(result["trials"])==4 and all(r["logical_state_equal"] for r in result["trials"])


def test_wrappers_export_reviewed_environment_before_first_python(tmp_path):
    import os
    bash=Path("C:/Program Files/Git/bin/bash.exe")
    fake=tmp_path/"python-sentinel.sh"
    fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$PYTHONHASHSEED" "$OMP_NUM_THREADS" "$MKL_NUM_THREADS" "$CUBLAS_WORKSPACE_CONFIG" > "$E3_TEST_OUTPUT"\nexit 42\n',encoding="utf-8",newline="\n")
    subprocess.run([str(bash),"-c",'chmod +x "'+fake.as_posix()+'"'],check=True)
    for script in ("e3_head.sbatch","e3_extract.sbatch"):
        output=tmp_path/(script+".txt")
        env=dict(os.environ,CS406_REPO=Path.cwd().as_posix(),CS406_PYTHON=fake.as_posix(),CS406_E3_REQUEST="unused.json",
            CS406_E3_INVENTORY="unused.json",CS406_E3_OUTPUT="unused",CS406_E3_ROOT_SEED="2001",CS406_E3_OPERATION="extract-cache",
            SLURM_CPUS_PER_TASK="4",SLURM_MEM_PER_NODE="16384",E3_TEST_OUTPUT=output.as_posix())
        result=subprocess.run([str(bash),"scripts/slurm/"+script],env=env,capture_output=True,text=True)
        assert result.returncode==42,result.stderr
        assert output.read_text().splitlines()==["2001","2","2",":4096:8"]
