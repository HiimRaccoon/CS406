"""CPU checks for the one-run deployment surface; never submit a job."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "scripts/slurm/e1_train.sbatch"


def bash():
    windows = Path("C:/Program Files/Git/bin/bash.exe")
    found = str(windows) if windows.is_file() else shutil.which("bash")
    if not found:
        pytest.skip("Bash is required for wrapper execution checks")
    return found


def cli(*args, env=None):
    return subprocess.run([sys.executable, "-m", "src.e1.train", *args], cwd=ROOT,
                          env=env, capture_output=True, text=True)


def test_check_config_is_cpu_only_and_rejects_overrides(run_config, tmp_path):
    config = tmp_path / "run.json"
    config.write_text(json.dumps(run_config), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    assert cli("--config", str(config), "--check-config", env=env).returncode == 0
    for args in [("--test",), ("--batch", "4"), ("--lr", "0.1"), ("--search",)]:
        assert cli("--config", str(config), "--check-config", *args, env=env).returncode != 0
    run_config["recipe"]["batch"]["microbatch"] = 4
    config.write_text(json.dumps(run_config), encoding="utf-8")
    assert cli("--config", str(config), "--check-config", env=env).returncode != 0


def test_check_config_never_calls_setup_or_training(run_config, tmp_path, monkeypatch):
    from src.e1 import train
    config = tmp_path / "run.json"
    config.write_text(json.dumps(run_config), encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("Configuration-only CLI allocated training resources")

    monkeypatch.setattr(train, "run", forbidden)
    monkeypatch.setattr(train, "_setup", forbidden)
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config), "--check-config"])
    train.main()


def test_check_config_rejects_missing_required_field(run_config, tmp_path):
    del run_config["seed"]
    config = tmp_path / "missing-seed.json"
    config.write_text(json.dumps(run_config), encoding="utf-8")
    assert cli("--config", str(config), "--check-config").returncode != 0


def test_training_cli_without_allocation_cannot_start(run_config, tmp_path):
    config = tmp_path / "run.json"
    config.write_text(json.dumps(run_config), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    result = cli("--config", str(config), env=env)
    assert result.returncode != 0
    assert not (Path(run_config["paths"]["runs_dir"]) / run_config["run_id"] / "last.pt").exists()


@pytest.fixture
def shell_run(tmp_path, run_config):
    config = tmp_path / "run.json"
    config.write_text(json.dumps(run_config), encoding="utf-8")
    capture = tmp_path / "capture.json"
    # srun is the only external boundary replaced: the real shell and Python
    # declaration validator run. No GPU allocation or training is possible here.
    (tmp_path / "capture.py").write_text(
        "import json,os,sys\nfrom pathlib import Path\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps({'argv':sys.argv[1:],"
        "'env':{k:os.environ.get(k) for k in "
        "['PYTHONHASHSEED','OMP_NUM_THREADS','MKL_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG']}}))\n",
        encoding="utf-8")
    srun = tmp_path / "srun"
    srun.write_text('#!/usr/bin/env bash\nexec "$CS406_PYTHON" "$CAPTURE_SCRIPT" "$@"\n', encoding="utf-8")
    srun.chmod(0o755)
    env = {**os.environ, "CS406_REPO": ROOT.as_posix(), "CS406_PYTHON": Path(sys.executable).as_posix(),
           "CS406_CONFIG": config.as_posix(), "CS406_E1_SEED": "1001", "CAPTURE": str(capture),
           "CAPTURE_SCRIPT": (tmp_path / "capture.py").as_posix(), "SLURM_CPUS_PER_TASK": "4",
           "SLURM_MEM_PER_NODE": "16384", "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]}

    def execute(*args, changes=None):
        capture.unlink(missing_ok=True)
        result = subprocess.run([bash(), WRAPPER.as_posix(), *args], env={**env, **(changes or {})},
                                cwd=ROOT, capture_output=True, text=True)
        return result, json.loads(capture.read_text()) if capture.exists() else None
    return execute, run_config


def test_wrapper_exact_allocation_and_single_run(shell_run):
    text = WRAPPER.read_text(encoding="utf-8")
    for directive in ["partition=normal", "account=uit", "qos=mig2-35", "gres=gpu:2g.35gb:1",
                      "ntasks=1", "cpus-per-task=4", "mem=16384M", "time=09:00:00"]:
        assert f"#SBATCH --{directive}\n" in text
    execute, _ = shell_run
    result, captured = execute()
    assert result.returncode == 0, result.stderr
    assert captured["argv"][:5] == ["--ntasks=1", "--unbuffered", Path(sys.executable).as_posix(), "-m", "src.e1.train"]
    assert captured["env"] == {"PYTHONHASHSEED": "1001", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}


@pytest.mark.parametrize("changes,args", [({"CS406_E1_SEED": "2001"}, ()),
    ({"CS406_E1_SEED": "3406"}, ()), ({"SLURM_CPUS_PER_TASK": "8"}, ()),
    ({}, ("--test",)), ({}, ("--resume", "best.pt")), ({}, ("--resume", "missing/last.pt"))])
def test_wrapper_rejects_wrong_seed_allocation_or_arguments(shell_run, changes, args):
    execute, _ = shell_run
    result, captured = execute(*args, changes=changes)
    assert result.returncode != 0
    assert captured is None


def test_wrapper_resume_requires_own_last_checkpoint(shell_run):
    execute, config = shell_run
    last = Path(config["paths"]["runs_dir"]) / config["run_id"] / "last.pt"
    last.parent.mkdir(parents=True)
    last.touch()  # Shell checks the path; the production runner checks contents.
    result, captured = execute("--resume", last.as_posix())
    assert result.returncode == 0, result.stderr
    assert captured["argv"][-2:] == ["--resume", last.as_posix()]
    other = last.parent.parent / "other" / "last.pt"
    other.parent.mkdir()
    other.touch()
    assert execute("--resume", other.as_posix())[0].returncode != 0


def test_smoke_evidence_cannot_approve_g2_or_become_official(recipe):
    from tests.e1.test_gpu_smoke import mechanics_config, g2_candidate
    before = json.dumps(recipe, sort_keys=True)
    fixture = mechanics_config(recipe)
    assert fixture["kind"] == "e1_t10_checkpoint_fixture"
    assert fixture["recipe"]["budget"]["planned_max_updates"] == 139992
    assert fixture["recipe"]["batch"]["updates_per_epoch"] == 1
    assert json.dumps(recipe, sort_keys=True) == before
    candidate = g2_candidate({"source_sha256": "fixture"}, {"name": "fixture"}, "evidence.json")
    assert candidate["gate"] == "G2" and candidate["approved"] is False
    from src.e1.config import validate_run
    with pytest.raises(ValueError):
        validate_run(fixture)
