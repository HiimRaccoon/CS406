"""Opt-in T10 acceptance probe. Never an official campaign run or approval.

See docs/E1/runbook.md. Checkpoints are deliberately bound to a test-only
configuration: one bounded two-image segment represents a mechanics boundary.
The canonical configuration and the full 139992-update schedule stay intact.
"""
from __future__ import annotations

import copy
import datetime as dt
import gc
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from src.e1.checkpoint import _rng, atomic_json, restore_checkpoint, save_checkpoint
from src.e1.config import CONTRACT_SHA256, ROOT, load_recipe, read_json, resolve_run, sha256_file
from src.e1.data import E1Dataset, E1Loader, EpochSampler, FixedManifest, derived_seed, seed_worker
from src.e1.engine import train_epoch
from src.e1.train import _allocation, _setup, _software, source_identity


def mechanics_config(recipe):
    # This is NOT a modified official recipe. The checkpoint API accepts a
    # minimal test fixture, as in the CPU mechanics tests. Campaign rejects it.
    return {"kind": "e1_t10_checkpoint_fixture", "contract_sha256": CONTRACT_SHA256,
            "recipe": {"budget": copy.deepcopy(recipe["budget"]),
                       "batch": {"updates_per_epoch": 1}},
            "boundary_unit": "one two-image smoke segment; never a full epoch"}


def g2_candidate(identity, device, evidence_path):
    return {"gate": "G2", "approved": False, "contract_sha256": CONTRACT_SHA256,
            "identity": identity, "device": device, "evidence": str(evidence_path),
            "status": "candidate_only_requires_user_readiness_and_declaration_review"}


class TrainPair(Dataset):
    """Select one existing train member per raw layout, using E1 preprocessing."""
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        epoch, i = index
        return self.dataset[epoch, self.indices[i]]


def pair_loader(manifest, config, indices):
    recipe, seed = config["recipe"], config["seed"]
    runtime = recipe["runtime"]
    dataset = TrainPair(E1Dataset(manifest, config["paths"]["data_root"], "train", seed,
                                  recipe["preprocessing"]["augmentation"]), indices)
    return E1Loader(dataset, sampler=EpochSampler(dataset, seed), batch_size=2,
                    num_workers=runtime["num_workers"], pin_memory=runtime["pin_memory"],
                    persistent_workers=runtime["persistent_workers"], prefetch_factor=runtime["prefetch_factor"],
                    drop_last=False, worker_init_fn=seed_worker,
                    generator=torch.Generator().manual_seed(derived_seed(seed, "loader", "train")))


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for k in left:
            assert_same(left[k], right[k])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def snapshot(context):
    return cpu_copy({"model": context["model"].state_dict(), "optimizer": context["optimizer"].state_dict(),
                     "scheduler": context["scheduler"].state_dict(), "rng": _rng(),
                     "loader_rng": {k: v.generator.get_state() for k, v in context["loaders"].items()}})


def segment(context, recipe, epoch):
    context["loaders"]["train"].sampler.set_epoch(epoch)
    observed = []

    def check_shape(module, args, output):
        assert list(args[0].shape) == [2, 3, 384, 896]
        assert list(output.shape) == [2, 124]
        assert args[0].dtype == output.dtype == torch.float32
        observed.append({"input": list(args[0].shape), "logits": list(output.shape)})

    hook = context["model"].register_forward_hook(check_shape)
    try:
        result = train_epoch(context["model"], context["loaders"]["train"], context["optimizer"],
                             context["scheduler"], None, device=context["device"], runtime=recipe["runtime"],
                             accumulation=recipe["batch"]["gradient_accumulation"],
                             clipping=recipe["optimizer"]["gradient_clipping"])
    finally:
        hook.remove()
    assert result["samples"] == 2 and result["optimizer_updates"] == 1 and len(observed) == 1
    assert math.isfinite(result["loss"])
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in context["model"].parameters() if p.requires_grad)
    return {**result, **observed[0], "finite_gradients": True}


def checked_identity(recipe, manifest):
    expected_env = {"PYTHONHASHSEED": "1001", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    assert all(os.environ.get(k) == v for k, v in expected_env.items()), "Export runtime variables BEFORE Python"
    assert os.environ.get("NVIDIA_TF32_OVERRIDE") in {None, "0"}
    software = _software()
    assert software == recipe["runtime"]["software_lock"], "Actual packages must match the frozen lock"
    allocation = _allocation(recipe)
    source = source_identity()
    pre = recipe["preprocessing"]
    assert sha256_file(ROOT / pre["implementation_path"]) == pre["implementation_sha256"]
    paths = recipe["deployment"]["paths"]
    weights = Path(paths["pretrained"])
    assert weights.name == recipe["model"]["pretrained"]["filename"]
    assert sha256_file(weights) == recipe["model"]["pretrained"]["sha256"]
    report_path = Path(os.environ.get("CS406_E1_CONFORMANCE_REPORT", str(
        ROOT / "reports/pilot/evidence/uit_2026-10-31/conformance/conformance_report.json")))
    report = read_json(report_path)
    assert report["status"] == "pass" and report["test_images_opened"] == 0
    assert report["dataset_root"] == paths["dataset"] and report["fingerprints"] == manifest.fingerprints
    assert report["preprocessing_sha256"] == pre["config_sha256"]
    assert report["image_hash_audit"] == {"train": 34997, "val": 7497}
    assert report["environment"]["packages"] == {k: software[k] for k in
        ("torch", "torchvision", "Pillow", "numpy", "pandas", "scikit-learn")}
    # Refresh byte/containment integrity before GPU allocation. This opens only
    # train/val bytes, decodes neither val nor test, and does not rerun the pilot.
    audit = manifest.verify_images(Path(paths["dataset"]))
    assert audit == {"train": 34997, "val": 7497}
    common = {**source, "software": software, "fingerprints": manifest.fingerprints,
              "preprocessing_sha256": pre["config_sha256"],
              "pretrained_sha256": recipe["model"]["pretrained"]["sha256"],
              "dataset_root": paths["dataset"], "conformance_sha256": sha256_file(report_path),
              "gpu_class": recipe["deployment"]["expected_gpu_class"]}
    assert torch.cuda.is_available(), "Opted-in UIT smoke requires CUDA"
    props = torch.cuda.get_device_properties(0)
    driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip().splitlines()[0]
    device = {"name": props.name, "total_bytes": props.total_memory, "driver": driver}
    assert device["name"] == recipe["deployment"]["expected_gpu_class"]
    return common, device, allocation, {"current_hash_audit": audit,
        "prior_preprocessing_conformance": str(report_path), "prior_conformance_sha256": sha256_file(report_path)}


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("CS406_E1_GPU_SMOKE") != "1", reason="Explicit UIT T10 opt-in required")
def test_uit_real_train_boundary_resume():
    recipe = load_recipe()
    output = Path(os.environ["CS406_E1_SMOKE_DIR"]).resolve()
    official = Path(recipe["deployment"]["paths"]["official_e1_runs"]).resolve()
    assert not output.is_relative_to(ROOT.resolve()) and not output.is_relative_to(official)
    output.mkdir(parents=True, exist_ok=False)  # Every attempt, including failure, remains separate.
    evidence = {"kind": "e1_t10_gpu_smoke", "status": "running", "contract_sha256": CONTRACT_SHA256,
                "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "official_run": False, "test_images_opened": 0, "validation_images_decoded": 0,
                "G2_approved": False, "G3a_approved": False, "G3b_approved": False}
    atomic_json(output / "evidence.json", evidence)
    try:
        manifest = FixedManifest()
        common, device, allocation, audit = checked_identity(recipe, manifest)
        evidence.update(identity=common, device=device, allocation=allocation, integrity=audit)
        atomic_json(output / "evidence.json", evidence)
        # Fixed deterministic members covering both layouts. No architecture,
        # learning-rate, batch, or validation-based search is performed.
        records = manifest.records("train")
        indices = [next(i for i, r in enumerate(records) if (r.width, r.height) == tuple(layout))
                   for layout in recipe["preprocessing"]["raw_layouts_wh"]]
        evidence["train_members"] = [{"relative_path": records[i].relative_path, "sha256": records[i].sha256,
                                       "raw_wh": [records[i].width, records[i].height]} for i in indices]
        paths = recipe["deployment"]["paths"]
        declaration = resolve_run(recipe, stage="tuning", lr=1e-4, seed=1001, paths={
            "data_root": paths["dataset"], "weights": paths["pretrained"], "runs_dir": paths["official_e1_runs"]})
        checked = {**common, "contract_sha256": CONTRACT_SHA256, "evidence_kind": "t10_gpu_smoke",
                   "accepted_device": device}
        fixture = mechanics_config(recipe)

        def fresh():
            # Reuse the production builder/runtime/optimizer/scheduler without
            # calling run()/preflight() or manufacturing stage authorizations.
            ctx = _setup(declaration, checked)
            ctx["loaders"] = {"train": pair_loader(manifest, declaration, indices)}
            ctx["config"] = fixture
            return ctx

        context = fresh()
        assert context["model_metadata"]["parameters"] == 27915484
        evidence["environment"] = context["environment"]
        evidence["parameters"] = context["model_metadata"]["parameters"]
        progress = {"completed_epoch": 0, "updates": 0, "history": [], "selected": None, "selected_weights": None}

        def args(ctx):
            return {k: ctx[k] for k in ("model", "optimizer", "scheduler", "scaler", "config", "identity", "loaders")}

        last = output / "last.pt"
        save_checkpoint(last, progress=progress, **args(context))
        first = segment(context, recipe, 0)
        # Sentinel selection is test mechanics only; no validation score exists.
        progress = {"completed_epoch": 1, "updates": 1, "history": [{"epoch": 1, "smoke": first}],
                    "selected": {"epoch": 1, "macro_f1_supported": 0.0, "fixture_sentinel": True},
                    "selected_weights": cpu_copy(context["model"].state_dict())}
        save_checkpoint(last, progress=progress, **args(context))
        committed_sha = sha256_file(last)
        reference = segment(context, recipe, 1)
        expected = snapshot(context)
        # Discard uncommitted segment and reconstruct model, optimizer, scheduler
        # and persistent workers, then replay solely from the saved boundary.
        del context
        gc.collect()
        torch.cuda.empty_cache()
        context = fresh()
        restored = restore_checkpoint(last, **args(context))
        assert_same(restored, progress)
        assert context["scheduler"].completed == 1
        replay = segment(context, recipe, 1)
        assert_same(snapshot(context), expected)
        assert reference["loss"] == replay["loss"]
        assert context["scheduler"].completed == 2 and context["scheduler"].total_updates == 139992
        assert sha256_file(last) == committed_sha
        evidence.update(status="pass", segments={"committed": first, "uncommitted_reference": reference, "replay": replay},
                        resume={"method": "discard uncommitted state; fresh model/optimizer/scheduler/workers; restore last.pt",
                                "boundary_units": "two-image mechanics segments, not full epochs",
                                "committed_updates": 1, "replayed_update": 1, "completed_after_replay": 2,
                                "model_optimizer_scheduler_rng_loader_bitwise_equal": True,
                                "loss_exactly_equal": True, "last_sha256": committed_sha,
                                "full_update_horizon": 139992, "official_epochs_completed": 0},
                        finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(output / "evidence.json", evidence)
        atomic_json(output / "g2_candidate.json", g2_candidate(common, device, output / "evidence.json"))
    except BaseException as exc:
        evidence.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                        finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(output / "evidence.json", evidence)
        raise
