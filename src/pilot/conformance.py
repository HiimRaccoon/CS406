"""CPU preprocessing contract checks on deterministic train-only samples."""
from __future__ import annotations

import datetime as dt
import json
import re
import uuid
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from .config import load_config, parser
from .dataset import FixedManifest, resolve_image
from .runtime import environment, write_json
from ..preprocessing.geoguessr_v1 import PREPROCESS_SHA256, Photometric, Preprocessor, half_up, sha256_file


def verify_sample(processor, image):
    raw, valid, layout = processor.processed_raw(image)
    expected_wh = layout["processed_wh"]
    assert list(raw.shape) == [3, expected_wh[1], expected_wh[0]]
    assert raw.dtype == torch.float32 and raw.device.type == "cpu"
    assert torch.equal(raw[:, ~valid], processor.mean[:, 0, 0, None].expand_as(raw[:, ~valid]))
    jittered, _, _ = processor.processed_raw(image, Photometric(1.07, .93))  # diagnostic, not training amplitudes
    assert torch.equal(jittered[:, ~valid], raw[:, ~valid])
    normalized_raw = processor.normalize(raw)
    assert torch.max(normalized_raw[:, ~valid].abs()).item() == 0.
    views = processor(image)
    again = processor(image)
    for key in views:
        assert torch.equal(views[key], again[key]), "Non-deterministic repeated preprocessing"
        assert torch.isfinite(views[key]).all()
    assert list(views["global"].shape) == [3, 384, 896]
    assert list(views["local"].shape) == [4, 3, 320, 320]
    for i, (left, top, width, height) in enumerate(layout["local_crops_xywh"]):
        reference = F.resize(raw[:, top:top + height, left:left + width], [320, 320],
                             interpolation=InterpolationMode.BILINEAR, antialias=True)
        assert torch.equal(views["local"][i], processor.normalize(reference)), "Local source is not processed raw"
    width, height = expected_wh
    scale = min(896 / width, 384 / height)
    rw, rh = half_up(width * scale), half_up(height * scale)
    assert [rw, rh] == layout["baseline_letterbox_content_wh"]
    left, top = (896 - rw) // 2, (384 - rh) // 2
    assert [left, top, 896 - rw - left, 384 - rh - top] == layout["baseline_letterbox_padding_ltrb"]
    padding = torch.ones(384, 896, dtype=torch.bool)
    padding[top:top + rh, left:left + rw] = False
    if padding.any():
        assert torch.max(views["global"][:, padding].abs()).item() == 0.
    for mask in layout["masks_xywh"]:
        x, y, w, h = mask["xywh"]
        px, py = left + half_up((x + w / 2) * scale), top + half_up((y + h / 2) * scale)
        assert views["global"][:, py, px].abs().max().item() < 1e-6, "Mask interior fill differs"
    return {"footer_wh": expected_wh, "global_shape": list(views["global"].shape),
            "local_shape": list(views["local"].shape), "crop_rectangles": layout["local_crops_xywh"],
            "fill_and_padding": "pass", "deterministic": True, "local_source_raw": True,
            "weights_transforms_used": False}


def run(cfg, verify_all=False):
    if not __debug__:
        raise RuntimeError("Conformance assertions must be enabled; do not run Python with -O")
    torch.set_num_threads(cfg.get("cpu_threads", 1))
    manifest = FixedManifest(cfg["artifacts_dir"])
    processor = Preprocessor("e2", config_path=cfg["preprocessing_config"])
    root = Path(cfg["paths"]["data_root"])
    samples = []
    for key in sorted(processor.config["layouts"]):
        records = sorted((r for r in manifest.for_split("train") if f"{r.width}x{r.height}" == key),
                         key=lambda r: r.sha256)[:cfg["samples_per_layout"]]
        if not records:
            raise ValueError(f"No train samples for {key}")
        for record in records:
            path = resolve_image(root, record.relative_path)
            if sha256_file(path) != record.sha256:
                raise ValueError(f"Local image hash mismatch: {path}")
            with Image.open(path) as image:
                if image.size != (record.width, record.height):
                    raise ValueError("Decoded raw dimensions differ from manifest")
                result = verify_sample(processor, image)
            samples.append({"path": record.relative_path, "split": "train", "layout": key, **result})
    audit = manifest.verify_images(root) if verify_all else None
    return {"purpose": "pilot", "stage": "conformance", "status": "pass",
            "preprocessing_sha256": PREPROCESS_SHA256, "fingerprints": manifest.fingerprints,
            "counts": dict(manifest.counts), "labels": len(manifest.labels),
            "detected_layouts": sorted(processor.config["layouts"]), "samples": samples,
            "test_images_opened": 0, "dataset_root": str(root.resolve()),
            "image_hash_audit": audit, "environment": environment()}


def main():
    p = parser(__doc__)
    p.add_argument("--verify-all-train-val-hashes", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config, data_root=args.data_root)
    if cfg["stage"] != "conformance":
        p.error("Expected conformance config")
    if args.check_config:
        print("Conformance config complete")
        return
    run_id = args.run_id or f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) or run_id in {".", ".."}:
        p.error("Unsafe run ID")
    output = Path(cfg["output_dir"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = run(cfg, args.verify_all_train_val_hashes)
    except Exception as exc:
        write_json(output / "conformance_report.json", {"purpose": "pilot", "stage": "conformance",
                   "status": "fail", "error": f"{type(exc).__name__}: {exc}"})
        raise
    write_json(output / "conformance_report.json", result)
    (output / "conformance_report.md").write_text(
        "# Preprocessing conformance — pilot only\n\nNo classifier scores or test images.\n\n"
        + "```json\n" + json.dumps(result, indent=2) + "\n```\n", encoding="utf-8")
    print(f"PASS: {output / 'conformance_report.json'}")


if __name__ == "__main__":
    main()
