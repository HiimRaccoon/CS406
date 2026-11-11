"""Lossless immutable FP32 chunks and exact source-head replay acceptance."""
from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .config import (DIMENSIONS, ROOT, contained, fingerprint, numeric, payload, policy, read_json, parse_json,
                     require_approval, sha256_file, validate_record, validate_source_binding, verify_payload)
from .data import FixedManifest
from .io import atomic_json, atomic_write, writer_lock
from ..e1.metrics import rank_logits


def compare_replay(reference, candidate, labels, policy):
    if reference.shape != candidate.shape or len(reference) != len(labels):
        raise ValueError("Replay coverage/logit shapes differ")
    if not torch.equal(rank_logits(reference)[:, :5], rank_logits(candidate)[:, :5]):
        raise ValueError("Replay discrete predictions/ordered Top-5 differ")
    for ref, got, tolerance in [
        (reference, candidate, policy["logits"]),
        (F.cross_entropy(reference, labels, reduction="none"),
         F.cross_entropy(candidate, labels, reduction="none"), policy["nll"])]:
        if not torch.isfinite(got).all() or not torch.isfinite(ref).all() or not torch.all(
                (got - ref).abs() <= tolerance["atol"] + tolerance["rtol"] * ref.abs()):
            raise ValueError("Replay float tolerance failed")
    return {"discrete_equal": True, "logits_within_tolerance": True,
            "nll_within_tolerance": True, "samples": len(labels)}


def _row_records(rows):
    return [{"row_index": i, "relative_path": r["relative_path"], "label_id": r["label_id"]}
            for i, r in enumerate(rows)]


def _build_payloads(root, rows, batches, *, dimension, identity):
    root = Path(root)
    with writer_lock(root):
        if (root / "index.json").exists():
            raise ValueError("Accepted/completed cache is immutable")
        for temporary in root.rglob("*.tmp.*"):
            if temporary.is_symlink() or not temporary.resolve().is_relative_to(root.resolve()):
                raise ValueError("Unsafe temporary cache path")
            if temporary.is_file(): temporary.unlink()
        identity = {"bindings": identity, "dimension": dimension, "rows_sha256": fingerprint(_row_records(rows))}
        if (root / "build_identity.json").exists():
            if read_json(root / "build_identity.json") != identity:
                raise ValueError("Cannot mix source/runtime/policy/feature dimensions on recovery")
        else:
            atomic_json(root / "build_identity.json", identity)
        raw = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in _row_records(rows)).encode()
        row_path = root / "rows.jsonl"
        if row_path.exists() and row_path.read_bytes() != raw:
            raise ValueError("Existing row index differs")
        if not row_path.exists():
            atomic_write(row_path, lambda stream: stream.write(raw))
        chunks = []
        source = identity["bindings"].get("source")
        source_digest = source["binary_sha256"] if isinstance(source, dict) else fingerprint(source)
        commit_identity = {"source_checkpoint_sha256": source_digest,
                           "extraction_identity_sha256": fingerprint(identity),
                           "row_index_sha256": sha256_file(row_path),
                           "cache_policy_sha256": fingerprint(policy()["storage"])}
        committed = sorted((root / "chunks").glob("commit_*.json"))
        cursor = 0
        for i, file in enumerate(committed):
            commit = read_json(file)
            chunk = commit["chunk"]
            validate_record(commit, "cache_chunk_commit")
            if (file.name != f"commit_{i:06d}.json" or any(commit[k] != v for k, v in commit_identity.items())
                    or chunk["row_start"] != cursor or chunk["chunk_index"] != i):
                raise ValueError("Stale/mixed/gapped chunk commit")
            chunks.append(chunk); cursor += chunk["row_count"]
        build = {"row_index": payload(row_path, root), "chunks": chunks}
        if chunks:
            _validate_payloads(root, build, rows[:cursor], dimension=dimension, full_row_index=rows)
        pending = []
        pending_rows = 0
        for features, labels, keys in batches:
            labels = torch.as_tensor(labels).tolist()
            count = len(features)
            expected = rows[cursor + pending_rows:cursor + pending_rows + count]
            if (len(expected) != count or list(keys) != [r["relative_path"] for r in expected]
                    or labels != [r["label_id"] for r in expected]
                    or features.dtype != torch.float32 or tuple(features.shape) != (count, dimension)
                    or not torch.isfinite(features).all()):
                raise ValueError("Feature batch has wrong coverage/order/labels/shape/values")
            pending.append(features.detach().cpu().contiguous().numpy())
            pending_rows += count
            while pending_rows >= 1024 or cursor + pending_rows == len(rows):
                array = np.concatenate(pending, axis=0)
                count = min(1024, pending_rows)
                array, remaining = array[:count].copy(), array[count:].copy()
                i = len(chunks)
                file = root / f"chunks/features_{i:06d}.npy"
                atomic_write(file, lambda stream: np.lib.format.write_array(stream, array, version=(2, 0), allow_pickle=False))
                chunk = {"chunk_index": i, "row_start": cursor, "row_count": count,
                         "feature_dimension": dimension, "payload": payload(file, root)}
                atomic_json(root / f"chunks/commit_{i:06d}.json",
                            {"kind": "e3_cache_chunk_commit_v1", **commit_identity, "chunk": chunk, "complete": True})
                chunks.append(chunk); cursor += count
                pending_rows = len(remaining); pending = [remaining] if pending_rows else []
                if not pending_rows:
                    break
        if cursor != len(rows) or pending_rows:
            raise ValueError("Incomplete cache; uncommitted batches receive no coverage credit")
        build = {"row_index": payload(row_path, root), "chunks": chunks}
        _validate_payloads(root, build, rows, dimension=dimension)
        return build


def _validate_payloads(root, build, rows, *, dimension, full_row_index=None):
    row_path = verify_payload(build["row_index"], root)
    indexed = [validate_record(parse_json(line),"row_index_record") for line in row_path.read_text(encoding="utf-8").splitlines()]
    if indexed != _row_records(rows if full_row_index is None else full_row_index):
        raise ValueError("Row coverage/order/labels differ from fixed manifest")
    cursor = 0
    for i, chunk in enumerate(build["chunks"]):
        if (chunk["chunk_index"] != i or chunk["row_start"] != cursor
                or chunk["feature_dimension"] != dimension or chunk["row_count"] != min(1024, len(rows) - cursor)):
            raise ValueError("Invalid chunk sequence/shape/coverage")
        file = verify_payload(chunk["payload"], root)
        if chunk["payload"]["relative_path"] != f"chunks/features_{i:06d}.npy":
            raise ValueError("Chunk payload path differs from approved layout")
        with file.open("rb") as stream:
            if np.lib.format.read_magic(stream) != (2, 0):
                raise ValueError("Expected NumPy v2.0 payload")
            shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream)
            if file.stat().st_size!=stream.tell()+int(np.prod(shape))*dtype.itemsize:
                raise ValueError("Truncated/trailing bytes in NumPy payload")
        array = np.load(file, allow_pickle=False, mmap_mode="r")
        if (array.dtype.str != "<f4" or not array.flags.c_contiguous
                or array.shape != (chunk["row_count"], dimension) or not np.isfinite(array).all()):
            raise ValueError("Invalid FP32 feature payload")
        cursor += len(array)
    if cursor != len(rows):
        raise ValueError("Incomplete chunk coverage")


def validate_cache(index_path: Path, *, expected: dict) -> dict:
    # Reject test/split confusion before opening any feature payload.
    if expected.get("split") not in {"train", "val"}:
        raise ValueError("Train/val split is required before cache access")
    index = validate_record(read_json(index_path), "cache_index")
    for key in ("family", "source_seed", "split", "source", "runtime", "cache_policy_sha256", "preprocessing_sha256"):
        if key in expected and index[key] != expected[key]:
            raise ValueError(f"Cache {key} identity mismatch")
    validate_source_binding(index["source"])
    if (index["family"] != index["source"]["family"] or index["source_seed"] != index["source"]["source_seed"]
            or index["feature_dimension"] != DIMENSIONS[index["family"]]
            or index["preprocessing_sha256"] != numeric()["preprocessing"]["config"]["sha256"]
            or index["cache_policy_sha256"] != fingerprint(policy()["storage"])):
        raise ValueError("Cache source/policy/feature boundary mismatch")
    rows = FixedManifest().records(index["split"])
    if len(rows) != index["row_count"]:
        raise ValueError("Cache must cover every fixed manifest row")
    root = Path(index_path).parent
    build_identity=read_json(root/"build_identity.json")
    declaration=validate_record(build_identity["bindings"],"cache_request")
    if (any(declaration[k]!=index[k] for k in ("family","source_seed","split","source","runtime","cache_policy_sha256","preprocessing_sha256"))
            or contained(ROOT,declaration["output_root"])!=root.resolve()
            or build_identity["dimension"]!=index["feature_dimension"]
            or build_identity["rows_sha256"]!=fingerprint(_row_records(rows))):
        raise ValueError("Original extraction declaration differs from published cache lineage")
    _validate_payloads(root, index, rows, dimension=index["feature_dimension"])
    for i, chunk in enumerate(index["chunks"]):
        commit = validate_record(read_json(root / f"chunks/commit_{i:06d}.json"), "cache_chunk_commit")
        if (commit["chunk"] != chunk or commit["source_checkpoint_sha256"] != index["source"]["binary_sha256"]
                or commit["row_index_sha256"] != index["row_index"]["sha256"]
                or commit["extraction_identity_sha256"] != fingerprint(build_identity)
                or commit["cache_policy_sha256"] != index["cache_policy_sha256"]):
            raise ValueError("Chunk commit lineage mismatch")
    expected_files={f"features_{i:06d}.npy" for i in range(len(index["chunks"]))}
    expected_commits={f"commit_{i:06d}.json" for i in range(len(index["chunks"]))}
    if ({p.name for p in (root/"chunks").glob("features_*.npy")}!=expected_files
            or {p.name for p in (root/"chunks").glob("commit_*.json")}!=expected_commits):
        raise ValueError("Extra/unlisted cache chunks or commits")
    replay = read_json(verify_payload(index["replay_evidence"], root))
    identity = {"source": index["source"], "runtime": index["runtime"],
                "row_index": index["row_index"], "chunks": index["chunks"]}
    if not replay.get("all_required_checks_passed") or replay.get("payload_identity_sha256") != fingerprint(identity):
        raise ValueError("Missing complete replay/source/payload binding")
    return index


class CacheDataset(Dataset):
    def __init__(self, index_path: Path, *, expected: dict):
        if (expected.get("split") not in {"train","val"} or expected.get("family") not in DIMENSIONS
                or expected.get("source_seed") not in {2001,3001,4001}):
            raise ValueError("Explicit cache family/seed/train-val scope required before access")
        require_approval(expected["cache_acceptance_receipt"], gate="E3-CACHE-ACCEPT",
                         expected={"subjects": {f"cache:{expected['family']}:{expected['source_seed']}:{expected['split']}": sha256_file(index_path)}})
        index = validate_cache(index_path, expected=expected)
        self.root, self.index = Path(index_path).parent, index
        self.rows = _row_records(FixedManifest().records(index["split"]))
        self.ends = [c["row_start"] + c["row_count"] for c in index["chunks"]]
        self.arrays = [np.load(verify_payload(c["payload"], self.root), mmap_mode="r", allow_pickle=False)
                       for c in index["chunks"]]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if not 0 <= i < len(self):
            raise IndexError(i)
        chunk = bisect.bisect_right(self.ends, i)
        start = self.index["chunks"][chunk]["row_start"]
        row = self.rows[i]
        return torch.from_numpy(self.arrays[chunk][i-start].copy()), row["label_id"], row["relative_path"]


def verify_original_replay(source_entry: dict, cache_index: dict, evaluation: dict, policy: dict) -> dict:
    if evaluation["source"] != source_entry or cache_index["source"] != source_entry:
        raise ValueError("Replay source mismatch")
    actual, reference = evaluation["actual"], evaluation["reference"]
    if actual["keys"] != reference["keys"] or actual["labels"] != reference["labels"]:
        raise ValueError("Replay membership/ground truth mismatch")
    if actual["top5"] != reference["top5"]:
        raise ValueError("Replay historical discrete mismatch")
    if len(actual["keys"]) != cache_index["row_count"]:
        raise ValueError("Replay must cover complete split")
    ref_nll, nll = reference["nll"], actual["nll"]
    if (not np.isfinite([ref_nll, nll]).all()
            or abs(nll-ref_nll) > policy["nll"]["atol"] + policy["nll"]["rtol"] * abs(ref_nll)):
        raise ValueError("Historical NLL unavailable or outside tolerance")
    identity = {k: cache_index[k] for k in ("source", "runtime", "row_index", "chunks")}
    return {"all_required_checks_passed": True, "payload_identity_sha256": fingerprint(identity),
            "source": source_entry, "rows": len(actual["keys"]), "discrete_equal": True,
            "reference_evidence_sha256": evaluation["reference_evidence_sha256"]}


def extract_cache(request: dict, *, source_model, loader, output: Path) -> dict:
    from .runtime import preflight
    checked = preflight(request, operation="bulk_cache")
    declaration = checked["declaration"]
    if Path(output).resolve() != contained(ROOT, declaration["output_root"]):
        raise ValueError("Output differs from approved declaration")
    if getattr(source_model, "source_binding", None) != declaration["source"]:
        raise ValueError("Source model is not bound to approved selected bytes")
    rows = FixedManifest().records(declaration["split"])
    committed_cursor = sum(read_json(p)["chunk"]["row_count"] for p in sorted((Path(output)/"chunks").glob("commit_*.json")))
    if tuple(loader.dataset.records) != rows[committed_cursor:]:
        raise ValueError("Extraction loader membership differs from fixed manifest")
    device = next(source_model.parameters()).device
    def batches():
        for views, labels, keys in loader:
            views = {k: v.to(device) for k, v in views.items()}
            features = source_model.encode(views)
            compare_replay(source_model(views), source_model.original_head(features), labels.to(device), policy()["replay"])
            yield features.cpu(), labels, keys
    build = _build_payloads(output, rows, batches(), dimension=DIMENSIONS[declaration["family"]], identity=declaration)
    index = {"kind": "e3_cache_index_v1", "family": declaration["family"], "source_seed": declaration["source_seed"],
             "split": declaration["split"], "row_count": len(rows), "feature_dimension": DIMENSIONS[declaration["family"]],
             "feature_dtype": "<f4", **{k: declaration[k] for k in ("source", "runtime", "cache_policy_sha256", "preprocessing_sha256")},
             **build, "complete": True}
    # Replay is computed from the written payloads; it cannot bind the future index hash.
    with writer_lock(output):
        if (Path(output)/"index.json").exists():
            raise ValueError("Completed index cannot be overwritten")
        _validate_payloads(output, index, rows, dimension=index["feature_dimension"])
        evaluation = checked["replay_evaluation"](index, source_model)
        evidence = verify_original_replay(declaration["source"], index, evaluation, policy()["replay"])
        atomic_json(Path(output) / "replay.json", evidence)
        index["replay_evidence"] = payload(Path(output) / "replay.json", output)
        validate_record(index, "cache_index")
        atomic_json(Path(output) / "index.json", index)
    return validate_cache(Path(output) / "index.json", expected=declaration)
