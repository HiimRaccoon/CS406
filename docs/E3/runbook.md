# E3 implementation and execution handoff

E3-G1 was explicitly approved on 2026-11-11 for T01–T12 implementation, nonofficial CPU fixtures and deployment preparation, using `executing-plans` inline/native. The exact approved preparation bytes and scoped user decision are retained under `artifacts/E3/approvals/v1/E3-G1/`. Current task status is in `tasks.md`; decisions and observed verification results are in `implementation_log.md`.

The numeric/source/cache-pilot/schema candidates remain the exact historical bytes reviewed at G1. Their pre-approval status fields do not grant execution permission. Executable declarations, external gate receipts and separate execution wrappers govern later operations. The official final head winner remains unselected. TARGET-AUTH, G2, CACHE-AUTH, CACHE-ACCEPT, PILOT-AUTH, HEAD-FREEZE and FINAL-AUTH remain separate approvals. Test images and features remain locked.

## Environment and CPU verification

Install the existing reviewed project dependencies and the pinned E3 schema dependencies in `requirements-e3.txt`. The local Windows CPU environment is preparation evidence; the reviewed target software/device/driver/allocation must be accepted separately.

```powershell
.venv/Scripts/python.exe -m pytest tests/e3 -q -m "not gpu" --basetemp=.tools/pytest-e3
.venv/Scripts/python.exe -m pytest -q -m "not gpu" --basetemp=.tools/pytest-e3-all
.venv/Scripts/python.exe -m compileall -q src/e3 tests/e3
```

On Windows, use Git Bash for the wrapper syntax checks. On the target, use Bash:

```bash
bash -n scripts/slurm/e3_extract.sbatch
bash -n scripts/slurm/e3_head.sbatch
```

## Immutable requests and explicit artifact lookup

Each public execution command takes one normalized `e3_stage_execution_request_v1` and one explicit input inventory. The wrapper joins the exact scientific declaration payload hash to the already-approved operation receipt. The receipt also binds the inventory's exact file-byte SHA-256 under role `input_inventory`.

The inventory has kind `e3_input_inventory_v1`, an `artifacts` mapping from SHA-256 to `{relative_path,bytes,sha256}`, and a `roles` mapping from named role to SHA-256. Payload paths resolve beneath the execution workspace with traversal/symlink containment checks. Actual source binary/report paths are explicitly supplied by the source binding; preparation observation paths never become defaults.

Declaration subject roles are `scientific_declaration` for bounded acceptance, `declaration:FAMILY:SEED:SPLIT` for extraction, and `declaration:RUN_ID` for a head. A cache acceptance receipt binds each index as `cache:FAMILY:SEED:SPLIT`. Final execution additionally binds the complete `final_declarations` manifest and HEAD-FREEZE's `common_profile` hash.

Required lookup roles include `current_data_audit` before bounded acceptance; `selected_source_analysis:FAMILY:SEED` for all six selected historical analyses; and `g2_receipt` for post-target operations. Runtime/source/cache/init records supply the other consumed artifact digests. A final execution inventory also supplies `final_declarations`.

`runtime.code_inventory()` inventories E3 source plus transitive repository imports and pinned E3 dependency requirements. Git commit, job IDs and harness inventories are operational evidence; changing only documentation does not change an otherwise identical consumed source closure. Changed consumed code or source/cache/runtime inputs require relevant reacceptance.

Before TARGET-AUTH, materialize and review the six actual selected complete checkpoints, reports, source configurations/history/full selected validation analyses/CSV files, explicit data root and an audit bound to actual root/content/software. The bounded target harness opens only its two listed train fixtures; it consumes an existing approved train/validation audit and historical validation metadata. It cannot silently perform a new unrestricted validation-image audit under train-only target permission.

## Commands

```bash
python -m src.e3 check-config --input /explicit/declaration-or-wrapper.json
python -m src.e3 accept-source --request /explicit/target-wrapper.json --inventory /explicit/inventory.json --output /declared/acceptance/root --gpu
python -m src.e3 extract-cache --request /explicit/cache-wrapper.json --inventory /explicit/inventory.json --output /declared/cache/root
python -m src.e3 run-head --request /explicit/head-wrapper.json --inventory /explicit/inventory.json --output /declared/run/root
python -m src.e3 run-head --request /explicit/head-wrapper.json --inventory /explicit/inventory.json --output /declared/run/root --resume /same/run/root/last.pt
python -m src.e3 report-selected --request /explicit/head-wrapper.json --inventory /explicit/inventory.json --output /declared/run/root
python -m src.e3 report-study --input /explicit/study-inputs.json
```

`check-config` inspects strict JSON, schema and scientific bindings without allocating models/GPU or reading images/features; it never reports execution acceptance. `accept-source` without `--gpu` reports an explicitly unexecuted target check. With GPU opt-in, absent/mismatched permission is an error.

The study-report input names all twelve complete final run directories, a materialized source-manifest file, twelve cache-index descriptors and the exact common protocol. The materialized manifest supplies `registry_sha256`, six accepted `sources`, and six `original_reference_indices` payloads in corresponding order. Each original index points to the actual unchanged historical `validation_analysis/analysis.json` with its complete source history and CSV files. The historical E1 envelope remains distinct from E2.

## Slurm wrappers and recovery

The wrappers require explicit `CS406_REPO`, `CS406_PYTHON`, `CS406_E3_REQUEST`, `CS406_E3_INVENTORY`, `CS406_E3_OUTPUT` and `CS406_E3_ROOT_SEED`. Extraction also requires `CS406_E3_OPERATION=accept-source|extract-cache`. They export PYTHONHASHSEED, OMP/MKL thread counts and CUBLAS workspace settings before the first Python interpreter. Neither wrapper submits jobs. Actual allocation/software/device/driver checks remain in the runner after stage permissions and input bytes have been verified.

Extraction uses the approved batch 16, two CPU workers, pinned memory, prefetch factor 2 and persistent workers. Source encoding stays in the main CUDA process. Workers preserve fixed manifest order. Only committed lossless FP32 chunks count; interruption discards uncommitted queues/buffers/temporary chunks and reconstructs the loader at the first uncommitted chunk. `index.json` is published after complete payload checks and original-head replay; accepted caches cannot be overwritten.

Head training uses batch 512, accumulation 1 and the approved complete 20-window/1380-update candidate budget. The standalone classifier owns all trainable tensors. `last.pt` atomically commits full state/history/RNG at a complete window; `best.pt`, history JSON and the normalized checkpoint index are derived publications. An interrupted publication is repaired from authoritative same-run last state. A completed budget never trains again. At most two same-identity attempts are allowed; stale writer locks require operator review.

The bounded recovery harness uses private full-dimension heads and synthetic feature rows. Its four logical family/arm fixtures use the reviewed 512/512/512/181 row schedule. A private fixture boundary after two updates permits discard/reconstruction without exceeding 32 total updates including replay; production head commits remain whole windows. These fixtures never create accepted caches or official pilot/final results.

## Stop boundary

T12 readiness is CPU/deployment preparation only. Present the verified source/input inventory and exact bounded request for E3-TARGET-AUTH before running T13. No G2 receipt, bulk cache, pilot result, final winner or test evaluation is produced by T01–T12.
