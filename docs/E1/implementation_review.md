# E1 T01-T09 independent review and corrections

Client date: 2026-11-01. Scope: G1-authorized implementation and CPU verification only. The frozen numeric contract remains SHA-256 `c9153c53534e95099c374e8ab7b54bc347588f11b200adcc92b3c23ad6ac4143`.

One independent read-only reviewer inspected the whole branch from `b7a4f6ba706bf38948166fb053532cdfdcf4d15b` through `466d0fe1904bd183e50bb40fd26c15c76befacbb`, including all E1 modules/tests, plan, contract, ledger, and implementation evidence. It found six important issues and no critical issues. No second review is claimed. The author corrected the findings in one test-driven pass.

| Finding | Correction | Regression evidence |
|---|---|---|
| Historical conformance did not establish current image bytes/path containment | Preflight verifies all current train/validation hashes and safe paths before model allocation, once per invocation | `test_preflight_verifies_current_bytes_and_paths_before_allocation`: changed same-length bytes, missing file, escaped path; image decoding/model allocation forbidden |
| Failure before initial last.pt stranded an immutable declaration | Same configuration with no committed boundary may initialize again; existing boundaries still require explicit last.pt resume | `test_unstarted_run_recovers_same_declaration_without_erasing_failure`: preflight and boundary-zero save failures retain lineage |
| Complete summary could precede terminal event and strand completed budget | Terminal complete event and attempt outcome precede publication of complete summary; authoritative boundary eight can finish publication with no training | `test_terminal_publication_crash_recovers_without_training_more` |
| Fresh persistent workers consumed an extra logical loader RNG draw after resume | E1Loader recreates worker bootstrap from its initial generator state, then restores the committed logical generator state; sample-keyed order/jitter remains unchanged | `test_persistent_worker_resume_preserves_loader_model_optimizer_and_rng`, plus actual image order/jitter and generator-state checks in `test_data.py` |
| LR candidates could mix accepted source/environment identities | All four tuning records must share the reviewed scientific identity, including the common acceptance binding; operational job IDs/UUIDs remain outside that identity | `test_tuning_cannot_mix_accepted_scientific_identities` |
| Latest environment alias erased prior allocation evidence; replay lacked observations | Each invocation has exclusive request/environment/start-boundary/outcome snapshots; events retain epoch starts and measured training before validation; abruptly terminated attempts explicitly retain unknown uncommitted work | `test_attempt_allocations_and_replayed_training_are_retained` |

The nine new regression cases failed against the reviewed revision, then passed after correction (22 passed when combined with runner/campaign checks). Three further artifact-parser tests cover the real successful completed-run/locked-selection path and reject rehashed incomplete histories or wrong selected weights. Those tests fabricate trusted record envelopes in temporary directories; they are not actual UIT evidence, configurations for execution, or real experiment results.

A final author regression also reproduced and corrected failure-recording when last.pt itself cannot be deserialized. The diagnostic probe now leaves the committed boundary explicitly unknown instead of masking the original load error. Final full CPU verification: **171 passed, 1 opt-in GPU test deselected**, no warnings; compilation passed.

The earlier whole CPU regression run passed 158 tests, with one opt-in GPU test deselected. Final expanded-suite results are recorded in [implementation_log.md](implementation_log.md).

## Review limits and rulings

- Real-weight/full-resolution CUDA correctness, memory, throughput, and target resume equivalence remain T10 checks. They are not inferred from CPU tests.
- Slurm wrapper/runbook and actual G2/G3 execution remain outside this T01-T09 delivery. No GPU job, official tuning, final seed, or test evaluation is authorized by this review.
- Mutually fabricated, self-consistent local approval/checkpoint records are outside this trusted-local-artifact interface's threat model. Production still rejects synthetic markers, incomplete budgets, wrong identities, mismatched hashes, invalid selection, and checkpoint lineage errors.
- Bitwise equivalence across different hardware/software stacks is not claimed; changed accepted identities are rejected.

No minor findings were reported by the independent reviewer. Validation histories and full learning-curve data are exported as CSV without adding plotting dependencies.
