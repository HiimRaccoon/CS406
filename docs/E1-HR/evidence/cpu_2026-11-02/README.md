# Local CPU verification transcripts

These are local CPU and synthetic-fixture transcripts for HR-T01-HR-T10, recorded on 2026-11-02. They contain no official HR results, actual Slurm allocation or target acceptance. GPU tests were deselected. Final HR suite: 106 passed, 1 deselected. Final full regression: 290 passed, 3 deselected in 121.54 seconds.

PowerShell captured the original transcripts as UTF-16; this archive preserves their text as UTF-8 with LF newlines. `cpu_readiness.json` records exact byte counts and SHA-256 hashes of these archived bytes. Source, frozen contract, approval receipt and protected-input hashes refer to their exact original file bytes without text normalization.

The observed RED tests, GREEN results, initial Windows sharing failures, isolated reproductions and subsequent clean runs remain visible. The final independent review found four Important issues; its single RED-to-GREEN fix pass and full regression are recorded in `review-*` transcripts and the implementation log. Earlier counts describe earlier task boundaries, rather than additional official runs.

`protected.json` is the original 159-file protection baseline. `review_package.md` records the review scope at commit 9b47464; `final_review.md` records its findings. `progress.md` preserves the local task ledger and rulings. Full task commits and current gate state are in `../../implementation_log.md` and `../../tasks.md`. Separate HR-T11 target-work authorization remains required.
