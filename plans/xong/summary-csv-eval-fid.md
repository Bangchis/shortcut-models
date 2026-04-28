---
status: done
started: 2026-04-28
finished: 2026-04-28
---

# Goal

Make `training_summary.csv` useful as a single per-step summary table, including train metrics, validation update metrics, eval metrics, and FID metrics.

# Approach

- Change summary CSV writing from append-only to step-based upsert/merge.
- Return scalar eval metrics from `helper_eval.eval_model`.
- Write eval/FID metrics into the same summary CSV row at eval steps.
- Keep W&B upload behavior after each CSV write.

# Tasks

- [x] Inspect current summary writer and eval/FID logging.
- [x] Implement step-based CSV upsert.
- [x] Collect FID/eval scalar metrics in `helper_eval.py`.
- [x] Merge eval metrics into CSV in `train.py`.
- [x] Run smoke checks.
