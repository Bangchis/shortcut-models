---
status: done
finished: 2026-04-22
started: 2026-04-22
---

# Goal

Reduce unnecessary WandB logging in `moe1-ablation` while producing a compact copyable report for later analysis.

# Approach

Keep local artifacts/stats and key summary tables, but log only essential images/tables to the master WandB run. Write a Markdown + JSON analysis packet with the run selections, ranking metrics, geometry/path stats, and paths to important local artifacts.

# Tasks

- [x] Add helpers to filter image logs to essential keys.
- [x] Add helpers to build `analysis_packet.json` and `analysis_packet.md`.
- [x] Apply slim logging to stage summary/master runs without deleting local stats.
- [x] Compile/check touched files and close the plan.
