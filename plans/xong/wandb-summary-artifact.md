---
status: done
started: 2026-04-23
finished: 2026-04-23
---

# Goal

Upload the final moe1-naive-k-ablation summary packet to W&B as an artifact.

# Approach

- Keep existing image logging unchanged.
- Add a W&B artifact to the summary run with JSON/CSV/MD summaries and summary plots.
- Include all generated visualization image paths in the same artifact when present.

# Tasks

- [x] Add artifact helper for relative file upload.
- [x] Call helper from summary W&B run.
- [x] Compile-check runner.
