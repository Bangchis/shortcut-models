---
status: done
started: 2026-04-23
finished: 2026-04-23
---

# Goal

Always retrain the greedy-selected best MoE config for the full main budget before comparing with `naive_reference`.

# Approach

- Add a final `final_best_moe_50k` run after greedy stages.
- Use the selected `final_config`, full `--max_steps`, and matching GMM stats.
- Include the final run in summary tables, combined plots, and analysis packet.

# Tasks

- [x] Add final best MoE retrain helper.
- [x] Wire final run into multihost flow before naive/reference summary.
- [x] Include final run in combined visualizations and summary JSON/CSV.
- [x] Compile-check touched files.
