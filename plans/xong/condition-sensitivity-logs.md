---
status: done
started: 2026-04-28
finished: 2026-04-28
---

# Goal
Add logs that show whether DiT output actually depends on MoE condition signals.

# Approach
- Add periodic sensitivity diagnostics by re-running DiT on the same `x_t,t` with altered conditions.
- Log output velocity deltas for zeroed condition, zeroed geometry, zeroed rho, and rolled full condition.
- Add cheap embedding norm/relative norm logs every step.
- Update docs/commands.

# Tasks
- [x] Add sensitivity config and train metrics.
- [x] Update markdown/run command.
- [x] Run checks, commit, and push.
