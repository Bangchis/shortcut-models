---
status: done
started: 2026-04-28
finished: 2026-04-28
---

# Goal
Implement conditional Flow Matching for `naive-moe-source` using latent condition `c=(k,a,rho)`, analytic conditional source `x0`, and markdown math/run commands.

# Approach
- Add DiT conditioning vector support for GMM/angular/radius/kappa condition while preserving existing label conditioning.
- Add GMM utility for conditional source sampling in standardized/local space.
- Replace current SourceCNN branch for `naive-moe-source` with analytic conditional source and conditional DiT call.
- Update inference/eval source sampling and model calls to carry the same condition.
- Add diagnostics for source/target geometry and entanglement risk.
- Write markdown documentation with math and Kaggle commands.

# Tasks
- [x] Inspect DiT call/model state and GMM utilities.
- [x] Add conditional embeddings/MLP to DiT.
- [x] Add analytic conditional source sampler.
- [x] Wire training/eval/inference.
- [x] Add metrics and CSV coverage.
- [x] Add markdown math/run guide.
- [x] Run syntax/smoke checks and push branch.
