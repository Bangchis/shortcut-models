---
status: done
started: 2026-04-24
finished: 2026-04-24
---

# Goal

Thêm một alignment loss trực tiếp cho `model.train_type=naive-moe-source`, để ép `conditioned_mode` khớp với source-cluster theo GMM, có flag bật/tắt qua command line, và cung cấp lệnh chạy đã bật flag.

# Approach

1. Thêm weight flag mới vào `model_config`.
2. Trong nhánh train `naive-moe-source`, tính soft source-cluster posterior từ `mu_x0` bằng GMM hiện có.
3. Thêm cross-entropy alignment loss giữa one-hot `conditioned_mode` và posterior cluster của source.
4. Log thêm metrics chẩn đoán và compile-check.

# Tasks

- [x] Thêm flag `model.loss_alignment_weight`.
- [x] Gắn alignment loss vào nhánh `naive-moe-source`.
- [x] Log metrics alignment trong training summary.
- [x] Compile-check và viết lệnh chạy Kaggle đã bật flag.
