# FRCRN_KD_MoE (chỉ phần dựng n̂)

Phần nhỏ của `FRCRN_KD_MoE` mà Single-CED (SplitFFN) cần để dựng cache n̂ = FRCRN(mixture):
model FRCRN (`src/model/frcrn`), bộ đọc dataset 36_labels (`scripts/load_dataset.py`) và
`scripts/frcrn_cache.py`. Code trùng từng byte với bản đã chạy trên molab cho ablation 2026-11-20.
`src/model/kdmoe` và `src/loss` có mặt chỉ vì `frcrn_cache.py` import chúng.

Checkpoint FRCRN không có trên git: chép `frcrn_best.pt` (SHA-256 `b8fdeb00a762...`) vào `model/frcrn_best.pt`.
Chạy qua `../CED_SingleMoE/build_nhat.py` (xem README của CED_SingleMoE, mục "Ablation").
