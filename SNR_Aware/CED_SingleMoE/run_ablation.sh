#!/bin/bash
# Single-CED (SplitFFN), mixup p = 0,8, alpha = 0,8 (final-report.xlsx, sheet "Đánh giá Mixup"): val acc 63,19.
#   bash run_ablation.sh              # mặc định variant mx0p8a0p8
#   bash run_ablation.sh mx1a1        # variant khác: base | ld0p3 ... | mx0p4a0p1 ...
# Cần trước: chép frcrn_best.pt vào ../FRCRN_KD_MoE/model/ (hoặc đặt SF_FRCRN_CKPT). Xem README.md, mục "Ablation".
set -euo pipefail
cd "$(dirname "$0")"
VARIANT=${1:-mx0p8a0p8}
PY=${PY:-$( [ -x /tmp/uv-venv/bin/python ] && echo /tmp/uv-venv/bin/python || command -v python )}
mkdir -p output/ablation_s2027
LOG=output/ablation_s2027/$VARIANT.log

# 1. Gói (torch có sẵn trên molab: torch 2.11.0+cu130). timm cài --no-deps để không kéo torch khác về.
if command -v uv >/dev/null; then PIP="uv pip install --python $PY -q"; else PIP="$PY -m pip install -q"; fi
$PIP soundfile==0.14.0 scipy scikit-learn pyyaml safetensors huggingface_hub tqdm requests pytest
$PIP --no-deps timm==1.0.30
if ! $PY -c "import torchaudio" 2>/dev/null; then  # torchaudio đúng bản torch (lượt gốc: 2.11.0, cu130)
  TV=$($PY -c "import torch; print(torch.__version__.split('+')[0])")
  CU=$($PY -c "import torch; print('cu' + torch.version.cuda.replace('.', ''))")
  $PIP --index-url https://download.pytorch.org/whl/$CU --no-deps torchaudio==$TV
fi
$PY -c "import torch, torchaudio, timm, soundfile, sklearn; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# 2. Dữ liệu + logit teacher (HF, kiểm SHA-256)
$PY -u download_data.py

# 3. Cache n_hat FRCRN (bỏ qua nếu đã có frcrn_{train,validation}.json)
$PY -u build_nhat.py

# 4. Kiểm nhanh: test recipe ce9 + 1 epoch 64 clip
$PY -m pytest -q -p no:cacheprovider tests/test_ce9_recipe.py
$PY -u ablation.py "$VARIANT" --stage smoke > output/ablation_s2027/smoke.log 2>&1 || { tail -30 output/ablation_s2027/smoke.log; exit 1; }

# 5. Train 15 epoch (early stop 5). Kết quả: output/ablation_s2027/$VARIANT/val_theo_snr.csv (dòng overall)
$PY -u ablation.py "$VARIANT" 2>&1 | tee "$LOG"
grep "best val top1" "$LOG" || true
