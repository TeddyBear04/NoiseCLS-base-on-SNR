#!/bin/bash
# Dual-CED-Mini, mixup p = 1, alpha = 1 (final-report.xlsx, sheet "Đánh giá Mixup"): val acc 61,78.
#   bash run.sh            # mặc định variant mx1a1
#   bash run.sh base       # hoặc variant khác: base | ld0p3 ... | mx0p4a0p1 ...
# PY = python dùng để chạy (molab: /tmp/uv-venv/bin/python). Log: $DUAL_OUT/<variant>.log
set -euo pipefail
cd "$(dirname "$0")"
VARIANT=${1:-mx1a1}
PY=${PY:-$( [ -x /tmp/uv-venv/bin/python ] && echo /tmp/uv-venv/bin/python || command -v python )}
OUT=${DUAL_OUT:-/marimo/dual_ablation}
mkdir -p "$OUT"

# 1. Gói (torch có sẵn trên molab: torch 2.11.0+cu130). timm cài --no-deps để không kéo torch khác về.
if command -v uv >/dev/null; then PIP="uv pip install --python $PY -q"; else PIP="$PY -m pip install -q"; fi
$PIP -r requirements.txt
$PIP --no-deps timm==1.0.30
$PY -c "import torch, timm, soundfile, sklearn; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# 2. Dữ liệu + teacher (HF phitruong0606/mix-dataset-v2, kiểm SHA-256)
$PY -u download_data.py

# 3. Train 15 epoch + chấm val (blend 0,85 + anchored gate). Kết quả: $OUT/$VARIANT/results_anchored.json
$PY -u dual_ablation.py "$VARIANT" 2>&1 | tee "$OUT/$VARIANT.log"
grep RESULT "$OUT/$VARIANT.log" || true
