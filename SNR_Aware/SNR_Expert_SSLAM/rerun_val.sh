#!/bin/bash
# Chạy lại Đợt 4-5 của Result.xlsx để có số VALIDATION (cùng config, seed 2026, code high-expert):
#   Đợt 4  config/train_config_beats_fullsnr.json   BEATs mọi SNR       (gốc: commit 57accbd)
#   Đợt 5  config/train_config_sslam_fullsnr.json   SSLAM mọi SNR       (gốc: commit e284ea3)
#   (tuỳ chọn) Đợt 3  config/train_config_high_beats.json  BEATs 15-20 dB (gốc: 2026-09-29): ROUNDS="... high_beats"
# Mỗi đợt: teacher36 -> 4 student (run1, run7, run3, run8) -> test36 trên test và --eval-split validation
# (Đợt 3 thêm --full-test để có đủ 6 mức) -> report36 cho cả hai. Tuần tự từng job một trên GPU.
#
#   DATASET=/marimo/dataset/mix-dataset bash rerun_val.sh        ROUNDS="beats_fullsnr" để chạy một phần
#
# Mỗi bước xong ghi output/rerun_val/done/<bước>; chạy lại script sẽ bỏ qua bước đã xong.
set -u
cd "$(dirname "$0")" || exit 1
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
DATASET=${DATASET:-/marimo/dataset/mix-dataset}
ROUNDS=${ROUNDS:-beats_fullsnr sslam_fullsnr}
RUNS="run1_baseline run7_attn run3_kd_crd run8_attn_kd_crd"
if [ -z "${PY:-}" ]; then [ -x /tmp/uv-venv/bin/python ] && PY=/tmp/uv-venv/bin/python || PY=python; fi
W=output/rerun_val; L=$W/logs; D=$W/done; C=$W/configs; mkdir -p $L $D $C
log() { echo "$(date -u +%F' '%H:%M) $*" >> $W/chain.log; }
step() {  # step <tên> <lệnh...>: bỏ qua nếu đã xong, dừng cả chuỗi nếu lỗi
  local name=$1; shift
  [ -f $D/$name ] && return 0
  log "START $name"; "$@" > $L/$name.log 2>&1; local rc=$?; log "rc=$rc $name"
  [ $rc -eq 0 ] && touch $D/$name || { log "STOP: $name lỗi, xem $L/$name.log"; exit $rc; }
}

[ -f "$DATASET/manifest.csv" ] || { echo "DATASET không hợp lệ: $DATASET (cần manifest.csv)"; exit 1; }
log "commit $(git rev-parse --short HEAD), dataset $DATASET"

# Môi trường (README): transformers<5, timm; torchvision bản cu130 thay bản PyPI hỏng
if ! $PY -c "import transformers, timm, torchaudio, soundfile, torchvision.ops; assert int(transformers.__version__.split('.')[0]) < 5" 2>/dev/null; then
  step env_install bash -c "uv pip install --python $PY 'transformers<5' timm torchaudio soundfile && \
    uv pip install --python $PY --index-url https://download.pytorch.org/whl/cu130 --no-deps \
      --reinstall-package torchvision torchvision"
fi
CK=../BEATs_Experts/checkpoint/pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
if [ ! -f $CK ]; then
  mkdir -p "$(dirname $CK)"
  curl -sL --fail --retry 3 -o $CK.part "https://huggingface.co/WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2/resolve/main/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?download=true" || exit 1
  [ "$(sha256sum $CK.part | cut -c1-16)" = "e5815275a04b6885" ] && mv $CK.part $CK || { log "BEATs cpt2 sai sha256"; exit 1; }
fi

# Config có đường dẫn dataset của máy này; smoke = cùng config, smoke_test bật, thư mục riêng
$PY - "$DATASET" "$C" <<'PYEOF' || exit 1
import json, sys
from pathlib import Path
dataset, out = sys.argv[1], Path(sys.argv[2])
for name in ("beats_fullsnr", "sslam_fullsnr", "high_beats"):
    c = json.loads(Path(f"config/train_config_{name}.json").read_text(encoding="utf-8"))
    c["dataset"]["path"] = dataset
    (out / f"{name}.json").write_text(json.dumps(c, indent=2, ensure_ascii=False), encoding="utf-8")
    c["runtime"]["smoke_test"] = True
    c["outputs"]["dir"] = c["outputs"]["dir"] + "_smoke"
    (out / f"{name}_smoke.json").write_text(json.dumps(c, indent=2, ensure_ascii=False), encoding="utf-8")
PYEOF

# Smoke: teacher + 1 student + test trên test và validation, phát hiện lỗi môi trường trước khi chạy dài
S=$C/beats_fullsnr_smoke.json
step smoke_teacher $PY -u main.py teacher36 --config $S
step smoke_student $PY -u main.py student36 --config $S --run run8_attn_kd_crd
step smoke_test    $PY -u main.py test36    --config $S --run run8_attn_kd_crd
step smoke_val     $PY -u main.py test36    --config $S --run run8_attn_kd_crd --eval-split validation
log "SMOKE_OK"

for round in $ROUNDS; do
  CFG=$C/$round.json
  FULL=""; [ $round = high_beats ] && FULL="--full-test"   # Đợt 3: validation đủ 6 mức (+ lát 15-20 dB)
  step ${round}_teacher $PY -u main.py teacher36 --config $CFG
  for run in $RUNS; do
    step ${round}_${run}_train $PY -u main.py student36 --config $CFG --run $run
    step ${round}_${run}_test  $PY -u main.py test36    --config $CFG --run $run
    step ${round}_${run}_val   $PY -u main.py test36    --config $CFG --run $run $FULL --eval-split validation
  done
  step ${round}_report_test $PY -u main.py report36 --config $CFG
  step ${round}_report_val  $PY -u main.py report36 --config $CFG $FULL --eval-split validation
  log "ROUND_DONE $round"
done
log "RERUN_DONE"
