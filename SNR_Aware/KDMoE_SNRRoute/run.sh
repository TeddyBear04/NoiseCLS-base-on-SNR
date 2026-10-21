#!/bin/bash
# Cả chuỗi của nhánh: cache n_hat + SNR FRCRN -> teacher cross-fit -> train các run -> evaluate. Chạy tuần tự.
#   DATASET=/path/to/mix-dataset bash run.sh        RUNS="..." để chỉ chạy một số run
set -u
cd "$(dirname "$0")" || exit 1
DATASET=${DATASET:-/marimo/dataset/mix-dataset}
RUNS=${RUNS:-panns_snrroute mn10_snrroute}
if [ -z "${PY:-}" ]; then [ -x /tmp/uv-venv/bin/python ] && PY="/tmp/uv-venv/bin/python -u" || PY="python -u"; fi
L=output/logs; C=output/configs; mkdir -p $L $C pretrained
NEED_FOLDS=   # nhánh cần đặc trưng teacher -> cần trọng số 5 fold
step() { echo "$(date -u +%F' '%H:%M) START $1" >> output/chain.log; local name=$1; shift
         "$@" > $L/$name.log 2>&1; local rc=$?; echo "$(date -u +%F' '%H:%M) rc=$rc $name" >> output/chain.log; return $rc; }

[ -d "$DATASET" ] && [ -f "$DATASET/manifest.csv" ] || { echo "DATASET không hợp lệ: $DATASET (cần manifest.csv)"; exit 1; }
echo "b8fdeb00a76289fa48451578a7a40aa437cdb9954683d578197dd0f8a184b14a  model/frcrn_best.pt" | sha256sum -c --quiet || { echo "thiếu/sai model/frcrn_best.pt"; exit 1; }
CK=pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
if [ ! -f $CK ]; then
  curl -sL --fail --retry 3 -o $CK.part "https://huggingface.co/WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2/resolve/main/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?download=true" || exit 1
  [ "$(sha256sum $CK.part | cut -c1-16)" = "e5815275a04b6885" ] && mv $CK.part $CK || { echo "BEATs cpt2 sai sha256"; exit 1; }
fi
REPO=/marimo/ext/EfficientAT   # bản clone thiếu metadata/ (sandbox khôi phục) -> clone lại
[ -d $REPO ] && [ ! -d $REPO/metadata ] && rm -rf $REPO

# bản config có đường dẫn dataset của máy này (output/configs/, ngoài git)
$PY - "$DATASET" <<'PYEOF' || exit 1
import json, sys
from pathlib import Path
dataset = sys.argv[1]
for path in Path("config").glob("*.json"):
    c = json.loads(path.read_text(encoding="utf-8"))
    if "paths" in c:
        c["paths"]["dataset"] = dataset
    if "dataset" in c:
        c["dataset"]["path"] = dataset
    Path("output/configs", path.name).write_text(json.dumps(c, indent=2, ensure_ascii=False), encoding="utf-8")
PYEOF

step cache $PY -c "
import logging, os, sys
from pathlib import Path
os.environ['MLNC_MIXTURE_36L'] = sys.argv[1]
import torch, online_pipeline as op
from scripts.frcrn_cache import build_frcrn_cache, build_frcrn_snr
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
device = torch.device('cuda')
sep = op.load_frcrn(Path('config/frcrn.yaml'), Path('model/frcrn_best.pt'), device)
splits = ('train', 'validation', 'test')
build_frcrn_cache(sep, device, logging.getLogger('cache'), num_workers=8, splits=splits)
build_frcrn_snr(logging.getLogger('snr'), num_workers=8, splits=splits)
" "$DATASET" || exit 1

T=output/teacher_unified
if [ ! -f $T/teacher_unified.pt ] || { [ -n "$NEED_FOLDS" ] && ! ls $T/fold*.pt > /dev/null 2>&1; }; then
  step teacher_crossfit $PY teacher_crossfit.py crossfit --config $C/teacher.json || exit 1
fi


for run in $RUNS; do
  [ -f output/runs/$run/train/stage2/result.json ] && { echo "bỏ qua $run (đã xong)" >> output/chain.log; continue; }
  step train_$run $PY main.py train --config $C/train_$run.json
done
step evaluate $PY evaluate.py --dataset "$DATASET" --out output/eval
echo "$(date -u +%F' '%H:%M) CHAIN_DONE" >> output/chain.log
