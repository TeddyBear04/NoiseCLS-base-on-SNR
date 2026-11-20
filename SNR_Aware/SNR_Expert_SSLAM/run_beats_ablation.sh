#!/bin/bash
# Ablation layer decay / mixup cho BEATs-BEATs + logit KD (run3c_kd_only, student nghe n_hat),
# sheet "Đánh giá Layer_decay" và "Đánh giá Mixup" của final-report.xlsx (user 2026-11-20).
# Mỗi lượt đổi MỘT trục so với gốc (layer decay 1.0, không mixup); 15 epoch, early stop 5, seed 2027.
#
#   CFG=config/train_config_beats_ablation_s2027.json bash run_beats_ablation.sh
#
# Thứ tự: smoke -> teacher -> gốc (= layer decay 1.0 = mixup p 0) -> layer decay 0.3/0.7/0.82/0.92
#          -> mixup p 0.4/0.8/1.0 x alpha 0.1/0.2/0.4/0.8/1.0. Tuần tự trên một GPU.
# Chỉ chạy một phần: ONLY="ld0p3 mx0p4a0p1" (tên lượt như trong danh sách VARIANTS).
# Mỗi bước xong ghi $W/done/<bước>; chạy lại script sẽ bỏ qua bước đã xong.
set -u
cd "$(dirname "$0")" || exit 1
CFG=${CFG:-config/train_config_beats_ablation_s2027.json}
if [ -z "${PY:-}" ]; then [ -x /tmp/uv-venv/bin/python ] && PY=/tmp/uv-venv/bin/python || PY=python; fi
NAME=$(basename "$CFG" .json)
W=${W:-output/$NAME}; mkdir -p "$W/logs" "$W/done"
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
RUN=run3c_kd_only
log() { echo "$(date -u +%F' '%H:%M) $*" >> "$W/chain.log"; }
step() {  # step <tên> <lệnh...>: bỏ qua nếu đã xong, dừng cả chuỗi nếu lỗi
  local name=$1; shift
  [ -f "$W/done/$name" ] && return 0
  log "START $name"; "$@" > "$W/logs/$name.log" 2>&1; local rc=$?; log "rc=$rc $name"
  [ $rc -eq 0 ] && touch "$W/done/$name" || { log "STOP: $name lỗi, xem $W/logs/$name.log"; exit $rc; }
}
log "ablation config=$CFG commit $(git rev-parse --short HEAD 2>/dev/null)"

CK=../BEATs_Experts/checkpoint/pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
if [ ! -f $CK ]; then
  mkdir -p "$(dirname $CK)"
  curl -sL --fail --retry 3 -o $CK.part "https://huggingface.co/WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2/resolve/main/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?download=true" || exit 1
  [ "$(sha256sum $CK.part | cut -c1-16)" = "e5815275a04b6885" ] && mv $CK.part $CK || { log "BEATs cpt2 sai sha256"; exit 1; }
fi

SM=$W/smoke.json
$PY - "$CFG" "$SM" <<'PYEOF' || exit 1
import json, sys
src, smoke = sys.argv[1:3]
c = json.load(open(src, encoding="utf-8"))
c["runtime"]["smoke_test"] = True
c["outputs"]["dir"] += "_smoke"
json.dump(c, open(smoke, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
PYEOF

# tên lượt -> cờ dòng lệnh
declare -A FLAGS=([base]="")
VARIANTS="base"
for d in 0.3 0.7 0.82 0.92; do t=ld${d/./p}; FLAGS[$t]="--layer-decay $d"; VARIANTS="$VARIANTS $t"; done
for p in 0.4 0.8 1; do for a in 0.1 0.2 0.4 0.8 1; do
  t=mx${p/./p}a${a/./p}; FLAGS[$t]="--mixup $p,$a"; VARIANTS="$VARIANTS $t"
done; done
[ -n "${ONLY:-}" ] && VARIANTS="$ONLY"

step smoke_teacher $PY -u main.py teacher36 --config $SM
step smoke_mixup   $PY -u main.py student36 --config $SM --run $RUN --mixup 1,0.4 --layer-decay 0.3
step smoke_val     $PY -u main.py test36    --config $SM --run $RUN --mixup 1,0.4 --layer-decay 0.3 --eval-split validation
log SMOKE_OK
step teacher $PY -u main.py teacher36 --config $CFG
log TEACHER_DONE

for v in $VARIANTS; do
  step ${v}_train $PY -u main.py student36 --config $CFG --run $RUN ${FLAGS[$v]}
  step ${v}_val   $PY -u main.py test36    --config $CFG --run $RUN ${FLAGS[$v]} --eval-split validation
done
log ALL_DONE
