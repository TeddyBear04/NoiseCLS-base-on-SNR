#!/bin/bash
# 6 run BEATs (CE, attention, KD logit, CRD, KD+CRD, attention+KD+CRD) + teacher theo một recipe,
# chấm trên validation sạch 3.064 clip (config: dataset.validation_exclude_noise_ytids), rồi report36.
#
#   CFG=config/train_config_beats_review3.json ROLE=all bash run_beats_recipe.sh     # cả 6 run trên một máy
#   CFG=config/train_config_beats_ce15.json    ROLE=all bash run_beats_recipe.sh
#
# Chia hai máy (mỗi máy tuần tự, không chạy song song trên một GPU):
#   ROLE=a: teacher -> run3_kd_crd -> run8_attn_kd_crd
#   ROLE=b: run1_baseline -> run7_attn -> (chờ teacher) -> run3c_kd_only -> run3b_crd_only
#   Máy b chờ file <outputs.dir>/teacher_noise_best.pt.ok: chép teacher_noise_best.pt từ máy a sang
#   cùng đường dẫn rồi ghi sha256 vào file .ok - để bốn run KD/CRD dùng CHUNG một teacher.
#   Báo cáo chung: gom student_*_{predictions.npz,test.json,history.json,params.json} của
#   <outputs.dir>_validation hai máy về một chỗ, chạy lại bước report (ROLE=report).
#
# Biến môi trường: PY (mặc định /tmp/uv-venv/bin/python nếu có, không thì python),
# DATASET (mặc định giữ dataset.path của config), W (thư mục log, mặc định output/<tên config>).
# Mỗi bước xong ghi $W/done/<bước>; chạy lại script sẽ bỏ qua bước đã xong.
set -u
cd "$(dirname "$0")" || exit 1
CFG=${CFG:?CFG=config/train_config_beats_<recipe>.json}
ROLE=${ROLE:?ROLE=all|a|b|report}
if [ -z "${PY:-}" ]; then [ -x /tmp/uv-venv/bin/python ] && PY=/tmp/uv-venv/bin/python || PY=python; fi
NAME=$(basename "$CFG" .json)
W=${W:-output/$NAME}; mkdir -p "$W/logs" "$W/done"
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
log() { echo "$(date -u +%F' '%H:%M) $*" >> "$W/chain.log"; }
step() {  # step <tên> <lệnh...>: bỏ qua nếu đã xong, dừng cả chuỗi nếu lỗi
  local name=$1; shift
  [ -f "$W/done/$name" ] && return 0
  log "START $name"; "$@" > "$W/logs/$name.log" 2>&1; local rc=$?; log "rc=$rc $name"
  [ $rc -eq 0 ] && touch "$W/done/$name" || { log "STOP: $name lỗi, xem $W/logs/$name.log"; exit $rc; }
}
log "ROLE=$ROLE config=$CFG commit $(git rev-parse --short HEAD 2>/dev/null)"

CK=../BEATs_Experts/checkpoint/pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
if [ ! -f $CK ]; then
  mkdir -p "$(dirname $CK)"
  curl -sL --fail --retry 3 -o $CK.part "https://huggingface.co/WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2/resolve/main/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?download=true" || exit 1
  [ "$(sha256sum $CK.part | cut -c1-16)" = "e5815275a04b6885" ] && mv $CK.part $CK || { log "BEATs cpt2 sai sha256"; exit 1; }
fi

# Config chạy thật (đổi dataset.path nếu có DATASET) + bản smoke (smoke_test bật, thư mục riêng).
RUN_CFG=$W/config.json; SM=$W/smoke.json
$PY - "$CFG" "$RUN_CFG" "$SM" "${DATASET:-}" <<'PYEOF' || exit 1
import json, sys
src, run, smoke, dataset = sys.argv[1:5]
c = json.load(open(src, encoding="utf-8"))
if dataset:
    c["dataset"]["path"] = dataset
json.dump(c, open(run, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
c["runtime"]["smoke_test"] = True
c["outputs"]["dir"] += "_smoke"
json.dump(c, open(smoke, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
PYEOF
OUT=$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['outputs']['dir'])" "$RUN_CFG")

train_eval() {  # train_eval <run>: train rồi chấm trên validation (đã lọc)
  step ${1}_train $PY -u main.py student36 --config $RUN_CFG --run $1
  step ${1}_val   $PY -u main.py test36    --config $RUN_CFG --run $1 --eval-split validation
}
teacher_part() {
  step smoke_teacher $PY -u main.py teacher36 --config $SM
  step smoke_student $PY -u main.py student36 --config $SM --run run8_attn_kd_crd
  step smoke_val     $PY -u main.py test36    --config $SM --run run8_attn_kd_crd --eval-split validation
  log SMOKE_OK
  step teacher $PY -u main.py teacher36 --config $RUN_CFG
  log TEACHER_DONE
}
ce_part() {
  step smoke_ce_student $PY -u main.py student36 --config $SM --run run7_attn
  step smoke_ce_val     $PY -u main.py test36    --config $SM --run run7_attn --eval-split validation
  log SMOKE_CE_OK
  train_eval run1_baseline
  train_eval run7_attn
}
report() { step report_val $PY -u main.py report36 --config $RUN_CFG --eval-split validation; }

case $ROLE in
  all)
    teacher_part
    ce_part
    for run in run3c_kd_only run3b_crd_only run3_kd_crd run8_attn_kd_crd; do train_eval $run; done
    report ;;
  a)
    teacher_part
    train_eval run3_kd_crd
    train_eval run8_attn_kd_crd ;;
  b)
    ce_part
    T=$OUT/teacher_noise_best.pt
    until [ -f $T.ok ]; do sleep 60; done
    log "teacher có mặt: $(cat $T.ok)"
    train_eval run3c_kd_only
    train_eval run3b_crd_only ;;
  report)
    rm -f "$W/done/report_val"; report ;;
  *) echo "ROLE phải là all|a|b|report"; exit 1 ;;
esac
log "ROLE_DONE $ROLE"
