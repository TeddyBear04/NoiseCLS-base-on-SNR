#!/bin/bash
# BEATs-BEATs KHÔNG FiLM (user 2026-11-24): chạy lại bảng thành phần + quét hệ số KD logit.
#
#   nohup bash run_beats_nofilm.sh > /dev/null 2>&1 &
#
# Teacher: một teacher nhiễu sạch của train_config_beats_ce15_rh.json (seed 2026), dùng chung cho mọi run.
# Thành phần (seed 2026, 15 epoch, không early stop; baseline CE vốn không FiLM nên không chạy lại):
#   mix_<run>  = ce15_rh        --run <run> --no-film
#   nhat_<run> = ce15_rh_frcrn  --run <run> --no-film --teacher-dir artifacts_beats_ce15_rh   (student nghe n_hat)
#   với <run> = run3c_kd_only (KD logit), run3b_crd_only (CRD), run3_kd_crd (KD + CRD)
# Quét λ_KD (KD logit, mixture, recipe ablation seed 2027 / early stop 5): kdw<W> = ce15_rh --run run3c_kd_only
#   --no-film --kd-weight W --seed 2027 --patience 5, W = 0.1 0.3 0.5 0.7 1.
# Thứ tự: smoke -> teacher -> KD logit mixture -> quét λ_KD -> KD logit n_hat -> 4 run có CRD (chậm ~320 s/epoch).
# Chỉ chạy một phần: ONLY="kdw0p1 nhat_run3_kd_crd". Mỗi bước xong ghi $W/done/<bước>; chạy lại bỏ qua bước đã xong.
# Cần: dataset /marimo/dataset/mix-dataset, cache n_hat /marimo/nhat_cache/frcrn_{train,validation}.npy (+ .json)
# dựng bằng FRCRN_KD_MoE scripts/frcrn_cache.py.
set -u
cd "$(dirname "$0")" || exit 1
if [ -z "${PY:-}" ]; then [ -x /tmp/uv-venv/bin/python ] && PY=/tmp/uv-venv/bin/python || PY=python; fi
MIX=config/train_config_beats_ce15_rh.json
NHAT=config/train_config_beats_ce15_rh_frcrn.json
TD=artifacts_beats_ce15_rh            # outputs.dir của $MIX = nơi có teacher_noise_best.pt
W=${W:-output/beats_nofilm}; mkdir -p "$W/logs" "$W/done"
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
log() { echo "$(date -u +%F' '%H:%M) $*" >> "$W/chain.log"; }
step() {  # step <tên> <lệnh...>: bỏ qua nếu đã xong, dừng cả chuỗi nếu lỗi
  local name=$1; shift
  [ -f "$W/done/$name" ] && return 0
  log "START $name"; "$@" > "$W/logs/$name.log" 2>&1; local rc=$?; log "rc=$rc $name"
  [ $rc -eq 0 ] && touch "$W/done/$name" || { log "STOP: $name lỗi, xem $W/logs/$name.log"; exit $rc; }
}
for f in /marimo/nhat_cache/frcrn_train.npy /marimo/nhat_cache/frcrn_validation.npy; do
  [ -e "$f" ] || { log "THIẾU $f"; exit 1; }
done
log "nofilm chain commit $(git rev-parse --short HEAD 2>/dev/null)"

CK=../BEATs_Experts/checkpoint/pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
if [ ! -f $CK ]; then
  mkdir -p "$(dirname $CK)"
  curl -sL --fail --retry 3 -o $CK.part "https://huggingface.co/WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2/resolve/main/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?download=true" || exit 1
  [ "$(sha256sum $CK.part | cut -c1-16)" = "e5815275a04b6885" ] && mv $CK.part $CK || { log "BEATs cpt2 sai sha256"; exit 1; }
fi

# Bản smoke của hai config (smoke_test bật, thư mục riêng *_smoke).
$PY - "$MIX" "$W/smoke_mix.json" "$NHAT" "$W/smoke_nhat.json" <<'PYEOF' || exit 1
import json, sys
for src, smoke in zip(sys.argv[1::2], sys.argv[2::2]):
    c = json.load(open(src, encoding="utf-8"))
    c["runtime"]["smoke_test"] = True
    c["outputs"]["dir"] += "_smoke"
    json.dump(c, open(smoke, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
PYEOF

train_eval() {  # train_eval <tên> <config> <cờ...>: train rồi chấm trên validation sạch 3.064 clip
  local name=$1 cfg=$2; shift 2
  step ${name}_train $PY -u main.py student36 --config $cfg "$@"
  step ${name}_val   $PY -u main.py test36    --config $cfg "$@" --eval-split validation
}
declare -A JOB
for run in run3c_kd_only run3b_crd_only run3_kd_crd; do
  JOB[mix_$run]="$MIX --run $run --no-film"
  JOB[nhat_$run]="$NHAT --run $run --no-film --teacher-dir $TD"
done
for w in 0.1 0.3 0.5 0.7 1; do
  JOB[kdw${w/./p}]="$MIX --run run3c_kd_only --no-film --kd-weight $w --seed 2027 --patience 5"
done
ORDER="mix_run3c_kd_only kdw0p1 kdw0p3 kdw0p5 kdw0p7 kdw1 nhat_run3c_kd_only
       mix_run3b_crd_only mix_run3_kd_crd nhat_run3b_crd_only nhat_run3_kd_crd"
ORDER=${ONLY:-$ORDER}

step smoke_teacher $PY -u main.py teacher36 --config $W/smoke_mix.json
train_eval smoke_mix  $W/smoke_mix.json  --run run3_kd_crd --no-film --kd-weight 0.5 --seed 2027 --patience 5
train_eval smoke_nhat $W/smoke_nhat.json --run run3c_kd_only --no-film --teacher-dir ${TD}_smoke
log SMOKE_OK
step teacher $PY -u main.py teacher36 --config $MIX
log TEACHER_DONE
for j in $ORDER; do
  [ -n "${JOB[$j]:-}" ] || { log "không có bước $j"; exit 1; }
  train_eval $j ${JOB[$j]}
done
log ALL_DONE
