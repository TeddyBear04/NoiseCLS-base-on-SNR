#!/bin/bash
# BEATs-CED-Small KHÔNG mask gate (user 2026-11-24): chạy lại 6 config bảng thành phần + quét hệ số KD.
#
#   nohup bash run_ced_small_nogate.sh > /dev/null 2>&1 &
#
# Thành phần (seed 2026, 15 epoch, không early stop = config ce9sa gốc, chỉ use_gate false):
#   c6_logit_crd c1_ce c2_logit c4_crd c3_mgd c5_logit_mgd -> output/components_ce9sa_nogate/ced_small/
# Quét λ_KD (gốc = KD logit + CRD 0.8, recipe ablation seed 2027 / early stop 5, use_gate false):
#   kd0p1 kd0p3 kd0p5 kd0p7 kd1 -> output/ablation_s2027_nogate/ced_small/
# Tuần tự trên một GPU. Mỗi lượt xong có DONE + result.json; chạy lại script bỏ qua lượt đã xong.
# Chỉ chạy một phần: ONLY="c6_logit_crd kd0p5".
# Cần sẵn (như đợt ablation 2026-11-20): model/frcrn_best.pt, output/frcrn_cache_fp32/, output/teacher_unified/
# (beats_teacher_logits_clean.pt + teacher_pooled.npy), dataset /marimo/dataset/mix-dataset.
set -u
cd "$(dirname "$0")" || exit 1
if [ -z "${PY:-}" ]; then [ -x /tmp/uv-venv/bin/python ] && PY=/tmp/uv-venv/bin/python || PY=python; fi
W=output/nogate_chain; mkdir -p "$W/logs"
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
log() { echo "$(date -u +%F' '%H:%M) $*" >> "$W/chain.log"; }
for f in model/frcrn_best.pt output/frcrn_cache_fp32 output/teacher_unified/beats_teacher_logits_clean.pt \
         output/teacher_unified/teacher_pooled.npy /marimo/dataset/mix-dataset; do
  [ -e "$f" ] || { log "THIẾU $f"; exit 1; }
done
VARIANTS=${ONLY:-"c6_logit_crd c1_ce c2_logit kd0p1 kd0p3 kd0p5 kd0p7 kd1 c4_crd c3_mgd c5_logit_mgd"}
log "nogate chain commit $(git rev-parse --short HEAD 2>/dev/null): $VARIANTS"

run() {  # run <variant> [--smoke]
  local tag=$1${2:+_smoke}
  log "START $tag"; $PY -u ablation_ced_small.py "$1" --no-gate ${2:-} > "$W/logs/$tag.log" 2>&1; local rc=$?
  log "rc=$rc $tag $(grep -h '^RESULT' "$W/logs/$tag.log" | tail -1)"
  [ $rc -eq 0 ] || { log "STOP: $tag lỗi, xem $W/logs/$tag.log"; exit $rc; }
}
run kd0p5 --smoke
run c3_mgd --smoke
log SMOKE_OK
for v in $VARIANTS; do run "$v"; done
log ALL_DONE
