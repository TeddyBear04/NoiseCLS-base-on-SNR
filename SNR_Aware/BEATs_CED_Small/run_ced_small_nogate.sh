#!/bin/bash
# BEATs-CED-Small KHÔNG mask gate (user 2026-11-24): 6 config bảng thành phần trong bài, chấm validation + test.
#
#   nohup bash run_ced_small_nogate.sh > /dev/null 2>&1 &
#
# Mặc định = 6 dòng tab:ced_small_components (seed 2026, 15 epoch, không early stop = config ce9sa gốc, use_gate false):
#   c6_logit_pcrd c1_ce c2_logit c4_pcrd c3_mgd c5_logit_mgd -> output/components_ce9sa_nogate/ced_small/<lượt>/
#   (+ logit KD + CRD patch = cấu hình báo cáo, CE, + logit KD, + CRD patch, + MGD, + logit KD + MGD).
# Mỗi lượt: result.json = validation sạch 3.064 (overall/by_snr) + "test" = test sạch 3.253 (user 2026-11-28),
# val_predictions.npz + test_predictions.npz; log có dòng RESULT (val) và TEST.
# Lượt khác vẫn chạy được qua ONLY: c4_crd c6_logit_crd (CRD gộp, cần thêm teacher_pooled.npy), kd<W>, crd<W>
# (quét hệ số, output/ablation_s2027_nogate/ced_small/), vd. ONLY="c6_logit_crd kd0p5".
# Tuần tự trên một GPU. Lượt xong có DONE + result.json; chạy lại bỏ qua lượt đã xong; lượt DONE mà result.json
# chưa có "test" thì chỉ chấm lại (không train lại).
# Cần sẵn: model/frcrn_best.pt, output/teacher_unified/beats_teacher_logits_clean.pt (HF Keshio1/frcrn-kdmoe-teacher)
# + teacher_hints_l12_g25x4.npy (token BEATs l12 lưới 25 x 4, cho CRD patch / MGD), dataset /marimo/dataset/mix-dataset.
# Cache n_hat output/frcrn_cache_fp32 (train/validation/test) tự dựng ở lượt đầu nếu chưa có.
set -u
cd "$(dirname "$0")" || exit 1
if [ -z "${PY:-}" ]; then [ -x /tmp/uv-venv/bin/python ] && PY=/tmp/uv-venv/bin/python || PY=python; fi
W=output/nogate_chain; mkdir -p "$W/logs"
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore
log() { echo "$(date -u +%F' '%H:%M) $*" >> "$W/chain.log"; }
for f in model/frcrn_best.pt output/teacher_unified/beats_teacher_logits_clean.pt \
         output/teacher_unified/teacher_hints_l12_g25x4.npy /marimo/dataset/mix-dataset; do
  [ -e "$f" ] || { log "THIẾU $f"; exit 1; }
done
VARIANTS=${ONLY:-"c6_logit_pcrd c1_ce c2_logit c4_pcrd c3_mgd c5_logit_mgd"}
log "nogate chain commit $(git rev-parse --short HEAD 2>/dev/null): $VARIANTS"

run() {  # run <variant> [--smoke]
  local tag=$1${2:+_smoke}
  log "START $tag"; $PY -u ablation_ced_small.py "$1" --no-gate ${2:-} > "$W/logs/$tag.log" 2>&1; local rc=$?
  log "rc=$rc $tag $(grep -h -e '^RESULT' -e '^TEST' -e '^smoke test' "$W/logs/$tag.log" | tail -2 | tr '\n' ' ')"
  [ $rc -eq 0 ] || { log "STOP: $tag lỗi, xem $W/logs/$tag.log"; exit $rc; }
}
run c6_logit_pcrd --smoke
run c3_mgd --smoke
log SMOKE_OK
for v in $VARIANTS; do run "$v"; done
log ALL_DONE
