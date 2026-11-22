"""Cache n_hat = FRCRN(mixture) fp32 cho train + validation của 36_labels (paths.nhat_dir của SplitFFN).

    python build_nhat.py

Giống hệt bước dựng cache của lượt gốc (nhat_build.py trên molab): code FRCRN_KD_MoE (thư mục anh em
../FRCRN_KD_MoE, chỉ phần FRCRN + bộ đọc dataset), checkpoint FRCRN `frcrn_best.pt` (SHA-256 b8fdeb00...),
fp32 không TF32. Ra frcrn_{split}.npy [N, 64000] + frcrn_{split}.json (sample_id) + frcrn_{split}.snr.npy.
Val 36_labels có 6.480 clip; SplitFFN lấy 3.064 clip của v2 eval theo sample_id. Khoảng 25 phút trên RTX PRO 6000.

    SF_FRCRN_CKPT  ../FRCRN_KD_MoE/model/frcrn_best.pt   (không có trên git / HF: chép tay vào đây)
    SF_DATA_36L    /marimo/dataset/mix-dataset
    SF_NHAT        /marimo/nhat_cache
"""
import hashlib
import logging
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRCRN_ROOT = HERE.parent / "FRCRN_KD_MoE"
CKPT = Path(os.environ.get("SF_FRCRN_CKPT", FRCRN_ROOT / "model" / "frcrn_best.pt"))
CKPT_SHA256 = "b8fdeb00a76289fa48451578a7a40aa437cdb9954683d578197dd0f8a184b14a"
CACHE = Path(os.environ.get("SF_NHAT", "/marimo/nhat_cache"))
os.environ["MLNC_MIXTURE_36L"] = os.environ.get("SF_DATA_36L", "/marimo/dataset/mix-dataset")


def main() -> None:
    if not CKPT.is_file():
        sys.exit(f"thiếu checkpoint FRCRN {CKPT}: chép frcrn_best.pt vào đó (hoặc đặt SF_FRCRN_CKPT)")
    got = hashlib.sha256(CKPT.read_bytes()).hexdigest()
    if got != CKPT_SHA256:
        sys.exit(f"SHA-256 của {CKPT} là {got}, khác checkpoint của lượt gốc ({CKPT_SHA256[:16]}...)")
    sys.path.insert(0, str(FRCRN_ROOT))
    os.chdir(FRCRN_ROOT)
    import torch
    from scripts.frcrn_cache import build_frcrn_cache, build_frcrn_snr
    from src.model.frcrn import FRCRNSeparator

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("nhat")
    device = torch.device("cuda")
    # = online_pipeline.load_frcrn của FRCRN_KD_MoE
    separator = FRCRNSeparator.from_yaml(FRCRN_ROOT / "config" / "frcrn.yaml")
    report = separator.load_pretrained(CKPT)
    if not report.is_clean:
        sys.exit(f"FRCRN checkpoint không khớp: missing={report.missing[:5]}, unexpected={report.unexpected[:5]}")
    separator = separator.to(device).freeze()
    build_frcrn_cache(separator, device, log, cache_dir=CACHE, num_workers=8, batch_size=64)
    build_frcrn_snr(log, cache_dir=CACHE, num_workers=8)
    print("CACHE_DONE", CACHE, flush=True)


if __name__ == "__main__":
    main()
