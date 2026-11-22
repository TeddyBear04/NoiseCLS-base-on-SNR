"""Ablation layer decay / mixup cho Dual-CED-Mini (final-report.xlsx, 2026-11-20).

    python dual_ablation.py <variant>     # base | ld0p3 | ld0p7 | ld0p82 | ld0p92 | mx<P>a<ALPHA> (vd. mx0p4a0p1, mx1a1)

Gốc = config_mini_dual_input.yaml (FRCRN -> Low expert, raw -> High expert, dual KD, router loss) với recipe ce9
của user thay cho recipe riêng: 15 epoch, early stop 5 (val acc), batch 64, AdamW wd 0.01, lr encoder 3e-5 /
head + router 3e-4, warm-up 2 epoch + cosine về 0, clip 1.0, không label smoothing / SpecAugment / SWA / TTA,
layer decay 1.0, không mixup, seed 2027. Mỗi lượt đổi đúng một trục. Layer decay theo quy tắc của user
(block i: lr x decay^(11 - i), embedding giữ lr encoder).
Chấm: blend beta 0.85 vào High expert + anchored gate c = clip(0.42 + 0.05 (alpha - 0.5)), một view.

Đường dẫn đọc từ biến môi trường (mặc định = máy molab đã chạy ra số trong Excel):
    DUAL_DATASET      /datasets/mix-dataset-v2      (train/, eval/, labels.txt)
    DUAL_FRCRN        /datasets/frcrn               (train.npy, validation.npy = n_hat FRCRN theo thứ tự manifest)
    DUAL_TEACHER_DIR  /marimo/pretrained/beat-ft    (mix_frcrn_ld0p3_best_model.pt, mix_original_ld0p8_weights.pt)
    DUAL_OUT          /marimo/dual_ablation         (kết quả, mỗi variant một thư mục)
"""
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from src.data import build_datasets  # noqa: E402
from src.metrics import classification_metrics, snr_metrics  # noqa: E402
from src.models import DualExpertCED  # noqa: E402
from src.runner import run_training  # noqa: E402

SEED = 2027
DATASET = os.environ.get("DUAL_DATASET", "/datasets/mix-dataset-v2")
FRCRN = os.environ.get("DUAL_FRCRN", "/datasets/frcrn")
TEACHER_DIR = Path(os.environ.get("DUAL_TEACHER_DIR", "/marimo/pretrained/beat-ft"))
OUT_ROOT = Path(os.environ.get("DUAL_OUT", "/marimo/dual_ablation"))
SHARED = OUT_ROOT / "_teacher_cache"
# Khoá đường dẫn: không so với config tham chiếu (chỉ khác máy, không khác phương pháp).
PATH_KEYS = {("dataset", "path"), ("dataset", "frcrn_root"), ("teacher", "beat_config"),
             ("teacher", "checkpoint_low"), ("teacher", "checkpoint_high")}


def variant_overrides(name: str) -> dict:
    if name == "base":
        return {}
    if name.startswith("ld"):
        return {"layer_decay": float(name[2:].replace("p", "."))}
    if name.startswith("mx"):
        p, a = name[2:].split("a")
        return {"mixup_prob": float(p.replace("p", ".")), "mixup_alpha": float(a.replace("p", "."))}
    raise ValueError(name)


def build_config(name: str) -> dict:
    cfg = yaml.safe_load(open(ROOT / "config" / "config_mini_dual_input.yaml"))
    cfg["dataset"]["path"] = DATASET
    cfg["dataset"]["frcrn_root"] = FRCRN
    cfg["teacher"]["beat_config"] = str(ROOT / "config" / "beat.yaml")
    cfg["teacher"]["checkpoint_low"] = str(TEACHER_DIR / "mix_frcrn_ld0p3_best_model.pt")
    cfg["teacher"]["checkpoint_high"] = str(TEACHER_DIR / "mix_original_ld0p8_weights.pt")
    cfg["experiment"]["seed"] = SEED
    ft = cfg["finetuning"]
    ft.update({
        "epochs": 15, "patience": 5, "selection": "acc",
        "batch_size": 64, "accumulation_steps": 1, "validation_batch_size": 64, "num_workers": 8,
        "encoder_lr": 3.0e-5, "head_lr": 3.0e-4, "router_lr": 3.0e-4, "weight_decay": 0.01,
        "warmup_fraction": 2.0 / 15.0, "lr_schedule": "warmup_cosine", "gradient_clip_norm": 1.0,
        "label_smoothing": 0.0, "augment": {"spec_ratio": 0.0},
        "layer_decay": 1.0, "layer_decay_embeddings": False,
        "mixup_prob": 0.0, "mixup_alpha": 0.4,
        "swa_enabled": False, "tta_enabled": False,
    })
    cfg["kd"]["label_smoothing"] = 0.0
    ft.update(variant_overrides(name))
    cfg["recipe"]["id"] = f"dual_ced_mini_ablation_ce9_s{SEED}_{name}"
    return cfg


def check_reference(cfg: dict) -> None:
    """mx1a1: config phải trùng config_used.json của lượt molab 2026-11-20 (val acc 61.78), trừ đường dẫn."""
    ref = json.load(open(ROOT / "config" / "reference_mx1a1_config_used.json"))
    diff = []
    for sec in sorted(set(ref) | set(cfg)):
        a, b = ref.get(sec, {}), cfg.get(sec, {})
        for k in sorted(set(a) | set(b)):
            if (sec, k) not in PATH_KEYS and a.get(k) != b.get(k):
                diff.append(f"{sec}.{k}: ref={a.get(k)!r} now={b.get(k)!r}")
    if diff:
        raise SystemExit("config lệch so với lượt tham chiếu:\n  " + "\n  ".join(diff))
    print("config khớp reference_mx1a1_config_used.json (trừ đường dẫn)", flush=True)


@torch.no_grad()
def anchored_eval(ckpt: Path, cfg: dict, out: Path) -> dict:
    device = torch.device("cuda")
    val = build_datasets(cfg, include_test=False)["validation"]
    loader = DataLoader(val, batch_size=64, shuffle=False, num_workers=8)
    model = DualExpertCED(num_classes=36, arch=cfg["student"]["arch"], embed_dim=int(cfg["student"]["embed_dim"]),
                          router_mode=cfg["router"]["mode"], transition_center_db=7.5, pretrained=False).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state", state), strict=True)
    model.eval()
    fused, low, high, alpha, tgt, snr, ids = [], [], [], [], [], [], []
    for batch in loader:
        raw, fr = batch["waveform_raw"].to(device), batch["waveform_frcrn"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = model(waveform_raw=0.85 * fr + 0.15 * raw, waveform_frcrn=fr, return_details=True)
        lo, hi, al = o["logits_low"].float(), o["logits_high"].float(), o["alpha"].float()
        c = (0.42 + 0.05 * (al - 0.5)).clamp(0, 1)
        fused.append(((1 - c) * lo + c * hi).cpu())
        low.append(lo.cpu()); high.append(hi.cpu()); alpha.append(al.cpu())
        tgt.append(batch["target_idx"]); snr.append(batch["snr_db"])
        ids.extend(batch.get("sample_id", [""] * raw.shape[0]))
    logits, target, snr_t = torch.cat(fused), torch.cat(tgt), torch.cat(snr)
    res = {"overall": classification_metrics(logits, target, num_classes=36),
           "by_snr": snr_metrics(logits, target, snr_t, num_classes=36),
           "n": int(target.numel()), "ckpt_epoch": str(state.get("epoch", "?")),
           "inference": "beta 0.85 blend -> High, anchored gate 0.42 + 0.05 (alpha - 0.5), 1 view"}
    np.savez_compressed(out / "val_predictions_anchored.npz", logits=logits.numpy(), target=target.numpy(),
                        snr=snr_t.numpy(), alpha=torch.cat(alpha).numpy(), sample_id=np.array(ids))
    return res


def main() -> None:
    name = sys.argv[1]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)
    cfg = build_config(name)
    if name == "mx1a1":
        check_reference(cfg)
    out = OUT_ROOT / name
    if (out / "DONE").is_file():
        print("already done", out, flush=True)
        return
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    SHARED.mkdir(parents=True, exist_ok=True)
    for f in SHARED.glob("teacher_logits_train_*.npy"):  # dùng chung logit teacher (tính một lần)
        os.symlink(f, out / f.name)
    json.dump(cfg, open(out / "config_used.json", "w"), indent=2)
    print("VARIANT", name, json.dumps(variant_overrides(name)), "OUT", out, flush=True)
    t0 = time.time()
    run_training(ROOT / "config", out / "ckpt", out, config_override=cfg)
    for f in out.glob("teacher_logits_train_*.npy"):
        if not f.is_symlink():
            shutil.move(str(f), SHARED / f.name)
            os.symlink(SHARED / f.name, f)
    res = anchored_eval(out / "best_model.pt", cfg, out)
    res["train_minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(res, open(out / "results_anchored.json", "w"), indent=2, default=float)
    o = res["overall"]
    print(f"RESULT {name} acc={o['acc']:.4f} prec={o['precision']:.4f} rec={o['recall']:.4f} "
          f"f1={o['macro_f1']:.4f} mAP={o['mAP']:.4f} epoch={res['ckpt_epoch']}", flush=True)
    (out / "DONE").touch()


if __name__ == "__main__":
    main()
