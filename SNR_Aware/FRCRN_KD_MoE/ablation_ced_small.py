"""Ablation layer decay / mixup cho BEATs-CED-Small (final-report.xlsx, user 2026-11-20).

    python -u ablation_ced_small.py <variant> [--smoke]   # base | ld0p3 | ld0p7 | ld0p82 | ld0p92 | mx<P>a<ALPHA> (vd. mx0p4a0p1, mx1a1)

Gốc = biến thể user chọn cho bài (KD logit + CRD, recipe ce9 + SpecAugment):
config/unified/train_ced_small_ce9sa_6_logit_crd.json + kdmoe_ced_small_meanpool_ce9sa_crd.yaml, chỉ đổi seed 2027 và early
stop patience 5 (15 epoch, chọn val acc). Mỗi lượt đổi MỘT trục: student.layer_decay (block i lr x d^(11 - i), block trên
cùng + embedding giữ lr) hoặc train.mixup_prob / mixup_alpha (KDMoETrainer._mixup). Cặp yaml/JSON của lượt được sinh vào
output/ablation_s2027/ced_small/<variant>/; chấm trọng số best_raw (stage2/best.pt) trên validation sạch 3.064 clip
(noise_ytid ngoài AudioSet train), ghi result.json + val_predictions.npz.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)

BASE_JSON = HERE / "config/unified/train_ced_small_ce9sa_6_logit_crd.json"
BASE_YAML = HERE / "config/unified/kdmoe_ced_small_meanpool_ce9sa_crd.yaml"
SEED, PATIENCE = 2027, 5
DATASET = "/marimo/dataset/mix-dataset"
OUT_ROOT = HERE / "output/ablation_s2027/ced_small"


def overrides(name: str) -> dict:
    if name == "base":
        return {}
    if name.startswith("ld"):
        return {"student.layer_decay": float(name[2:].replace("p", "."))}
    if name.startswith("mx"):
        p, a = name[2:].split("a")
        return {"train.mixup_prob": float(p.replace("p", ".")), "train.mixup_alpha": float(a.replace("p", "."))}
    raise ValueError(name)


def build(name: str, out: Path) -> dict:
    spec = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    spec["train"].update({"patience": PATIENCE, "seed": SEED, "epochs": 15, "mixup_prob": 0.0, "mixup_alpha": 0.0})
    spec["student"]["layer_decay"] = 1.0
    for key, value in overrides(name).items():
        section, field = key.split(".")
        spec[section][field] = value
    yaml_path = out / "kdmoe.yaml"
    yaml_path.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")
    config = json.loads(BASE_JSON.read_text(encoding="utf-8"))
    config["paths"].update({"student_config": str(yaml_path), "output": str(out / "train"), "dataset": DATASET})
    config["runtime"]["seed"] = SEED
    config["_ablation"] = {"variant": name, "overrides": overrides(name), "base_json": BASE_JSON.name,
                           "base_yaml": BASE_YAML.name, "seed": SEED, "patience": PATIENCE}
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return config


def run_dir(out: Path) -> Path:
    """online_pipeline.train ghi vào <paths.output>/train/stage2 -> tìm stage2 ở cả hai mức."""
    for d in (out / "train" / "train", out / "train"):
        if (d / "stage2" / "best.pt").is_file():
            return d
    return out / "train" / "train"


def evaluate_best(out: Path) -> dict:
    import torch

    os.environ["MLNC_MIXTURE_36L"] = DATASET
    from evaluate import SNRS, build_model, metrics, predict, weight_sets

    device = torch.device("cuda")
    model = build_model(out / "kdmoe.yaml").to(device)
    model.load_state_dict(weight_sets(run_dir(out))["best_raw"])
    model.eval()
    p = predict(model, "validation", "fp32_cache", None, device)
    clean = ~p["leaked"]
    result = {"subset": "validation clean (3064)", "weights": "best_raw",
              "overall": metrics(p, clean),
              "by_snr": {str(s): metrics(p, clean & (p["snr"] == s)) for s in SNRS}}
    np.savez_compressed(out / "val_predictions.npz", prob=p["prob"][clean], label=p["label"][clean],
                        snr=p["snr"][clean], sample_id=p["id"][clean])
    stage2 = run_dir(out) / "stage2" / "result.json"
    if stage2.is_file():
        result["train_result"] = json.loads(stage2.read_text(encoding="utf-8"))
    return result


def main() -> None:
    name = sys.argv[1]
    smoke = "--smoke" in sys.argv
    eval_only = "--eval-only" in sys.argv
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    out = OUT_ROOT / (name + ("_smoke" if smoke else ""))
    if (out / "DONE").is_file():
        print("already done", out, flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    os.environ["MLNC_MIXTURE_36L"] = DATASET
    from online_pipeline import train

    trained = (run_dir(out) / "stage2" / "result.json").is_file()
    if eval_only or trained:
        print("skip training:", "eval-only" if eval_only else "already trained", flush=True)
    else:
        config = build(name, out)
        print("VARIANT", name, json.dumps(overrides(name)), "->", out, flush=True)
        train(config, smoke=smoke)
    if smoke:
        print("SMOKE_TRAIN_OK", flush=True)
        return
    result = evaluate_best(out)
    (out / "result.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    o = result["overall"]
    print(f"RESULT {name} acc={o['accuracy']:.4f} prec={o['precision_macro']:.4f} rec={o['recall_macro']:.4f} "
          f"f1={o['macro_f1']:.4f} mAP={o['mAP']:.4f} n={o['n']}", flush=True)
    (out / "DONE").touch()


if __name__ == "__main__":
    main()
