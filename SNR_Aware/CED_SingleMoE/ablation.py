"""Ablation layer decay / mixup của Single-CED (SplitFFN) trong final-report.xlsx (2026-11-20).

    python ablation.py <variant> [--stage train|smoke]
        variant: base | ld0p3 | ld0p7 | ld0p82 | ld0p92 | mx<P>a<ALPHA> (vd. mx0p8a0p8, mx1a1)

= main.py <stage> --config config/train_config_overlap_ce9_mini_ablation_s2027.json + --set đúng một trục,
như lượt chạy molab (CED-Mini overlap ce9, seed 2027, 15 epoch, early stop 5 theo val acc).
Với mx0p8a0p8 (val acc 63,19), config sau khi ghi đè được so với config/reference_mx0p8a0p8_thong_so.csv
(thong_so.csv của lượt gốc) và dừng nếu lệch. Đường dẫn lấy từ biến môi trường SF_DATA_V2, SF_NHAT, SF_TEACHER.
Kết quả: output/ablation_s2027/<variant>/ (val_theo_snr.csv dòng overall, lich_su.csv, thong_so.csv).
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config" / "train_config_overlap_ce9_mini_ablation_s2027.json"
REFERENCE = {"mx0p8a0p8": HERE / "config" / "reference_mx0p8a0p8_thong_so.csv"}
SKIP = {"teacher.logits", "commit"}  # đường dẫn / commit: khác máy, không khác phương pháp


def variant_sets(v: str) -> list[str]:
    sets = [f"paths.output_dir=output/ablation_s2027/{v}"]
    if v.startswith("ld"):
        sets.append(f"train.layer_decay={v[2:].replace('p', '.')}")
    elif v.startswith("mx"):
        p, a = v[2:].split("a")
        sets += [f"train.mixup_prob={p.replace('p', '.')}", f"train.mixup_alpha={a.replace('p', '.')}"]
    elif v != "base":
        raise SystemExit(f"variant không hợp lệ: {v}")
    return sets


def path_sets() -> list[str]:
    env = {"SF_DATA_V2": "paths.data_root", "SF_NHAT": "paths.nhat_dir", "SF_TEACHER": "teacher.logits"}
    return [f"{key}={os.environ[name]}" for name, key in env.items() if os.environ.get(name)]


def apply(cfg: dict, item: str) -> None:  # = apply_override của main.py
    key, _, raw = item.partition("=")
    *parents, leaf = key.split(".")
    node = cfg
    for name in parents:
        node = node[name]
    if leaf not in node:
        raise KeyError(key)
    try:
        node[leaf] = json.loads(raw)
    except json.JSONDecodeError:
        node[leaf] = raw


def check_reference(v: str, sets: list[str]) -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in sets:
        apply(cfg, item)
    sections = ("experiment", "teacher", "model", "routing", "loss", "feature_kd", "train")
    now = {f"{sec}.{k}": json.dumps(val) for sec in sections if sec in cfg for k, val in cfg[sec].items()}
    with open(REFERENCE[v], encoding="utf-8") as f:
        ref = {row["muc"]: row["gia_tri"] for row in csv.DictReader(f)}
    diff = [f"{k}: ref={ref.get(k)} now={now.get(k)}" for k in sorted(set(ref) | set(now))
            if k not in SKIP and json.loads(ref.get(k, "null")) != json.loads(now.get(k, "null"))]
    if diff:
        raise SystemExit("config lệch so với lượt tham chiếu:\n  " + "\n  ".join(diff))
    print(f"config khớp {REFERENCE[v].name} (trừ đường dẫn teacher)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("--stage", default="train", choices=["train", "smoke"])
    args = ap.parse_args()
    sets = variant_sets(args.variant) + path_sets()
    if args.variant in REFERENCE:
        check_reference(args.variant, sets)
    if args.stage == "smoke":
        sets[0] = "paths.output_dir=output/ablation_s2027/smoke"
    cmd = [sys.executable, "-u", "main.py", args.stage, "--config", str(CONFIG)]
    for item in sets:
        cmd += ["--set", item]
    print(" ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=HERE))


if __name__ == "__main__":
    main()
