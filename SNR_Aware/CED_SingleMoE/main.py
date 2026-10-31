"""CED-Tiny single-backbone MoE + KD từ teacher BEATs 5 fold (nhiễu sạch); student chỉ nghe n_hat.

    python -u main.py check   --config config/train_config.json
    python -u main.py fetch   --config config/train_config.json    tải dataset (HF) + logit teacher (HF private)
    python -u main.py smoke   --config config/train_config.json
    python -u main.py train   --config config/train_config.json
    python -u main.py eval    --config config/train_config.json    student TTA +-50 ms trên val
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.dataset import Mixture36Dataset, NHatDataset, build_dataloader, load_teacher_logits  # noqa: E402
from src.kd_trainer import KDTrainer  # noqa: E402
from src.metrics import compute_snr_breakdown  # noqa: E402
from src.models.split_moe import CEDSingleMoESplitFFN  # noqa: E402

SPLITS = ("train", "validation")


def resolve(cfg: dict[str, Any], key: str) -> Path:
    path = Path(cfg["paths"][key])
    return path if path.is_absolute() else ROOT / path


def nhat_path(cfg: dict[str, Any], split: str) -> Path:
    return resolve(cfg, "nhat_dir") / cfg["paths"]["nhat_pattern"].format(split=split)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def build_model(cfg: dict[str, Any]) -> CEDSingleMoESplitFFN:
    m = cfg["model"]
    return CEDSingleMoESplitFFN(
        num_classes=cfg["experiment"]["num_classes"],
        freq_bins=m["freq_bins"],
        frames=m["frames"],
        split_blocks=tuple(m["split_blocks"]),
        dropout=m["dropout"],
        load_pretrained=m["load_pretrained"],
    )


def build_dataset(cfg: dict[str, Any], split: str, with_teacher: bool, max_samples: int | None = None) -> NHatDataset:
    base = Mixture36Dataset(resolve(cfg, "data_root"), split, max_samples)
    teacher = load_teacher_logits(resolve(cfg, "teacher_logits"), split, base.rows) if with_teacher else None
    return NHatDataset(base, nhat_path(cfg, split), teacher)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        # Hợp mọi cột theo thứ tự gặp: dòng "overall" có thêm "auc" mà dòng theo SNR không có.
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def cmd_check(cfg: dict[str, Any]) -> None:
    model = build_model(cfg)
    print("params:", model.get_num_params())
    with torch.no_grad():
        out = model.eval()(torch.randn(2, 64000), return_dict=True)
    print("logits", tuple(out["logits"].shape), "alpha", out["alpha"].flatten().tolist())
    print("teacher_logits", resolve(cfg, "teacher_logits"), resolve(cfg, "teacher_logits").is_file())
    for split in SPLITS:
        dataset = build_dataset(cfg, split, with_teacher=True)
        labels = np.array([int(dataset.base[i]["target_idx"]) for i in range(len(dataset))])
        acc = float((dataset.teacher.argmax(1) == labels).mean())
        print(f"{split}: {len(dataset)} clip | n_hat {nhat_path(cfg, split).name} ok | acc teacher (nhiễu sạch) {acc:.4f}")


def cmd_fetch(cfg: dict[str, Any]) -> None:
    from huggingface_hub import hf_hub_download

    target = resolve(cfg, "nhat_dir")
    target.mkdir(parents=True, exist_ok=True)
    for name in cfg["hf"]["dataset_files"]:
        print("tải", name, flush=True)
        hf_hub_download(cfg["hf"]["dataset_repo"], name, repo_type="dataset", local_dir=target)
    teacher = resolve(cfg, "teacher_logits")
    teacher.parent.mkdir(parents=True, exist_ok=True)
    print("tải logit teacher", flush=True)
    hf_hub_download(cfg["hf"]["teacher_repo"], cfg["hf"]["teacher_file"], local_dir=teacher.parent)
    (teacher.parent / cfg["hf"]["teacher_file"]).replace(teacher) if teacher.name != cfg["hf"]["teacher_file"] else None
    print("Giải nén các *.zip trong", target, "rồi chỉnh paths.data_root / nhat_dir cho khớp thư mục thật.")


def run_train(cfg: dict[str, Any], smoke: bool) -> None:
    t, loss = dict(cfg["train"]), cfg["loss"]
    max_samples = None
    out_dir = resolve(cfg, "output_dir")
    if smoke:
        s = cfg["smoke"]
        t.update(epochs=s["epochs"], batch_size=s["batch_size"], num_workers=0)
        max_samples = s["max_samples"]
        out_dir = out_dir / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_all(cfg["experiment"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== {cfg['experiment']['name']} | {device} | smoke={smoke} ===", flush=True)

    train_set = build_dataset(cfg, "train", True, max_samples)
    val_set = build_dataset(cfg, "validation", True, max_samples)
    train_loader = build_dataloader(train_set, t["batch_size"], True, t["num_workers"])
    val_loader = build_dataloader(val_set, t["batch_size"], False, t["num_workers"])

    trainer = KDTrainer(
        model=build_model(cfg), train_loader=train_loader, val_loader=val_loader, device=device,
        epochs=t["epochs"], lr_backbone=t["lr_backbone"], lr_experts=t["lr_experts"], lr_router=t["lr_router"],
        weight_decay=t["weight_decay"], warmup_epochs=t["warmup_epochs"], mixup_alpha=t["mixup_alpha"],
        mixup_prob=t["mixup_prob"], label_smoothing=loss["label_smoothing"], lambda_kd=loss["lambda_kd"],
        lambda_route=loss["lambda_route"], kd_temperature=loss["kd_temperature"],
        kd_fused_weight=loss["kd_fused_weight"], kd_expert_weight=loss["kd_expert_weight"],
        snr_center_db=loss["snr_center_db"], snr_tau=loss["snr_tau"], grad_clip=t["grad_clip"],
        num_classes=cfg["experiment"]["num_classes"], amp_dtype=t["amp_dtype"], out_dir=out_dir,
    )

    history, best_snr = [], {}
    for epoch in range(t["epochs"]):
        train_res = trainer.train_epoch(epoch)
        val_res = trainer.evaluate()
        overall = val_res["overall"]
        history.append({
            "epoch": epoch + 1, **train_res, "val_top1": overall["top1"], "val_map": overall["map"],
            "val_auc": overall.get("auc", float("nan")), "mean_alpha": val_res["mean_alpha"],
        })
        print(" | ".join(f"{k} {v:.4f}" if k != "epoch" else f"epoch {v:02d}" for k, v in history[-1].items()), flush=True)
        is_best = overall["top1"] > trainer.best_top1_acc
        if is_best:
            trainer.best_top1_acc, trainer.best_results = overall["top1"], overall
            best_snr = {k: v for k, v in val_res.items() if k.endswith("dB")}
        trainer.save_checkpoint(out_dir / "last.pt", is_best=is_best)

    write_csv(out_dir / "lich_su.csv", history)
    write_csv(out_dir / "val_theo_snr.csv", [{"snr": k, **v} for k, v in {**best_snr, "overall": trainer.best_results}.items()])
    params = [{"muc": f"{sec}.{k}", "gia_tri": json.dumps(v)} for sec in ("experiment", "model", "loss", "train") for k, v in cfg[sec].items()]
    write_csv(out_dir / "thong_so.csv", params + [{"muc": "commit", "gia_tri": git_commit()}])
    print(f"best val top1 {trainer.best_top1_acc:.4f} -> {out_dir}", flush=True)


def cmd_eval(cfg: dict[str, Any]) -> None:
    from src.pipeline import load_pipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = resolve(cfg, "output_dir")
    m = cfg["model"]
    pipe = load_pipeline(out_dir / "best_model.pt", device, num_classes=cfg["experiment"]["num_classes"],
                         freq_bins=m["freq_bins"], frames=m["frames"], split_blocks=tuple(m["split_blocks"]))
    loader = build_dataloader(build_dataset(cfg, "validation", False), cfg["train"]["batch_size"], False, 0)
    probs, targets, snrs = [], [], []
    for batch in loader:
        probs.append(pipe.predict_tta(batch["waveform"])[0])
        targets.append(batch["target_idx"].numpy())
        snrs.append(batch["snr_db"].numpy())
    results = compute_snr_breakdown(np.concatenate(probs), np.concatenate(targets), np.concatenate(snrs))
    write_csv(out_dir / "val_tta_theo_snr.csv", [{"snr": k, **v} for k, v in results.items()])
    o = results["overall"]
    print(f"student TTA: top1 {o['top1']:.4f} | mAP {o['map']:.4f} | AUC {o['auc']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["check", "fetch", "smoke", "train", "eval"])
    parser.add_argument("--config", default=str(ROOT / "config" / "train_config.json"))
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    stages = {
        "check": cmd_check, "fetch": cmd_fetch, "eval": cmd_eval,
        "smoke": lambda c: run_train(c, True), "train": lambda c: run_train(c, False),
    }
    stages[args.stage](cfg)


if __name__ == "__main__":
    main()
