"""CED-Tiny single-backbone MoE + KD từ teacher BEATs; student chỉ nghe n_hat.

Teacher (`teacher.mode` trong config):
    clean5fold   teacher 5 fold fine-tune trên nhiễu sạch, logit out-of-fold đóng gói sẵn (mọi expert cùng teacher)
    fold0_ema    hai teacher của bản gốc: Fold0 và EMA nghe cùng n_hat với student; `teacher.expert_teachers`
                 gán teacher cho từng expert SNR (model.experts: low, [mid,] high)

    python -u main.py check   --config config/train_config.json
    python -u main.py logits  --config config/train_config_fold0_ema.json   chỉ fold0_ema: logit teacher trên n_hat
    python -u main.py smoke   --config config/train_config.json
    python -u main.py train   --config config/train_config.json
    python -u main.py eval    --config config/train_config.json    student TTA +-50 ms (fold0_ema: thêm trộn teacher)
    python -u main.py diag    --config config/train_config.json    expert có chuyên môn hoá không: ép từng expert theo SNR
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
DUAL_TEACHERS = (("t0", "fold0"), ("tema", "ema"))  # (tên logit, khoá checkpoint trong config teacher)


def as_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def resolve(cfg: dict[str, Any], key: str) -> Path:
    return as_path(cfg["paths"][key])


def nhat_path(cfg: dict[str, Any], split: str) -> Path:
    return resolve(cfg, "nhat_dir") / cfg["paths"]["nhat_pattern"].format(split=split)


def dual_logits_path(cfg: dict[str, Any], split: str, name: str) -> Path:
    return as_path(cfg["teacher"]["logits_dir"]) / f"{split}_{name}_logits.npy"


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
        expert_names=tuple(m["experts"]),
        router_input=m["router_input"],
        patch_stride=tuple(m.get("patch_stride", (16, 16))),
        specaug=m.get("specaug"),
        ced_variant=m.get("ced_variant", "tiny"),
    )


def teacher_logits(cfg: dict[str, Any], split: str, rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    """Logit của từng teacher theo thứ tự manifest: {"clean": ...} hoặc {"t0": ..., "tema": ...}."""
    teacher = cfg["teacher"]
    if teacher["mode"] == "clean5fold":
        return {"clean": load_teacher_logits(as_path(teacher["logits"]), split, rows)}
    if teacher["mode"] == "fold0_ema":
        return {name: np.load(dual_logits_path(cfg, split, name), mmap_mode="r")[: len(rows)] for name, _ in DUAL_TEACHERS}
    raise ValueError(f"teacher.mode không hợp lệ: {teacher['mode']}")


def expert_teachers(cfg: dict[str, Any], logits: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Một mảng logit mỗi expert, theo thứ tự model.experts."""
    if cfg["teacher"]["mode"] == "clean5fold":
        return [logits["clean"]] * len(cfg["model"]["experts"])
    mapping = cfg["teacher"]["expert_teachers"]
    return [logits[mapping[name]] for name in cfg["model"]["experts"]]


def feature_kd_on(cfg: dict[str, Any]) -> bool:
    fkd = cfg.get("feature_kd", {})
    return bool(fkd.get("enabled")) and float(fkd.get("weight", 0.0)) > 0.0


def build_dataset(cfg: dict[str, Any], split: str, with_teacher: bool, max_samples: int | None = None) -> NHatDataset:
    base = Mixture36Dataset(resolve(cfg, "data_root"), split, max_samples)
    teachers = expert_teachers(cfg, teacher_logits(cfg, split, base.rows)) if with_teacher else None
    # GFD-Cosine chỉ là loss lúc train: đặc trưng teacher chỉ xuất cho tập train (fold out-of-fold)
    features = None
    if with_teacher and split == "train" and feature_kd_on(cfg):
        features = [as_path(tap["teacher_features"]) for tap in cfg["feature_kd"]["taps"]]
    return NHatDataset(base, nhat_path(cfg, split), teachers, with_snr_est=frcrn_routing(cfg), teacher_features=features)


def frcrn_routing(cfg: dict[str, Any]) -> bool:
    return cfg["routing"].get("source", "router") == "frcrn_snr"


def frcrn_gates(cfg: dict[str, Any], snr_est: torch.Tensor) -> torch.Tensor:
    """One-hot khoảng SNR của SNR ước lượng từ FRCRN [B, K] (cách route của FRCRN_KD_MoE)."""
    import torch.nn.functional as F

    from src.router import snr_band_index

    return F.one_hot(snr_band_index(snr_est, tuple(cfg["loss"]["snr_edges_db"])), len(cfg["model"]["experts"])).float()


def labels_of(dataset: NHatDataset) -> np.ndarray:
    return np.array([int(dataset.base[i]["target_idx"]) for i in range(len(dataset))])


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
    print("logits", tuple(out["logits"].shape), "gates", out["gates"].tolist())
    print("teacher.mode:", cfg["teacher"]["mode"], "| experts:", cfg["model"]["experts"], "| ngưỡng SNR:", cfg["loss"]["snr_edges_db"])
    for split in SPLITS:
        dataset = build_dataset(cfg, split, with_teacher=True)
        labels = labels_of(dataset)
        accs = {name: float((np.asarray(lg).argmax(1) == labels).mean()) for name, lg in teacher_logits(cfg, split, dataset.base.rows).items()}
        print(f"{split}: {len(dataset)} clip | n_hat {nhat_path(cfg, split).name} ok | acc teacher {accs}")
        for path, position in zip(dataset.feat_paths, dataset.feat_position):
            feats = np.load(path, mmap_mode="r")
            rows = position[position >= 0]
            picked = np.asarray(feats[np.sort(rows[:: max(1, len(rows) // 2000)])], dtype=np.float32)
            # mỗi clip một vector: file lưới token (MGD, [N, 25, 4, 768]) lấy trung bình token, không thì ma trận cos
            # (clip x token)^2 cỡ 2e5 x 2e5 làm tràn bộ nhớ
            sample = picked.reshape(len(picked), -1, 768).mean(axis=1)
            unit = sample / np.linalg.norm(sample, axis=1, keepdims=True)
            print(f"  đặc trưng teacher {path.name} {feats.shape}: có cho {int((position >= 0).sum())}/{len(dataset)} clip"
                  f" | cos trung bình giữa các clip {float((unit @ unit.T).mean()):.3f}")


@torch.no_grad()
def cmd_logits(cfg: dict[str, Any]) -> None:
    """fold0_ema: chạy Fold0 và EMA trên n_hat (như pipeline gốc), lưu logit theo thứ tự manifest v2."""
    from huggingface_hub import hf_hub_download

    from src.frontend import rms_normalize
    from src.models.beat import load_fold_teacher

    teacher = cfg["teacher"]
    if teacher["mode"] != "fold0_ema":
        raise SystemExit("stage logits chỉ dùng cho teacher.mode = fold0_ema")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    as_path(teacher["logits_dir"]).mkdir(parents=True, exist_ok=True)
    for name, key in DUAL_TEACHERS:
        checkpoint = as_path(teacher[key])
        if not checkpoint.is_file():
            hf_hub_download(teacher["hf_repo"], checkpoint.name, repo_type="dataset", local_dir=checkpoint.parent)
        model, meta = load_fold_teacher(checkpoint, as_path(teacher["beat_config"]), device)
        print(f"teacher {name}: {checkpoint.name} {meta}", flush=True)
        for split in SPLITS:
            dataset = build_dataset(cfg, split, with_teacher=False)
            target = dual_logits_path(cfg, split, name)
            result = np.zeros((len(dataset), cfg["experiment"]["num_classes"]), dtype=np.float32)
            row = 0
            for batch in build_dataloader(dataset, teacher["batch_size"], False, cfg["train"]["num_workers"]):
                wave = batch["waveform"].to(device)
                if teacher["rms_normalize"]:
                    wave = rms_normalize(wave)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(wave).logits.float()
                result[row : row + len(logits)] = logits.cpu().numpy()
                row += len(logits)
            np.save(target, result)
            acc = float((result.argmax(1) == labels_of(dataset)).mean())
            print(f"  {split}: {row} clip -> {target.name} | acc teacher {acc:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()


def run_train(cfg: dict[str, Any], smoke: bool) -> None:
    t, loss = dict(cfg["train"]), cfg["loss"]
    max_samples = None
    out_dir = resolve(cfg, "output_dir")
    if smoke:
        s = cfg["smoke"]
        t.update(epochs=s["epochs"], batch_size=s["batch_size"], num_workers=0)
        max_samples = s["max_samples"]
        out_dir = out_dir / "smoke"
    if loss.get("kd_style", "splitffn") == "kdmoe" and cfg["teacher"]["mode"] != "clean5fold":
        raise ValueError("loss.kd_style kdmoe (KD-MoE: một teacher trên logit cuối) cần teacher.mode clean5fold")
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_all(cfg["experiment"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== {cfg['experiment']['name']} | {device} | teacher {cfg['teacher']['mode']} | smoke={smoke} ===", flush=True)

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
        snr_edges_db=tuple(loss["snr_edges_db"]), snr_tau=loss["snr_tau"], grad_clip=t["grad_clip"],
        band_labels=cfg["routing"]["band_labels"], route_source=cfg["routing"].get("source", "router"),
        oracle_prob_start=cfg["routing"]["oracle_prob_start"],
        oracle_prob_end=cfg["routing"]["oracle_prob_end"],
        num_classes=cfg["experiment"]["num_classes"], amp_dtype=t["amp_dtype"], out_dir=out_dir,
        feature_kd=cfg.get("feature_kd") if feature_kd_on(cfg) else None,
        lr_head=t.get("lr_head"),
        kd_style=loss.get("kd_style", "splitffn"),
        ce_weight=float(loss.get("ce_weight", 1.0)),
        kd_weight=float(loss.get("kd_weight", 1.0)),
    )

    history, best_snr, best_pred = [], {}, []
    for epoch in range(t["epochs"]):
        train_res = trainer.train_epoch(epoch)
        val_res = trainer.evaluate()
        overall = val_res["overall"]
        history.append({
            "epoch": epoch + 1, **train_res, "val_top1": overall["top1"], "val_map": overall["map"],
            "val_auc": overall.get("auc", float("nan")), "val_precision_macro": overall["precision_macro"],
            "val_recall_macro": overall["recall_macro"], "val_macro_f1": overall["macro_f1"],
            "router_band_acc": val_res["router_band_acc"],
            "oracle_prob": trainer.oracle_probability(epoch),
            **{f"gate_{name}": g for name, g in zip(cfg["model"]["experts"], val_res["mean_gates"])},
        })
        print(" | ".join(f"{k} {v:.4f}" if k != "epoch" else f"epoch {v:02d}" for k, v in history[-1].items()), flush=True)
        is_best = overall["top1"] > trainer.best_top1_acc
        if is_best:
            trainer.best_top1_acc, trainer.best_results = overall["top1"], overall
            best_snr = {k: v for k, v in val_res.items() if k.endswith("dB")}
            p = val_res["predictions"]
            best_pred = [{"sample_id": row["sample_id"], "snr_db": float(s), "label": int(y), "pred": int(q), "epoch": epoch + 1}
                         for row, s, y, q in zip(val_set.base.rows, p["snr"], p["target"], p["pred"])]
        trainer.save_checkpoint(out_dir / "last.pt", is_best=is_best)

    write_csv(out_dir / "lich_su.csv", history)
    write_csv(out_dir / "val_du_doan_best.csv", best_pred)  # dự đoán val (không TTA) của epoch tốt nhất
    write_csv(out_dir / "val_theo_snr.csv", [{"snr": k, **v} for k, v in {**best_snr, "overall": trainer.best_results}.items()])
    sections = tuple(sec for sec in ("experiment", "teacher", "model", "routing", "loss", "feature_kd", "train") if sec in cfg)
    params = [{"muc": f"{sec}.{k}", "gia_tri": json.dumps(v)} for sec in sections for k, v in cfg[sec].items()]
    write_csv(out_dir / "thong_so.csv", params + [{"muc": "commit", "gia_tri": git_commit()}])
    print(f"best val top1 {trainer.best_top1_acc:.4f} -> {out_dir}", flush=True)


def cmd_eval(cfg: dict[str, Any]) -> None:
    from src.pipeline import ensemble_with_teachers, load_pipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = resolve(cfg, "output_dir")
    m = cfg["model"]
    pipe = load_pipeline(out_dir / "best_model.pt", device, num_classes=cfg["experiment"]["num_classes"],
                         freq_bins=m["freq_bins"], frames=m["frames"], split_blocks=tuple(m["split_blocks"]),
                         expert_names=tuple(m["experts"]), router_input=m["router_input"],
                         patch_stride=tuple(m.get("patch_stride", (16, 16))),
                         ced_variant=m.get("ced_variant", "tiny"))
    dataset = build_dataset(cfg, "validation", False)
    probs, alphas, targets, snrs = [], [], [], []
    for batch in build_dataloader(dataset, cfg["train"]["batch_size"], False, 0):
        p, a = pipe.predict_tta(batch["waveform"], frcrn_gates(cfg, batch["snr_est"]) if frcrn_routing(cfg) else None)
        probs.append(p)
        alphas.append(a)
        targets.append(batch["target_idx"].numpy())
        snrs.append(batch["snr_db"].numpy())
    p_student, alpha = np.concatenate(probs), np.concatenate(alphas)
    targets, snrs = np.concatenate(targets), np.concatenate(snrs)

    modes = {"student_tta": p_student}
    if cfg["teacher"]["mode"] == "fold0_ema":
        logits = teacher_logits(cfg, "validation", dataset.base.rows)
        modes["student_tta+teacher"] = ensemble_with_teachers(p_student, alpha, np.asarray(logits["t0"]), np.asarray(logits["tema"]))
    rows = []
    for name, p in modes.items():
        results = compute_snr_breakdown(p, targets, snrs)
        rows += [{"che_do": name, "snr": k, **v} for k, v in results.items()]
        o = results["overall"]
        print(f"{name:>20}: top1 {o['top1']:.4f} | mAP {o['map']:.4f} | AUC {o['auc']:.4f}")
    write_csv(out_dir / "val_tta_theo_snr.csv", rows)


@torch.no_grad()
def cmd_diag(cfg: dict[str, Any]) -> None:
    """Expert có làm đúng vai chuyên gia không: accuracy theo SNR khi dùng router, khi ép toàn bộ sang từng expert,
    và khi route theo khoảng SNR thật (oracle). Chuyên môn hoá thật: expert k thắng ở khoảng của nó."""
    import torch.nn.functional as F

    from src.router import snr_band_index

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = resolve(cfg, "output_dir")
    names, edges = cfg["model"]["experts"], tuple(cfg["loss"]["snr_edges_db"])
    model = build_model({**cfg, "model": {**cfg["model"], "load_pretrained": False}})
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location="cpu", weights_only=False)["model_state_dict"])
    model.to(device).eval()
    dataset = build_dataset(cfg, "validation", False)
    modes = ["router", *names, "oracle"]
    preds = {m: [] for m in modes}
    labels, snrs, gates = [], [], []
    for batch in build_dataloader(dataset, 256, False, cfg["train"]["num_workers"]):
        wave, snr = batch["waveform"].to(device), batch["snr_db"].to(device)
        for m in modes:
            if m == "router":  # routing thật lúc suy luận: router học, hoặc khoảng SNR ước lượng từ FRCRN
                oracle = frcrn_gates(cfg, batch["snr_est"]).to(device) if frcrn_routing(cfg) else None
            elif m == "oracle":
                oracle = F.one_hot(snr_band_index(snr, edges), len(names)).float()
            else:
                oracle = F.one_hot(torch.full_like(snr, names.index(m), dtype=torch.long), len(names)).float()
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(wave, return_dict=True, oracle_gates=oracle)
            preds[m].append(out["logits"].float().argmax(1).cpu().numpy())
            if m == "router":
                gates.append(out["gates_used"].float().cpu().numpy())
        labels.append(batch["target_idx"].numpy())
        snrs.append(batch["snr_db"].numpy())
    y, s, g = np.concatenate(labels), np.concatenate(snrs), np.concatenate(gates)
    band = snr_band_index(torch.from_numpy(s), edges).numpy()
    rows = []
    for v in [*sorted(set(s.tolist())), None]:
        mask = np.ones_like(s, bool) if v is None else np.isclose(s, v)
        row = {"snr": "all" if v is None else v, "count": int(mask.sum())}
        row.update({f"acc_{m}": float((np.concatenate(preds[m])[mask] == y[mask]).mean()) for m in modes})
        row.update({f"gate_{n}": float(g[mask, k].mean()) for k, n in enumerate(names)})
        row["router_band_acc"] = float((g[mask].argmax(1) == band[mask]).mean())
        rows.append(row)
    write_csv(out_dir / "val_chan_doan_expert.csv", rows)
    print(" | ".join(rows[0]))
    for row in rows:
        print(" | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row.values()))
    first, last = (np.concatenate(preds[n]) for n in (names[0], names[-1]))
    print(f"dự đoán của expert {names[0]} và {names[-1]} trùng nhau: {(first == last).mean():.3f}")


def apply_override(cfg: dict[str, Any], item: str) -> None:
    """`a.b=v`: đặt cfg["a"]["b"] = json(v) (chuỗi không phải JSON giữ nguyên); khoá phải có sẵn để tránh gõ nhầm."""
    key, _, raw = item.partition("=")
    *parents, leaf = key.split(".")
    node = cfg
    for name in parents:
        node = node[name]
    if leaf not in node:
        raise KeyError(f"--set {key}: không có trong config")
    try:
        node[leaf] = json.loads(raw)
    except json.JSONDecodeError:
        node[leaf] = raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["check", "logits", "smoke", "train", "eval", "diag"])
    parser.add_argument("--config", default=str(ROOT / "config" / "train_config.json"))
    parser.add_argument("--set", action="append", default=[], metavar="MỤC=GIÁ_TRỊ",
                        help="ghi đè config, vd. --set feature_kd.weight=1.0 --set paths.output_dir=output/x (giá trị JSON)")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for item in args.set:
        apply_override(cfg, item)
    stages = {
        "check": cmd_check, "logits": cmd_logits, "eval": cmd_eval, "diag": cmd_diag,
        "smoke": lambda c: run_train(c, True), "train": lambda c: run_train(c, False),
    }
    stages[args.stage](cfg)


if __name__ == "__main__":
    main()
