"""Đánh giá các run KD-MoE của thư mục này trên validation + test, toàn bộ và theo từng mức SNR.

    python -u evaluate.py --dataset /marimo/dataset/mix-dataset --out output/eval [--only mn10]

Hai bộ trọng số mỗi run, vì trainer lưu trọng số thường ở best.pt nhưng chấm (và chọn epoch) bằng EMA:
  best_raw  stage2/best.pt            trọng số thường của epoch được chọn
  ema_last  stage2/resume.pt["ema"]   EMA của epoch cuối (số log in ở dòng "validation" cuối)
Mỗi run giữ đầu vào lúc train:
  fp32_cache   n_hat = FRCRN(mixture) fp32, không TF32 (output/frcrn_cache_fp32; cache test dựng ở đây)
  bf16_online  FRCRN chạy trong autocast bf16 của student (bản KDMoE_SNRRoute đời đầu)
Tập con: all / clean (clip nhiễu không nằm trong AudioSet train) / leaked. Ghi metrics_by_snr.csv và
predictions_<run>_<weights>_<split>.csv (cho McNemar) vào --out.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from online_pipeline import HERE, load_frcrn, student_type
from scripts.frcrn_cache import FRCRN_CACHE_DIR, build_frcrn_cache, build_frcrn_snr
from scripts.load_dataset import KDMoEMixDataset, build_dataloader
from src.model.kdmoe import KDMoEModel, frcrn_snr_db

OUTPUT, CONFIG = HERE / "output", HERE / "config"
RUNS = {  # run: (thư mục output/runs/<run>/train, đầu vào n_hat fp32 cache, yaml model)
    "panns_snrroute": (OUTPUT / "runs" / "panns_snrroute" / "train", "fp32_cache", CONFIG / "kdmoe_panns_cnn10_snrroute.yaml"),
    "mn10_snrroute": (OUTPUT / "runs" / "mn10_snrroute" / "train", "fp32_cache", CONFIG / "kdmoe_mn10_snrroute.yaml"),
}
# noise_ytid nằm trong AudioSet balanced+unbalanced train: backbone/teacher pretrain có nhãn AudioSet đã thấy
LEAKED = {line.strip() for line in (CONFIG / "audioset_train_noise_ytids.txt").read_text().splitlines()
          if line.strip() and not line.startswith("#")}
SPLITS = ("validation", "test")
SNRS = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)


def build_model(yaml_path: Path) -> KDMoEModel:
    """KDMoEModel; ReviewKDMoE khi yaml có mục reviewkd; EfficientATKDMoE khi student.type = efficientat."""
    import yaml

    if "reviewkd" in yaml.safe_load(yaml_path.read_text(encoding="utf-8")):
        if student_type(yaml_path) == "efficientat":
            from efficientat import EfficientATReviewKDMoE

            return EfficientATReviewKDMoE.from_yaml(yaml_path)
        from reviewkd import ReviewKDMoE

        return ReviewKDMoE.from_yaml(yaml_path)
    if student_type(yaml_path) == "efficientat":
        from efficientat import EfficientATKDMoE

        return EfficientATKDMoE.from_yaml(yaml_path)
    return KDMoEModel.from_yaml(yaml_path)


def weight_sets(run_dir: Path) -> dict[str, dict]:
    stage2 = run_dir / "stage2"
    sets = {"best_raw": torch.load(stage2 / "best.pt", map_location="cpu", weights_only=False)["model"]}
    resume = stage2 / "resume.pt"
    if resume.is_file():
        data = torch.load(resume, map_location="cpu", weights_only=False)
        state = dict(data["model"])
        state.update(data["ema"])
        sets["ema_last"] = state
    return sets


@torch.no_grad()
def predict(model: KDMoEModel, split: str, pipeline: str, separator, device) -> dict[str, np.ndarray]:
    cache = FRCRN_CACHE_DIR / f"frcrn_{split}.npy" if pipeline == "fp32_cache" else None
    edges = model.config.snr_band_edges if model.config.route_by == "snr" else None
    dataset = KDMoEMixDataset(split=split, load_teacher=False, frcrn_cache=cache, snr_band_edges=edges)
    probs, routers, labels, groups, snrs, ids, ytids = [], [], [], [], [], [], []
    for batch in build_dataloader(dataset, 64, False, 4):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            waveform = batch["waveform"].to(device)
            snr_est = batch["snr_est"].to(device) if "snr_est" in batch else None
            if pipeline == "bf16_online":
                mixture, waveform = waveform, separator(waveform)
                snr_est = frcrn_snr_db(mixture, waveform)
            output = model(waveform, None, snr_est)
        probs.append(output.logits.float().softmax(-1).cpu().numpy())
        routers.append(output.router_logits.argmax(-1).cpu().numpy())
        labels.append(batch["label"].numpy())
        groups.append(batch["group"].numpy())
        snrs.append(batch["snr_db"].numpy())
        ids += list(batch["sample_id"])
        ytids += list(batch["noise_ytid"])
    return {"prob": np.concatenate(probs), "router": np.concatenate(routers), "label": np.concatenate(labels),
            "group": np.concatenate(groups), "snr": np.concatenate(snrs), "id": np.array(ids),
            "leaked": np.array([y in LEAKED for y in ytids])}


def metrics(p: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    prob, label = p["prob"][mask], p["label"][mask]
    pred = prob.argmax(1)
    classes = np.arange(prob.shape[1])
    precision, recall, f1, _ = precision_recall_fscore_support(label, pred, labels=classes, zero_division=0)
    onehot = np.eye(prob.shape[1])[label]
    present = onehot.sum(0) > 0
    ap = [average_precision_score(onehot[:, c], prob[:, c]) for c in classes[present]]
    top3 = (np.argsort(-prob, 1)[:, :3] == label[:, None]).any(1).mean()
    return {"n": int(mask.sum()), "accuracy": float((pred == label).mean()), "top3": float(top3),
            "macro_f1": float(f1.mean()), "precision_macro": float(precision.mean()),
            "recall_macro": float(recall.mean()), "mAP": float(np.mean(ap)),
            "router_acc": float((p["router"][mask] == p["group"][mask]).mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate finished KDMoE_SNRRoute runs on validation and test")
    parser.add_argument("--dataset", required=True, help="Mixture-36l root (manifest.csv, labels.txt, wav)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", default=None, help="chỉ các run có tên chứa chuỗi này")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    os.environ["MLNC_MIXTURE_36L"] = arguments.dataset
    out = arguments.out if arguments.out.is_absolute() else HERE / arguments.out
    out.mkdir(parents=True, exist_ok=True)
    runs = {run: spec for run, spec in RUNS.items() if not arguments.only or arguments.only in run}

    device = torch.device("cuda")
    separator = load_frcrn(CONFIG / "frcrn.yaml", HERE / "model" / "frcrn_best.pt", device)
    build_frcrn_cache(separator, device, logging.getLogger("eval"), splits=SPLITS)
    build_frcrn_snr(logging.getLogger("eval"), splits=SPLITS)

    rows = []
    for run, (run_dir, pipeline, yaml_path) in runs.items():
        if not (run_dir / "stage2" / "best.pt").is_file():
            print(f"skip {run}: no stage2/best.pt", flush=True)
            continue
        finished = (run_dir / "stage2" / "result.json").is_file()
        for weights, state in weight_sets(run_dir).items():
            model = build_model(yaml_path).to(device)
            model.load_state_dict(state)
            model.eval()
            for split in SPLITS:
                p = predict(model, split, pipeline, separator, device)
                tag = f"{run} | {weights} | {split}"
                subsets = {"all": np.ones(len(p["label"]), bool), "clean": ~p["leaked"], "leaked": p["leaked"]}
                for subset, base in subsets.items():
                    for snr in ("all",) + SNRS:
                        mask = base if snr == "all" else base & (p["snr"] == snr)
                        rows.append({"run": run, "weights": weights, "pipeline": pipeline, "finished": finished,
                                     "router_groups": f"{model.config.route_by}{model.config.num_groups}",
                                     "split": split, "subset": subset, "snr_db": snr, **metrics(p, mask)})
                for row in (rows[-21], rows[-14]):  # (all, toàn bộ SNR) và (clean, toàn bộ SNR)
                    print(tag, row["subset"], {k: round(v, 4) for k, v in row.items() if isinstance(v, float)},
                          flush=True)
                safe = run.split(" (")[0].replace(" ", "_")
                with open(out / f"predictions_{safe}_{weights}_{split}.csv", "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["sample_id", "snr_db", "label", "pred", "pred_prob", "group", "router_pred",
                                "audioset_train_leaked"])
                    for i in range(len(p["label"])):
                        w.writerow([p["id"][i], p["snr"][i], p["label"][i], int(p["prob"][i].argmax()),
                                    round(float(p["prob"][i].max()), 5), p["group"][i], p["router"][i],
                                    int(p["leaked"][i])])

    with open(out / "metrics_by_snr.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out / "metrics_by_snr.csv", len(rows), "rows")


if __name__ == "__main__":
    main()
