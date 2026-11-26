"""Baseline cổ điển: đặc trưng STFT / MFCC / LFCC + LR, KNN, SVM, RF, ET, LightGBM.

    python -u classic_baselines.py --dataset /marimo/dataset/mix-dataset --input nhat --out output/classic/nhat
    python -u classic_baselines.py --dataset ... --input mixture --out output/classic/mixture

Tín hiệu vào: mixture thô (không FRCRN, không chuẩn hoá biên độ, không distillation); `--input nhat` (n_hat của FRCRN, cache fp32) chỉ để
đối chiếu khi cần. Đặc trưng ở 16 kHz, n_fft 512, hop 160 (front-end của CED): STFT = log10 công suất (257 bin);
MFCC 40 hệ số trên 64 mel 0-8 kHz; LFCC 40 hệ số trên 64 bộ lọc tuyến tính. Mỗi clip lấy mean + std theo thời gian. Chuẩn hoá z-score theo train.
Siêu tham số chọn trên val clean (accuracy); chỉ train/validation, không đụng test. Ghi metrics_by_snr.csv (cùng cột với evaluate.py) và dự đoán
validation từng clip (McNemar).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio

HERE = Path(__file__).resolve().parent
SAMPLE_RATE, N_FFT, HOP, N_MELS, N_COEF = 16_000, 512, 160, 64, 40
FEATURES = ("stft", "mfcc", "lfcc")
CLASSIFIERS = ("lr", "knn", "svm", "rf", "et", "lgbm")
SNRS = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)


class FeatureExtractor:
    """Waveform (B, N) -> {stft: (B, 514), mfcc: (B, 80), lfcc: (B, 80)} (mean + std theo thời gian)."""

    def __init__(self, preemph: float = 0.0) -> None:
        self.preemph = preemph  # y[n] = x[n] - preemph * x[n-1] trước STFT / MFCC / LFCC (0 = tắt)
        self.window = torch.hann_window(N_FFT)
        self.mfcc = torchaudio.transforms.MFCC(SAMPLE_RATE, n_mfcc=N_COEF, melkwargs={
            "n_fft": N_FFT, "hop_length": HOP, "n_mels": N_MELS, "f_min": 0.0, "f_max": 8000.0})
        self.lfcc = torchaudio.transforms.LFCC(SAMPLE_RATE, n_filter=N_MELS, n_lfcc=N_COEF, f_min=0.0, f_max=8000.0,
                                               speckwargs={"n_fft": N_FFT, "hop_length": HOP})

    @staticmethod
    def pool(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(dim=-1), x.std(dim=-1)], dim=-1)

    @torch.no_grad()
    def __call__(self, wave: torch.Tensor) -> dict[str, np.ndarray]:
        wave = wave.float()
        if self.preemph:
            wave = torch.cat([wave[:, :1], wave[:, 1:] - self.preemph * wave[:, :-1]], dim=1)
        spec = torch.stft(wave, N_FFT, HOP, N_FFT, window=self.window, center=True, return_complex=True)
        stft = 10.0 * torch.log10(spec.abs().square().clamp_min(1e-10))                     # (B, 257, T)
        return {"stft": self.pool(stft).numpy(), "mfcc": self.pool(self.mfcc(wave)).numpy(),
                "lfcc": self.pool(self.lfcc(wave)).numpy()}


def load_split(dataset_root: str, split: str, source: str, leaked: set[str], workers: int, preemph: float = 0.0) -> dict[str, np.ndarray]:
    os.environ["MLNC_MIXTURE_36L"] = dataset_root
    from scripts.frcrn_cache import FRCRN_CACHE_DIR
    from scripts.load_dataset import KDMoEMixDataset, build_dataloader

    cache = FRCRN_CACHE_DIR / f"frcrn_{split}.npy" if source == "nhat" else None
    dataset = KDMoEMixDataset(split=split, load_teacher=False, frcrn_cache=cache)
    extractor, store = FeatureExtractor(preemph), {name: [] for name in FEATURES}
    labels, snrs, ids, ytids = [], [], [], []
    for batch in build_dataloader(dataset, 128, False, workers):
        for name, value in extractor(batch["waveform"]).items():
            store[name].append(value)
        labels.append(batch["label"].numpy())
        snrs.append(batch["snr_db"].numpy())
        ids += list(batch["sample_id"])
        ytids += list(batch["noise_ytid"])
    out = {name: np.concatenate(parts).astype(np.float32) for name, parts in store.items()}
    out.update(label=np.concatenate(labels), snr=np.concatenate(snrs), id=np.array(ids),
               leaked=np.array([y in leaked for y in ytids]))
    return out


def build_classifier(name: str, params: dict, jobs: int):
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

    if name == "lr":
        return LogisticRegression(C=params["C"], max_iter=500, n_jobs=jobs)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=params["k"], weights="distance", n_jobs=jobs)
    if name == "svm":
        return SVC(C=params["C"], kernel="rbf", gamma="scale", decision_function_shape="ovr")
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, n_jobs=jobs, random_state=2026)
    if name == "et":
        return ExtraTreesClassifier(n_estimators=300, n_jobs=jobs, random_state=2026)
    raise ValueError(name)


GRIDS = {"lr": [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}], "knn": [{"k": 5}, {"k": 15}, {"k": 31}], "svm": [{"C": 1.0}, {"C": 10.0}],
         "rf": [{}], "et": [{}], "lgbm": [{}]}


def scores_of(model, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x) if hasattr(model, "predict_proba") else model.decision_function(x)


def fit_one(name: str, params: dict, train, val_clean, jobs: int):
    """Trả (model, scores_fn). LightGBM dừng sớm trên val clean."""
    x, y = train
    if name == "lgbm":
        import lightgbm as lgb

        model = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.5, n_jobs=jobs, random_state=2026, verbose=-1)
        model.fit(x, y, eval_set=[val_clean], callbacks=[lgb.early_stopping(30, verbose=False)])
        return model
    model = build_classifier(name, params, jobs)
    model.fit(x, y)
    return model


def metrics(scores: np.ndarray, label: np.ndarray, mask: np.ndarray, classes: int = 36) -> dict:
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support

    s, y = scores[mask], label[mask]
    pred = s.argmax(1)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, labels=np.arange(classes), zero_division=0)
    onehot = np.eye(classes)[y]
    present = onehot.sum(0) > 0
    ap = [average_precision_score(onehot[:, c], s[:, c]) for c in np.arange(classes)[present]]
    top3 = (np.argsort(-s, 1)[:, :3] == y[:, None]).any(1).mean()
    return {"n": int(mask.sum()), "accuracy": float((pred == y).mean()), "top3": float(top3), "macro_f1": float(f1.mean()),
            "precision_macro": float(precision.mean()), "recall_macro": float(recall.mean()), "mAP": float(np.mean(ap)),
            "router_acc": float("nan")}


def main() -> None:
    parser = argparse.ArgumentParser(description="STFT/MFCC/LFCC + LR/KNN/SVM/RF/ET/LGBM")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input", choices=("nhat", "mixture"), default="mixture")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--feats", nargs="+", default=list(FEATURES))
    parser.add_argument("--clfs", nargs="+", default=list(CLASSIFIERS))
    parser.add_argument("--preemph", type=float, default=0.0, help="hệ số pre-emphasis (0.97 chuẩn), 0 = tắt")
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else HERE / args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.jobs)
    leaked = {l.strip() for l in (HERE / "config" / "audioset_train_noise_ytids.txt").read_text().splitlines()
              if l.strip() and not l.startswith("#")}

    data = {}
    for split in ("train", "validation"):
        path = out / f"features_{split}.npz"
        if path.is_file():
            data[split] = dict(np.load(path, allow_pickle=False))
        else:
            start = time.time()
            data[split] = load_split(args.dataset, split, args.input, leaked, args.workers, args.preemph)
            np.savez(path, **data[split])
            print(f"features {split}: {len(data[split]['label'])} clips in {time.time() - start:.0f}s", flush=True)
    train, val = data["train"], data["validation"]
    clean = ~val["leaked"]
    rows, summary = [], []
    for feat in args.feats:
        mean, std = train[feat].mean(0), train[feat].std(0) + 1e-6
        norm = lambda x: (x - mean) / std
        x_train, x_val = norm(train[feat]), norm(val[feat])
        for clf in args.clfs:
            run = f"Classic {args.input}{'+pre' if args.preemph else ''} {feat} {clf}"
            done = out / f"done_{feat}_{clf}.json"
            if done.is_file():
                continue
            best = None
            for params in GRIDS[clf]:
                start = time.time()
                model = fit_one(clf, params, (x_train, train["label"]), (x_val[clean], val["label"][clean]), args.jobs)
                scores = scores_of(model, x_val)
                acc = float((scores.argmax(1) == val["label"])[clean].mean())
                print(f"{run} {params}: val clean acc {acc:.4f} ({time.time() - start:.0f}s)", flush=True)
                if best is None or acc > best[0]:
                    best = (acc, params, scores)
            acc, params, scores = best
            subsets = {"all": np.ones(len(val["label"]), bool), "clean": clean, "leaked": ~clean}
            for subset, base in subsets.items():
                for snr in ("all",) + SNRS:
                    mask = base if snr == "all" else base & (val["snr"] == snr)
                    rows.append({"run": run, "weights": "best_raw", "pipeline": args.input, "finished": True, "router_groups": "none",
                                 "split": "validation", "subset": subset, "snr_db": snr, **metrics(scores, val["label"], mask)})
            with open(out / f"predictions_{run.replace(' ', '_')}_best_raw_validation.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["sample_id", "snr_db", "label", "pred", "pred_prob", "group", "router_pred", "audioset_train_leaked"])
                for i in range(len(val["label"])):
                    w.writerow([val["id"][i], val["snr"][i], val["label"][i], int(scores[i].argmax()), round(float(scores[i].max()), 5),
                                -1, -1, int(val["leaked"][i])])
            m_clean = metrics(scores, val["label"], clean)
            summary.append({"input": args.input, "feature": feat, "classifier": clf, "params": json.dumps(params),
                            "val_clean_acc": m_clean["accuracy"], "val_clean_macro_f1": m_clean["macro_f1"], "val_clean_mAP": m_clean["mAP"],
                            "val_all_acc": metrics(scores, val["label"], subsets["all"])["accuracy"]})
            done.write_text(json.dumps({"run": run, "params": params, "summary": summary[-1]}), encoding="utf-8")
    if rows:  # nối vào CSV của các lần chạy trước
        existing = out / "metrics_by_snr.csv"
        old = list(csv.DictReader(open(existing, encoding="utf-8"))) if existing.is_file() else []
        with open(existing, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(old + rows)
    with open(out / "classic_summary.csv", "w", newline="", encoding="utf-8") as f:
        items = [json.loads(p.read_text(encoding="utf-8"))["summary"] for p in sorted(out.glob("done_*.json"))]
        if items:
            w = csv.DictWriter(f, fieldnames=list(items[0]))
            w.writeheader()
            w.writerows(items)
    print("done", flush=True)


if __name__ == "__main__":
    main()
