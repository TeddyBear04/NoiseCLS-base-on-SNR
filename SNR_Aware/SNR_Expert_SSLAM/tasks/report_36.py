"""Turn every finished run into CSV tables.

    python -u main.py report36 --config config/train_config_high.json

Reads each `artifacts/student_<run>_predictions.npz` and writes three CSVs under
`artifacts/results/`:

    metrics_by_run.csv    one row per run x slice, all eight repo metrics
    metrics_per_class.csv one row per run x slice x label
    comparison.csv        run-vs-run deltas plus the paired McNemar test
    thong_so.csv          one row per run: every effective parameter it ran with
    theo_seed.csv         every seed of every run, then mean/std and the paired
                          per-seed delta against run1_baseline (only if >1 seed)

The eight metrics match `pipeline_config_4_branches.json` so these tables sit
beside the other branches' numbers without translation: accuracy,
macro_precision, macro_recall, macro_f1, micro_f1, macro_map, balanced_accuracy,
macro_auc_ovr.

mAP and macro-AUC need scores rather than argmax, so a run only reports them if
its `.npz` carries `logits`. Runs saved before that was added report the
argmax-based metrics and leave the other two empty rather than guessing.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

# Not imported from tasks.run_36: that pulls in torch/torchaudio, and the report
# has to run anywhere the .npz predictions are - including a laptop.
HERE = Path(__file__).resolve().parents[1]

METRIC_COLUMNS = [
    "accuracy", "macro_precision", "macro_recall", "macro_f1", "micro_f1",
    "macro_map", "balanced_accuracy", "macro_auc_ovr", "top3_accuracy",
]

# The results spreadsheet's header, matched exactly so these rows paste in as-is.
SHEET_COLUMNS = [
    "Phuong phap", "Phien ban", "SNR (dB)", "So mau (support)",
    "Top-1 Accuracy", "Top-3 Accuracy", "Balanced Acc",
    "Precision Macro", "Recall Macro", "Macro-F1", "Micro-F1",
    "mAP", "Macro-AUC", "SI-SDR (dB)", "SI-SDR improvement (dB)",
]

# SI-SDR measures waveform separation quality. This branch deliberately never
# separates a waveform - CRD pulls the mixture embedding toward the clean-noise
# embedding instead, which is what keeps it clear of the artifacts that sank the
# DPCRN branch to 0.144. So those two columns stay empty here rather than being
# filled with a number that would not mean what the column says.
METHOD_NAMES = {
    "published_baseline": "BEATs-Mixture (Result.xlsx, published)",
    "beats_baseline": "BEATs baseline (on-disk checkpoint metrics file)",
    "run1_baseline": "SSLAM expert - CE only, no FiLM (rebuilt baseline)",
    "run2_ce_only": "SSLAM expert - CE only + FiLM (control)",
    "run3_kd_crd": "SSLAM expert - KD + CRD",
    "run3b_crd_only": "SSLAM expert - CRD only",
    "run3c_kd_only": "SSLAM expert - KD only",
    "run5_remix": "SSLAM expert - KD + CRD + remix augmentation",
    "run6_grl": "SSLAM expert - KD + CRD + GRL speech-adversarial",
    "run7_attn": "expert - soft attention mask (CE only, no FiLM)",
    "run8_attn_kd_crd": "expert - KD + CRD + soft attention mask",
}
# ("full", "band" = the SNR band this project is judged on) then one slice per
# SNR level. The band's own label ("15-20" etc.) is filled in from config at
# report time - see `slice_names()`.
SLICE_NAMES_STATIC = {"full": "all"}

# The "Theo tung SNR" sheet in Result.xlsx, column for column, so these rows paste
# straight in beside the other branches. One row per method per SNR level only -
# that sheet carries no "all" or "band" aggregate rows.
SNR_SHEET_COLUMNS = [
    "Phuong phap", "Phien ban", "SNR (dB)", "So mau (support)",
    "Top-1 Accuracy", "Top-3 Accuracy", "Balanced Acc",
    "Precision Macro", "Recall Macro", "Macro-F1", "Micro-F1",
    "mAP", "Macro-AUC", "SI-SDR (dB)", "SI-SDR improvement",
    "Emb cos(mixture, noise sach)", "Emb cos(khac clip)", "Emb gap",
    "Emb retrieval@1", "Ghi chu",
]

PURITY_COLUMNS = {
    "Emb cos(mixture, noise sach)": "emb_cos_pos",
    "Emb cos(khac clip)": "emb_cos_neg",
    "Emb gap": "emb_gap",
    "Emb retrieval@1": "emb_retrieval_top1",
}

# Sheet method prefix ("BEATs-MidExpert", "SSLAM-HighExpert", ...) comes from
# config["report"]["sheet_method"] so the same code serves any band/backbone.
SHEET_VERSION = {
    "run1_baseline": "run1-baseline",
    "run2_ce_only": "run2-ce-only",
    "run3b_crd_only": "run3b-crd-only",
    "run3c_kd_only": "run3c-kd-only",
    "run3_kd_crd": "run3-kd-crd",
    "run5_remix": "run5-remix",
    "run6_grl": "run6-grl",
    "run7_attn": "run7-attn",
    "run8_attn_kd_crd": "run8-attn-kd-crd",
}
SHEET_NOTE = {
    "run1_baseline": "Baseline dung lai: CE thuan, KHONG FiLM, chon checkpoint theo toan bo validation. Khong tach waveform nen khong co SI-SDR.",
    "run2_ce_only": "CE thuan + FiLM theo SNR, chon checkpoint theo lat band. Control cho run3.",
    "run3b_crd_only": "CE + CRD (b=0.8), khong KD. Setting chinh cua paper CRD.",
    "run3c_kd_only": "CE + KD (a=1.0, rho=4), khong CRD.",
    "run3_kd_crd": "CE + KD + CRD. Phuong phap day du. Teacher nhin noise sach (privileged info), student chi nhin mixture.",
    "run5_remix": "run3_kd_crd + remix augmentation (clean + gain*noise moi tren band).",
    "run6_grl": "run3_kd_crd + head phu doan speech qua gradient reversal, ep embedding vut bo speech.",
    "run7_attn": "Huong 2: run1_baseline + soft attention mask A(t,f) thay mean-pool (gated attention, Ilse et al. ICML 2018). CE thuan, khong KD/CRD/FiLM.",
    "run8_attn_kd_crd": "run3_kd_crd + soft attention mask A(t,f). So cap voi run3_kd_crd de tach phan dong gop cua mask khi da co KD+CRD.",
}


def snr_sheet_row(row: dict, sheet_method: str, purity: dict | None = None) -> dict:
    version = SHEET_VERSION.get(row["run"], row["run"])
    method = sheet_method if row["run"] in SHEET_VERSION else row["run"]
    def value(key):
        v = row.get(key, "")
        return round(v, 4) if isinstance(v, float) else v
    extra = {name: (round(purity[key], 4) if purity and key in purity else "")
             for name, key in PURITY_COLUMNS.items()}
    return {
        "Phuong phap": method,
        "Phien ban": version,
        "SNR (dB)": int(row["slice"].replace("snr_", "")),
        "So mau (support)": row.get("samples", ""),
        "Top-1 Accuracy": value("accuracy"),
        "Top-3 Accuracy": value("top3_accuracy"),
        "Balanced Acc": value("balanced_accuracy"),
        "Precision Macro": value("macro_precision"),
        "Recall Macro": value("macro_recall"),
        "Macro-F1": value("macro_f1"),
        "Micro-F1": value("micro_f1"),
        "mAP": value("macro_map"),
        "Macro-AUC": value("macro_auc_ovr"),
        "SI-SDR (dB)": "",
        "SI-SDR improvement": "",
        **extra,
        "Ghi chu": SHEET_NOTE.get(row["run"], ""),
    }


def sheet_row(row: dict, band_label: str) -> dict:
    slice_name = row["slice"]
    slice_names = {**SLICE_NAMES_STATIC, "band": band_label}
    snr = slice_names.get(slice_name, slice_name.replace("snr_", ""))
    def value(key):
        v = row.get(key, "")
        return round(v, 6) if isinstance(v, float) else v
    return {
        "Phuong phap": METHOD_NAMES.get(row["run"], row["run"]),
        "Phien ban": row["run"],
        "SNR (dB)": snr,
        "So mau (support)": row.get("samples", ""),
        "Top-1 Accuracy": value("accuracy"),
        "Top-3 Accuracy": value("top3_accuracy"),
        "Balanced Acc": value("balanced_accuracy"),
        "Precision Macro": value("macro_precision"),
        "Recall Macro": value("macro_recall"),
        "Macro-F1": value("macro_f1"),
        "Micro-F1": value("micro_f1"),
        "mAP": value("macro_map"),
        "Macro-AUC": value("macro_auc_ovr"),
        "SI-SDR (dB)": "",
        "SI-SDR improvement (dB)": "",
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def top_k_accuracy(target: np.ndarray, probabilities: np.ndarray, k: int) -> float:
    """Share of clips whose true class is among the k highest-scoring."""
    top = np.argpartition(-probabilities, kth=k - 1, axis=1)[:, :k]
    return float((top == target[:, None]).any(axis=1).mean())


def compute_metrics(target: np.ndarray, predicted: np.ndarray,
                    probabilities: np.ndarray | None, class_count: int) -> dict:
    """The repo's eight metrics. Score-based ones stay empty without probabilities."""
    from sklearn.metrics import (
        average_precision_score, balanced_accuracy_score, f1_score,
        precision_recall_fscore_support, roc_auc_score,
    )

    labels = np.arange(class_count)
    precision, recall, f1, _ = precision_recall_fscore_support(
        target, predicted, labels=labels, zero_division=0
    )
    result = {
        "accuracy": float((predicted == target).mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "micro_f1": float(f1_score(target, predicted, average="micro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "macro_map": "",
        "macro_auc_ovr": "",
        "top3_accuracy": "",
    }
    if probabilities is not None:
        result["top3_accuracy"] = top_k_accuracy(target, probabilities, min(3, class_count))
        present = np.unique(target)
        one_hot = np.zeros((target.size, class_count), dtype=np.float32)
        one_hot[np.arange(target.size), target] = 1.0
        # Restrict to classes actually present: average_precision and roc_auc are
        # undefined for a class with no positive example, and including them would
        # silently drag the macro average toward zero.
        try:
            result["macro_map"] = float(average_precision_score(
                one_hot[:, present], probabilities[:, present], average="macro"))
        except ValueError:
            pass
        try:
            result["macro_auc_ovr"] = float(roc_auc_score(
                target, probabilities[:, present] /
                probabilities[:, present].sum(axis=1, keepdims=True),
                multi_class="ovr", average="macro", labels=present))
        except ValueError:
            pass
    return result


def per_class_rows(target: np.ndarray, predicted: np.ndarray,
                   probabilities: np.ndarray | None, labels: list[str]) -> list[dict]:
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support

    indices = np.arange(len(labels))
    precision, recall, f1, support = precision_recall_fscore_support(
        target, predicted, labels=indices, zero_division=0
    )
    rows = []
    for index, name in enumerate(labels):
        row = {"label": name, "precision": round(float(precision[index]), 6),
               "recall": round(float(recall[index]), 6),
               "f1": round(float(f1[index]), 6), "support": int(support[index]),
               "average_precision": ""}
        if probabilities is not None and support[index] > 0:
            row["average_precision"] = round(float(average_precision_score(
                (target == index).astype(int), probabilities[:, index])), 6)
        rows.append(row)
    return rows


def slices(snr: np.ndarray, band: list[float],
           band_only: bool = False) -> list[tuple[str, np.ndarray]]:
    """full, the SNR band we are judged on, then one slice per SNR level.

    With `band_only` the predictions hold nothing but the band, so "full" would be
    a duplicate of "band" and levels outside the band cannot occur: keep the band
    and its own levels only.
    """
    in_band = (snr >= band[0]) & (snr <= band[1])
    out = [] if band_only else [("full", np.ones(snr.shape, dtype=bool))]
    out.append(("band", in_band))
    for level in sorted(set(snr.tolist())):
        if band_only and not band[0] <= level <= band[1]:
            continue
        out.append((f"snr_{int(level)}", snr == level))
    return out


def load_run(path: Path) -> dict:
    payload = np.load(path, allow_pickle=True)
    data = {"target": payload["target"].astype(int),
            "predicted": payload["predicted"].astype(int),
            "snr": payload["snr"].astype(int),
            "probabilities": None,
            "labels": None}
    if "logits" in payload.files:
        data["probabilities"] = softmax(payload["logits"].astype(np.float32))
    if "labels" in payload.files:
        data["labels"] = [str(x) for x in payload["labels"]]
    data["sample_id"] = ([str(x) for x in payload["sample_id"]]
                         if "sample_id" in payload.files else None)
    return data


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Paired test. The band slice is 2,160 clips, where the standard error on
    accuracy is about 1.0 point - comparing two independent proportions cannot
    resolve the effect sizes at stake here, but this can."""
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    row = {"both_correct": int((correct_a & correct_b).sum()),
           "both_wrong": int((~correct_a & ~correct_b).sum()),
           "only_a_correct": b, "only_b_correct": c,
           "chi2": "", "p_value": "", "significant_at_0.05": ""}
    if b + c > 0:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p = math.erfc(math.sqrt(chi2 / 2))
        row["chi2"] = round(chi2, 6)
        row["p_value"] = round(p, 6)
        row["significant_at_0.05"] = "yes" if p < 0.05 else "no"
    return row


def published_rows(config: dict, band: list[float]) -> list[dict]:
    """The baseline the branch is judged against, as published in Result.xlsx.

    Only the band's own SNR levels plus their support-weighted band aggregate.
    Result.xlsx carries accuracy and F1 only, so every other column stays empty.
    """
    published = config.get("gates", {}).get("published_baseline")
    if not published:
        return []
    levels = {int(float(k)): v for k, v in published["per_snr"].items()
              if band[0] <= float(k) <= band[1]}
    if not levels:
        return []
    def row(slice_name, samples, accuracy, macro_f1, micro_f1):
        return {"run": "published_baseline", "slice": slice_name, "samples": samples,
                "accuracy": accuracy, "macro_precision": "", "macro_recall": "",
                "macro_f1": macro_f1, "micro_f1": micro_f1, "macro_map": "",
                "balanced_accuracy": "", "macro_auc_ovr": "", "top3_accuracy": ""}
    total = sum(v["samples"] for v in levels.values())
    rows = [row("band", total,
                *(sum(v[key] * v["samples"] for v in levels.values()) / total
                  for key in ("accuracy", "macro_f1", "micro_f1")))]
    for level, v in sorted(levels.items()):
        rows.append(row(f"snr_{level}", v["samples"], v["accuracy"], v["macro_f1"],
                        v["micro_f1"]))
    return rows


def baseline_rows(labels: list[str], band: list[float]) -> list[dict]:
    """The BEATs baseline from its own metrics file. It has no per-clip predictions
    here, so its slices carry only what that file recorded.

    This is the on-disk checkpoint's baseline, not the published-in-Result.xlsx
    one - same gap the mid-expert project hit: the checkpoint behind the
    published numbers is gone (*.pt is gitignored). Keep both straight in any
    report: gate against the published numbers in `config["gates"]`, use this
    row only as an unpaired sanity check."""
    path = HERE.parent / "BEATs_Experts/checkpoint/test_metrics_36.json"
    if not path.exists():
        return []
    saved = json.loads(path.read_text(encoding="utf-8"))
    rows = [{"run": "beats_baseline", "slice": "full", "samples": saved.get("samples", ""),
             "accuracy": saved.get("accuracy", ""),
             "macro_precision": saved.get("precision", ""),
             "macro_recall": saved.get("recall", ""),
             "macro_f1": saved.get("macro_f1", ""),
             "micro_f1": saved.get("micro_f1", ""),
             "macro_map": saved.get("test_map", ""),
             "balanced_accuracy": saved.get("test_balanced_accuracy", ""),
             "macro_auc_ovr": saved.get("test_macro_auc", "")}]
    per_snr = saved.get("per_snr", {})
    band_levels = [v for k, v in per_snr.items() if band[0] <= float(k) <= band[1]]
    if band_levels:
        total = sum(v["samples"] for v in band_levels)
        rows.append({"run": "beats_baseline", "slice": "band", "samples": total,
                     "accuracy": sum(v["accuracy"] * v["samples"] for v in band_levels) / total,
                     "macro_precision": "", "macro_recall": "",
                     "macro_f1": sum(v["macro_f1"] * v["samples"] for v in band_levels) / total,
                     "micro_f1": sum(v["micro_f1"] * v["samples"] for v in band_levels) / total,
                     "macro_map": "", "balanced_accuracy": "", "macro_auc_ovr": ""})
    for level, value in per_snr.items():
        rows.append({"run": "beats_baseline", "slice": f"snr_{int(float(level))}",
                     "samples": value["samples"], "accuracy": value["accuracy"],
                     "macro_precision": "", "macro_recall": "",
                     "macro_f1": value["macro_f1"], "micro_f1": value["micro_f1"],
                     "macro_map": "", "balanced_accuracy": "", "macro_auc_ovr": ""})
    return rows


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
    print(f"wrote {path}  ({len(rows)} rows)", flush=True)


def flatten_params(tree: dict, prefix: str = "", skip: tuple = ()) -> dict:
    """Lam phang config long nhau thanh {"a.b.c": gia_tri} cho mot hang CSV.

    Bo khoa ghi chu (bat dau bang "_") va moi nhanh nam trong `skip`. List ghi
    thanh chuoi JSON de mot o CSV van doc lai duoc.
    """
    flat = {}
    for key, value in tree.items():
        if key.startswith("_"):
            continue
        path = f"{prefix}{key}"
        if path in skip:
            continue
        if isinstance(value, dict):
            flat.update(flatten_params(value, f"{path}.", skip))
        elif isinstance(value, list):
            flat[path] = json.dumps(value)
        else:
            flat[path] = value
    return flat


PARAM_LEAD = ["run", "seed", "source", "commit", "dirty", "epochs_run", "best_epoch",
              "started", "torch", "device"]
# Branches that are results or bookkeeping, not conditions a run was trained under.
PARAM_SKIP = ("gates.published_baseline", "report", "run")


def run_params(folder: Path, name: str, config: dict) -> dict:
    """One `thong_so.csv` row: the run's effective parameters plus how long it trained.

    Prefers `student_<run>_params.json`, written when training started. Runs from
    before that file existed are rebuilt from their history (which recorded the
    effective student and loss sections) and the config, and say so in `source`.
    """
    history_path = folder / f"student_{name}_history.json"
    history = (json.loads(history_path.read_text(encoding="utf-8"))
               if history_path.exists() else {})
    params_path = folder / f"student_{name}_params.json"
    if params_path.exists():
        saved = json.loads(params_path.read_text(encoding="utf-8"))
        effective = saved["config"]
        row = {"source": "params.json", **{k: saved.get(k, "") for k in
                                           ("commit", "dirty", "started", "torch", "device")}}
    else:
        effective = json.loads(json.dumps(config))
        if "student_config" in history:
            effective["student"] = history["student_config"]
        if "loss_config" in history:
            effective["loss"] = history["loss_config"]
        row = {"source": "reconstructed (history + config)", "commit": "unknown"}
    epochs = history.get("history", [])
    best = history.get("best_validation", {})
    best_epoch = best.get("epoch", "")
    if best_epoch == "" and epochs:
        # Older histories did not record it: the kept checkpoint is the epoch whose
        # validation score equals the best one, or epoch 0 if none improved on it.
        target = best.get("band", {}).get("macro_f1")
        best_epoch = next((e["epoch"] for e in epochs
                           if target is not None
                           and abs(e.get("val_band_macro_f1", -1) - target) < 1e-9), 0)
    row.update({"run": name, "seed": effective["experiment"]["seed"],
                "epochs_run": len(epochs), "best_epoch": best_epoch})
    row.update(flatten_params(effective, skip=PARAM_SKIP))
    return row


def seed_folders(config: dict) -> list[tuple[int, Path]]:
    """The base outputs folder and every `<base>_s<seed>` beside it."""
    outputs = config["outputs"]
    base = outputs.get("seed_base_dir", outputs["dir"])
    base_folder = HERE / base
    found = [(outputs.get("seed_base_seed", config["experiment"]["seed"]), base_folder)]
    for folder in sorted(base_folder.parent.glob(f"{base_folder.name}_s*")):
        suffix = folder.name[len(base_folder.name) + 2:]
        if suffix.isdigit():
            found.append((int(suffix), folder))
    return [(seed, folder) for seed, folder in found if folder.is_dir()]


def seed_rows(config: dict, band: list[float]) -> list[dict]:
    """Every seed of every run, then per-run mean/std, then the per-seed paired
    delta against run1_baseline. One seed says nothing about run-to-run spread;
    this table is what a claim of "better than the baseline" has to survive."""
    per_run: dict[str, dict[int, dict]] = {}
    for seed, folder in seed_folders(config):
        for path in sorted(folder.glob("student_*_predictions.npz")):
            name = path.name[len("student_"):-len("_predictions.npz")]
            run = load_run(path)
            ok = run["predicted"] == run["target"]
            entry = {}
            for slice_name, mask in slices(run["snr"], band, band_only=True):
                if mask.any():
                    entry[slice_name] = float(ok[mask].mean())
            per_run.setdefault(name, {})[seed] = entry
    if not any(len(seeds) > 1 for seeds in per_run.values()):
        return []
    rows = []
    keys = sorted({k for seeds in per_run.values() for e in seeds.values() for k in e},
                  key=lambda k: (k != "band", k))
    for name, seeds in sorted(per_run.items()):
        for seed, entry in sorted(seeds.items()):
            rows.append({"run": name, "seed": seed, "kind": "seed",
                         **{f"acc_{k}": round(v, 6) for k, v in entry.items()}})
        if len(seeds) > 1:
            stats = {"run": name, "seed": f"n={len(seeds)}", "kind": "mean+-std"}
            for k in keys:
                values = np.array([e[k] for e in seeds.values() if k in e])
                stats[f"acc_{k}"] = (f"{values.mean():.4f} +- {values.std(ddof=1):.4f}"
                                     if len(values) > 1 else "")
            rows.append(stats)
        base = per_run.get("run1_baseline", {})
        common = sorted(set(seeds) & set(base))
        if name != "run1_baseline" and len(common) > 1:
            delta = {"run": name, "seed": f"paired n={len(common)}",
                     "kind": "delta vs run1_baseline (mean+-std, wins)"}
            for k in keys:
                d = np.array([seeds[s][k] - base[s][k] for s in common
                              if k in seeds[s] and k in base[s]])
                if len(d) > 1:
                    delta[f"acc_{k}"] = (f"{100 * d.mean():+.2f}pt +- {100 * d.std(ddof=1):.2f}"
                                         f" ({int((d > 0).sum())}/{len(d)} wins)")
            rows.append(delta)
    return rows


def command_report36(config: dict) -> None:
    out_dir = HERE / config["outputs"]["dir"]
    results_dir = out_dir / "results"
    band = config["band_db"]
    band_only = config.get("evaluation", {}).get("band_only", False)
    report_cfg = config.get("report", {})
    sheet_method = report_cfg.get("sheet_method", "SSLAM-Expert")
    band_label = report_cfg.get("band_label", f"{int(band[0])}-{int(band[1])}")

    found = sorted(out_dir.glob("student_*_predictions.npz"))
    if not found:
        raise FileNotFoundError(
            f"No student_*_predictions.npz under {out_dir}. Run test36 first."
        )
    runs = {path.name[len("student_"):-len("_predictions.npz")]: load_run(path)
            for path in found}
    print("runs:", ", ".join(runs), flush=True)

    labels = next((r["labels"] for r in runs.values() if r["labels"]), None)
    if labels is None:
        labels_file = Path(config["dataset"]["path"]) / config["dataset"]["labels_file"]
        labels = labels_file.read_text(encoding="utf-8").splitlines()

    summary, per_class = [], []
    for name, run in runs.items():
        for slice_name, mask in slices(run["snr"], band, band_only):
            if not mask.any():
                continue
            probabilities = (run["probabilities"][mask]
                             if run["probabilities"] is not None else None)
            row = {"run": name, "slice": slice_name, "samples": int(mask.sum())}
            row.update(compute_metrics(run["target"][mask], run["predicted"][mask],
                                       probabilities, len(labels)))
            summary.append(row)
            if slice_name in ("full", "band"):
                for entry in per_class_rows(run["target"][mask], run["predicted"][mask],
                                            probabilities, labels):
                    per_class.append({"run": name, "slice": slice_name, **entry})

    # The published Result.xlsx numbers when the config has them; the on-disk
    # checkpoint's metrics file is a different training run and only a fallback.
    summary.extend(published_rows(config, band) or baseline_rows(labels, band))
    summary.sort(key=lambda r: (r["slice"] != "band", r["slice"], r["run"]))
    write_csv(results_dir / "metrics_by_run.csv", summary,
              ["run", "slice", "samples", *METRIC_COLUMNS])
    write_csv(results_dir / "ket_qua_tong_hop.csv",
              [sheet_row(r, band_label) for r in summary], SHEET_COLUMNS)
    # Only the SNR levels inside `band_db`: that is the band this branch was
    # assigned, and the other levels are not what the report claims anything about.
    wanted = {f"snr_{int(level)}" for level in band}
    snr_rows = [r for r in summary if r["slice"] in wanted and r["run"] in SHEET_VERSION]
    snr_rows.sort(key=lambda r: (list(SHEET_VERSION).index(r["run"]),
                                 int(r["slice"].replace("snr_", ""))))
    purity_by_run = {}
    for path in out_dir.glob("student_*_test.json"):
        name = path.name[len("student_"):-len("_test.json")]
        purity_by_run[name] = json.loads(path.read_text(encoding="utf-8")).get("purity", {})
    write_csv(results_dir / "theo_tung_snr.csv",
              [snr_sheet_row(r, sheet_method, purity_by_run.get(r["run"], {}).get(r["slice"]))
               for r in snr_rows], SNR_SHEET_COLUMNS)
    write_csv(results_dir / "metrics_per_class.csv", per_class,
              ["run", "slice", "label", "precision", "recall", "f1", "support",
               "average_precision"])

    comparisons = []
    names = list(runs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if not np.array_equal(runs[a]["target"], runs[b]["target"]):
                print(f"skip {a} vs {b}: different test order", flush=True)
                continue
            for slice_name, mask in slices(runs[a]["snr"], band, band_only):
                if not mask.any():
                    continue
                ok_a = (runs[a]["predicted"] == runs[a]["target"])[mask]
                ok_b = (runs[b]["predicted"] == runs[b]["target"])[mask]
                row = {"run_a": a, "run_b": b, "slice": slice_name,
                       "samples": int(mask.sum()),
                       "accuracy_a": round(float(ok_a.mean()), 6),
                       "accuracy_b": round(float(ok_b.mean()), 6),
                       "delta_accuracy": round(float(ok_b.mean() - ok_a.mean()), 6)}
                row.update(mcnemar(ok_a, ok_b))
                comparisons.append(row)
    comparisons.sort(key=lambda r: (r["slice"] != "band", r["slice"], r["run_a"], r["run_b"]))
    write_csv(results_dir / "comparison.csv", comparisons,
              ["run_a", "run_b", "slice", "samples", "accuracy_a", "accuracy_b",
               "delta_accuracy", "both_correct", "both_wrong", "only_a_correct",
               "only_b_correct", "chi2", "p_value", "significant_at_0.05"])

    # Every seed folder's runs too: their rows differ only in `seed` and the
    # epochs they trained, which is exactly what a fair comparison has to see.
    params = [run_params(folder, path.name[len("student_"):-len("_predictions.npz")], config)
              for _, folder in seed_folders(config)
              for path in sorted(folder.glob("student_*_predictions.npz"))]
    teacher_folder = HERE / config["outputs"].get("teacher_dir", config["outputs"]["dir"])
    teacher_params = teacher_folder / "teacher_params.json"
    if teacher_params.exists():
        saved = json.loads(teacher_params.read_text(encoding="utf-8"))
        params.append({"run": "teacher", "source": "params.json",
                       "seed": saved["config"]["experiment"]["seed"],
                       **{k: saved.get(k, "") for k in
                          ("commit", "dirty", "started", "torch", "device")},
                       **flatten_params(saved["config"], skip=PARAM_SKIP)})
    extra = sorted({k for row in params for k in row} - set(PARAM_LEAD))
    write_csv(results_dir / "thong_so.csv", params, PARAM_LEAD + extra)

    by_seed = seed_rows(config, band)
    if by_seed:
        columns = ["run", "seed", "kind"] + sorted(
            {k for row in by_seed for k in row if k.startswith("acc_")},
            key=lambda k: (k != "acc_band", k))
        write_csv(results_dir / "theo_seed.csv", by_seed, columns)

    print(f"\n=== band slice ({band_label} dB) ===", flush=True)
    header = f"{'run':<18}{'acc':>9}{'macro_f1':>10}{'mAP':>9}{'AUC':>9}"
    print(header, flush=True)
    for row in summary:
        if row["slice"] != "band":
            continue
        def fmt(key):
            value = row.get(key, "")
            return f"{value:>9.4f}" if isinstance(value, float) else f"{'-':>9}"
        print(f"{row['run']:<18}{fmt('accuracy')}{fmt('macro_f1')}"
              f"{fmt('macro_map')}{fmt('macro_auc_ovr')}", flush=True)


# ------------------------------------------------------------ backbone comparison

COMPARE_COLUMNS = ["Backbone", "Phien ban", "SNR (dB)", "So mau (support)",
                   "Top-1 Accuracy", "Top-3 Accuracy", "Macro-F1", "Micro-F1",
                   "mAP", "Macro-AUC", "Delta Top-1 vs Result.xlsx"]
MCNEMAR_COLUMNS = ["Phien ban", "SNR (dB)", "So mau (support)", "Backbone A",
                   "Backbone B", "Top-1 A", "Top-1 B", "Delta (B - A)",
                   "Chi A dung", "Chi B dung", "p_value", "Co y nghia (p<0.05)"]


def command_compare36(config_paths: list[Path], out_dir: Path) -> None:
    """Two or more backbones side by side, same runs, same test clips.

    Each config's runs are read from its own `outputs.dir`. Rows are matched by
    `sample_id`, never by file order, before the paired McNemar test between the
    same run on different backbones. Writes two CSVs and nothing else.
    """
    configs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in config_paths]
    band = configs[0]["band_db"]
    band_label = configs[0].get("report", {}).get("band_label",
                                                  f"{int(band[0])}-{int(band[1])}")
    published = {r["slice"]: r["accuracy"] for r in published_rows(configs[0], band)}

    def slice_label(name):
        return band_label if name == "band" else name.replace("snr_", "")

    backbones = {}
    for config in configs:
        label = config.get("report", {}).get("sheet_method", config["experiment"]["name"])
        folder = HERE / config["outputs"]["dir"]
        runs = {path.name[len("student_"):-len("_predictions.npz")]: load_run(path)
                for path in sorted(folder.glob("student_*_predictions.npz"))}
        if not runs:
            raise FileNotFoundError(f"no predictions under {folder}")
        backbones[label] = runs
        print(f"{label}: {', '.join(runs)}", flush=True)

    rows = []
    for row in published_rows(configs[0], band):
        rows.append({"Backbone": "Result.xlsx", "Phien ban": "published_baseline",
                     "SNR (dB)": slice_label(row["slice"]), "So mau (support)": row["samples"],
                     "Top-1 Accuracy": round(row["accuracy"], 4),
                     "Macro-F1": round(row["macro_f1"], 4), "Micro-F1": round(row["micro_f1"], 4)})
    for label, runs in backbones.items():
        for run_name, run in runs.items():
            labels_count = len(run["labels"]) if run["labels"] else 36
            for slice_name, mask in slices(run["snr"], band, band_only=True):
                if not mask.any():
                    continue
                probs = run["probabilities"][mask] if run["probabilities"] is not None else None
                m = compute_metrics(run["target"][mask], run["predicted"][mask], probs,
                                    labels_count)
                def r4(v):
                    return round(v, 4) if isinstance(v, float) else v
                rows.append({"Backbone": label, "Phien ban": run_name,
                             "SNR (dB)": slice_label(slice_name),
                             "So mau (support)": int(mask.sum()),
                             "Top-1 Accuracy": r4(m["accuracy"]),
                             "Top-3 Accuracy": r4(m["top3_accuracy"]),
                             "Macro-F1": r4(m["macro_f1"]), "Micro-F1": r4(m["micro_f1"]),
                             "mAP": r4(m["macro_map"]), "Macro-AUC": r4(m["macro_auc_ovr"]),
                             "Delta Top-1 vs Result.xlsx":
                                 r4(m["accuracy"] - published[slice_name])
                                 if slice_name in published else ""})
    order = {band_label: 0}
    rows.sort(key=lambda r: (order.get(r["SNR (dB)"], 1), str(r["SNR (dB)"]),
                             r["Phien ban"] != "published_baseline", r["Phien ban"],
                             r["Backbone"]))
    write_csv(out_dir / "so_sanh_backbone.csv", rows, COMPARE_COLUMNS)

    tests = []
    labels = list(backbones)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            for run_name in sorted(set(backbones[a]) & set(backbones[b])):
                ra, rb = backbones[a][run_name], backbones[b][run_name]
                if ra["sample_id"] is None or rb["sample_id"] is None:
                    print(f"skip {run_name}: no sample_id to pair on", flush=True)
                    continue
                index_b = {sid: k for k, sid in enumerate(rb["sample_id"])}
                common = [(k, index_b[sid]) for k, sid in enumerate(ra["sample_id"])
                          if sid in index_b]
                ia = np.array([c[0] for c in common]); ib = np.array([c[1] for c in common])
                ok_a = ra["predicted"][ia] == ra["target"][ia]
                ok_b = rb["predicted"][ib] == rb["target"][ib]
                snr = ra["snr"][ia]
                for slice_name, mask in slices(snr, band, band_only=True):
                    if not mask.any():
                        continue
                    t = mcnemar(ok_a[mask], ok_b[mask])
                    tests.append({"Phien ban": run_name, "SNR (dB)": slice_label(slice_name),
                                  "So mau (support)": int(mask.sum()),
                                  "Backbone A": a, "Backbone B": b,
                                  "Top-1 A": round(float(ok_a[mask].mean()), 4),
                                  "Top-1 B": round(float(ok_b[mask].mean()), 4),
                                  "Delta (B - A)": round(float(ok_b[mask].mean() - ok_a[mask].mean()), 4),
                                  "Chi A dung": t["only_a_correct"], "Chi B dung": t["only_b_correct"],
                                  "p_value": t["p_value"], "Co y nghia (p<0.05)": t["significant_at_0.05"]})
    write_csv(out_dir / "so_sanh_mcnemar.csv", tests, MCNEMAR_COLUMNS)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare backbones run by run.")
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=HERE / "results_compare")
    arguments = parser.parse_args()
    command_compare36(arguments.configs, arguments.out)
