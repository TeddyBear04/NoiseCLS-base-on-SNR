"""Turn every finished run into CSV tables.

    python -u main.py report36 --config config/train_config.json

Reads each `artifacts/student_<run>_predictions.npz` and writes three CSVs under
`artifacts/results/`:

    metrics_by_run.csv    one row per run x slice, all eight repo metrics
    metrics_per_class.csv one row per run x slice x label
    comparison.csv        run-vs-run deltas plus the paired McNemar test

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

from tasks.run_36 import HERE

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
    "beats_baseline": "BEATs baseline (mixture)",
    "run2_ce_only": "Mid expert - CE only (control)",
    "run3_kd_crd": "Mid expert - KD + CRD",
    "run3b_crd_only": "Mid expert - CRD only",
    "run3c_kd_only": "Mid expert - KD only",
    "run4_remix": "Mid expert - KD + CRD + remix",
}
SLICE_NAMES = {"full": "all", "mid": "5-10"}


def sheet_row(row: dict) -> dict:
    slice_name = row["slice"]
    snr = SLICE_NAMES.get(slice_name, slice_name.replace("snr_", ""))
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


def slices(snr: np.ndarray, band: list[float]) -> list[tuple[str, np.ndarray]]:
    """full, the mid band we are judged on, then one slice per SNR level."""
    out = [("full", np.ones(snr.shape, dtype=bool)),
           ("mid", (snr >= band[0]) & (snr <= band[1]))]
    for level in sorted(set(snr.tolist())):
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
    return data


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Paired test. The mid slice is 2,160 clips, where the standard error on
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


def baseline_rows(labels: list[str]) -> list[dict]:
    """The BEATs baseline from its own metrics file. It has no per-clip predictions
    here, so its slices carry only what that file recorded."""
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
    mid = [v for k, v in per_snr.items() if 5 <= float(k) <= 10]
    if len(mid) == 2:
        total = sum(v["samples"] for v in mid)
        rows.append({"run": "beats_baseline", "slice": "mid", "samples": total,
                     "accuracy": sum(v["accuracy"] * v["samples"] for v in mid) / total,
                     "macro_precision": "", "macro_recall": "",
                     "macro_f1": sum(v["macro_f1"] * v["samples"] for v in mid) / total,
                     "micro_f1": sum(v["micro_f1"] * v["samples"] for v in mid) / total,
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


def command_report36(config: dict) -> None:
    out_dir = HERE / config["outputs"]["dir"]
    results_dir = out_dir / "results"
    band = config["mid_band_db"]

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
        for slice_name, mask in slices(run["snr"], band):
            if not mask.any():
                continue
            probabilities = (run["probabilities"][mask]
                             if run["probabilities"] is not None else None)
            row = {"run": name, "slice": slice_name, "samples": int(mask.sum())}
            row.update(compute_metrics(run["target"][mask], run["predicted"][mask],
                                       probabilities, len(labels)))
            summary.append(row)
            if slice_name in ("full", "mid"):
                for entry in per_class_rows(run["target"][mask], run["predicted"][mask],
                                            probabilities, labels):
                    per_class.append({"run": name, "slice": slice_name, **entry})

    summary.extend(baseline_rows(labels))
    summary.sort(key=lambda r: (r["slice"] != "mid", r["slice"], r["run"]))
    write_csv(results_dir / "metrics_by_run.csv", summary,
              ["run", "slice", "samples", *METRIC_COLUMNS])
    write_csv(results_dir / "ket_qua_tong_hop.csv",
              [sheet_row(r) for r in summary], SHEET_COLUMNS)
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
            for slice_name, mask in slices(runs[a]["snr"], band):
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
    comparisons.sort(key=lambda r: (r["slice"] != "mid", r["slice"], r["run_a"], r["run_b"]))
    write_csv(results_dir / "comparison.csv", comparisons,
              ["run_a", "run_b", "slice", "samples", "accuracy_a", "accuracy_b",
               "delta_accuracy", "both_correct", "both_wrong", "only_a_correct",
               "only_b_correct", "chi2", "p_value", "significant_at_0.05"])

    print("\n=== mid slice ===", flush=True)
    header = f"{'run':<18}{'acc':>9}{'macro_f1':>10}{'mAP':>9}{'AUC':>9}"
    print(header, flush=True)
    for row in summary:
        if row["slice"] != "mid":
            continue
        def fmt(key):
            value = row.get(key, "")
            return f"{value:>9.4f}" if isinstance(value, float) else f"{'-':>9}"
        print(f"{row['run']:<18}{fmt('accuracy')}{fmt('macro_f1')}"
              f"{fmt('macro_map')}{fmt('macro_auc_ovr')}", flush=True)
