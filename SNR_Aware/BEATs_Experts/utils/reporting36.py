"""Artifacts for the 36-class single-label BEATs workflow."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

# (header, key) pairs. The per-SNR table reads `key`; the "All" row reads `test_<key>`.
SNR_COLUMNS = (
    ("Top-1 Accuracy", "accuracy"), ("Top-3 Accuracy", "top3_accuracy"),
    ("Balanced Acc", "balanced_accuracy"), ("Precision Macro", "precision"),
    ("Recall Macro", "recall"), ("Macro-F1", "macro_f1"), ("Micro-F1", "micro_f1"),
    ("mAP", "map"), ("Macro-AUC", "macro_auc"), ("SI-SDR (dB)", "si_sdr_db"),
)
TEST_COLUMNS = (
    ("Test Accuracy", "test_accuracy"), ("Test Precision", "test_precision"),
    ("Test Recall", "test_recall"), ("Test Macro-F1", "test_macro_f1"),
    ("Test Micro-F1", "test_micro_f1"), ("Test mAP", "test_map"),
    ("Test Balanced Acc", "test_balanced_accuracy"), ("Test Macro-AUC", "test_macro_auc"),
)
CLASS_COLUMNS = (
    ("Support", "support"), ("Precision", "precision"), ("Recall", "recall"),
    ("F1", "f1"), ("AP", "ap"), ("AUC", "auc"),
)


def top_k_accuracy(expected: np.ndarray, probabilities: np.ndarray, k: int = 3) -> float:
    top = np.argsort(-probabilities, axis=1)[:, :k]
    return float((top == expected[:, None]).any(axis=1).mean())


def full_metrics(expected: np.ndarray, predicted: np.ndarray, probabilities: np.ndarray) -> dict:
    one_hot = np.eye(probabilities.shape[1], dtype=np.int32)[expected]
    try:
        macro_auc = float(roc_auc_score(one_hot, probabilities, multi_class="ovr", average="macro"))
        mean_ap = float(average_precision_score(one_hot, probabilities, average="macro"))
    except ValueError:
        # A slice can lack one or more of the 36 classes.
        macro_auc, mean_ap = float("nan"), float("nan")
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "top3_accuracy": top_k_accuracy(expected, probabilities),
        "balanced_accuracy": float(balanced_accuracy_score(expected, predicted)),
        "precision": float(precision_score(expected, predicted, average="macro", zero_division=0)),
        "recall": float(recall_score(expected, predicted, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(expected, predicted, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(expected, predicted, average="micro", zero_division=0)),
        "map": mean_ap,
        "macro_auc": macro_auc,
    }


def per_class_metrics(expected, predicted, probabilities, labels) -> dict:
    one_hot = np.eye(len(labels), dtype=np.int32)[expected]
    precision, recall, f1, support = precision_recall_fscore_support(
        expected, predicted, labels=range(len(labels)), zero_division=0
    )
    rows = {}
    for index, label in enumerate(labels):
        try:
            ap = float(average_precision_score(one_hot[:, index], probabilities[:, index]))
            auc = float(roc_auc_score(one_hot[:, index], probabilities[:, index]))
        except ValueError:
            ap, auc = float("nan"), float("nan")
        rows[label] = {
            "support": int(support[index]), "precision": float(precision[index]),
            "recall": float(recall[index]), "f1": float(f1[index]), "ap": ap, "auc": auc,
        }
    return rows


def format_cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def markdown_table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def snr_rows(result: dict) -> list[list[str]]:
    rows = []
    for snr, values in result["per_snr"].items():
        rows.append([snr, *(format_cell(values.get(key)) for _, key in SNR_COLUMNS)])
    rows.append(["All", *(format_cell(result.get(f"test_{key}")) for _, key in SNR_COLUMNS)])
    return rows


def print_report(result: dict) -> None:
    print(markdown_table([h for h, _ in TEST_COLUMNS], [[format_cell(result[k]) for _, k in TEST_COLUMNS]]))
    print("\n" + markdown_table(["SNR (dB)", *(h for h, _ in SNR_COLUMNS)], snr_rows(result)))
    class_rows = [
        [label, *(format_cell(values[key]) for _, key in CLASS_COLUMNS)]
        for label, values in result["per_class"].items()
    ]
    print("\n" + markdown_table(["Label", *(h for h, _ in CLASS_COLUMNS)], class_rows))


def save_labeled_confusion_matrix(
    path: Path, labels: list[str], matrix: np.ndarray
) -> None:
    """Write the matrix with explicit true/predicted class labels."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])


def save_top_confusions(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    """Save the 50 most common off-diagonal classification mistakes."""
    confusions = []
    for true_index, true_label in enumerate(labels):
        support = int(matrix[true_index].sum())
        for predicted_index, predicted_label in enumerate(labels):
            count = int(matrix[true_index, predicted_index])
            if true_index == predicted_index or count == 0:
                continue
            confusions.append(
                {
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "count": count,
                    "true_class_samples": support,
                    "percent_of_true_class": 100 * count / support,
                }
            )
    fields = [
        "true_label",
        "predicted_label",
        "count",
        "true_class_samples",
        "percent_of_true_class",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            sorted(confusions, key=lambda row: row["count"], reverse=True)[:50]
        )


def save_per_class_metrics(
    path: Path, labels: list[str], matrix: np.ndarray, extra: dict | None = None
) -> None:
    """Write one row of classification metrics for every label."""
    fields = [
        "label",
        "support",
        "correct_predictions",
        "incorrect_predictions",
        "precision",
        "recall",
        "f1",
        *(("ap", "auc") if extra else ()),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, label in enumerate(labels):
            correct = int(matrix[index, index])
            support = int(matrix[index].sum())
            predicted = int(matrix[:, index].sum())
            precision = correct / predicted if predicted else 0.0
            recall = correct / support if support else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            writer.writerow(
                {
                    "label": label,
                    "support": support,
                    "correct_predictions": correct,
                    "incorrect_predictions": support - correct,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    **({"ap": extra[label]["ap"], "auc": extra[label]["auc"]} if extra else {}),
                }
            )


def save_evaluation_artifacts(output_dir: Path, result: dict, labels, expected, predicted) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["accuracy", "precision", "recall", "macro_f1", "micro_f1"])
        writer.writeheader(); writer.writerow({key: result[key] for key in writer.fieldnames})
    if result.get("per_snr"):
        with (output_dir / "snr_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["snr_db", "samples", *(key for _, key in SNR_COLUMNS)]
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for snr, values in result["per_snr"].items():
                writer.writerow({"snr_db": snr, **{key: values.get(key) for key in fields[1:]}})
            writer.writerow({
                "snr_db": "All", "samples": result["samples"],
                **{key: result.get(f"test_{key}") for _, key in SNR_COLUMNS},
            })
    matrix = confusion_matrix(expected, predicted, labels=np.arange(len(labels)))
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    save_labeled_confusion_matrix(
        output_dir / "confusion_matrix_labeled.csv", labels, matrix
    )
    save_top_confusions(output_dir / "top_confusions.csv", labels, matrix)
    save_per_class_metrics(
        output_dir / "per_class_metrics.csv", labels, matrix, result.get("per_class")
    )
    try:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(20, 18))
        image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
        figure.colorbar(image, ax=axis)
        positions = np.arange(len(labels))
        axis.set(
            title=f"{len(labels)}-label confusion matrix",
            xlabel="Predicted label",
            ylabel="True label",
            xticks=positions,
            yticks=positions,
            xticklabels=labels,
            yticklabels=labels,
        )
        axis.tick_params(axis="x", labelrotation=90, labelsize=7)
        axis.tick_params(axis="y", labelsize=7)
        figure.tight_layout(); figure.savefig(output_dir / "confusion_matrix.png", dpi=180); plt.close(figure)
    except ImportError:
        pass


def save_training_artifacts(output_dir: Path, labels, result: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy", "validation_macro_f1"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in result["history"]:
            validation = row["validation"]
            train = row.get("train", {})
            writer.writerow({"epoch": row["epoch"], "train_loss": train.get("loss"), "train_accuracy": train.get("accuracy"), "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"], "validation_macro_f1": validation["macro_f1"]})
