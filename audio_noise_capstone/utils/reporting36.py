"""Artifacts for the 36-class single-label BEATs workflow."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix


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


def save_evaluation_artifacts(output_dir: Path, result: dict, labels, expected, predicted) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["accuracy", "precision", "recall", "macro_f1", "micro_f1"])
        writer.writeheader(); writer.writerow({key: result[key] for key in writer.fieldnames})
    if result.get("per_snr"):
        with (output_dir / "snr_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["snr_db", "samples", "accuracy", "macro_f1", "micro_f1"]
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for snr, values in result["per_snr"].items():
                writer.writerow({"snr_db": snr, **{key: values[key] for key in fields[1:]}})
    matrix = confusion_matrix(expected, predicted, labels=np.arange(len(labels)))
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    save_labeled_confusion_matrix(
        output_dir / "confusion_matrix_labeled.csv", labels, matrix
    )
    save_top_confusions(output_dir / "top_confusions.csv", labels, matrix)
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
