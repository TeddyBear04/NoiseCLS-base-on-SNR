"""Reusable metrics, reports, CSV files, and plots for the 36-label workflow."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


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


def save_per_class_metrics(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    """Write one row of classification metrics for every label."""
    fields = [
        "label",
        "support",
        "correct_predictions",
        "incorrect_predictions",
        "precision",
        "recall",
        "f1",
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
                }
            )


def classification_report_text(
    labels: list[str], expected: np.ndarray, predicted: np.ndarray
) -> str:
    """Match sklearn's per-label report, including micro and samples averages."""
    classes = np.arange(len(labels))
    expected_one_hot = np.eye(len(labels), dtype=np.int8)[expected]
    predicted_one_hot = np.eye(len(labels), dtype=np.int8)[predicted]
    return classification_report(
        expected_one_hot,
        predicted_one_hot,
        labels=classes,
        target_names=labels,
        digits=4,
        zero_division=0,
    )


def save_snr_metrics(path: Path, per_snr: dict) -> None:
    fields = ["snr_db", "samples", "accuracy", "macro_f1", "micro_f1"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for snr, values in per_snr.items():
            writer.writerow({"snr_db": snr, **{key: values[key] for key in fields[1:]}})


def save_snr_plot(path: Path, per_snr: dict) -> None:
    if not per_snr:
        return
    import matplotlib.pyplot as plt

    snrs = list(per_snr)
    accuracy = [per_snr[snr]["accuracy"] for snr in snrs]
    macro_f1 = [per_snr[snr]["macro_f1"] for snr in snrs]
    positions = np.arange(len(snrs))
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(positions, accuracy, marker="o", label="accuracy")
    axis.plot(positions, macro_f1, marker="o", label="macro F1")
    axis.set(xticks=positions, xticklabels=snrs, xlabel="SNR (dB)", ylabel="Score")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_learning_curves(path: Path, history: list[dict]) -> None:
    """Plot train/validation loss and validation classification quality."""
    if not history:
        return
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    train_loss = [row.get("train", {}).get("loss", row.get("train_loss")) for row in history]
    validation_loss = [row["validation"].get("loss") for row in history]
    train_accuracy = [row.get("train", {}).get("accuracy", row.get("train_accuracy")) for row in history]
    validation_accuracy = [row["validation"].get("accuracy") for row in history]
    validation_f1 = [row["validation"].get("macro_f1") for row in history]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if any(value is not None for value in train_loss):
        axes[0].plot(epochs, train_loss, marker="o", label="train loss")
    if any(value is not None for value in validation_loss):
        axes[0].plot(epochs, validation_loss, marker="o", label="validation loss")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    axes[0].grid(alpha=0.3)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend()
    if any(value is not None for value in train_accuracy):
        axes[1].plot(epochs, train_accuracy, marker="o", label="train accuracy")
    axes[1].plot(epochs, validation_accuracy, marker="o", label="validation accuracy")
    axes[1].plot(epochs, validation_f1, marker="o", label="validation macro F1")
    axes[1].set(xlabel="Epoch", ylabel="Score", ylim=(0, 1), title="Validation quality")
    axes[1].grid(alpha=0.3)
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_evaluation_artifacts(
    output_dir: Path,
    result: dict,
    labels: list[str],
    expected: np.ndarray,
    predicted: np.ndarray,
    *,
    split: str,
) -> str:
    """Write split-specific evaluation outputs and return the text report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(expected, predicted, labels=np.arange(len(labels)))
    report = classification_report_text(labels, expected, predicted)
    (output_dir / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if split == "test":
        (output_dir / "classification_report_test.txt").write_text(
            report, encoding="utf-8"
        )
        (output_dir / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["accuracy", "precision", "recall", "macro_f1", "micro_f1"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow({key: result[key] for key in fields})
        save_per_class_metrics(output_dir / "test_per_label.csv", labels, matrix)
        save_snr_metrics(output_dir / "test_snr_metrics.csv", result.get("per_snr", {}))
        save_snr_plot(output_dir / "snr_metrics.png", result.get("per_snr", {}))
        np.savetxt(output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
        save_labeled_confusion_matrix(
            output_dir / "confusion_matrix_labeled.csv", labels, matrix
        )
        save_top_confusions(output_dir / "top_confusions.csv", labels, matrix)
    elif split == "validation":
        save_per_class_metrics(output_dir / "validation_per_label_best.csv", labels, matrix)
        save_snr_metrics(output_dir / "validation_snr_metrics.csv", result.get("per_snr", {}))
    else:
        raise ValueError(f"Unsupported split for reports: {split}")
    return report


def save_training_artifacts(output_dir: Path, labels: list[str], result: dict) -> None:
    """Save stable artifacts generated by either training stage."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    best = result["best_validation"]
    fields = [
        "stage", "best_epoch", "train_samples", "validation_samples", "accuracy",
        "precision", "recall", "macro_f1", "micro_f1", "elapsed_seconds", "checkpoint",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "stage": result["stage"],
                "best_epoch": result["best_epoch"],
                "train_samples": result["train_samples"],
                "validation_samples": result["validation_samples"],
                **{key: best.get(key) for key in ("accuracy", "precision", "recall", "macro_f1", "micro_f1")},
                "elapsed_seconds": result.get("elapsed_seconds"),
                "checkpoint": result["checkpoint"],
            }
        )
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "epoch", "train_loss", "train_accuracy", "validation_loss",
            "validation_accuracy", "validation_macro_f1",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result["history"]:
            validation = row["validation"]
            train = row.get("train", {})
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "train_loss": train.get("loss", row.get("train_loss")),
                    "train_accuracy": train.get("accuracy", row.get("train_accuracy")),
                    "validation_loss": validation.get("loss"),
                    "validation_accuracy": validation["accuracy"],
                    "validation_macro_f1": validation["macro_f1"],
                }
            )
    save_learning_curves(output_dir / "learning_curves.png", result["history"])
