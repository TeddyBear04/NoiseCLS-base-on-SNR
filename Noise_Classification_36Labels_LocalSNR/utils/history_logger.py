from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

logger = logging.getLogger(__name__)

SNR_TABLE_COLUMNS = [
    ("top-1 acc", "top1_accuracy"),
    ("top-3 acc", "top3_accuracy"),
    ("mAP", "mAP"),
    ("macro-AUC", "macro_auc"),
    ("macro-F1", "macro_f1"),
    ("micro-F1", "micro_f1"),
    ("precision", "precision_macro"),
    ("recall", "recall_macro"),
    ("exact", "subset_accuracy"),
]


def format_snr_table(snr_metrics: Dict[str, Dict[str, Any]]) -> str:
    """Render the per-SNR breakdown as a fixed-width table for logs and reports."""
    if not snr_metrics:
        return "(no SNR bands matched any clip)"
    header = f"  {'band':>10s} {'clips':>7s}" + "".join(
        f" {title:>10s}" for title, _ in SNR_TABLE_COLUMNS
    )
    lines = [header, "  " + "-" * (len(header) - 2)]
    for name, values in snr_metrics.items():
        row = f"  {name:>10s} {values['samples']:>7d}"
        row += "".join(f" {values[key]:>10.4f}" for _, key in SNR_TABLE_COLUMNS)
        lines.append(row)
    return "\n".join(lines)


class HistoryLogger:
    def __init__(self, log_dir: str, label_names: Sequence[str]) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.label_names = list(label_names)
        self.history_path = self.log_dir / "history.csv"
        self.headers = [
            "epoch",
            "train_loss",
            "train_top1_accuracy",
            "train_top3_accuracy",
            "train_mAP",
            "train_macro_f1",
            "train_micro_f1",
            "train_subset_accuracy",
            "val_loss",
            "val_top1_accuracy",
            "val_top3_accuracy",
            "val_mAP",
            "val_macro_f1",
            "val_micro_f1",
            "val_subset_accuracy",
            "is_best",
        ]
        with self.history_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.headers).writeheader()

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_statistics: Dict[str, Any],
        val_statistics: Dict[str, Any],
        is_best: bool,
    ) -> None:
        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "train_top1_accuracy": f"{train_statistics['top1_accuracy']:.6f}",
            "train_top3_accuracy": f"{train_statistics['top3_accuracy']:.6f}",
            "train_mAP": f"{train_statistics['mAP']:.6f}",
            "train_macro_f1": f"{train_statistics['f1_macro']:.6f}",
            "train_micro_f1": f"{train_statistics['f1_micro']:.6f}",
            "train_subset_accuracy": f"{train_statistics['subset_accuracy']:.6f}",
            "val_loss": f"{val_statistics['loss']:.6f}",
            "val_top1_accuracy": f"{val_statistics['top1_accuracy']:.6f}",
            "val_top3_accuracy": f"{val_statistics['top3_accuracy']:.6f}",
            "val_mAP": f"{val_statistics['mAP']:.6f}",
            "val_macro_f1": f"{val_statistics['f1_macro']:.6f}",
            "val_micro_f1": f"{val_statistics['f1_micro']:.6f}",
            "val_subset_accuracy": f"{val_statistics['subset_accuracy']:.6f}",
            "is_best": int(is_best),
        }
        with self.history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.headers)
            writer.writerow(row)
        if is_best:
            self.save_per_label_metrics("validation_per_label_best.csv", val_statistics)

    def save_per_label_metrics(self, filename: str, statistics: Dict[str, Any]) -> None:
        path = self.log_dir / filename
        rows = []
        for index, label in enumerate(self.label_names):
            matrix = statistics["confu_matrix"][index]
            rows.append(
                {
                    "model_index": index,
                    "label": label,
                    "average_precision": statistics["average_precision"][index],
                    "auc": statistics["auc"][index],
                    "top1_recall": statistics["per_label_top1_recall"][index],
                    "accuracy": statistics["per_label_accuracy"][index],
                    "tn": int(matrix[0, 0]),
                    "fp": int(matrix[0, 1]),
                    "fn": int(matrix[1, 0]),
                    "tp": int(matrix[1, 1]),
                }
            )
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    SNR_COLUMNS = [
        "snr_band",
        "snr_min_db",
        "snr_max_db",
        "samples",
        "top1_accuracy",
        "top3_accuracy",
        "balanced_accuracy",
        "mAP",
        "macro_auc",
        "macro_f1",
        "micro_f1",
        "precision_macro",
        "recall_macro",
        "subset_accuracy",
    ]

    def save_snr_metrics(self, filename: str, statistics: Dict[str, Any]) -> None:
        """Write one row per SNR band so the breakdown is readable without JSON."""
        bands = statistics.get("snr_metrics") or {}
        if not bands:
            logger.warning("No SNR band metrics to write to %s", filename)
            return
        with (self.log_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.SNR_COLUMNS)
            writer.writeheader()
            for name, values in bands.items():
                row = {"snr_band": name}
                row.update({key: values[key] for key in self.SNR_COLUMNS if key in values})
                writer.writerow(row)

    def plot_snr_metrics(self, statistics: Dict[str, Any], filename: str = "snr_metrics.png") -> None:
        bands = statistics.get("snr_metrics") or {}
        if not bands:
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = list(bands)
        series = [
            ("Top-1 acc", "top1_accuracy"),
            ("mAP", "mAP"),
            ("Macro F1", "macro_f1"),
            ("Micro F1", "micro_f1"),
        ]
        positions = np.arange(len(names), dtype=float)
        width = 0.8 / len(series)

        figure, axis = plt.subplots(figsize=(1.9 * len(names) + 4.0, 5))
        for index, (title, key) in enumerate(series):
            values = [bands[name][key] for name in names]
            offset = (index - (len(series) - 1) / 2) * width
            bars = axis.bar(positions + offset, values, width=width, label=title)
            axis.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)

        axis.set_xticks(positions)
        axis.set_xticklabels([f"{name}\nn={bands[name]['samples']}" for name in names])
        axis.set_xlabel("Target SNR band (dB)")
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", alpha=0.3)
        axis.legend(ncol=len(series), loc="upper left")
        axis.set_title("Test metrics by SNR band")
        figure.tight_layout()
        figure.savefig(self.log_dir / filename, dpi=150)
        plt.close(figure)

    def save_confusion_matrix(self, filename: str, statistics: Dict[str, Any]) -> None:
        """Write the square confusion matrix as counts, one row per true label."""
        matrix = statistics.get("confusion_matrix")
        if matrix is None:
            logger.warning("No confusion matrix to write to %s", filename)
            return
        with (self.log_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["true_label", *self.label_names])
            for label, row in zip(self.label_names, np.asarray(matrix)):
                writer.writerow([label, *(int(value) for value in row)])

    def plot_confusion_matrix(
        self,
        statistics: Dict[str, Any],
        filename: str = "confusion_matrix_test.png",
        title: str = "Test confusion matrix",
    ) -> None:
        """Draw the confusion matrix of the single-label task.

        The classes are unbalanced, so raw counts alone hide which labels the
        model actually confuses: the cell colour is the share of the true class
        (its row sums to 1) while the printed number stays the clip count.
        """
        matrix = statistics.get("confusion_matrix")
        if matrix is None:
            return
        matrix = np.asarray(matrix, dtype=np.int64)
        if matrix.size == 0 or not matrix.sum():
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        row_totals = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(
            matrix,
            row_totals,
            out=np.zeros(matrix.shape, dtype=np.float64),
            where=row_totals > 0,
        )
        size = len(matrix)
        names = self.label_names[:size]
        # imshow keeps the cells square, so the figure is sized from the label
        # count and constrained layout is left to close the gap to the colorbar.
        side = 0.42 * size + 3.0
        figure, axis = plt.subplots(figsize=(side + 2.0, side), layout="constrained")
        image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
        figure.colorbar(image, ax=axis, shrink=0.82, label="Share of the true label")

        for row in range(size):
            for column in range(size):
                if not matrix[row, column]:
                    continue
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:d}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if normalized[row, column] > 0.5 else "black",
                )

        axis.set_xticks(np.arange(size))
        axis.set_yticks(np.arange(size))
        axis.set_xticklabels(names, rotation=90, fontsize=7)
        axis.set_yticklabels(names, fontsize=7)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_title(f"{title} - {int(matrix.sum())} clips, top-1 acc {np.trace(matrix) / matrix.sum():.4f}")
        # Minor ticks only exist to draw the cell borders.
        axis.set_xticks(np.arange(size + 1) - 0.5, minor=True)
        axis.set_yticks(np.arange(size + 1) - 0.5, minor=True)
        axis.grid(which="minor", color="white", linewidth=0.5)
        axis.tick_params(which="minor", length=0)
        figure.savefig(self.log_dir / filename, dpi=150)
        plt.close(figure)

    @staticmethod
    def _summary_values(prefix: str, statistics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            f"{prefix}_loss": statistics["loss"],
            f"{prefix}_top1_accuracy": statistics["top1_accuracy"],
            f"{prefix}_top3_accuracy": statistics["top3_accuracy"],
            f"{prefix}_balanced_accuracy": statistics["balanced_accuracy"],
            f"{prefix}_mAP": statistics["mAP"],
            f"{prefix}_macro_f1": statistics["f1_macro"],
            f"{prefix}_micro_f1": statistics["f1_micro"],
            f"{prefix}_macro_auc": statistics["macro_auc"],
            f"{prefix}_subset_accuracy": statistics["subset_accuracy"],
            f"{prefix}_clips": statistics["num_clips"],
            f"{prefix}_windows": statistics["num_windows"],
        }

    def _build_test_report(
        self,
        val_statistics: Dict[str, Any],
        test_statistics: Dict[str, Any],
    ) -> str:
        """Bundle the per-label report, headline metrics and SNR breakdown in one file."""
        sections = [
            f"Test classification report (argmax over {len(self.label_names)} labels)",
            "=" * 78,
            test_statistics["message"].strip("\n"),
            "",
            "Overall test metrics",
            "-" * 78,
            self._format_overall_metrics(test_statistics),
            "",
            "Test metrics by SNR band",
            "-" * 78,
            format_snr_table(test_statistics.get("snr_metrics") or {}),
            "",
            "Validation metrics by SNR band (best epoch)",
            "-" * 78,
            format_snr_table(val_statistics.get("snr_metrics") or {}),
            "",
        ]
        return "\n".join(sections)

    @staticmethod
    def _format_overall_metrics(statistics: Dict[str, Any]) -> str:
        rows = [
            ("top-1 accuracy", statistics["top1_accuracy"]),
            ("top-3 accuracy", statistics["top3_accuracy"]),
            ("balanced accuracy (top-1)", statistics["balanced_accuracy"]),
            ("subset accuracy (exact match)", statistics["subset_accuracy"]),
            ("mAP", statistics["mAP"]),
            ("macro AUC", statistics["macro_auc"]),
            ("macro F1", statistics["f1_macro"]),
            ("micro F1", statistics["f1_micro"]),
            ("macro precision", statistics["precision_macro"]),
            ("macro recall", statistics["recall_macro"]),
        ]
        lines = [f"  {name:<30s} {value:.4f}" for name, value in rows]
        lines.append(f"  {'clips':<30s} {statistics['num_clips']:d}")
        lines.append(f"  {'windows':<30s} {statistics['num_windows']:d}")
        return "\n".join(lines)

    def save_summary(
        self,
        training_time: float,
        inference_time_ms: float,
        best_epoch: int,
        val_statistics: Dict[str, Any],
        test_statistics: Dict[str, Any],
    ) -> None:
        summary = {
            "training_time_seconds": training_time,
            "inference_time_ms_per_window": inference_time_ms,
            "best_epoch": best_epoch,
            **self._summary_values("val", val_statistics),
            **self._summary_values("test", test_statistics),
        }
        with (self.log_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow(summary)

        details = {
            "summary": summary,
            "validation_snr_metrics": val_statistics["snr_metrics"],
            "test_snr_metrics": test_statistics["snr_metrics"],
            "loss": "cross_entropy",
            "prediction_rule": "argmax",
        }
        with (self.log_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(details, handle, indent=2, ensure_ascii=False)
        self.save_per_label_metrics("test_per_label.csv", test_statistics)
        self.save_snr_metrics("validation_snr_metrics.csv", val_statistics)
        self.save_snr_metrics("test_snr_metrics.csv", test_statistics)
        self.plot_snr_metrics(test_statistics)
        self.save_confusion_matrix("confusion_matrix_test.csv", test_statistics)
        self.plot_confusion_matrix(test_statistics)
        self.plot_confusion_matrix(
            val_statistics,
            filename="confusion_matrix_validation.png",
            title="Validation confusion matrix (best epoch)",
        )
        (self.log_dir / "classification_report_test.txt").write_text(
            self._build_test_report(val_statistics, test_statistics), encoding="utf-8"
        )
        logger.info("Saved training summary to %s", self.log_dir)

    def plot_history(self) -> None:
        rows = []
        with self.history_path.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [int(row["epoch"]) for row in rows]
        figure, axes = plt.subplots(2, 2, figsize=(13, 9))
        pairs = [
            ("Loss", "train_loss", "val_loss"),
            ("mAP", "train_mAP", "val_mAP"),
            ("Macro F1", "train_macro_f1", "val_macro_f1"),
            ("Top-1 accuracy", "train_top1_accuracy", "val_top1_accuracy"),
        ]
        for axis, (title, train_key, val_key) in zip(axes.ravel(), pairs):
            axis.plot(epochs, [float(row[train_key]) for row in rows], label="train")
            axis.plot(epochs, [float(row[val_key]) for row in rows], label="validation")
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.3)
            axis.legend()
        figure.suptitle(f"{len(self.label_names)}-label noise classification")
        figure.tight_layout()
        figure.savefig(self.log_dir / "learning_curves.png", dpi=150)
        plt.close(figure)
