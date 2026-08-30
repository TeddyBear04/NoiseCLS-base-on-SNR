from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

logger = logging.getLogger(__name__)

SNR_TABLE_COLUMNS = [
    ("mAP", "mAP"),
    ("macro-AUC", "macro_auc"),
    ("macro-F1", "macro_f1"),
    ("micro-F1", "micro_f1"),
    ("precision", "precision_macro"),
    ("recall", "recall_macro"),
    ("acc", "hamming_accuracy"),
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
    def __init__(self, log_dir: str, label_names: Sequence[str], threshold: float = 0.5) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.label_names = list(label_names)
        self.threshold = threshold
        self.history_path = self.log_dir / "history.csv"
        self.headers = [
            "epoch",
            "train_loss",
            "train_mAP",
            "train_macro_f1",
            "train_micro_f1",
            "train_hamming_accuracy",
            "train_subset_accuracy",
            "train_local_snr_mae_db",
            "train_local_snr_rmse_db",
            "val_loss",
            "val_mAP",
            "val_macro_f1",
            "val_micro_f1",
            "val_hamming_accuracy",
            "val_subset_accuracy",
            "val_local_snr_mae_db",
            "val_local_snr_rmse_db",
            "val_local_snr_pearson",
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
            "train_mAP": f"{train_statistics['mAP']:.6f}",
            "train_macro_f1": f"{train_statistics['f1_macro']:.6f}",
            "train_micro_f1": f"{train_statistics['f1_micro']:.6f}",
            "train_hamming_accuracy": f"{train_statistics['hamming_accuracy']:.6f}",
            "train_subset_accuracy": f"{train_statistics['subset_accuracy']:.6f}",
            "train_local_snr_mae_db": f"{train_statistics['local_snr_mae_db']:.6f}",
            "train_local_snr_rmse_db": f"{train_statistics['local_snr_rmse_db']:.6f}",
            "val_loss": f"{val_statistics['loss']:.6f}",
            "val_mAP": f"{val_statistics['mAP']:.6f}",
            "val_macro_f1": f"{val_statistics['f1_macro']:.6f}",
            "val_micro_f1": f"{val_statistics['f1_micro']:.6f}",
            "val_hamming_accuracy": f"{val_statistics['hamming_accuracy']:.6f}",
            "val_subset_accuracy": f"{val_statistics['subset_accuracy']:.6f}",
            "val_local_snr_mae_db": f"{val_statistics['local_snr_mae_db']:.6f}",
            "val_local_snr_rmse_db": f"{val_statistics['local_snr_rmse_db']:.6f}",
            "val_local_snr_pearson": f"{val_statistics['local_snr_pearson']:.6f}",
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
        "mAP",
        "macro_auc",
        "macro_f1",
        "micro_f1",
        "precision_macro",
        "recall_macro",
        "hamming_accuracy",
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
            ("mAP", "mAP"),
            ("Macro F1", "macro_f1"),
            ("Micro F1", "micro_f1"),
            ("Hamming acc", "hamming_accuracy"),
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

    @staticmethod
    def _summary_values(prefix: str, statistics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            f"{prefix}_loss": statistics["loss"],
            f"{prefix}_mAP": statistics["mAP"],
            f"{prefix}_macro_f1": statistics["f1_macro"],
            f"{prefix}_micro_f1": statistics["f1_micro"],
            f"{prefix}_macro_auc": statistics["macro_auc"],
            f"{prefix}_hamming_accuracy": statistics["hamming_accuracy"],
            f"{prefix}_subset_accuracy": statistics["subset_accuracy"],
            f"{prefix}_clips": statistics["num_clips"],
            f"{prefix}_windows": statistics["num_windows"],
            f"{prefix}_local_snr_mae_db": statistics["local_snr_mae_db"],
            f"{prefix}_local_snr_rmse_db": statistics["local_snr_rmse_db"],
            f"{prefix}_local_snr_pearson": statistics["local_snr_pearson"],
            f"{prefix}_local_snr_valid_segments": statistics["local_snr_valid_segments"],
        }

    def _build_test_report(
        self,
        val_statistics: Dict[str, Any],
        test_statistics: Dict[str, Any],
    ) -> str:
        """Bundle the per-label report, headline metrics and SNR breakdown in one file."""
        sections = [
            f"Test classification report (threshold={self.threshold:.2f})",
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
            ("subset accuracy (exact match)", statistics["subset_accuracy"]),
            ("hamming accuracy", statistics["hamming_accuracy"]),
            ("mAP", statistics["mAP"]),
            ("macro AUC", statistics["macro_auc"]),
            ("macro F1", statistics["f1_macro"]),
            ("micro F1", statistics["f1_micro"]),
            ("macro precision", statistics["precision_macro"]),
            ("macro recall", statistics["recall_macro"]),
            ("local SNR MAE (dB)", statistics["local_snr_mae_db"]),
            ("local SNR RMSE (dB)", statistics["local_snr_rmse_db"]),
            ("local SNR Pearson", statistics["local_snr_pearson"]),
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
            "threshold": self.threshold,
        }
        with (self.log_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(details, handle, indent=2, ensure_ascii=False)
        self.save_per_label_metrics("test_per_label.csv", test_statistics)
        self.save_snr_metrics("validation_snr_metrics.csv", val_statistics)
        self.save_snr_metrics("test_snr_metrics.csv", test_statistics)
        self.plot_snr_metrics(test_statistics)
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
        figure, axes = plt.subplots(2, 3, figsize=(17, 9))
        pairs = [
            ("Loss", "train_loss", "val_loss"),
            ("mAP", "train_mAP", "val_mAP"),
            ("Macro F1", "train_macro_f1", "val_macro_f1"),
            ("Hamming accuracy", "train_hamming_accuracy", "val_hamming_accuracy"),
            ("Local SNR MAE (dB)", "train_local_snr_mae_db", "val_local_snr_mae_db"),
            ("Local SNR RMSE (dB)", "train_local_snr_rmse_db", "val_local_snr_rmse_db"),
        ]
        for axis, (title, train_key, val_key) in zip(axes.ravel(), pairs):
            axis.plot(epochs, [float(row[train_key]) for row in rows], label="train")
            axis.plot(epochs, [float(row[val_key]) for row in rows], label="validation")
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.3)
            axis.legend()
        figure.suptitle(f"{len(self.label_names)}-label noise classification + local SNR")
        figure.tight_layout()
        figure.savefig(self.log_dir / "learning_curves.png", dpi=150)
        plt.close(figure)
