"""Jointly fine-tune separator, last BEATs blocks and fusion head.

Total loss: L = L_class + lambda * L_sep, where L_sep is negative SI-SDR between
the separator output and ``oracle_noise`` (training only).
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config.paths import BEATS_CHECKPOINT
from models.fusion import (
    TEMPORAL_POOLING,
    FusionClassifier,
    encode_branches,
    supervised_contrastive_loss,
)
from models.separator import build_separator, separation_loss, separator_payload, si_sdr
from noise_pipeline.mix_data import MixNoiseDataset, load_mix_manifest
from utils.reporting36 import save_evaluation_artifacts, save_training_artifacts
from utils.training36 import (
    SEED,
    SNRS,
    add_si_sdr_metrics,
    load_beats,
    metrics,
    mixed_precision_context,
    require_finite,
    seed_everything,
    select_rows,
)


def load_model(
    device: torch.device,
    checkpoint: dict,
    trainable_blocks: int,
    dropout: float = 0.1,
    noise_dropout: float = 0.2,
    conditioned_dropout: float = 0.2,
):
    """Build encoder, separator and classifier from a head or fine-tuned checkpoint."""
    checkpoint_pooling = checkpoint.get("temporal_pooling", "mean_v1")
    if checkpoint_pooling != TEMPORAL_POOLING:
        raise ValueError(
            f"Checkpoint pooling {checkpoint_pooling!r} is incompatible with "
            f"{TEMPORAL_POOLING!r}. Rerun head36 to regenerate embeddings and weights."
        )
    encoder, _ = load_beats(device)
    if trainable_blocks < 0 or trainable_blocks > len(encoder.encoder.layers):
        raise ValueError("trainable_blocks must be between 0 and the encoder layer count")
    if trainable_blocks:
        for layer in encoder.encoder.layers[-trainable_blocks:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
    if "encoder_delta" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder_delta"], strict=False)
    separator = build_separator(checkpoint["separator"]).to(device)
    labels = checkpoint["labels"]
    fusion_weight = checkpoint["classifier"].get("fusion.weight")
    if fusion_weight is None or fusion_weight.shape[1] != 3 * 768:
        raise ValueError(
            "This checkpoint predates the speech-WDRC conditioned branch. "
            "Run `python main.py head36 --config config/train_config.json --force-extract` "
            "and then rerun finetune36."
        )
    classifier = FusionClassifier(
        len(labels), dropout=dropout, noise_dropout=noise_dropout,
        conditioned_dropout=conditioned_dropout,
    )
    try:
        classifier.load_state_dict(checkpoint["classifier"])
    except RuntimeError as error:
        raise ValueError(
            "This checkpoint predates the quality-gated three-view classifier. "
            "Retrain head36, then finetune36, before evaluating it."
        ) from error
    classifier.to(device)
    return encoder, separator, classifier, labels


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool, device: torch.device,
                seed: int = SEED):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def parse_snr_weights(raw: str) -> torch.Tensor:
    """Parse one weight for each fixed training SNR group."""
    weights = [float(value.strip()) for value in raw.split(",")]
    if len(weights) != len(SNRS) or any(weight <= 0 for weight in weights):
        raise ValueError(f"snr_group_weights must contain {len(SNRS)} positive comma-separated values")
    return torch.tensor(weights, dtype=torch.float32)


class SNRRobustLoss:
    """Per-sample CE with fixed SNR weights or exponentiated Group DRO weights."""

    def __init__(self, mode: str, fixed_weights: torch.Tensor, eta: float, device: torch.device):
        if mode not in {"weighted_ce", "group_dro"}:
            raise ValueError("snr_loss_mode must be 'weighted_ce' or 'group_dro'")
        self.mode = mode
        self.fixed_weights = fixed_weights.to(device)
        self.eta = eta
        self.log_q = torch.zeros(len(SNRS), device=device)

    def __call__(self, per_sample_loss: torch.Tensor, snrs: torch.Tensor) -> torch.Tensor:
        group_losses = torch.stack([
            per_sample_loss[snrs == snr].mean() if (snrs == snr).any()
            else torch.full((), float("nan"), device=per_sample_loss.device)
            for snr in SNRS
        ])
        if self.mode == "weighted_ce":
            weights = torch.ones_like(per_sample_loss)
            for index, snr in enumerate(SNRS):
                weights = torch.where(snrs == snr, self.fixed_weights[index], weights)
            return (per_sample_loss * weights).mean() / self.fixed_weights.mean()

        present = torch.isfinite(group_losses)
        with torch.no_grad():
            self.log_q[present] += self.eta * group_losses[present]
            self.log_q -= torch.logsumexp(self.log_q, dim=0)
        q = torch.softmax(self.log_q, dim=0)
        return (q[present] * group_losses[present]).sum() / q[present].sum()

    def weights(self) -> dict[str, float]:
        weights = self.fixed_weights if self.mode == "weighted_ce" else torch.softmax(self.log_q, dim=0)
        return {str(snr): float(weights[index]) for index, snr in enumerate(SNRS)}


def validation_score(result: dict, metric_name: str) -> float:
    if metric_name == "macro_f1":
        return result["macro_f1"]
    if metric_name == "worst_group_macro_f1":
        return min(values["macro_f1"] for values in result["per_snr"].values())
    raise ValueError("selection_metric must be 'macro_f1' or 'worst_group_macro_f1'")


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.float()
    right = right.float()
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.norm() * right.norm()
    return None if denominator <= 1e-8 else float((left * right).sum() / denominator)


def branch_quality_metrics(
    weights: torch.Tensor, noise_si_sdr: torch.Tensor, leakage_si_sdr: torch.Tensor,
    correct: torch.Tensor, snrs: torch.Tensor,
) -> dict:
    """Report gate reliability correlations overall and for every SNR group."""
    result = {"overall": {}, "per_snr": {}}

    def summarise(mask: torch.Tensor) -> dict:
        selected = weights[mask]
        noise_gate = selected[:, 1]
        conditioned_gate = selected[:, 2]
        return {
            "samples": int(mask.sum()),
            "mean_branch_weights": {
                "mixture": float(selected[:, 0].mean()),
                "noise": float(noise_gate.mean()),
                "conditioned": float(conditioned_gate.mean()),
            },
            "noise_gate_si_sdr_correlation": _correlation(noise_gate, noise_si_sdr[mask]),
            "noise_gate_leakage_si_sdr_correlation": _correlation(noise_gate, leakage_si_sdr[mask]),
            "noise_gate_correct_correlation": _correlation(noise_gate, correct[mask]),
            "conditioned_gate_correct_correlation": _correlation(conditioned_gate, correct[mask]),
        }

    all_samples = torch.ones_like(snrs, dtype=torch.bool)
    result["overall"] = summarise(all_samples)
    for snr in SNRS:
        mask = snrs == snr
        if mask.any():
            result["per_snr"][str(snr)] = summarise(mask)
    return result


def set_training_mode(encoder, separator, classifier, trainable_blocks: int, train_separator: bool) -> None:
    # Frozen blocks remain deterministic; dropout is active only in blocks that
    # receive gradient updates.
    encoder.eval()
    if trainable_blocks:
        for layer in encoder.encoder.layers[-trainable_blocks:]:
            layer.train()
    separator.train(train_separator)
    classifier.train()


@torch.inference_mode()
def evaluate(encoder, separator, classifier, loader, device, labels, sep_loss_weight: float,
             return_predictions: bool = False):
    encoder.eval()
    separator.eval()
    classifier.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    all_logits, all_targets, all_snrs, all_si_sdrs = [], [], [], []
    all_weights, all_leakage_si_sdrs, all_correct = [], [], []
    class_loss_total = 0.0
    for batch in loader:
        mixture = batch["mixture"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        oracle = batch["oracle_noise"].to(device, non_blocking=True)
        z_mix, z_noise, z_conditioned, noise, noise_ratio_db, speech_ratio_db = encode_branches(
            encoder, separator, mixture, device
        )
        with mixed_precision_context(device):
            logits, details = classifier(
                z_mix, z_noise, z_conditioned, noise_ratio_db, speech_ratio_db, return_details=True
            )
            loss = criterion(logits, targets)
        require_finite(loss, "validation loss")
        class_loss_total += loss.item()
        all_logits.append(logits.float().cpu())
        all_targets.append(targets.cpu())
        all_snrs.append(batch["snr"].cpu())
        all_si_sdrs.append(si_sdr(noise, oracle).cpu())
        oracle_speech = mixture - oracle
        all_leakage_si_sdrs.append(si_sdr(noise, oracle_speech).cpu())
        all_weights.append(details["branch_weights"].float().cpu())
        all_correct.append((logits.argmax(dim=1) == targets).float().cpu())
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    snrs = torch.cat(all_snrs)
    si_sdrs = torch.cat(all_si_sdrs)
    branch_weights = torch.cat(all_weights)
    leakage_si_sdrs = torch.cat(all_leakage_si_sdrs)
    correct = torch.cat(all_correct)
    result = add_si_sdr_metrics(metrics(logits, targets, snrs, labels), si_sdrs, snrs)
    result["worst_group_macro_f1"] = min(
        values["macro_f1"] for values in result["per_snr"].values()
    )
    result["branch_quality"] = branch_quality_metrics(
        branch_weights, si_sdrs, leakage_si_sdrs, correct, snrs
    )
    result["class_loss"] = class_loss_total / len(loader.dataset)
    result["separation_loss"] = -result["si_sdr"]
    result["loss"] = result["class_loss"] + sep_loss_weight * result["separation_loss"]
    if return_predictions:
        return result, targets.numpy(), logits.argmax(dim=1).numpy()
    return result


def train_epoch(encoder, separator, classifier, loader, optimizer, device, args, train_separator: bool,
                robust_loss: SNRRobustLoss):
    set_training_mode(encoder, separator, classifier, args.trainable_blocks, train_separator)
    optimizer.zero_grad(set_to_none=True)
    totals = {
        "loss": 0.0, "class_loss": 0.0, "noise_aux_loss": 0.0,
        "consistency_loss": 0.0, "contrastive_loss": 0.0,
        "noise_separation_loss": 0.0, "residual_separation_loss": 0.0,
    }
    correct = 0
    seen = 0
    trainable_parameters = [
        parameter
        for module in (encoder, separator, classifier)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    for batch_index, batch in enumerate(loader, start=1):
        mixture = batch["mixture"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        oracle = batch["oracle_noise"].to(device, non_blocking=True)
        snrs = batch["snr"].to(device, non_blocking=True)
        z_mix, z_noise, z_conditioned, noise, noise_ratio_db, speech_ratio_db = encode_branches(
            encoder, separator, mixture, device
        )
        with mixed_precision_context(device):
            logits, details = classifier(
                z_mix, z_noise, z_conditioned, noise_ratio_db, speech_ratio_db, return_details=True
            )
            class_loss = robust_loss(F.cross_entropy(logits, targets, reduction="none"), snrs)
            noise_aux_loss = F.cross_entropy(details["noise_logits"], targets)
            consistency_loss = F.mse_loss(details["mix_logits"], details["conditioned_logits"])
            second_noise_projection = classifier.noise_projection(classifier.noise_norm(z_noise))
            contrastive_loss = supervised_contrastive_loss(
                torch.cat([details["noise_projection"], second_noise_projection]),
                targets.repeat(2), args.contrastive_temperature,
            )
        if train_separator:
            noise_sep_loss = separation_loss(noise, oracle, args.spectral_loss_weight)
            residual_sep_loss = separation_loss(
                mixture - noise, mixture - oracle, args.spectral_loss_weight
            )
        else:
            noise_sep_loss = logits.new_zeros(())
            residual_sep_loss = logits.new_zeros(())
        loss = (
            class_loss
            + args.noise_aux_weight * noise_aux_loss
            + args.view_consistency_weight * consistency_loss
            + args.contrastive_weight * contrastive_loss
            + args.sep_loss_weight * noise_sep_loss
            + args.residual_sep_loss_weight * residual_sep_loss
        )
        require_finite(loss, "training loss")
        (loss / args.accumulation_steps).backward()
        if batch_index % args.accumulation_steps == 0 or batch_index == len(loader):
            nn.utils.clip_grad_norm_(trainable_parameters, 5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        size = targets.shape[0]
        totals["loss"] += loss.item() * size
        totals["class_loss"] += class_loss.item() * size
        totals["noise_aux_loss"] += noise_aux_loss.item() * size
        totals["consistency_loss"] += consistency_loss.item() * size
        totals["contrastive_loss"] += contrastive_loss.item() * size
        totals["noise_separation_loss"] += noise_sep_loss.item() * size
        totals["residual_separation_loss"] += residual_sep_loss.item() * size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += size
        if batch_index == 1 or batch_index % 200 == 0 or batch_index == len(loader):
            memory = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
            print(
                f"train batch={batch_index}/{len(loader)} loss={totals['loss']/seen:.4f} "
                f"class_loss={totals['class_loss']/seen:.4f} "
                f"noise_aux={totals['noise_aux_loss']/seen:.4f} "
                f"sep_loss={totals['noise_separation_loss']/seen:.4f} "
                f"accuracy={correct/seen:.4f} max_vram_gb={memory:.2f}",
                flush=True,
            )
    return {**{key: value / seen for key, value in totals.items()}, "accuracy": correct / seen}


def trainable_encoder_state(encoder, trainable_blocks: int):
    if not trainable_blocks:
        return {}
    layer_count = len(encoder.encoder.layers)
    prefixes = tuple(
        f"encoder.layers.{index}." for index in range(layer_count - trainable_blocks, layer_count)
    )
    return {
        key: value.detach().cpu().clone()
        for key, value in encoder.state_dict().items()
        if key.startswith(prefixes)
    }


def snapshot(encoder, separator, classifier, trainable_blocks: int) -> dict:
    return {
        "encoder_delta": trainable_encoder_state(encoder, trainable_blocks),
        "separator": separator_payload(separator),
        "classifier": {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("../../36_labels"))
    parser.add_argument("--head-checkpoint", type=Path, default=Path("checkpoint/beats_fusion_head_36.pt"))
    parser.add_argument("--output", type=Path, default=Path("checkpoint/audio_best_36_fusion.pt"))
    parser.add_argument("--results", type=Path, default=Path("checkpoint/summary_36_fusion.json"))
    parser.add_argument("--trainable-blocks", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--fusion-lr", type=float, help="default: same value used in head36")
    parser.add_argument("--fusion-weight-decay", type=float, help="default: same value used in head36")
    parser.add_argument("--encoder-lr", type=float, default=0.0)
    parser.add_argument("--separator-lr", type=float, default=0.0,
                        help="0 keeps the pre-trained separator frozen")
    parser.add_argument("--sep-loss-weight", type=float, default=0.05,
                        help="lambda in L = L_class + lambda * L_sep")
    parser.add_argument("--spectral-loss-weight", type=float, default=0.25,
                        help="weight of multi-resolution STFT regularisation in L_sep")
    parser.add_argument("--residual-sep-loss-weight", type=float, default=0.05,
                        help="lambda for separation of mixture - n_hat from oracle speech")
    parser.add_argument("--noise-aux-weight", type=float, default=0.1,
                        help="lambda for the noise-view auxiliary classifier")
    parser.add_argument("--view-consistency-weight", type=float, default=0.02,
                        help="lambda for mixture/conditioned-logit consistency")
    parser.add_argument("--contrastive-weight", type=float, default=0.02,
                        help="lambda for supervised contrastive noise-view learning")
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--snr-loss-mode", choices=("weighted_ce", "group_dro"), default="weighted_ce")
    parser.add_argument("--snr-group-weights", type=str, default="1,1,1,1,1.5,2")
    parser.add_argument("--group-dro-eta", type=float, default=0.01)
    parser.add_argument("--selection-metric", choices=("macro_f1", "worst_group_macro_f1"),
                        default="worst_group_macro_f1")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-per-class", type=int)
    parser.add_argument("--validation-per-class", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke_test:
        args.train_per_class = 6
        args.validation_per_class = 6
        args.epochs = 1
        args.patience = 1
        args.batch_size = 4
        args.validation_batch_size = 8
        args.accumulation_steps = 2
        args.workers = 0
        args.head_checkpoint = Path("checkpoint/beats_fusion_head_36_smoke.pt")
        args.output = Path("checkpoint/audio_best_36_fusion_smoke.pt")
        args.results = Path("checkpoint/summary_36_fusion_smoke.json")

    rows = load_mix_manifest(args.data_root)
    train_dataset = MixNoiseDataset(
        args.data_root, "train", rows=select_rows(rows, "train", args.train_per_class)
    )
    validation_dataset = MixNoiseDataset(
        args.data_root, "validation",
        rows=select_rows(rows, "validation", args.validation_per_class),
    )
    head_checkpoint = torch.load(args.head_checkpoint, map_location="cpu", weights_only=True)
    encoder, separator, classifier, labels = load_model(
        device, head_checkpoint, args.trainable_blocks,
        dropout=head_checkpoint["args"].get("dropout", 0.1),
        noise_dropout=head_checkpoint["args"].get("noise_dropout", 0.2),
        conditioned_dropout=head_checkpoint["args"].get("conditioned_dropout", 0.2),
    )
    train_separator = args.separator_lr > 0
    for parameter in separator.parameters():
        parameter.requires_grad = train_separator

    # head36 trains the fusion projection slowly with strong decay toward its
    # zero (mixture-only) start; keep that same protection here, otherwise the
    # 36-class head's normal lr/decay lets fusion drift and overfit as soon as
    # the encoder starts moving too. Default to whatever head36 actually used.
    args.fusion_lr = args.fusion_lr if args.fusion_lr is not None else head_checkpoint["args"].get("fusion_lr", 1e-4)
    args.fusion_weight_decay = (
        args.fusion_weight_decay
        if args.fusion_weight_decay is not None
        else head_checkpoint["args"].get("fusion_weight_decay", 0.05)
    )
    fusion_parameters = list(classifier.fusion.parameters())
    fusion_ids = {id(parameter) for parameter in fusion_parameters}

    train_loader = make_loader(train_dataset, args.batch_size, args.workers, True, device, args.seed)
    validation_loader = make_loader(
        validation_dataset, args.validation_batch_size, args.workers, False, device
    )
    robust_loss = SNRRobustLoss(
        args.snr_loss_mode, parse_snr_weights(args.snr_group_weights), args.group_dro_eta, device
    )
    encoder_parameters = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    parameter_groups = [
        {
            "params": [p for p in classifier.parameters() if id(p) not in fusion_ids],
            "lr": args.head_lr,
            "weight_decay": 1e-4,
        },
        {"params": fusion_parameters, "lr": args.fusion_lr, "weight_decay": args.fusion_weight_decay},
    ]
    if encoder_parameters:
        parameter_groups.insert(0, {"params": encoder_parameters, "lr": args.encoder_lr})
    if train_separator:
        parameter_groups.append({"params": separator.parameters(), "lr": args.separator_lr})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-7
    )
    print(
        f"device={device} train={len(train_dataset)} validation={len(validation_dataset)} "
        f"trainable_encoder_params={sum(p.numel() for p in encoder_parameters)} "
        f"train_separator={train_separator} sep_loss_weight={args.sep_loss_weight} "
        f"fusion_lr={args.fusion_lr} fusion_weight_decay={args.fusion_weight_decay} "
        f"snr_loss_mode={args.snr_loss_mode} selection_metric={args.selection_metric}",
        flush=True,
    )

    initial = evaluate(encoder, separator, classifier, validation_loader, device, labels, args.sep_loss_weight)
    print(
        f"epoch=0 val_loss={initial['loss']:.4f} val_accuracy={initial['accuracy']:.4f} "
        f"val_macro_f1={initial['macro_f1']:.4f} val_mAP={initial['mAP']:.4f} val_si_sdr={initial['si_sdr']:.2f}",
        flush=True,
    )
    best_metrics = initial
    best_epoch = 0
    best_state = snapshot(encoder, separator, classifier, args.trainable_blocks)
    history = [{"epoch": 0, "validation": initial}]
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        train_result = train_epoch(
            encoder, separator, classifier, train_loader, optimizer, device, args, train_separator,
            robust_loss,
        )
        train_result["snr_group_weights"] = robust_loss.weights()
        validation_result = evaluate(
            encoder, separator, classifier, validation_loader, device, labels, args.sep_loss_weight
        )
        score = validation_score(validation_result, args.selection_metric)
        scheduler.step(score)
        history.append(
            {
                "epoch": epoch,
                "train": train_result,
                "validation": validation_result,
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
            }
        )
        print(
            f"epoch={epoch} train_loss={train_result['loss']:.4f} "
            f"train_accuracy={train_result['accuracy']:.4f} "
            f"val_loss={validation_result['loss']:.4f} "
            f"val_accuracy={validation_result['accuracy']:.4f} "
            f"val_macro_f1={validation_result['macro_f1']:.4f} val_mAP={validation_result['mAP']:.4f} "
            f"val_worst_snr_f1={validation_score(validation_result, 'worst_group_macro_f1'):.4f} "
            f"val_si_sdr={validation_result['si_sdr']:.2f}",
            flush=True,
        )
        if score > validation_score(best_metrics, args.selection_metric) + 1e-4:
            best_metrics = validation_result
            best_epoch = epoch
            best_state = snapshot(encoder, separator, classifier, args.trainable_blocks)
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early_stop={epoch}", flush=True)
                break

    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **best_state,
            "labels": labels,
            "base_encoder_checkpoint": str(BEATS_CHECKPOINT),
            "base_head_checkpoint": str(args.head_checkpoint),
            "trainable_blocks": args.trainable_blocks,
            "temporal_pooling": TEMPORAL_POOLING,
            "dropout": head_checkpoint["args"].get("dropout", 0.1),
            "noise_dropout": head_checkpoint["args"].get("noise_dropout", 0.2),
            "conditioned_dropout": head_checkpoint["args"].get("conditioned_dropout", 0.2),
            "best_epoch": best_epoch,
            "validation_metrics": best_metrics,
            "args": serializable_args,
        },
        args.output,
    )
    result = {
        "stage": (
            "frozen_encoder_mixture_noise_fusion_finetune"
            if args.trainable_blocks == 0
            else "beats_mixture_noise_fusion_joint_finetune"
        ),
        "input_kind": "mixture+separated_noise",
        "temporal_pooling": TEMPORAL_POOLING,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "trainable_encoder_parameters": sum(parameter.numel() for parameter in encoder_parameters),
        "best_epoch": best_epoch,
        "best_validation": best_metrics,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(args.output),
        "args": serializable_args,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save_training_artifacts(args.output.parent, labels, result)

    encoder.load_state_dict(best_state["encoder_delta"], strict=False)
    if "state" in best_state["separator"]:
        separator.load_state_dict(best_state["separator"]["state"])
    classifier.load_state_dict(best_state["classifier"])
    best_validation, expected, predicted = evaluate(
        encoder, separator, classifier, validation_loader, device, labels,
        args.sep_loss_weight, return_predictions=True,
    )
    save_evaluation_artifacts(
        args.output.parent, best_validation, labels, expected, predicted, split="validation"
    )
    print(
        f"best_epoch={best_epoch} val_accuracy={best_metrics['accuracy']:.4f} "
        f"val_macro_f1={best_metrics['macro_f1']:.4f} val_mAP={best_metrics['mAP']:.4f} val_si_sdr={best_metrics['si_sdr']:.2f}",
        flush=True,
    )
    print(f"checkpoint={args.output} results={args.results}", flush=True)

    del encoder, separator, classifier
    gc.collect()


if __name__ == "__main__":
    main()
