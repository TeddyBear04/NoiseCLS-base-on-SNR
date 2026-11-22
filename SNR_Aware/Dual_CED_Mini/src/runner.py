"""
Runner module for Dual-Expert CED with Dual BEATs Teachers Knowledge Distillation (KD):
- Student Model (Inference & Deployment): Dual-Expert CED (~10.85M parameters)
    * Expert 1 (Low-SNR):  ViTSBias CED-Tiny (~5.42M)
    * Expert 2 (High-SNR): ViTSBias CED-Tiny (~5.42M)
    * Router: Soft MoE Gating Router (arXiv:2308.00951)
    * Inference cost: 10.85M parameters, fast & lightweight, NO Teacher needed.
- Teacher Models (Training Only):
    * Low-SNR Teacher:  BEATs Fold 0 teacher (beats_teacher_fold0_weights.pt, ~90.3M)
    * High-SNR Teacher: BEATs EMA teacher    (beats_teacher_ema_standalone.pt, ~90.3M)
    * Used ONLY during training for Logit-Only KD, completely excluded from validation/inference.
- Centered MoE SNR Transition at 7.5 dB for balanced 50/50 specialist allocation.
- Equalized Multi-Task KD (50% Fused KD + 50% Specialized Expert KD).
- Gentle Layer-wise Learning Rate Decay (LLRD, layer_decay = 0.92) to preserve low-level features.
- 5-Way Multi-Scale Temporal TTA (center, +-50ms, +-100ms) for robust inference.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .data import build_datasets, load_class_mids, load_class_names
from .metrics import classification_metrics, snr_metrics
from .frontend import rms_normalize
from .mgd import MGDHead, student_tokens_to_map, teacher_tokens_to_map
from .models import DualExpertCED
from .teachers import get_or_compute_dual_teacher_logits, load_beats_teacher

LOGGER = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _amp_context(device: torch.device, amp: str):
    enabled = device.type == "cuda" and amp.casefold() in {"bf16", "bfloat16", "fp16", "float16"}
    dtype = torch.float16 if amp.casefold() in {"fp16", "float16"} else torch.bfloat16
    return torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def apply_mixup_with_teachers(
    waveform_raw: Tensor,
    waveform_frcrn: Tensor,
    target: Tensor,
    snr_db: Tensor,
    t0_logits: Tensor,
    tema_logits: Tensor,
    alpha: float = 0.4,
    prob: float = 0.8,
    temp: float = 2.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Apply Mixup data augmentation simultaneously across dual waveforms (raw and frcrn),
    one-hot ground-truth labels, continuous SNR values, and soft teacher distributions (Beyer et al., CVPR 2022).
    """
    p_t0 = F.softmax(t0_logits / temp, dim=-1)
    p_tema = F.softmax(tema_logits / temp, dim=-1)

    if np.random.rand() > prob or alpha <= 0.0:
        return waveform_raw, waveform_frcrn, target, snr_db, p_t0, p_tema

    batch_size = waveform_raw.shape[0]
    perm = torch.randperm(batch_size, device=waveform_raw.device)
    lam = float(np.random.beta(alpha, alpha))

    mixed_wave_raw = lam * waveform_raw + (1.0 - lam) * waveform_raw[perm]
    mixed_wave_frcrn = lam * waveform_frcrn + (1.0 - lam) * waveform_frcrn[perm]
    mixed_target = lam * target + (1.0 - lam) * target[perm]
    mixed_snr = lam * snr_db + (1.0 - lam) * snr_db[perm]
    mixed_pt0 = lam * p_t0 + (1.0 - lam) * p_t0[perm]
    mixed_ptema = lam * p_tema + (1.0 - lam) * p_tema[perm]

    return mixed_wave_raw, mixed_wave_frcrn, mixed_target, mixed_snr, mixed_pt0, mixed_ptema


def _build_param_groups(
    model: DualExpertCED,
    encoder_lr: float = 6.0e-5,
    head_lr: float = 1.5e-4,
    layer_decay: float = 0.92,
    weight_decay: float = 0.01,
    trainable_blocks: int = 12,
    router_lr: float | None = None,
    decay_embeddings: bool = True,
) -> list[dict[str, Any]]:
    """
    Constructs Layer-wise Learning Rate Decay (LLRD) parameter groups for both experts.
    Applies gentle decay layer_decay^(11 - l) to blocks 0..11, and handles router and heads.
    """
    num_blocks = 12
    first_trainable = max(0, num_blocks - trainable_blocks)
    if router_lr is None:
        router_lr = 2.5e-4

    layer_scales = [layer_decay ** (num_blocks - 1 - i) for i in range(num_blocks)]
    # decay_embeddings False: patch/pos embeddings keep the full encoder lr, as the top block does
    # (layer-decay rule of SNR_Expert_SSLAM / FRCRN_KD_MoE, used for the cross-method ablation).
    embed_scale = layer_decay ** num_blocks if decay_embeddings else 1.0

    grouped_params: dict[tuple[float, float], list[nn.Parameter]] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # No weight decay for 1D biases and LayerNorm parameters
        is_no_decay = param.ndim <= 1 or name.endswith(".bias")
        wd = 0.0 if is_no_decay else weight_decay

        if "head" in name:
            lr = head_lr
        elif "router" in name or "acoustic_extractor" in name:
            lr = router_lr
        elif ".blocks." in name:
            parts = name.split(".")
            block_idx = int(parts[parts.index("blocks") + 1])
            if block_idx < first_trainable:
                param.requires_grad_(False)
                continue
            lr = encoder_lr * layer_scales[block_idx]
        elif "patch_embed" in name or "pos_embed" in name or "cls_token" in name:
            lr = encoder_lr * embed_scale
        elif ".norm." in name:
            lr = encoder_lr
        else:
            lr = encoder_lr

        grouped_params.setdefault((round(lr, 10), wd), []).append(param)

    groups = []
    for (lr_val, wd), params in grouped_params.items():
        groups.append({"params": params, "lr": lr_val, "weight_decay": wd})

    if not groups:
        groups.append({"params": list(model.parameters()), "lr": encoder_lr, "weight_decay": weight_decay})
    return groups


def run_tta_predict(
    model: DualExpertCED,
    wave_raw: Tensor,
    wave_frcrn: Tensor | None = None,
    device: torch.device | None = None,
    shift_samples: int = 800,
) -> Tensor:
    """5-Way Multi-Scale Temporal TTA (center, +-50ms, +-100ms) for Dual-Input Student inference."""
    if device is None:
        device = wave_raw.device
    if wave_frcrn is None:
        wave_frcrn = wave_raw

    if shift_samples <= 0 or wave_raw.shape[-1] <= 2 * shift_samples:
        return model(waveform_raw=wave_raw.to(device), waveform_frcrn=wave_frcrn.to(device))

    views_raw = [
        wave_raw,
        F.pad(wave_raw[:, shift_samples:], (0, shift_samples), mode="reflect"),
        F.pad(wave_raw[:, :-shift_samples], (shift_samples, 0), mode="reflect"),
        F.pad(wave_raw[:, 2 * shift_samples:], (0, 2 * shift_samples), mode="reflect"),
        F.pad(wave_raw[:, :-2 * shift_samples], (2 * shift_samples, 0), mode="reflect"),
    ]
    views_frcrn = [
        wave_frcrn,
        F.pad(wave_frcrn[:, shift_samples:], (0, shift_samples), mode="reflect"),
        F.pad(wave_frcrn[:, :-shift_samples], (shift_samples, 0), mode="reflect"),
        F.pad(wave_frcrn[:, 2 * shift_samples:], (0, 2 * shift_samples), mode="reflect"),
        F.pad(wave_frcrn[:, :-2 * shift_samples], (2 * shift_samples, 0), mode="reflect"),
    ]

    batch_size = wave_raw.shape[0]
    stacked_raw = torch.cat(views_raw, dim=0).to(device)
    stacked_frcrn = torch.cat(views_frcrn, dim=0).to(device)
    with torch.no_grad():
        logits = model(waveform_raw=stacked_raw, waveform_frcrn=stacked_frcrn)
    v0, v1, v2, v3, v4 = logits.reshape(5, batch_size, -1).unbind(0)
    return 0.40 * v0 + 0.15 * (v1 + v2) + 0.15 * (v3 + v4)


@torch.no_grad()
def _evaluate(
    model: DualExpertCED,
    loader: DataLoader,
    device: torch.device,
    description: str,
    amp: str = "bf16",
    use_tta: bool = False,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Inference and evaluation on Student Dual-Expert CED only (NO Teacher)."""
    model.eval()
    all_logits = []
    all_targets = []
    all_snrs = []

    for batch in tqdm(loader, desc=f"Evaluating {description}", leave=False):
        wave_raw = batch.get("waveform_raw", batch["waveform"]).to(device, non_blocking=True)
        wave_frcrn = batch.get("waveform_frcrn", batch["waveform"]).to(device, non_blocking=True)
        targets = batch["target_idx"].to(device, non_blocking=True)
        snrs = batch["snr_db"].to(device, non_blocking=True)

        with _amp_context(device, amp):
            if use_tta:
                logits = run_tta_predict(model, wave_raw, wave_frcrn, device, shift_samples=800)
            else:
                logits = model(waveform_raw=wave_raw, waveform_frcrn=wave_frcrn)

        all_logits.append(logits.float())
        all_targets.append(targets)
        all_snrs.append(snrs)

    concat_logits = torch.cat(all_logits, dim=0)
    concat_targets = torch.cat(all_targets, dim=0)
    concat_snrs = torch.cat(all_snrs, dim=0)

    overall = classification_metrics(concat_logits, concat_targets, num_classes=36)
    by_snr = snr_metrics(concat_logits, concat_targets, concat_snrs, num_classes=36)
    return overall, by_snr


def run_smoke_test(
    config_dir: str | Path,
    checkpoint_dir: str | Path,
    max_samples: int = 4,
    config_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verification forward pass of Student Dual-Expert CED (~10.85M params)."""
    config = config_override if config_override is not None else load_config(config_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = build_datasets(config, max_samples=max_samples)

    labels_path = Path(config["dataset"]["path"]) / config["dataset"]["labels_file"]
    class_names = load_class_names(labels_path)
    class_mids = load_class_mids(datasets["train"])

    student_cfg = config.get("student", {})
    arch = student_cfg.get("arch", "ced_mini")
    embed_dim = int(student_cfg.get("embed_dim", 256))
    pretrained_src = student_cfg.get("pretrained_source", "mispeech/ced-mini")

    model = DualExpertCED(
        num_classes=int(config["experiment"]["num_classes"]),
        arch=arch,
        embed_dim=embed_dim,
        router_mode=config.get("router", {}).get("mode", "soft_moe_cosine"),
        transition_center_db=float(config.get("router", {}).get("transition_center_db", 7.5)),
        pretrained=True,
        pretrained_source=pretrained_src,
        class_names=class_names,
        class_mids=class_mids,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    low_params = sum(p.numel() for p in model.low_expert.parameters())
    high_params = sum(p.numel() for p in model.high_expert.parameters())
    router_params = sum(p.numel() for p in model.router.parameters())

    loader = DataLoader(datasets["validation"], batch_size=min(2, max_samples), shuffle=False)
    batch = next(iter(loader))

    with torch.no_grad():
        out = model(batch["waveform"].to(device), return_details=True)

    logits = out["logits"]
    alpha = out["alpha"]
    combine_weights = out["combine_weights"]

    return {
        "status": "ok",
        "device": str(device),
        "model": "DualExpertCED (Student)",
        "student_parameters": {
            "total": total_params,
            "total_m": round(total_params / 1e6, 2),
            "low_expert_m": round(low_params / 1e6, 2),
            "high_expert_m": round(high_params / 1e6, 2),
            "router_params": router_params,
        },
        "inference_role": "ONLY Student DualExpertCED is executed during inference/validation.",
        "logits_shape": list(logits.shape),
        "soft_moe_combine_weights": combine_weights.cpu().numpy().tolist(),
        "mean_high_alpha": float(alpha.mean().item()),
    }


def run_training(
    config_dir: str | Path,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    config_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Supervised training of Student Dual-Expert CED with Dual-BEATs Teacher KD.
    Validation and final inference evaluate ONLY the Student (~10.85M params).
    """
    config = config_override if config_override is not None else load_config(config_dir)
    seed = int(config.get("experiment", {}).get("seed", 2026))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pkg_dir = str(Path(__file__).resolve().parent.parent)
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)
    cur_pypath = os.environ.get("PYTHONPATH", "")
    if pkg_dir not in cur_pypath.split(os.pathsep):
        os.environ["PYTHONPATH"] = f"{pkg_dir}{os.pathsep}{cur_pypath}" if cur_pypath else pkg_dir

    try:
        mp.set_sharing_strategy("file_system")
    except Exception:
        pass

    datasets = build_datasets(config)
    train_cfg = config.get("finetuning", {})
    batch_size = int(train_cfg.get("batch_size", 16))
    val_batch_size = int(train_cfg.get("validation_batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 0))
    epochs = int(train_cfg.get("epochs", 15))
    accumulation_steps = max(1, int(train_cfg.get("accumulation_steps", 2)))
    patience = int(train_cfg.get("patience", 4))
    encoder_lr = float(train_cfg.get("encoder_lr", 6.0e-5))
    head_lr = float(train_cfg.get("head_lr", 1.5e-4))
    router_lr = float(train_cfg.get("router_lr", 2.5e-4))
    layer_decay = float(train_cfg.get("layer_decay", 0.92))
    trainable_blocks = int(train_cfg.get("trainable_blocks", 12))
    amp = str(train_cfg.get("amp", "bf16"))

    mixup_alpha = float(train_cfg.get("mixup_alpha", 0.4))
    mixup_prob = float(train_cfg.get("mixup_prob", 0.8))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.05))

    # Optional OOF split: train on one parity of the train indices only (teacher logits stay indexed by the full set).
    kd_opts = config.get("kd", {})
    train_train_ds = datasets["train"]
    if kd_opts.get("train_parity") is not None:
        parity = int(kd_opts["train_parity"])
        keep = [i for i in range(len(train_train_ds)) if i % 2 == parity]
        train_train_ds = torch.utils.data.Subset(train_train_ds, keep)
        LOGGER.info("OOF split: training on %d of %d train clips (index parity %d)", len(keep), len(datasets["train"]), parity)
    difficulty_w: Tensor | None = None
    if kd_opts.get("high_difficulty_weights"):
        difficulty_w = torch.from_numpy(np.load(kd_opts["high_difficulty_weights"])).float().to(device)
        LOGGER.info("High expert KD weighted by per-clip difficulty (mean %.3f, max %.3f)", float(difficulty_w.mean()), float(difficulty_w.max()))
    train_loader = DataLoader(
        train_train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )
    val_loader = DataLoader(
        datasets["validation"],
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    labels_path = Path(config["dataset"]["path"]) / config["dataset"]["labels_file"]
    class_names = load_class_names(labels_path)
    class_mids = load_class_mids(datasets["train"])

    router_cfg = config.get("router", {})
    trans_center = float(router_cfg.get("transition_center_db", 7.5))
    trans_tau = float(router_cfg.get("transition_tau", 3.0))

    # 1. Student Model: Dual-Expert CED
    student_cfg = config.get("student", {})
    arch = student_cfg.get("arch", "ced_mini")
    embed_dim = int(student_cfg.get("embed_dim", 256))
    pretrained_src = student_cfg.get("pretrained_source", "mispeech/ced-mini")

    student = DualExpertCED(
        num_classes=int(config["experiment"]["num_classes"]),
        arch=arch,
        embed_dim=embed_dim,
        router_mode=router_cfg.get("mode", "soft_moe_cosine"),
        transition_center_db=trans_center,
        pretrained=True,
        pretrained_source=pretrained_src,
        class_names=class_names,
        class_mids=class_mids,
    ).to(device)

    total_student_params = sum(p.numel() for p in student.parameters())
    low_expert_params = sum(p.numel() for p in student.low_expert.parameters())
    high_expert_params = sum(p.numel() for p in student.high_expert.parameters())
    router_params = sum(p.numel() for p in student.router.parameters())

    LOGGER.info("================================================================================")
    LOGGER.info("STUDENT MODEL: Dual-Expert CED (%s, ~%.2fM parameters)", arch.upper(), total_student_params / 1e6)
    LOGGER.info("  * Expert 1 (Low-SNR):  ViTSBias %s (%.2fM parameters)", arch.upper(), low_expert_params / 1e6)
    LOGGER.info("  * Expert 2 (High-SNR): ViTSBias %s (%.2fM parameters)", arch.upper(), high_expert_params / 1e6)
    LOGGER.info("  * Soft MoE Router:     Puigcerver et al. (%d parameters)", router_params)
    LOGGER.info("  * Centered MoE SNR:    %.1f dB (tau: %.1f dB) for 50/50 specialist allocation", trans_center, trans_tau)
    LOGGER.info("  * Layer Decay:         %.2f (Gentle LLRD across both experts)", layer_decay)
    LOGGER.info("  * Encoder LR:          %.2e | Head LR: %.2e | Router LR: %.2e | Epochs: %d", encoder_lr, head_lr, router_lr, epochs)
    LOGGER.info("  * Inference Role:      ONLY Student runs during validation and evaluation!",)
    LOGGER.info("================================================================================")

    # 2. Knowledge Distillation: Precompute or load Teacher Logits [N, 2, C]
    kd_cfg = config.get("kd", {})
    kd_enabled = bool(kd_cfg.get("enabled", True))
    teacher_logits = None
    if kd_enabled:
        LOGGER.info("Preparing Dual BEATs Teachers (Fold0 + EMA) for Logit-Only Distillation...")
        teacher_logits = get_or_compute_dual_teacher_logits(
            config=config,
            train_dataset=datasets["train"],
            device=device,
            output_dir=output_dir,
            logger_inst=LOGGER,
        )
        LOGGER.info("Teacher logits ready: shape %s. Teachers fully unloaded from GPU memory.", list(teacher_logits.shape))

    # 3. Optimizer with Layer-wise Learning Rate Decay (LLRD)
    param_groups = _build_param_groups(
        student,
        encoder_lr=encoder_lr,
        head_lr=head_lr,
        router_lr=router_lr,
        layer_decay=layer_decay,
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        trainable_blocks=trainable_blocks,
        decay_embeddings=bool(train_cfg.get("layer_decay_embeddings", True)),
    )
    # Optional Masked Generative Distillation of the Low expert's last block from BEATs layer 12 (FRCRN teacher).
    mgd_alpha = float(kd_cfg.get("mgd_alpha", 0.0))
    mgd_head: MGDHead | None = None
    mgd_teacher: nn.Module | None = None
    mgd_feat: dict[str, Tensor] = {}
    if mgd_alpha > 0.0 and teacher_logits is not None:
        t_cfg = config.get("teacher", {})
        beat_cfg_path = Path(t_cfg.get("beat_config", "imp/dual-expert/config/beat.yaml"))
        if not beat_cfg_path.is_file():
            beat_cfg_path = Path("/marimo/imp/dual-expert/config/beat.yaml")
        mgd_teacher, _ = load_beats_teacher(t_cfg["checkpoint_low"], beat_cfg_path, device)
        mgd_head = MGDHead(
            student_dim=student.embed_dim, teacher_dim=768, alpha=mgd_alpha, mask_ratio=float(kd_cfg.get("mgd_mask_ratio", 0.5))
        ).to(device)
        student.low_expert.vit.blocks[-1].register_forward_hook(lambda _m, _i, out: mgd_feat.__setitem__("low", out))
        param_groups.append(
            {"params": list(mgd_head.parameters()), "lr": head_lr, "weight_decay": float(train_cfg.get("weight_decay", 0.01))}
        )
        LOGGER.info("MGD enabled: alpha=%.1e, mask=%.2f, Low expert last block <- BEATs layer 12 (%s)", mgd_alpha, mgd_head.mask_ratio, t_cfg["checkpoint_low"])
    optimizer = torch.optim.AdamW(param_groups)

    steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, round(total_steps * float(train_cfg.get("warmup_fraction", 0.08))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / warmup_steps
        if step < warmup_steps
        else 0.5
        * (1 + math.cos(math.pi * min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)))),
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp.casefold() in {"fp16"}))
    router_loss_weight = float(router_cfg.get("router_loss_weight", 0.20))
    kd_tau = float(kd_cfg.get("temperature", 2.0))
    lambda_kd = float(kd_cfg.get("lambda_kd", 0.65))
    fused_kd_weight = float(kd_cfg.get("fused_kd_weight", 0.50))
    expert_kd_weight = float(kd_cfg.get("specialized_kd_weight", 0.50))
    # Share of the Fold0/FRCRN teacher in the High expert's KD target (0 = EMA/raw teacher only, the original recipe).
    high_teacher_mix = float(kd_cfg.get("high_teacher_mix", 0.0))
    transition_center = float(kd_cfg.get("snr_transition_center_db", trans_center))
    transition_tau = float(kd_cfg.get("snr_transition_tau_db", trans_tau))

    selection_key = str(train_cfg.get("selection", "composite")).casefold()
    best_score = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    history: list[dict[str, Any]] = []
    top_checkpoints: list[dict[str, Any]] = []
    top_k_max = int(train_cfg.get("swa_top_k", 3))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    best_path = output_path / "best_model.pt"

    LOGGER.info("Starting Dual-Expert CED training for %d epochs (Student: %.2fM params)...", epochs, total_student_params / 1e6)

    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0.0
        total_ce = 0.0
        total_kd = 0.0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for step_idx, batch in enumerate(pbar):
            wave_raw = batch.get("waveform_raw", batch["waveform"]).to(device, non_blocking=True)
            wave_frcrn = batch.get("waveform_frcrn", wave_raw).to(device, non_blocking=True)
            target_idx = batch["target_idx"].to(device, non_blocking=True)
            snrs = batch["snr_db"].to(device, non_blocking=True)
            indices = batch["index"].long()
            sample_w = difficulty_w[indices.to(device)] if difficulty_w is not None else None

            target_one_hot = F.one_hot(target_idx, num_classes=36).float()

            if teacher_logits is not None:
                tl = teacher_logits[indices].to(device)  # [B, 2, C]
                t0_log = tl[:, 0, :]
                tema_log = tl[:, 1, :]
            else:
                t0_log = target_one_hot
                tema_log = target_one_hot

            # Synchronized Mixup across Dual Waveforms (Raw & FRCRN), Targets, SNR, and Teacher Logits
            wave_raw_m, wave_frcrn_m, target_m, snr_m, pt0_m, ptema_m = apply_mixup_with_teachers(
                wave_raw,
                wave_frcrn,
                target_one_hot,
                snrs,
                t0_log,
                tema_log,
                alpha=mixup_alpha,
                prob=mixup_prob,
                temp=kd_tau,
            )

            # Label smoothing regularizer
            if label_smoothing > 0.0:
                target_smooth = target_m * (1.0 - label_smoothing) + label_smoothing / 36.0
            else:
                target_smooth = target_m

            # Continuous SNR blend weight centered at 7.5 dB
            w_snr = torch.sigmoid((snr_m.float() - transition_center) / transition_tau).unsqueeze(-1)
            p_teacher_blend = (1.0 - w_snr) * pt0_m + w_snr * ptema_m

            with _amp_context(device, amp):
                details = student(waveform_raw=wave_raw_m, waveform_frcrn=wave_frcrn_m, snr_db=snr_m, return_details=True)
                logits = details["logits"]
                logits_low = details["logits_low"]
                logits_high = details["logits_high"]
                alpha = details["alpha"]

                # 1. Supervised Task Cross-Entropy with smoothed targets
                log_p_fused = F.log_softmax(logits, dim=-1)
                loss_ce = -(target_smooth * log_p_fused).sum(dim=-1).mean()

                # 2. Dual-Teacher Blended KD (T^2 scaling)
                loss_kd = torch.tensor(0.0, device=device)
                if teacher_logits is not None:
                    kd_scale = kd_tau ** 2
                    # Fused KD matches blended teacher distribution
                    log_p_fused_kd = F.log_softmax(logits / kd_tau, dim=-1)
                    loss_kd_fused = kd_scale * -(p_teacher_blend * log_p_fused_kd).sum(dim=-1).mean()

                    # Specialized KD: low expert specializes on Fold0, high expert on EMA
                    log_p_low_kd = F.log_softmax(logits_low / kd_tau, dim=-1)
                    log_p_high_kd = F.log_softmax(logits_high / kd_tau, dim=-1)
                    loss_low_kd = kd_scale * -(pt0_m * log_p_low_kd).sum(dim=-1)
                    p_teacher_high = (1.0 - high_teacher_mix) * ptema_m + high_teacher_mix * pt0_m
                    loss_high_kd = kd_scale * -(p_teacher_high * log_p_high_kd).sum(dim=-1)
                    if sample_w is None:
                        loss_kd_experts = ((1.0 - w_snr.squeeze(-1)) * loss_low_kd + w_snr.squeeze(-1) * loss_high_kd).mean()
                    else:
                        # Difficulty weighting applies only to the High expert's KD term (mixup: weight of the primary clip).
                        loss_kd_experts = ((1.0 - w_snr.squeeze(-1)) * loss_low_kd).mean() + (sample_w * w_snr.squeeze(-1) * loss_high_kd).mean()

                    loss_kd = fused_kd_weight * loss_kd_fused + expert_kd_weight * loss_kd_experts

                # 3. Soft MoE Router guidance loss
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    loss_router = F.binary_cross_entropy(alpha.float().clamp(1e-6, 1.0 - 1e-6), w_snr.float())

                # Combined Multi-Task Objective
                raw_loss = (1.0 - lambda_kd) * loss_ce + lambda_kd * loss_kd + router_loss_weight * loss_router
                if mgd_head is not None and mgd_teacher is not None:
                    with torch.no_grad():
                        teacher_hidden = mgd_teacher(rms_normalize(wave_frcrn_m, target_rms=0.07925), layers=(12,)).hidden[0]
                    freq_p, time_p = student.low_expert.freq_patches, student.low_expert.time_patches
                    loss_mgd = mgd_head(
                        student_tokens_to_map(mgd_feat["low"], freq_p, time_p), teacher_tokens_to_map(teacher_hidden, freq_p, time_p)
                    )
                    raw_loss = raw_loss + loss_mgd
                loss = raw_loss / accumulation_steps

            scaler.scale(loss).backward()
            if (step_idx + 1) % accumulation_steps == 0 or (step_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                clip_params = list(student.parameters()) + (list(mgd_head.parameters()) if mgd_head is not None else [])
                torch.nn.utils.clip_grad_norm_(clip_params, float(train_cfg.get("gradient_clip_norm", 5.0)))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += raw_loss.item()
            total_ce += loss_ce.item()
            total_kd += loss_kd.item()
            pbar.set_postfix({"loss": f"{raw_loss.item():.4f}", "ce": f"{loss_ce.item():.4f}", "kd": f"{loss_kd.item():.4f}"})

        # Validation after epoch (evaluates STUDENT ONLY)
        val_metrics, val_by_snr = _evaluate(student, val_loader, device, f"Epoch {epoch} val", amp=amp)
        macro_f1 = val_metrics["macro_f1"]
        val_acc = val_metrics["acc"]
        LOGGER.info(
            "Epoch %d/%d | Train Loss: %.4f | Val Acc: %.4f | Val Macro-F1: %.4f | Val mAP: %.4f",
            epoch, epochs, total_loss / max(1, len(train_loader)), val_acc, macro_f1, val_metrics["mAP"]
        )

        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(1, len(train_loader)),
            "val_acc": val_acc,
            "val_macro_f1": macro_f1,
            "val_mAP": val_metrics["mAP"],
        })

        if selection_key in ("acc", "accuracy", "val_acc"):
            score = val_acc
        elif selection_key in ("composite", "harmonic"):
            score = 0.5 * (val_acc + macro_f1)
        else:
            score = macro_f1

        # Record checkpoint state for SWA Weight-Space Averaging
        cpu_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
        top_checkpoints.append({"epoch": epoch, "score": score, "state": cpu_state})
        top_checkpoints.sort(key=lambda x: x["score"], reverse=True)
        if len(top_checkpoints) > top_k_max:
            top_checkpoints = top_checkpoints[:top_k_max]

        if score > best_score:
            best_score = score
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": student.state_dict(),
                "val_macro_f1": macro_f1,
                "val_acc": val_acc,
                "val_metrics": val_metrics,
                "val_by_snr": val_by_snr,
            }, best_path)
            LOGGER.info("=> New best checkpoint saved at epoch %d (%s: %.4f, val acc: %.4f, val macro-F1: %.4f)", epoch, selection_key, score, val_acc, macro_f1)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                LOGGER.info("Early stopping triggered after %d epochs without improvement.", patience)
                break

    # Restore best model for final evaluation
    if best_path.is_file():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["model_state"])

    # 1. Single Best Model Evaluation (Overlapping Window Inference)
    use_tta = bool(train_cfg.get("tta_enabled", True))
    swa_enabled = bool(train_cfg.get("swa_enabled", True))
    val_single, val_by_snr_single = _evaluate(student, val_loader, device, f"Best Epoch {best_epoch} (Overlapping)", amp=amp, use_tta=use_tta)
    LOGGER.info("=> Single Best Model (Epoch %d): Acc=%.4f | Macro-F1=%.4f | mAP=%.4f", best_epoch, val_single["acc"], val_single["macro_f1"], val_single["mAP"])

    final_val = val_single
    final_val_by_snr = val_by_snr_single
    ensemble_info = f"Single Best Checkpoint (Epoch {best_epoch})"

    # 2. SWA / Checkpoint Weight-Space Averaging (Zero extra params, Zero latency increase)
    if swa_enabled and len(top_checkpoints) >= 2:
        top_epochs = [c["epoch"] for c in top_checkpoints]
        LOGGER.info("Applying SWA Weight-Space Averaging across top %d checkpoints (Epochs %s)...", len(top_checkpoints), top_epochs)
        swa_state = {}
        for key in top_checkpoints[0]["state"]:
            swa_state[key] = sum(c["state"][key].to(device).float() for c in top_checkpoints) / len(top_checkpoints)

        student.load_state_dict(swa_state)
        torch.save({"epochs": top_epochs, "model_state": swa_state}, output_path / "swa_model.pt")

        val_swa, val_by_snr_swa = _evaluate(student, val_loader, device, f"SWA Top-{len(top_checkpoints)} Averaged (Overlapping)", amp=amp, use_tta=use_tta)
        LOGGER.info("=> SWA Ensemble Model (Top %d: Epochs %s): Acc=%.4f | Macro-F1=%.4f | mAP=%.4f", len(top_checkpoints), top_epochs, val_swa["acc"], val_swa["macro_f1"], val_swa["mAP"])

        if val_swa["acc"] >= val_single["acc"] or val_swa["macro_f1"] >= val_single["macro_f1"]:
            LOGGER.info("=> SWA Ensemble outperformed or matched Single Best! Promoting SWA to final champion model.")
            final_val = val_swa
            final_val_by_snr = val_by_snr_swa
            ensemble_info = f"SWA Weight-Space Ensemble (Top {len(top_checkpoints)} Epochs: {top_epochs})"
            torch.save({"epoch": f"SWA-{top_epochs}", "model_state": swa_state, "val_metrics": val_swa, "val_by_snr": val_by_snr_swa}, best_path)
        else:
            # Restore single best if it was slightly higher
            student.load_state_dict(ckpt["model_state"])

    param_breakdown = {
        "student_total": total_student_params,
        "student_total_m": round(total_student_params / 1e6, 2),
        "low_expert_m": round(low_expert_params / 1e6, 2),
        "high_expert_m": round(high_expert_params / 1e6, 2),
        "soft_moe_router_params": router_params,
        "teacher_fold0_m": 90.34,
        "teacher_ema_m": 90.34,
        "inference_only_student": True,
    }

    results = {
        "model": f"Dual-Expert CED ({arch.upper()}) (Soft MoE)",
        "teachers": "BEATs Fold0 (Low) + BEATs EMA (High) [KD Training Only]",
        "parameters_m": round(total_student_params / 1e6, 2),
        "param_breakdown": param_breakdown,
        "selection_metric": f"validation {selection_key}",
        "best_epoch": best_epoch,
        "history": history,
        "validation": final_val,
        "validation_by_snr": final_val_by_snr,
    }

    (output_path / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def show_result(results_path: str | Path) -> None:
    """Print validation metrics grouped by SNR with Student parameter counts."""
    path = Path(results_path)
    if not path.is_file():
        raise FileNotFoundError(f"Results file not found: {path}")

    results = json.loads(path.read_text(encoding="utf-8"))
    columns = ("acc", "precision", "recall", "macro_f1", "micro_f1", "mAP")

    params_m = results.get("parameters_m") or results.get("param_breakdown", {}).get("student_total_m", "10.85")
    best_epoch = results.get("best_epoch", "N/A")
    model_name = results.get("model", "Dual-Expert CED (Soft MoE)")
    teachers_name = results.get("teachers", "BEATs Fold0 + EMA (KD Training Only)")

    print("=" * 100)
    print(f"Model: {model_name} | Inference Params (Student): {params_m}M | Best Epoch: {best_epoch}")
    print(f"Teachers (KD Training Only): {teachers_name} | Zero Teacher Footprint in Inference")
    if "param_breakdown" in results:
        pb = results["param_breakdown"]
        print(f"Student Breakdown: Low Expert: {pb.get('low_expert_m')}M | High Expert: {pb.get('high_expert_m')}M | Router: {pb.get('soft_moe_router_params')} params")
    print("=" * 100)

    print("| SNR (dB) | " + " | ".join(columns) + " |")
    print("| -------- | " + " | ".join("--------" for _ in columns) + " |")

    snr_dict = results.get("validation_by_snr") or results.get("test_by_snr") or {}
    order = ["-5", "0", "5", "10", "15", "20", "ALL"]
    for snr in order:
        if snr in snr_dict:
            m = snr_dict[snr]
            vals = [f"{m[c]:.4f}" if c in m else "—" for c in columns]
            print(f"| {snr:<8} | " + " | ".join(f"{v:<8}" for v in vals) + " |")
    print("=" * 100)

