"""
Dual BEATs-12L Teacher Utilities for Logit-Only Knowledge Distillation (KD).

Teachers are FROZEN, run in inference mode only during precomputation/training,
and are completely excluded from evaluation and inference.

  * Low-SNR Expert Teacher  <- BEATs fold0 teacher (beats_teacher_fold0_weights.pt)
  * High-SNR Expert Teacher <- BEATs EMA teacher   (beats_teacher_ema_standalone.pt)

During inference and validation, ONLY the Student Dual-Expert CED (~10.83M params) runs.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import src.beats as _beats_pkg
from src.beats.beat import BEATClassifier
from src.frontend import rms_normalize

logger = logging.getLogger(__name__)

FOLD_PREFIX = "encoder.model."


def _register_legacy_aliases() -> None:
    """Teacher checkpoints were pickled with original module paths."""
    for name in ("src.model", "src.model.beat", "src.model.beat.beat"):
        sys.modules.setdefault(name, _beats_pkg)


def resolve_teacher_checkpoint(ckpt_path_or_name: str | Path) -> Path:
    """Locate teacher checkpoint locally or in standard deployment directories."""
    cand = Path(ckpt_path_or_name)
    if cand.is_file():
        return cand.resolve()

    cand_str = str(ckpt_path_or_name).replace("\\", "/")
    for prefix in ("/D:/PROJECT/MLNC/", "D:/PROJECT/MLNC/", "/D:/", "D:/"):
        if cand_str.startswith(prefix):
            sub = cand_str[len(prefix):].lstrip("/")
            cand_linux = Path("/marimo") / sub
            if cand_linux.is_file():
                return cand_linux.resolve()
            cand_root = Path("/") / sub
            if cand_root.is_file():
                return cand_root.resolve()

    filename = cand.name
    normalized_path = cand_str.casefold()
    checkpoint_aliases: list[str] = []
    if "mix_original_ld0p8" in normalized_path:
        checkpoint_aliases.append("mix_original_ld0p8_weights.pt")
    if "mix_frcrn_ld0p3" in normalized_path:
        checkpoint_aliases.append("mix_frcrn_ld0p3_best_model.pt")
    if not checkpoint_aliases and filename == "best_model.pt":
        checkpoint_aliases.append("mix_frcrn_ld0p3_best_model.pt")
    search_dirs = [
        Path("/marimo/imp/beats/outputs/four_config_ld_20261109/mix_original_ld0p8"),
        Path("/marimo/imp/beats/outputs/four_config_ld_20261109/mix_frcrn_ld0p3"),
        Path("/marimo/pretrained/beat-ft"),
        Path("pretrained/beat-ft"),
        Path.cwd() / "pretrained" / "beat-ft",
        Path.cwd().parent / "pretrained" / "beat-ft",
        Path("imp/beats/outputs/four_config_ld_20261109/mix_original_ld0p8"),
        Path("imp/beats/outputs/four_config_ld_20261109/mix_frcrn_ld0p3"),
        Path("/D:/PROJECT/MLNC/imp/beats/outputs/four_config_ld_20261109/mix_original_ld0p8"),
        Path("/D:/PROJECT/MLNC/imp/beats/outputs/four_config_ld_20261109/mix_frcrn_ld0p3"),
    ]

    for d in search_dirs:
        p = d / filename
        if p.is_file():
            return p.resolve()
        for alias in checkpoint_aliases:
            alt = d / alias
            if alt.is_file():
                return alt.resolve()

    # Fallback to HuggingFace dataset download if not found
    try:
        from huggingface_hub import hf_hub_download
        logger.info("Downloading %s from HuggingFace dataset phitruong0606/mix-dataset-v2...", filename)
        dl_path = hf_hub_download("phitruong0606/mix-dataset-v2", filename, repo_type="dataset")
        return Path(dl_path).resolve()
    except Exception as exc:
        raise FileNotFoundError(f"Cannot resolve teacher checkpoint {ckpt_path_or_name}: {exc}")


def load_beats_teacher(
    checkpoint_path: str | Path,
    config_path: str | Path,
    device: torch.device,
) -> tuple[BEATClassifier, dict[str, Any]]:
    """Loads a frozen BEATs teacher from fold0, standalone EMA, or fine-tuned checkpoint."""
    _register_legacy_aliases()
    resolved_ckpt = resolve_teacher_checkpoint(checkpoint_path)
    ckpt = torch.load(resolved_ckpt, map_location="cpu", weights_only=False)
    teacher = BEATClassifier.from_yaml(config_path)

    if isinstance(ckpt, dict) and "model" in ckpt and any(k.startswith("backbone.") for k in ckpt["model"]):
        fmt = "standalone_ema"
        state = {k: v.float() if torch.is_tensor(v) else v for k, v in ckpt["model"].items()}
        result = teacher.load_state_dict(state, strict=False)
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        fmt = "finetuned_beats"
        sd = ckpt["model_state_dict"]
        backbone = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        converted = teacher.convert_checkpoint({"model": backbone})
        converted["head.fc.weight"] = sd["head.weight"]
        converted["head.fc.bias"] = sd["head.bias"]
        result = teacher.load_state_dict(converted, strict=True)
    else:
        fmt = "fold"
        sd = ckpt.get("state_dict")
        if sd is None or not any(k.startswith(FOLD_PREFIX) for k in sd):
            raise ValueError(f"{resolved_ckpt}: expected 'model_state_dict', 'model', or '{FOLD_PREFIX}*' keys")
        backbone = {k[len(FOLD_PREFIX):]: v for k, v in sd.items() if k.startswith(FOLD_PREFIX)}
        converted = teacher.convert_checkpoint({"model": backbone})
        converted["head.fc.weight"] = sd["head.weight"]
        converted["head.fc.bias"] = sd["head.bias"]
        result = teacher.load_state_dict(converted, strict=False)

    if result.missing_keys or result.unexpected_keys:
        logger.warning("Teacher load warning: missing=%s, unexpected=%s", result.missing_keys[:3], result.unexpected_keys[:3])

    for p in teacher.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in teacher.parameters())
    return teacher.to(device).eval(), {"format": fmt, "params": n_params, "path": str(resolved_ckpt)}


@torch.no_grad()
def compute_teacher_logits(
    teacher: nn.Module,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    input_rms: float = 0.07925,
    amp: bool = True,
    desc: str = "Teacher Logits",
    waveform_key: str = "waveform",
) -> np.ndarray:
    """Compute teacher logits [N, C] over dataset with RMS normalization."""
    from tqdm import tqdm
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    out = []
    use_amp = amp and device.type == "cuda"
    for batch in tqdm(loader, desc=desc, leave=False):
        wave_data = batch[waveform_key] if waveform_key in batch else batch["waveform"]
        wave = rms_normalize(wave_data.to(device), target_rms=input_rms)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = teacher(wave).logits
        out.append(logits.float().cpu().numpy())
    arr = np.concatenate(out, axis=0)
    if not np.isfinite(arr).all():
        raise FloatingPointError("Non-finite teacher logits produced")
    return arr


def get_or_compute_dual_teacher_logits(
    config: dict[str, Any],
    train_dataset: Dataset,
    device: torch.device,
    output_dir: str | Path,
    logger_inst: logging.Logger | None = None,
) -> Tensor:
    """
    Load or precompute dual-teacher logits [N, 2, C] for the training dataset:
      Index 0: Teacher Low (e.g. mix_original_ld0p8 on Raw Mix)
      Index 1: Teacher High (e.g. mix_frcrn_ld0p3 on FRCRN Mix)
    """
    log = logger_inst or logger
    n_samples = len(train_dataset)
    num_classes = int(config.get("experiment", {}).get("num_classes", 36))

    # Fast path for synthetic / small smoke tests
    if n_samples <= 32 or getattr(train_dataset, "is_synthetic", False):
        log.info("Smoke test / small dataset detected: using random teacher logits for KD pipeline check.")
        rng = np.random.default_rng(2026)
        arr = rng.standard_normal((n_samples, 2, num_classes)).astype(np.float32)
        return torch.from_numpy(arr).float()

    kd_cfg = config.get("kd", {})
    t_cfg = config.get("teacher", {})
    ckpt_low = t_cfg.get("checkpoint_low", "pretrained/beat-ft/beats_teacher_fold0_weights.pt")
    ckpt_high = t_cfg.get("checkpoint_high", "pretrained/beat-ft/beats_teacher_ema_standalone.pt")
    wf_low = t_cfg.get("waveform_key_low", "waveform_raw")
    wf_high = t_cfg.get("waveform_key_high", "waveform_frcrn")

    tag0 = Path(str(ckpt_low)).stem
    tag1 = Path(str(ckpt_high)).stem
    tag = f"{tag0}_{wf_low}_{tag1}_{wf_high}"

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"teacher_logits_train_{n_samples}_{tag}.npy"

    if cache_path.is_file():
        arr = np.load(cache_path)
        if arr.shape == (n_samples, 2, num_classes) and np.isfinite(arr).all():
            log.info("Loaded cached dual-teacher logits from %s %s", cache_path.name, arr.shape)
            return torch.from_numpy(arr).float()

    # Teacher paths and configs
    beat_cfg_path = t_cfg.get("beat_config", "imp/dual-expert/config/beat.yaml")
    cand_beat_cfg = Path(beat_cfg_path)
    if not cand_beat_cfg.is_file():
        alt = Path("/marimo/imp/dual-expert/config/beat.yaml")
        if alt.is_file():
            beat_cfg_path = alt
        else:
            beat_cfg_path = Path.cwd() / "imp" / "dual-expert" / "config" / "beat.yaml"

    teacher_bs = int(kd_cfg.get("teacher_batch_size", 64))
    input_rms = float(kd_cfg.get("input_rms", 0.07925))

    # Check if both teachers share identical checkpoint AND input waveform
    same_teacher = (str(ckpt_low) == str(ckpt_high) or tag0 == tag1) and (wf_low == wf_high)

    # 1. Compute Teacher 0 (Low-SNR Specialist on wf_low)
    log.info("Loading Teacher 0 from %s (input: %s)...", ckpt_low, wf_low)
    teacher0, info0 = load_beats_teacher(ckpt_low, beat_cfg_path, device)
    log.info("Teacher 0 loaded: %s params (%.2fM). Computing train logits on %s...", f"{info0['params']:,}", info0['params'] / 1e6, wf_low)
    logits_low = compute_teacher_logits(teacher0, train_dataset, teacher_bs, device, input_rms=input_rms, desc=f"Teacher 0 ({wf_low})", waveform_key=wf_low)
    del teacher0
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 2. Compute Teacher 1 (High-SNR Specialist on wf_high)
    if same_teacher:
        log.info("Teacher 0 and Teacher 1 share identical checkpoint and input. Reusing computed logits!")
        logits_high = logits_low
    else:
        log.info("Loading Teacher 1 from %s (input: %s)...", ckpt_high, wf_high)
        teacher1, info1 = load_beats_teacher(ckpt_high, beat_cfg_path, device)
        log.info("Teacher 1 loaded: %s params (%.2fM). Computing train logits on %s...", f"{info1['params']:,}", info1['params'] / 1e6, wf_high)
        logits_high = compute_teacher_logits(teacher1, train_dataset, teacher_bs, device, input_rms=input_rms, desc=f"Teacher 1 ({wf_high})", waveform_key=wf_high)
        del teacher1
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 3. Stack and save to cache
    arr = np.stack([logits_low, logits_high], axis=1)  # [N, 2, 36]
    np.save(cache_path, arr)
    log.info("Saved dual-teacher logits to cache: %s (shape: %s)", cache_path.name, list(arr.shape))
    return torch.from_numpy(arr).float()

