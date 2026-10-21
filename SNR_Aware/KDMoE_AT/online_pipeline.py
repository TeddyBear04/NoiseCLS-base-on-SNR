from __future__ import annotations

import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # src/ và scripts/ của thư mục này (bản chép từ MLNC), không dùng thư mục khác
    sys.path.insert(0, str(HERE))

from scripts.load_dataset import (  # noqa: E402
    KDMoEMixDataset,
    PairedClipSampler,
    build_dataloader,
    subsample,
)
from src.loss import KDMoELossConfig  # noqa: E402
from src.model.frcrn import FRCRNSeparator  # noqa: E402
from src.model.kdmoe import KDMoEConfig, KDMoEModel  # noqa: E402
from src.train import KDMoETrainConfig, KDMoETrainer  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (HERE / path).resolve()


def student_type(student_config: Path) -> str | None:
    """Mục `student.type` của yaml student (vd. "efficientat"); None = encoder của chính KD-MoE."""
    import yaml

    raw = yaml.safe_load(Path(student_config).read_text(encoding="utf-8"))
    return (raw.get("student") or {}).get("type")


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def validate_config(
    config: dict[str, Any], require_dataset: bool = True
) -> dict[str, Path]:
    """paths.teacher_cache: logit teacher đóng gói sẵn (.pt, `teacher_crossfit.py` hoặc model/teacher_midexpert.pt).
    paths.frcrn_cache_dir (tuỳ chọn): n_hat fp32 dựng một lần, thay FRCRN online trong autocast bf16."""
    paths = config["paths"]
    required = {
        "frcrn_checkpoint": resolve_path(paths["frcrn_checkpoint"]),
        "frcrn_config": resolve_path(paths["frcrn_config"]),
        "student_config": resolve_path(paths["student_config"]),
        "group_map": resolve_path(paths["group_map"]),
        "membership": resolve_path(paths["membership"]),
        "teacher_cache": resolve_path(paths["teacher_cache"]),
    }
    if require_dataset:
        required["dataset"] = resolve_path(paths["dataset"])
    missing = [
        f"{name}: {path}" for name, path in required.items() if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))
    if paths.get("frcrn_cache_dir"):
        required["frcrn_cache_dir"] = resolve_path(paths["frcrn_cache_dir"])
    if paths.get("teacher_hints"):  # hint FitNets (.npy + .ids.json) cho train
        required["teacher_hints"] = resolve_path(paths["teacher_hints"])
        if not required["teacher_hints"].is_file():
            raise FileNotFoundError(required["teacher_hints"])
    if paths.get("teacher_review"):  # ReviewKD: token teacher từng layer (.npy + .ids.json), chỉ cho train
        required["teacher_review"] = [resolve_path(item) for item in paths["teacher_review"]]
        missing = [str(item) for item in required["teacher_review"] if not item.is_file()]
        if missing:
            raise FileNotFoundError("Missing teacher_review features:\n" + "\n".join(missing))
    if paths.get("validation_exclude_noise_ytids"):
        required["validation_exclude_noise_ytids"] = resolve_path(paths["validation_exclude_noise_ytids"])
        if not required["validation_exclude_noise_ytids"].is_file():
            raise FileNotFoundError(required["validation_exclude_noise_ytids"])
    required["output"] = resolve_path(paths["output"])
    return required


def load_frcrn(
    config_path: Path, checkpoint_path: Path, device: torch.device
) -> FRCRNSeparator:
    separator = FRCRNSeparator.from_yaml(config_path)
    report = separator.load_pretrained(checkpoint_path)
    if not report.is_clean:
        raise ValueError(
            f"FRCRN checkpoint is incompatible: missing={report.missing[:5]}, unexpected={report.unexpected[:5]}"
        )
    return separator.to(device).freeze()


def build_loaders(
    dataset_root: Path,
    group_map: Path,
    train_config: KDMoETrainConfig,
    data_config: dict[str, Any],
    max_samples: int | None = None,
    model_config: KDMoEConfig | None = None,
    frcrn_cache: Path | None = None,
    teacher_cache: Path | None = None,
    validation_exclude: Path | None = None,
    teacher_hints: Path | None = None,
    review_features: list[Path] | None = None,
) -> dict[str, Any]:
    """frcrn_cache: n_hat fp32 đã tính sẵn thay mixture; teacher_cache: logit teacher đã cache.
    validation_exclude: bỏ khỏi validation các clip nhiễu backbone pretrain đã thấy (chọn epoch trên phần sạch).
    teacher_hints: hint FitNets (.npy) cho train; loss hint chỉ tính trên train.
    review_features: đặc trưng teacher nhiều layer cho ReviewKD (train).

    model_config.route_by == "snr": nhóm expert là khoảng SNR (snr_band_edges) thay cho nhóm lớp.
    """
    edges = model_config.snr_band_edges if model_config and model_config.route_by == "snr" else None

    def dataset(split: str) -> KDMoEMixDataset:
        review = review_features if review_features and split == data_config["train_split"] else None
        if review:
            from reviewkd import ReviewKDMixDataset
        cls, extra = (ReviewKDMixDataset, {"review_features": review}) if review else (KDMoEMixDataset, {})
        return cls(
            **extra,
            root=dataset_root,
            split=split,
            duration=float(data_config["clip_seconds"]),
            teacher_cache=teacher_cache / f"teacher_{split}.npz" if teacher_cache else None,
            load_teacher=teacher_cache is not None,
            require_teacher=teacher_cache is not None,
            group_map=group_map,
            frcrn_cache=frcrn_cache / f"frcrn_{split}.npy" if frcrn_cache else None,
            snr_band_edges=edges,
            exclude_noise_ytids=validation_exclude if split == data_config["validation_split"] else None,
            teacher_hints=teacher_hints if split == data_config["train_split"] else None,
        )

    train = subsample(dataset(data_config["train_split"]), max_samples)
    validation = subsample(dataset(data_config["validation_split"]), max_samples)
    workers = int(data_config["num_workers"])
    sampler = PairedClipSampler(
        train,
        units_per_batch=int(data_config["paired_units_per_batch"]),
        seed=train_config.seed,
    )
    return {
        "train": build_dataloader(
            train, None, num_workers=workers, batch_sampler=sampler
        ),
        "validation": build_dataloader(
            validation, train_config.eval_batch_size, False, workers
        ),
    }


def unpack_teacher(packed: Path, out_dir: Path) -> Path:
    """Logit teacher đã đóng gói (.pt) -> teacher_{split}.npz cho KDMoEMixDataset."""
    from scripts.teacher_cache import save_teacher_cache

    import numpy as np

    data = torch.load(packed, map_location="cpu", weights_only=False)
    for split in ("train", "validation"):
        part = data[split]
        path = save_teacher_cache(out_dir / f"teacher_{split}.npz", part["mix_id"], part["logits36"], part["snr_db"])
        if "at_maps" in part:  # bản đồ Attention Transfer của teacher, đi kèm logit
            with np.load(path) as saved:
                arrays = {name: saved[name] for name in saved.files}
            np.savez(path, **arrays, at_maps=np.asarray(part["at_maps"], dtype=np.float16))
    return out_dir


def _run_stage(
    model: KDMoEModel,
    stage: int,
    output_dir: Path,
    loaders: dict[str, Any],
    train_config: KDMoETrainConfig,
    loss_config: KDMoELossConfig,
    membership: Tensor,
    device: torch.device,
    logger: logging.Logger,
    separator: FRCRNSeparator | None,
    review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stage_dir = output_dir / f"stage{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    best = stage_dir / "best.pt"
    resume = stage_dir / "resume.pt"
    result_path = stage_dir / "result.json"
    trainer_cls, extra = KDMoETrainer, {}
    if review is not None:
        from reviewkd import ReviewKDTrainer

        trainer_cls = ReviewKDTrainer
        extra = {"review_weight": float(review["weight"]), "warmup_epochs": float(review["warmup_epochs"])}
    trainer = trainer_cls(
        model,
        train_config,
        loss_config,
        membership,
        device,
        logger,
        f"online-kdmoe-stage{stage}",
        best,
        stage=stage,
        resume_path=resume,
        separator=separator,
        **extra,
    )
    if resume.is_file():
        trainer.load_resume(resume)
    fit = trainer.fit(loaders["train"], loaders["validation"])
    result: dict[str, Any] = {**fit.to_dict(), "history": trainer.history}
    if stage == 2:
        trainer._in_fit = False
        result["validation"] = trainer.evaluate(
            loaders["validation"], desc="online-kdmoe validation"
        )
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_student(paths: dict[str, Any], review: dict[str, Any] | None, logger: logging.Logger) -> KDMoEModel:
    """KD-MoE theo yaml: ReviewKDMoE (reviewkd.py) khi config có `reviewkd`, EfficientATKDMoE (efficientat.py) khi
    yaml có student.type = efficientat, còn lại KDMoEModel; nạp trọng số pretrained của encoder."""
    if review is not None and student_type(paths["student_config"]) == "efficientat":
        from efficientat import EfficientATReviewKDMoE

        model = EfficientATReviewKDMoE.from_yaml(paths["student_config"])
        loaded = model.load_efficientat_pretrained()
        logger.info("student: KD-MoE EfficientAT %s + ReviewKD (stages %s), %d pretrained tensors",
                    model._student["variant"], model._review_stages, loaded)
    elif review is not None:
        from reviewkd import ReviewKDMoE

        model = ReviewKDMoE.from_yaml(paths["student_config"])
        logger.info("student: KD-MoE PANNs CNN10 + ReviewKD, weight %s, warm-up %s epochs, teacher layers %d",
                    review["weight"], review["warmup_epochs"], len(paths.get("teacher_review", [])))
    elif student_type(paths["student_config"]) == "efficientat":
        from efficientat import EfficientATKDMoE

        model = EfficientATKDMoE.from_yaml(paths["student_config"])
        loaded = model.load_efficientat_pretrained()
        logger.info("student: KD-MoE EfficientAT %s, AudioSet weights, %d tensors", model._student["variant"], loaded)
    else:
        model = KDMoEModel.from_yaml(paths["student_config"])
    if model.config.encoder == "panns_cnn10" and model.config.panns_checkpoint:
        from scripts.frcrn_cache import download_checkpoint

        loaded = model.load_panns_pretrained(download_checkpoint(model.config.panns_checkpoint))
        logger.info("student: PANNs CNN10 AudioSet weights, %d tensors", loaded)
    return model


def train(config: dict[str, Any], smoke: bool = False) -> dict[str, Any]:
    paths = validate_config(config)
    device = resolve_device(config["runtime"]["device"])
    seed = int(config["runtime"]["seed"])
    set_seed(seed)
    logger = logging.getLogger("frcrn-kdmoe")
    student_config = replace(
        KDMoETrainConfig.from_yaml(paths["student_config"]), seed=seed
    )
    max_samples = None
    data_config = dict(config["data"])
    if smoke:
        student_config = replace(
            student_config,
            epochs=1,
            patience=1,
            stage1_epochs=1,
            batch_size=4,
            eval_batch_size=4,
        )
        max_samples = int(config["runtime"]["smoke_samples"])
        data_config["num_workers"] = 0
        data_config["paired_units_per_batch"] = 2

    separator = load_frcrn(paths["frcrn_config"], paths["frcrn_checkpoint"], device)
    output_dir = paths["output"] / ("smoke" if smoke else "train")
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher_cache = unpack_teacher(paths["teacher_cache"], output_dir / "teacher_cache")
    logger.info("teacher: cached logits %s", paths["teacher_cache"])

    frcrn_cache = None
    if "frcrn_cache_dir" in paths:
        import os

        from scripts.frcrn_cache import build_frcrn_cache, build_frcrn_snr

        os.environ["MLNC_MIXTURE_36L"] = str(paths["dataset"])
        frcrn_cache = paths["frcrn_cache_dir"] / ("smoke" if smoke else "")
        build_frcrn_cache(
            separator, device, logger, cache_dir=frcrn_cache, max_samples=max_samples,
            num_workers=int(data_config["num_workers"]),
        )
        build_frcrn_snr(logger, cache_dir=frcrn_cache, num_workers=int(data_config["num_workers"]))
        separator = None  # student đọc n_hat fp32 từ cache
        logger.info("frcrn: fp32 cache %s (no online FRCRN)", frcrn_cache)

    import numpy as np

    membership = torch.from_numpy(np.load(paths["membership"])).float()
    review = config.get("reviewkd")  # ReviewKD: PANNs CNN10 pretrained + ReviewModule (reviewkd.py)
    model = build_student(paths, review, logger)
    loaders = build_loaders(
        paths["dataset"],
        paths["group_map"],
        student_config,
        data_config,
        max_samples,
        model_config=model.config,
        frcrn_cache=frcrn_cache,
        teacher_cache=teacher_cache,
        # smoke: cache FRCRN lấy mẫu trên validation chưa lọc, nên không lọc để hai bên cùng tập clip
        validation_exclude=None if smoke else paths.get("validation_exclude_noise_ytids"),
        teacher_hints=paths.get("teacher_hints"),
        review_features=paths.get("teacher_review"),
    )

    full_loss = KDMoELossConfig.from_yaml(paths["student_config"])
    if review is not None and "kd_weight" in review:  # ghi đè kd_weight của yaml cho riêng run này
        full_loss = replace(full_loss, kd_weight=float(review["kd_weight"]))
    front_only = replace(full_loss, kd_weight=0.0, cons_weight=0.0, route_ce_weight=0.0, at_weight=0.0,
                         hint_weight=0.0)
    stage1_config = replace(student_config, epochs=student_config.stage1_epochs)
    stage1 = _run_stage(
        model,
        1,
        output_dir,
        loaders,
        stage1_config,
        front_only,
        membership,
        device,
        logger,
        separator,
    )
    model.load_pretrained(output_dir / "stage1" / "best.pt")
    stage2 = _run_stage(
        model,
        2,
        output_dir,
        loaders,
        student_config,
        full_loss,
        membership,
        device,
        logger,
        separator,
        review,
    )
    return {"device": str(device), "stage1": stage1, "stage2": stage2}
