"""Cache n_hat = FRCRN(mixture) fp32 và tải checkpoint pretrained (PANNs) cho KDMoE_ReviewKD.

`build_frcrn_cache` và `download_checkpoint` chép nguyên văn từ MLNC (scripts/run_kdmoe.py,
scripts/load_pretrain.py), bỏ phần Drive/finetune của MLNC không dùng ở đây.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from tqdm import tqdm

from scripts.load_dataset import Mixture36Dataset, build_dataloader, subsample
from src.model.frcrn import FRCRNSeparator
from src.model.kdmoe import frcrn_snr_db

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation")
CLIP_SECONDS = 4.0
FRCRN_CACHE_DIR = ROOT / "output" / "frcrn_cache_fp32"
PRETRAINED_DIR = ROOT / "pretrained"

logger = logging.getLogger(__name__)


def build_frcrn_cache(
    separator: FRCRNSeparator,
    device: torch.device,
    logger: logging.Logger,
    *,
    cache_dir: str | Path = FRCRN_CACHE_DIR,
    max_samples: int | None = None,
    num_workers: int = 4,
    batch_size: int = 64,
    splits: tuple[str, ...] = SPLITS,
) -> Path:
    """n_hat = FRCRN(mixture) in fp32 for every clip of `splits`: frcrn_{split}.npy [N, samples] + .json sample ids.

    Mixtures are read whole (4 s, no random crop) and FRCRN is frozen, so n_hat per clip is fixed and the
    student can read it instead of running FRCRN every step. A split is skipped when its ids match.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    separator.to(device).eval()
    for split in splits:
        dataset = Mixture36Dataset(split=split, duration=CLIP_SECONDS)
        chosen = subsample(dataset, max_samples)
        indices = chosen.indices if hasattr(chosen, "indices") else range(len(dataset))
        ids = [dataset.rows[i]["sample_id"] for i in indices]
        array_path, ids_path = cache_dir / f"frcrn_{split}.npy", cache_dir / f"frcrn_{split}.json"
        if ids_path.is_file() and json.loads(ids_path.read_text(encoding="utf-8")) == ids:
            logger.info("frcrn cache %s: already built (%d clips)", split, len(ids))
            continue

        (cache_dir / f"frcrn_{split}.snr.npy").unlink(missing_ok=True)  # SNR của cache cũ không còn đúng
        partial = cache_dir / f"frcrn_{split}.partial.npy"
        out = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=(len(ids), dataset.num_samples))
        row = 0
        tf32 = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False  # fp32 thật, không TF32
        try:
            for batch in build_dataloader(chosen, batch_size, False, num_workers):
                with torch.autocast(device_type=device.type, enabled=False):
                    n_hat = separator(batch["waveform"].to(device).float())
                out[row : row + len(n_hat)] = n_hat.cpu().numpy()
                row += len(n_hat)
        finally:
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf32
        out.flush()
        del out
        partial.replace(array_path)
        ids_path.write_text(json.dumps(ids), encoding="utf-8")  # viết sau cùng: có .json là cache đủ
        logger.info("frcrn cache %s: %d clips (fp32) -> %s", split, row, array_path)
    return cache_dir


def build_frcrn_snr(
    logger: logging.Logger,
    *,
    cache_dir: str | Path = FRCRN_CACHE_DIR,
    num_workers: int = 4,
    batch_size: int = 64,
    splits: tuple[str, ...] = SPLITS,
) -> Path:
    """SNR (dB) từ năng lượng FRCRN cho mỗi clip đã cache: frcrn_{split}.snr.npy [N], cùng thứ tự frcrn_{split}.json.

    frcrn_snr_db(mixture, n_hat) ở mức gốc, trước rms_normalize, cho route_source frcrn_snr. Chỉ đọc mixture và
    cache n_hat (CPU). Bỏ qua split đã có; build_frcrn_cache xoá file này khi dựng lại cache.
    """
    cache_dir = Path(cache_dir)
    for split in splits:
        snr_path = cache_dir / f"frcrn_{split}.snr.npy"
        if snr_path.is_file():
            logger.info("frcrn snr %s: already built", split)
            continue
        ids = json.loads((cache_dir / f"frcrn_{split}.json").read_text(encoding="utf-8"))
        dataset = Mixture36Dataset(split=split, duration=CLIP_SECONDS)
        position = {row["sample_id"]: i for i, row in enumerate(dataset.rows)}
        chosen = torch.utils.data.Subset(dataset, [position[sample_id] for sample_id in ids])
        n_hat = np.load(cache_dir / f"frcrn_{split}.npy", mmap_mode="r")
        out = np.empty(len(ids), dtype=np.float32)
        row = 0
        for batch in build_dataloader(chosen, batch_size, False, num_workers):
            size = len(batch["sample_id"])
            if list(batch["sample_id"]) != ids[row : row + size]:
                raise ValueError(f"{split}: mixture order differs from frcrn_{split}.json at row {row}")
            out[row : row + size] = frcrn_snr_db(batch["waveform"], torch.from_numpy(np.array(n_hat[row : row + size]))).numpy()
            row += size
        np.save(snr_path, out)
        logger.info("frcrn snr %s: %d clips, mean %.2f dB -> %s", split, row, float(out.mean()), snr_path)
    return cache_dir


def download_checkpoint(url: str, out_dir: Path = PRETRAINED_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / Path(urlparse(url).path).name
    if destination.exists():
        logger.info("Checkpoint already exists: %s", destination)
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with (
            open(partial, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=destination.name) as progress,
        ):
            for chunk in response.iter_content(1 << 20):
                f.write(chunk)
                progress.update(len(chunk))
    partial.rename(destination)
    return destination
