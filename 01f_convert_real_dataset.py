"""
Real-world audio dataset converter (step 1f)

Converts WAV files + metadata.csv into the .npz + index.csv format expected by
the existing ML pipeline (scripts 02-06).

Mapping:
  metadata.csv split       -> index.csv split
  metadata.csv label       -> index.csv noise_type   (6 environment classes)
  metadata.csv snr_db      -> index.csv noise_intensity
  metadata.csv speech_src  -> run_id (stable integer hash for group-aware split)

Output structure:
  Real_Noise_Exp/
  â””â”€â”€ {timestamp}_real_data/
      â”œâ”€â”€ metadata/index.csv
      â””â”€â”€ data/noisy/{label}/intensity_{snr_db}/run_{id}.npz
"""

from __future__ import annotations

import os
import hashlib
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import soundfile as sf

# =============================================================================
# User settings
# =============================================================================

DATASET_INPUT_DIR = "dataset_out"  # filepath in CSV is relative to outer dataset_out/
METADATA_CSV = os.path.join("dataset_out", "dataset_out", "metadata.csv")
OUTPUT_BASE = "Real_Noise_Exp"
TARGET_SR = 32000
N_SAMPLES = 96000


def speech_src_to_run_id(speech_src: str) -> int:
    """Stable integer run_id from speech_src via MD5 (mod 1e6)."""
    h = hashlib.md5(speech_src.encode("utf-8")).digest()
    return int.from_bytes(h[:4], byteorder="little", signed=False) % 1_000_000


def convert_dataset() -> Tuple[str, pd.DataFrame]:
    """Read WAVs, save .npz, build and return (output_root, index_df)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = os.path.join(OUTPUT_BASE, f"{timestamp}_real_data")
    metadata_dir = os.path.join(output_root, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    print(f"[1/3] Loading: {METADATA_CSV}")
    df = pd.read_csv(METADATA_CSV)
    print(f"      {len(df)} rows  |  labels: {sorted(df['label'].unique())}")

    index_rows = []
    used_paths = set()
    fail_count = 0

    for k, row in enumerate(df.itertuples(index=False), start=1):
        wav_path = os.path.join(DATASET_INPUT_DIR, row.filepath)
        label = str(row.label)
        snr_db = float(row.snr_db)
        split = str(row.split)
        speech_src = str(row.speech_src)
        run_id = speech_src_to_run_id(speech_src)

        npz_rel = os.path.join(
            "data", "noisy", label,
            f"intensity_{snr_db:.2f}",
            f"run_{run_id:06d}.npz",
        )
        npz_abs = os.path.join(output_root, npz_rel)

        # Guard against MD5 collisions (extremely unlikely with 2757 sources)
        if npz_rel in used_paths:
            raise RuntimeError(f"NPZ path collision: {npz_rel}")
        used_paths.add(npz_rel)

        try:
            audio, sr = sf.read(wav_path, dtype="float64")
            if sr != TARGET_SR:
                print(f"      WARN: sr={sr} in {wav_path}")
            n = len(audio)
            t_eval = np.arange(n, dtype=np.float64) / float(sr)
            os.makedirs(os.path.dirname(npz_abs), exist_ok=True)
            np.savez(npz_abs, t_eval=t_eval, noisy=audio)
        except Exception as exc:
            fail_count += 1
            if fail_count <= 5:
                print(f"      FAILED: {wav_path}  ({exc})")
            continue

        index_rows.append({
            "noise_type": label,
            "split": "noisy",  # signal type (all real data has noise)
            "noise_intensity": snr_db, "run_id": run_id,
            "rep": 0, "signal_id": str(run_id),
            "npz_path": npz_rel, "n_samples": n,
            "speech_src": speech_src,
            "dataset_split": split,  # original train/val/test reference
        })

        if k % 500 == 0:
            print(f"      ... {k}/{len(df)} files converted")

    if fail_count:
        print(f"      WARNING: {fail_count} files failed")

    index_df = pd.DataFrame(index_rows)
    index_csv = os.path.join(metadata_dir, "index.csv")
    print(f"\n[2/3] Saving index.csv: {index_csv}")
    print(f"      {len(index_df)} rows")
    index_df.to_csv(index_csv, index=False)

    print(f"\n[3/3] Done. Output: {output_root}")
    print(f"      Runs: {len(index_df)}  |  Unique run_ids: {index_df['run_id'].nunique()}")
    for s in ["train", "val", "test"]:
        print(f"      {s}: {(index_df['split'] == s).sum()}")
    return output_root, index_df


if __name__ == "__main__":
    convert_dataset()
