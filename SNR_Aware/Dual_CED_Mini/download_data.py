"""Tải dữ liệu + teacher cho Dual-CED-Mini từ HF dataset phitruong0606/mix-dataset-v2 (công khai).

    python download_data.py

- 2 teacher BEATs fine-tune: mix_frcrn_ld0p3_best_model.pt (Low, nghe n_hat FRCRN), mix_original_ld0p8_weights.pt (High, nghe mixture)
- cache-nhat.zip -> train.npy / validation.npy (n_hat FRCRN, đúng thứ tự manifest)
- mix-dataset-v2.zip -> train/, eval/ (wav + manifest.csv)
Đường dẫn giống dual_ablation.py (DUAL_DATASET, DUAL_FRCRN, DUAL_TEACHER_DIR). Kiểm SHA-256; file đã có thì bỏ qua.
"""
import hashlib
import os
import shutil
import sys
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "phitruong0606/mix-dataset-v2"
DATASET = Path(os.environ.get("DUAL_DATASET", "/datasets/mix-dataset-v2"))
FRCRN = Path(os.environ.get("DUAL_FRCRN", "/datasets/frcrn"))
TEACHER_DIR = Path(os.environ.get("DUAL_TEACHER_DIR", "/marimo/pretrained/beat-ft"))
ZIP_DIR = Path(os.environ.get("DUAL_ZIP_DIR", "/marimo/zip"))
SHA256 = {
    "mix_frcrn_ld0p3_best_model.pt": "6aa647b652e9da7e99abfd18a916fda7b2f01edf1daf123e909f333970abb2b0",
    "mix_original_ld0p8_weights.pt": "aeae3b2da6d8bde3aca32ac9468ac07fd17ca9650aced34f5192f6ef06249ddb",
    "cache-nhat.zip": "45c29f251b96ddd4e117799bfe94f9f0200480c8c08233fa47ac89e393b50b3b",
    "mix-dataset-v2.zip": "cfbc6ec89b463e59604ffe69d7a8a41a958bd28b92e5f281a44e54e42e13ffa7",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(16 << 20), b""):
            h.update(buf)
    return h.hexdigest()


def fetch(name: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = Path(hf_hub_download(REPO, name, repo_type="dataset", local_dir=str(dest_dir)))
    got = sha256(path)
    if got != SHA256[name]:
        sys.exit(f"SHA-256 sai cho {name}: {got}")
    print(f"ok {name} ({path.stat().st_size / 2**20:.0f} MB)", flush=True)
    return path


def main() -> None:
    for name in ("mix_frcrn_ld0p3_best_model.pt", "mix_original_ld0p8_weights.pt"):
        p = TEACHER_DIR / name
        if p.is_file() and sha256(p) == SHA256[name]:
            print(f"có sẵn {p}", flush=True)
        else:
            fetch(name, TEACHER_DIR)

    if (FRCRN / "train.npy").is_file() and (FRCRN / "validation.npy").is_file():
        print(f"có sẵn n_hat FRCRN ở {FRCRN}", flush=True)
    else:
        z = fetch("cache-nhat.zip", ZIP_DIR)
        FRCRN.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            for member in zf.namelist():
                if member.endswith(("train.npy", "validation.npy", ".done")):
                    with zf.open(member) as src, open(FRCRN / Path(member).name, "wb") as dst:
                        shutil.copyfileobj(src, dst, 16 << 20)
        z.unlink()
        print(f"giải nén n_hat FRCRN -> {FRCRN}", flush=True)

    if (DATASET / "train" / "manifest.csv").is_file() and (DATASET / "eval" / "manifest.csv").is_file():
        print(f"có sẵn dataset ở {DATASET}", flush=True)
    else:
        z = fetch("mix-dataset-v2.zip", ZIP_DIR)
        with zipfile.ZipFile(z) as zf:
            # zip có thể chứa thư mục gốc mix-dataset-v2/ hoặc không
            target = DATASET.parent if zf.namelist()[0].startswith("mix-dataset-v2") else DATASET
            target.mkdir(parents=True, exist_ok=True)
            zf.extractall(target)
        z.unlink()
        print(f"giải nén dataset -> {DATASET}", flush=True)

    for p in (DATASET / "train" / "manifest.csv", DATASET / "eval" / "manifest.csv",
              FRCRN / "train.npy", FRCRN / "validation.npy"):
        if not p.is_file():
            sys.exit(f"thiếu {p}")
    print("DATA_OK", flush=True)


if __name__ == "__main__":
    main()
