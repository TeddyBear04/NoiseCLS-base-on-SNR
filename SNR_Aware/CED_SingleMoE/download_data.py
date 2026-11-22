"""Tải dữ liệu + logit teacher cho ablation SplitFFN (final-report.xlsx). Mọi nguồn trên HF đều công khai, có kiểm SHA-256.

    python download_data.py

- mix-dataset-v2 (phitruong0606/mix-dataset-v2): train 30.240 + eval 3.064 clip -> SF_DATA_V2 (paths.data_root)
- 36_labels (Keshio1/36_labels): chỉ giải nén mixture + manifest của train / validation -> SF_DATA_36L,
  dùng để dựng cache n_hat FRCRN (build_nhat.py), đúng như lượt gốc
- logit OOF teacher BEATs 5 fold nghe nhiễu sạch (Keshio1/frcrn-kdmoe-teacher, rev a231940) -> SF_TEACHER (teacher.logits)
File / thư mục đã có thì bỏ qua.
"""
import hashlib
import os
import shutil
import sys
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

DATA_V2 = Path(os.environ.get("SF_DATA_V2", "/marimo/dataset_v2/mix-dataset-v2"))
DATA_36L = Path(os.environ.get("SF_DATA_36L", "/marimo/dataset/mix-dataset"))
TEACHER = Path(os.environ.get("SF_TEACHER", "/marimo/teacher/beats_teacher_logits_clean.pt"))
ZIP_DIR = Path(os.environ.get("SF_ZIP_DIR", "/marimo/zip"))
TEACHER_REV = "a231940eee11a4cc7e55ec311a091749b79ef589"
SHA256 = {
    "beats_teacher_logits_clean.pt": "a52d083a801e7b9fc4c16f5f84213b755436b0951715b9535199c46e36de42b1",
    "mix-dataset-v2.zip": "cfbc6ec89b463e59604ffe69d7a8a41a958bd28b92e5f281a44e54e42e13ffa7",
    "mix-dataset.zip": "528fca42a957da0fe5613acbe165ef30b53a9a219cb6d6307a30c518d6c7ad1c",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(16 << 20), b""):
            h.update(buf)
    return h.hexdigest()


def fetch(repo: str, name: str, repo_type: str, dest_dir: Path, revision: str | None = None) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = Path(hf_hub_download(repo, name, repo_type=repo_type, revision=revision, local_dir=str(dest_dir)))
    got = sha256(path)
    if got != SHA256[name]:
        sys.exit(f"SHA-256 sai cho {name}: {got}")
    print(f"ok {name} ({path.stat().st_size / 2**20:.0f} MB)", flush=True)
    return path


def main() -> None:
    # 1. logit teacher
    if TEACHER.is_file() and sha256(TEACHER) == SHA256[TEACHER.name]:
        print(f"có sẵn {TEACHER}", flush=True)
    else:
        p = fetch("Keshio1/frcrn-kdmoe-teacher", "beats_teacher_logits_clean.pt", "model", TEACHER.parent, TEACHER_REV)
        if p != TEACHER:
            shutil.move(str(p), TEACHER)

    # 2. mix-dataset-v2 (train / eval)
    if (DATA_V2 / "train" / "manifest.csv").is_file() and (DATA_V2 / "eval" / "manifest.csv").is_file():
        print(f"có sẵn {DATA_V2}", flush=True)
    else:
        z = fetch("phitruong0606/mix-dataset-v2", "mix-dataset-v2.zip", "dataset", ZIP_DIR)
        with zipfile.ZipFile(z) as zf:
            target = DATA_V2.parent if zf.namelist()[0].startswith("mix-dataset-v2") else DATA_V2
            target.mkdir(parents=True, exist_ok=True)
            zf.extractall(target)
        z.unlink()
        print(f"giải nén -> {DATA_V2}", flush=True)

    # 3. 36_labels: chỉ mixture + manifest (FRCRN chỉ đọc mixture), bỏ clean / noise / test
    if all((DATA_36L / s / "manifest.csv").is_file() for s in ("train", "validation")):
        print(f"có sẵn {DATA_36L}", flush=True)
    else:
        z = fetch("Keshio1/36_labels", "mix-dataset.zip", "dataset", ZIP_DIR)
        keep = ("mix-dataset/train/mixture/", "mix-dataset/validation/mixture/")

        def wanted(n: str) -> bool:
            parts = n.split("/")
            if n.endswith("/"):
                return False
            if len(parts) <= 3:  # manifest, labels.txt, summary ở gốc và ở từng split
                return True
            return n.startswith(keep)

        with zipfile.ZipFile(z) as zf:
            members = [n for n in zf.namelist() if wanted(n)]
            zf.extractall(DATA_36L.parent, members=members)
        z.unlink()
        print(f"giải nén {len(members)} file -> {DATA_36L}", flush=True)

    for p in (DATA_V2 / "train" / "manifest.csv", DATA_V2 / "eval" / "manifest.csv",
              DATA_36L / "train" / "manifest.csv", DATA_36L / "validation" / "manifest.csv", TEACHER):
        if not p.is_file():
            sys.exit(f"thiếu {p}")
    print("DATA_OK", flush=True)


if __name__ == "__main__":
    main()
