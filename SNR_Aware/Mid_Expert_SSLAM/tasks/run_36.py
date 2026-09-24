"""Mid-SNR expert on an SSLAM backbone.

    python -u main.py teacher36 --config config/train_config.json
    python -u main.py student36 --config config/train_config.json --run <run>
    python -u main.py test36    --config config/train_config.json --run <run>
    python -u main.py report36  --config config/train_config.json

Same experiment as `../Mid_Expert`, two things changed:

  * SSLAM replaces BEATs (0.502 vs 0.480 mAP on AudioSet, and pretrained on
    mixtures rather than isolated clips).
  * CRD aligns PATCH TOKENS instead of one pooled vector. The BEATs run threw the
    time-frequency structure away at `sequence.mean(dim=1)` before the loss saw
    it, and that is the most likely reason CRD contributed so little there
    (+0.37 points against KD's +0.92).

There is no `bank36` stage here. A patch-level teacher bank would be
201 tokens x 768 x 43,200 rows, about 13 GB, so the frozen teacher runs online
beside the student instead. That costs roughly 50% more time per epoch, removes
the staleness the BEATs project had to caveat, and matches how SSLAM itself runs
its teacher.

Requires `transformers<5` and `timm`; transformers 5.x cannot load the hub's
remote code. See `models/sslam.py`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent.parent
# This project deliberately does NOT put BEATs_Experts on sys.path. That directory
# carries `config` and `tasks` packages whose names collide with this one's, and
# the collision is silent until an import quietly resolves to the wrong package.
# The two loader functions we used from it live in `models/audio_io.py` instead.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from models.audio_io import load_float_audio, load_mix_manifest  # noqa: E402

from mid_expert_lib import (  # noqa: E402
    CRDLoss, FiLM, Projection, film_deviation, mid_slice_mask, sample_negatives,
)
from models.sslam import SSLAMEncoder, load_sslam, unfreeze_last_blocks  # noqa: E402


#   run1_baseline  the recipe without any privileged information: CE only, no
#                  FiLM, checkpoint chosen on the whole validation split. The
#                  control every other run is measured against, because a
#                  published baseline from a different training run cannot enter
#                  a paired test and cannot separate method from run-to-run noise.
RUNS = {
    "run1_baseline": {"a_kd": 0.0, "b_crd": 0.0, "film": False, "select_on": "full"},
    "run2_ce_only": {"a_kd": 0.0, "b_crd": 0.0, "film": True, "select_on": "mid"},
    "run3_kd_crd": {"a_kd": 1.0, "b_crd": 0.8, "film": True, "select_on": "mid"},
    "run3b_crd_only": {"a_kd": 0.0, "b_crd": 0.8, "film": True, "select_on": "mid"},
    "run3c_kd_only": {"a_kd": 1.0, "b_crd": 0.0, "film": True, "select_on": "mid"},
}
RUN_REQUIRED = ("student36", "test36")
STAGES = ("teacher36", "student36", "test36", "report36")


# --------------------------------------------------------------------------- setup


def load_config(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seed_everything(seed: int) -> torch.device:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    return device


def autocast(device: torch.device):
    return torch.autocast(
        device_type=device.type, dtype=torch.bfloat16,
        enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
    )


def metrics(logits: torch.Tensor, targets: torch.Tensor, snrs: torch.Tensor,
            labels) -> dict:
    """Copied rather than imported: the BEATs version lives in a module that pulls
    in a colliding `config` package. Same computation, same field names, so the
    two projects' numbers line up."""
    predictions = logits.argmax(dim=1).cpu().numpy()
    expected = targets.cpu().numpy()
    indices = np.arange(len(labels))
    precision, recall, class_f1, support = precision_recall_fscore_support(
        expected, predictions, labels=indices, zero_division=0)
    result = {
        "accuracy": float((predictions == expected).mean()),
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "macro_f1": float(class_f1.mean()),
        "micro_f1": float(f1_score(expected, predictions, average="micro",
                                   zero_division=0)),
        "per_class": {label: {"f1": float(class_f1[i]), "support": int(support[i])}
                      for i, label in enumerate(labels)},
        "per_snr": {},
    }
    snr_values = snrs.cpu().numpy()
    for snr in sorted(set(snr_values.tolist())):
        mask = snr_values == snr
        _, _, snr_f1, _ = precision_recall_fscore_support(
            expected[mask], predictions[mask], labels=indices, zero_division=0)
        result["per_snr"][str(snr)] = {
            "samples": int(mask.sum()),
            "accuracy": float((predictions[mask] == expected[mask]).mean()),
            "macro_f1": float(snr_f1.mean()),
            "micro_f1": float(f1_score(expected[mask], predictions[mask],
                                       average="micro", zero_division=0)),
        }
    return result


# ------------------------------------------------------------------------- dataset


class ClipDataset(Dataset):
    """One audio column plus label, SNR and the manifest row it came from.

    Returns both the mixture and the clean noise when `paired` is set, which is
    what lets the frozen teacher run beside the student in the same step.
    """

    def __init__(self, root: Path, rows, row_index, label_to_index, samples, rate,
                 column, paired_column=None):
        self.root = Path(root)
        self.rows = rows
        self.row_index = row_index
        self.label_to_index = label_to_index
        self.samples = samples
        self.rate = rate
        self.column = column
        self.paired_column = paired_column

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        item = {
            "audio": load_float_audio(self.root / row[self.column], self.rate,
                                      self.samples),
            "target": self.label_to_index[row["label_names"]],
            "snr": int(float(row["target_snr_db"])),
            "row_index": self.row_index[index],
        }
        if self.paired_column:
            item["paired"] = load_float_audio(self.root / row[self.paired_column],
                                              self.rate, self.samples)
        return item


class Corpus:
    def __init__(self, config: dict, column: str, paired_column=None):
        dataset = config["dataset"]
        self.root = Path(dataset["path"])
        self.rows = load_mix_manifest(self.root)
        self.labels = (self.root / dataset["labels_file"]).read_text(
            encoding="utf-8").splitlines()
        self.label_to_index = {name: i for i, name in enumerate(self.labels)}
        self.index_of = {id(row): i for i, row in enumerate(self.rows)}
        self.samples = int(dataset["clip_seconds"] * dataset["sample_rate"])
        self.rate = dataset["sample_rate"]
        self.column = column
        self.paired_column = paired_column
        self.by_split: dict[str, list] = {}
        for row in self.rows:
            self.by_split.setdefault(row["split"], []).append(row)

    def loader(self, rows, batch_size, workers, shuffle, device, seed=2026):
        dataset = ClipDataset(self.root, rows,
                              [self.index_of[id(r)] for r in rows],
                              self.label_to_index, self.samples, self.rate,
                              self.column, self.paired_column)
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            generator=torch.Generator().manual_seed(seed) if shuffle else None,
            num_workers=workers, pin_memory=device.type == "cuda")

    def audit(self) -> dict:
        train = self.by_split["train"]
        unique = len({row["noise_path"] for row in train})
        report = {"rows": len(self.rows),
                  "per_split": {k: len(v) for k, v in self.by_split.items()},
                  "labels": len(self.labels), "unique_noise": unique,
                  "reuse_ratio": round(len(train) / unique, 4)}
        print(json.dumps(report, indent=2), flush=True)
        return report


# --------------------------------------------------------------------------- model


class Classifier(nn.Module):
    """SSLAM encoder, optional SNR-FiLM on the pooled vector, then 36 classes.

    `forward` hands back the patch tokens too, because the patch-level CRD term
    is the whole reason this project exists separately from the BEATs one.

    Unlike the BEATs pipeline, the head starts from scratch: SSLAM_pretrain ships
    no classifier, so there are no AudioSet predictor rows to warm-start from.
    Expect the head stage to need its epochs rather than peaking at epoch 1.
    """

    def __init__(self, encoder: SSLAMEncoder, dim: int, classes: int,
                 film: FiLM | None):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(dim, classes)
        self.film = film

    def forward(self, waveform, snr_db=None):
        patches, pooled = self.encoder(waveform)
        if self.film is not None and snr_db is not None:
            pooled = self.film(pooled, snr_db)
        return self.head(pooled), pooled, patches


def build_model(config: dict, device: torch.device, film_enabled: bool,
                trainable_blocks: int) -> Classifier:
    backbone = config["backbone"]
    raw = load_sslam(device, backbone["model_id"])
    trainable = unfreeze_last_blocks(raw, trainable_blocks)
    print(f"unfroze {trainable/1e6:.1f}M params in the last {trainable_blocks} blocks",
          flush=True)
    encoder = SSLAMEncoder(raw, backbone["num_mel_bins"], backbone["norm_divisor"],
                           config["dataset"]["sample_rate"])
    film = FiLM(backbone["embedding_dim"],
                config["student"]["film"]["hidden"]) if film_enabled else None
    model = Classifier(encoder, backbone["embedding_dim"],
                       config["experiment"]["num_classes"], film)
    return model.to(device)


def trainable_parameters(model: Classifier):
    return [p for p in model.parameters() if p.requires_grad]


def snapshot(model: Classifier) -> dict:
    return {k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point}


# -------------------------------------------------------------------------- shared


@torch.inference_mode()
def collect(model: Classifier, loader, device, use_snr: bool) -> dict:
    model.eval()
    logits, targets, snrs, rows, pooled = [], [], [], [], []
    for batch in loader:
        snr = batch["snr"].float().to(device, non_blocking=True) if use_snr else None
        with autocast(device):
            out, vec, _ = model(batch["audio"].to(device, non_blocking=True), snr)
        logits.append(out.float().cpu())
        pooled.append(vec.float().cpu())
        targets.append(batch["target"])
        snrs.append(batch["snr"])
        rows.append(batch["row_index"])
    return {"logits": torch.cat(logits), "target": torch.cat(targets),
            "snr": torch.cat(snrs), "row_index": torch.cat(rows),
            "pooled": torch.cat(pooled)}


def split_metrics(out: dict, labels, band) -> dict:
    full = metrics(out["logits"], out["target"], out["snr"].long(), labels)
    mask = mid_slice_mask(out["snr"].float(), band[0], band[1])
    mid = metrics(out["logits"][mask], out["target"][mask],
                  out["snr"][mask].long(), labels)
    mid["samples"] = int(mask.sum())
    return {"full": full, "mid": mid}


def resolve_stage(name: str):
    from tasks import stages
    return {"teacher36": stages.command_teacher36,
            "student36": stages.command_student36,
            "test36": stages.command_test36}[name] if name != "report36" else (
        __import__("tasks.report_36", fromlist=["command_report36"]).command_report36)


def apply_run(config: dict, name: str) -> dict:
    switches = RUNS[name]
    config["run"] = name
    config["loss"]["a_kd"] = switches["a_kd"]
    config["loss"]["b_crd"] = switches["b_crd"]
    config["student"]["film"]["enabled"] = switches["film"]
    config["student"]["select_on"] = switches["select_on"]
    print(f"run={name} a_kd={switches['a_kd']} b_crd={switches['b_crd']} "
          f"film={switches['film']} select_on={switches['select_on']} "
          f"crd_level={config['loss']['crd_level']}", flush=True)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--config", type=Path,
                        default=HERE / "config/train_config.json")
    parser.add_argument("--run", choices=sorted(RUNS),
                        help="Required for student36 and test36.")
    arguments = parser.parse_args()

    if arguments.stage in RUN_REQUIRED and arguments.run is None:
        parser.error(f"{arguments.stage} needs --run: {', '.join(sorted(RUNS))}")
    if arguments.stage not in RUN_REQUIRED and arguments.run is not None:
        parser.error(f"{arguments.stage} ignores --run; drop it.")

    config = load_config(arguments.config)
    print(f"stage={arguments.stage} config={arguments.config}", flush=True)
    if arguments.run is not None:
        config = apply_run(config, arguments.run)
    resolve_stage(arguments.stage)(config)


if __name__ == "__main__":
    main()
