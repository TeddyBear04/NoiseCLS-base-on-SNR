"""Task 5  -  train the teacher on clean noise, then precompute its embedding bank.

Two stages, run as two commands so a failure in one does not cost the other:

    python -u main.py teacher36 --config config/train_config.json
    python -u main.py bank36    --config config/train_config.json

The teacher sees ``noise_path``  -  the clean noise waveform. That is privileged
information (Lopez-Paz, Bottou, Scholkopf, Vapnik, ICLR 2016): it exists only at
training time. The student trained later sees the mixture and nothing else, so
nothing here leaks into inference.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

HERE = Path(__file__).resolve().parent.parent
BEATS_PROJECT = HERE.parent / "BEATs_Experts"

# Order matters. Both projects have a `config/` directory, and `train_beats_head`
# does `from config.paths import BEATS_CHECKPOINT`  -  so BEATs_Experts has to come
# first or our own `config/` shadows it and that import dies. Our code never
# imports `config` as a module; it reads the JSON by path.
for _path in (HERE, BEATS_PROJECT):
    if str(_path) in sys.path:
        sys.path.remove(str(_path))
    sys.path.insert(0, str(_path))

from noise_pipeline.mix_data import load_float_audio, load_mix_manifest  # noqa: E402
from train_beats_head import initialize_head, load_beats, metrics  # noqa: E402


# --------------------------------------------------------------------------- setup


def load_config(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_pretrained(candidates: list[str]) -> Path:
    """First existing candidate, or a failure that says why it is missing."""
    for candidate in candidates:
        path = (HERE / candidate).resolve() if not Path(candidate).is_absolute() else Path(candidate)
        if path.exists():
            print(f"pretrained={path} ({path.stat().st_size / 1e6:.0f} MB)", flush=True)
            return path
        print(f"  not found: {path}", flush=True)
    raise FileNotFoundError(
        "No BEATs pretrained checkpoint found.\n"
        "`*.pt` is gitignored, so it never arrives with `git clone`  -  the file has to be\n"
        "on this machine already. Add the right path to pretrained.candidates in the config."
    )


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
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
    )


# ------------------------------------------------------------------------- dataset


class NoiseOnlyDataset(Dataset):
    """Clean-noise waveform plus label, and the manifest row it came from.

    ``row_index`` is what keeps the bank aligned with ``manifest.csv``. The student
    looks the bank up by that index; a one-row shift would pair every mixture with
    some other clip's noise and raise nothing at all  -  the run would just score
    worse and the method would take the blame.
    """

    def __init__(self, root: Path, rows, row_index, label_to_index, samples, rate, column):
        self.root = Path(root)
        self.rows = rows
        self.row_index = row_index
        self.label_to_index = label_to_index
        self.samples = samples
        self.rate = rate
        self.column = column

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        return {
            "audio": load_float_audio(self.root / row[self.column], self.rate, self.samples),
            "target": self.label_to_index[row["label_names"]],
            "snr": int(float(row["target_snr_db"])),
            "row_index": self.row_index[index],
        }


class Corpus:
    """Manifest, labels, and the row-order the bank must preserve."""

    def __init__(self, config: dict, column: str | None = None):
        dataset = config["dataset"]
        self.root = Path(dataset["path"])
        self.rows = load_mix_manifest(self.root)
        self.labels = (self.root / dataset["labels_file"]).read_text(
            encoding="utf-8"
        ).splitlines()
        self.label_to_index = {name: i for i, name in enumerate(self.labels)}
        self.index_of = {id(row): i for i, row in enumerate(self.rows)}
        self.samples = int(dataset["clip_seconds"] * dataset["sample_rate"])
        self.rate = dataset["sample_rate"]
        self.column = column or config["teacher"]["source_column"]
        self.by_split: dict[str, list] = {}
        for row in self.rows:
            self.by_split.setdefault(row["split"], []).append(row)

        self.label_to_mid: dict[str, str] = {}
        for row in self.rows:
            self.label_to_mid.setdefault(row["label_names"], row["label_mids"])

    def dataset(self, rows) -> NoiseOnlyDataset:
        return NoiseOnlyDataset(
            self.root, rows, [self.index_of[id(row)] for row in rows],
            self.label_to_index, self.samples, self.rate, self.column,
        )

    def loader(self, rows, batch_size, workers, shuffle, device, seed=2026):
        return DataLoader(
            self.dataset(rows), batch_size=batch_size, shuffle=shuffle,
            generator=torch.Generator().manual_seed(seed) if shuffle else None,
            num_workers=workers, pin_memory=device.type == "cuda",
        )

    def audit(self) -> dict:
        train = self.by_split["train"]
        unique = len({row[self.column] for row in train})
        report = {
            "rows": len(self.rows),
            "per_split": {k: len(v) for k, v in self.by_split.items()},
            "labels": len(self.labels),
            "unique_source": unique,
            "reuse_ratio": round(len(train) / unique, 4),
        }
        print(json.dumps(report, indent=2), flush=True)
        if abs(report["reuse_ratio"] - 1.0) > 0.01:
            print(
                f"WARNING reuse_ratio={report['reuse_ratio']} differs from the audited 1.0; "
                "revisit the no-dedup decision before trusting this run.",
                flush=True,
            )
        return report


# -------------------------------------------------------------------------- stages


@torch.inference_mode()
def extract(model, corpus, rows, device, batch_size, workers, tag):
    loader = corpus.loader(rows, batch_size, workers, shuffle=False, device=device)
    xs, ys, snrs, idx = [], [], [], []
    started = time.perf_counter()
    for step, batch in enumerate(loader, start=1):
        with autocast(device):
            sequence, _ = model.extract_features(batch["audio"].to(device, non_blocking=True))
        xs.append(sequence.mean(dim=1).float().cpu())
        ys.append(batch["target"].long())
        snrs.append(batch["snr"].long())
        idx.append(batch["row_index"].long())
        if step == 1 or step % 100 == 0 or step == len(loader):
            print(f"extract {tag} {step}/{len(loader)}", flush=True)
    return {
        "x": torch.cat(xs), "y": torch.cat(ys), "snr": torch.cat(snrs),
        "row_index": torch.cat(idx), "seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def head_logits(head, data, device, chunk=2048):
    return torch.cat([
        head(data["x"][i:i + chunk].to(device)).cpu()
        for i in range(0, len(data["x"]), chunk)
    ])


def train_head(head, train_data, validation_data, labels, device, config):
    teacher = config["teacher"]
    loader = DataLoader(
        TensorDataset(train_data["x"], train_data["y"]),
        batch_size=teacher["head_batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(config["experiment"]["seed"]),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=teacher["head_lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )

    best = metrics(head_logits(head, validation_data, device),
                   validation_data["y"], validation_data["snr"], labels)
    best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    history, stale = [], 0
    epochs = 1 if config["runtime"]["smoke_test"] else teacher["head_epochs"]
    print(f"head epoch=0 val_accuracy={best['accuracy']:.4f} "
          f"val_macro_f1={best['macro_f1']:.4f}", flush=True)

    for epoch in range(1, epochs + 1):
        head.train()
        total = correct = seen = 0
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            logits = head(features)
            loss = criterion(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * targets.shape[0]
            correct += (logits.argmax(1) == targets).sum().item()
            seen += targets.shape[0]
        head.eval()
        current = metrics(head_logits(head, validation_data, device),
                          validation_data["y"], validation_data["snr"], labels)
        scheduler.step(current["macro_f1"])
        history.append({"epoch": epoch, "train_loss": total / seen,
                        "train_accuracy": correct / seen,
                        "val_accuracy": current["accuracy"],
                        "val_macro_f1": current["macro_f1"]})
        print(f"head epoch={epoch} loss={total / seen:.4f} "
              f"train_accuracy={correct / seen:.4f} "
              f"val_accuracy={current['accuracy']:.4f} "
              f"val_macro_f1={current['macro_f1']:.4f}", flush=True)
        if current["macro_f1"] > best["macro_f1"] + 1e-4:
            best, stale = current, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
            if stale >= teacher["head_patience"]:
                print(f"head early_stop={epoch}", flush=True)
                break
    head.load_state_dict(best_state)
    return best, history


def encoder_state(model, trainable_blocks: int) -> dict:
    first = len(model.encoder.layers) - trainable_blocks
    prefixes = tuple(f"encoder.layers.{i}." for i in range(first, len(model.encoder.layers)))
    return {k: v.detach().cpu().clone()
            for k, v in model.state_dict().items() if k.startswith(prefixes)}


@torch.inference_mode()
def evaluate(model, head, loader, device, labels):
    model.eval()
    head.eval()
    all_logits, all_targets, all_snrs = [], [], []
    for batch in loader:
        with autocast(device):
            sequence, _ = model.extract_features(batch["audio"].to(device, non_blocking=True))
            logits = head(sequence.mean(dim=1))
        all_logits.append(logits.float().cpu())
        all_targets.append(batch["target"])
        all_snrs.append(batch["snr"])
    return metrics(torch.cat(all_logits), torch.cat(all_targets), torch.cat(all_snrs), labels)


def finetune(model, head, corpus, train_rows, validation_rows, device, config):
    teacher = config["teacher"]
    blocks = teacher["trainable_blocks"]
    for parameter in model.parameters():
        parameter.requires_grad = False
    for layer in model.encoder.layers[-blocks:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True

    train_loader = corpus.loader(train_rows, teacher["batch_size"], teacher["workers"],
                                 True, device, config["experiment"]["seed"])
    validation_loader = corpus.loader(validation_rows, teacher["validation_batch_size"],
                                      teacher["workers"], False, device)
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad],
          "lr": teacher["encoder_lr"]},
         {"params": head.parameters(), "lr": teacher["head_ft_lr"]}],
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in list(model.parameters()) + list(head.parameters())
                 if p.requires_grad]

    best = evaluate(model, head, validation_loader, device, corpus.labels)
    best_state = {"encoder": encoder_state(model, blocks),
                  "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
    print(f"finetune epoch=0 val_accuracy={best['accuracy']:.4f} "
          f"val_macro_f1={best['macro_f1']:.4f}", flush=True)

    history, stale = [], 0
    epochs = 1 if config["runtime"]["smoke_test"] else teacher["finetune_epochs"]
    for epoch in range(1, epochs + 1):
        model.eval()
        for layer in model.encoder.layers[-blocks:]:
            layer.train()
        head.train()
        optimizer.zero_grad(set_to_none=True)
        total = correct = seen = 0
        for step, batch in enumerate(train_loader, start=1):
            waveforms = batch["audio"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            with autocast(device):
                sequence, _ = model.extract_features(waveforms)
                logits = head(sequence.mean(dim=1))
                loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss; refusing to save a NaN checkpoint")
            (loss / teacher["accumulation_steps"]).backward()
            if step % teacher["accumulation_steps"] == 0 or step == len(train_loader):
                nn.utils.clip_grad_norm_(trainable, teacher["gradient_clip_norm"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total += loss.item() * targets.shape[0]
            correct += (logits.argmax(1) == targets).sum().item()
            seen += targets.shape[0]
            if step == 1 or step % 200 == 0 or step == len(train_loader):
                vram = (torch.cuda.max_memory_allocated() / 1024 ** 3
                        if device.type == "cuda" else 0.0)
                print(f"finetune epoch={epoch} batch={step}/{len(train_loader)} "
                      f"loss={total / seen:.4f} accuracy={correct / seen:.4f} "
                      f"max_vram_gb={vram:.2f}", flush=True)
        current = evaluate(model, head, validation_loader, device, corpus.labels)
        history.append({"epoch": epoch, "train_loss": total / seen,
                        "train_accuracy": correct / seen,
                        "val_accuracy": current["accuracy"],
                        "val_macro_f1": current["macro_f1"]})
        print(f"finetune epoch={epoch} val_accuracy={current['accuracy']:.4f} "
              f"val_macro_f1={current['macro_f1']:.4f}", flush=True)
        if current["macro_f1"] > best["macro_f1"] + 1e-4:
            best, stale = current, 0
            best_state = {"encoder": encoder_state(model, blocks),
                          "head": {k: v.detach().cpu().clone()
                                   for k, v in head.state_dict().items()}}
        else:
            stale += 1
            if stale >= teacher["patience"]:
                print(f"finetune early_stop={epoch}", flush=True)
                break

    model.load_state_dict(best_state["encoder"], strict=False)
    head.load_state_dict(best_state["head"])
    return best, history, best_state


def soft_label_report(logits: torch.Tensor, rho: float) -> dict:
    """Confidence and entropy of the teacher's soft labels at temperature ``rho``.

    This is the number KD actually consumes, and it is NOT the same thing as
    validation accuracy. The 2026-09-24 run made that concrete: the teacher hit
    train accuracy 1.0000 with train loss 0.0001  -  total memorisation  -  while
    validation accuracy sat at 0.7799 and the accuracy gate happily said PASS.
    Accuracy cannot see memorisation; entropy can.
    """
    probabilities = torch.softmax(logits.float() / rho, dim=1)
    entropy = -(probabilities * torch.log(probabilities + 1e-12)).sum(dim=1).mean()
    return {"rho": rho,
            "max_prob": float(probabilities.max(dim=1).values.mean()),
            "entropy": float(entropy),
            "entropy_max": float(np.log(logits.shape[1]))}


def report_gate(accuracy: float, config: dict, train_logits: torch.Tensor | None = None) -> str:
    """The stop gate from DESIGN.md ?10. Prints a verdict the log makes obvious."""
    gates = config["gates"]
    floor = gates["teacher_acc_floor"]
    baseline = gates["baseline_mid_accuracy"]
    print("", flush=True)
    print(f"teacher val_accuracy       = {accuracy:.4f}", flush=True)
    print(f"BEATs baseline (mid slice) = {baseline:.4f}", flush=True)
    print(f"headroom                   = {accuracy - baseline:+.4f}", flush=True)

    if train_logits is not None:
        print("\nsoft labels on TRAIN rows (what KD reads):", flush=True)
        for rho in (1.0, 2.0, 4.0, 8.0):
            r = soft_label_report(train_logits, rho)
            share = r["entropy"] / r["entropy_max"]
            print(f"  rho={rho:>4}  max_prob={r['max_prob']:.4f}  "
                  f"entropy={r['entropy']:.4f}/{r['entropy_max']:.4f} ({share:.0%})",
                  flush=True)
        print("  Pick rho so entropy lands near 50-70% of the maximum. Measured on the\n"
              "  2026-09-24 teacher, rho=4 gave 60% while rho=8 gave 94%  -  nearly uniform,\n"
              "  which teaches noise rather than class similarity. Higher is NOT safer.",
              flush=True)

    if accuracy < floor:
        print(f"\nGATE=STOP  accuracy below {floor:.2f}. The teacher sees CLEAN noise and "
              "still barely beats a baseline that sees the mixture, so the "
              "privileged-information premise has failed. Do not run the student.",
              flush=True)
        return "STOP"
    print("\nGATE=PASS", flush=True)
    return "PASS"


# ------------------------------------------------------------------------ commands


def command_teacher36(config: dict) -> None:
    device = seed_everything(config["experiment"]["seed"])
    outputs = config["outputs"]
    out_dir = HERE / outputs["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher = config["teacher"]
    runtime = config["runtime"]

    corpus = Corpus(config)
    audit = corpus.audit()
    train_rows = corpus.by_split[config["dataset"]["train_split"]]
    validation_rows = corpus.by_split[config["dataset"]["validation_split"]]
    if runtime["smoke_test"]:
        train_rows = train_rows[:runtime["smoke_train_rows"]]
        validation_rows = validation_rows[:runtime["smoke_validation_rows"]]
        print("SMOKE TEST  -  results are not reportable", flush=True)
    print(f"device={device} teacher_train={len(train_rows)} "
          f"validation={len(validation_rows)} dedup={teacher['dedup']}", flush=True)

    pretrained = resolve_pretrained(config["pretrained"]["candidates"])
    model, raw_checkpoint = load_beats(device, pretrained)

    cache_path = out_dir / outputs["embedding_cache"]
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        train_data, validation_data = cached["train"], cached["validation"]
        print(f"cache={cache_path}", flush=True)
    else:
        train_data = extract(model, corpus, train_rows, device,
                             teacher["extract_batch_size"], teacher["workers"], "train")
        validation_data = extract(model, corpus, validation_rows, device,
                                  teacher["extract_batch_size"], teacher["workers"],
                                  "validation")
        torch.save({"train": train_data, "validation": validation_data}, cache_path)
        print(f"cached -> {cache_path}", flush=True)

    head = initialize_head(raw_checkpoint, corpus.labels, corpus.label_to_mid, device)
    head_best, head_history = train_head(head, train_data, validation_data,
                                         corpus.labels, device, config)
    print(f"head stage val_accuracy={head_best['accuracy']:.4f} "
          f"val_macro_f1={head_best['macro_f1']:.4f}", flush=True)

    del train_data, validation_data
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    best, finetune_history, best_state = finetune(
        model, head, corpus, train_rows, validation_rows, device, config
    )
    torch.save({**best_state, "labels": corpus.labels, "validation_metrics": best,
                "pretrained": str(pretrained)}, out_dir / outputs["checkpoint"])
    (out_dir / outputs["history"]).write_text(
        json.dumps({"audit": audit, "head_stage": head_history,
                    "finetune_stage": finetune_history, "best_validation": best},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    gate = report_gate(best["accuracy"], config)
    summary = {"teacher_val_accuracy": round(float(best["accuracy"]), 4),
               "teacher_val_macro_f1": round(float(best["macro_f1"]), 4),
               "teacher_gate": gate, "audit": audit,
               "checkpoint": str(out_dir / outputs["checkpoint"])}
    (out_dir / outputs["summary"]).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("", flush=True)
    print("--- paste into STATUS.md ---", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    if gate == "STOP":
        raise SystemExit(2)


def command_bank36(config: dict) -> None:
    """Precompute the teacher's embeddings and logits for EVERY manifest row.

    The teacher is frozen from here on, so this bank is exact and never goes stale  - 
    unlike the rolling memory buffer the CRD reference has to use.
    """
    device = seed_everything(config["experiment"]["seed"])
    outputs = config["outputs"]
    out_dir = HERE / outputs["dir"]
    teacher = config["teacher"]
    runtime = config["runtime"]

    corpus = Corpus(config)
    checkpoint_path = out_dir / outputs["checkpoint"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} is missing  -  run `main.py teacher36` first."
        )
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    pretrained = resolve_pretrained(config["pretrained"]["candidates"])
    model, _ = load_beats(device, pretrained)
    model.load_state_dict(saved["encoder"], strict=False)
    head = nn.Linear(768, len(corpus.labels))
    head.load_state_dict(saved["head"])
    head.to(device)
    model.eval()
    head.eval()

    rows = corpus.rows[:runtime["smoke_bank_rows"]] if runtime["smoke_test"] else corpus.rows
    loader = corpus.loader(rows, teacher["validation_batch_size"], teacher["workers"],
                           False, device)

    embeddings, logits_all, targets, snrs, indices = [], [], [], [], []
    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            with autocast(device):
                sequence, _ = model.extract_features(batch["audio"].to(device, non_blocking=True))
                pooled = sequence.mean(dim=1).float()
                logits = head(pooled)
            embeddings.append(pooled.half().cpu())
            logits_all.append(logits.float().half().cpu())
            targets.append(batch["target"].long())
            snrs.append(batch["snr"].long())
            indices.append(batch["row_index"].long())
            if step == 1 or step % 100 == 0 or step == len(loader):
                print(f"bank {step}/{len(loader)}", flush=True)

    bank = {"z": torch.cat(embeddings), "logits": torch.cat(logits_all),
            "y": torch.cat(targets), "snr": torch.cat(snrs),
            "row_index": torch.cat(indices),
            "sample_id": [row["sample_id"] for row in rows]}

    expected_y = torch.tensor([corpus.label_to_index[r["label_names"]] for r in rows])
    expected_snr = torch.tensor([int(float(r["target_snr_db"])) for r in rows])
    assert bank["z"].shape[0] == len(rows), "bank row count differs from the manifest"
    assert torch.equal(bank["row_index"], torch.arange(len(rows))), "row_index is not monotonic"
    assert torch.equal(bank["y"], expected_y), "bank labels are shifted against the manifest"
    assert torch.equal(bank["snr"], expected_snr), "bank snr is shifted against the manifest"
    assert bank["sample_id"] == [r["sample_id"] for r in rows], "sample_id is shifted"
    assert torch.isfinite(bank["z"].float()).all(), "bank holds NaN or Inf"

    torch.save(bank, out_dir / outputs["bank"])
    megabytes = (bank["z"].numel() + bank["logits"].numel()) * 2 / 1e6
    print(f"bank -> {out_dir / outputs['bank']} ({megabytes:.0f} MB), "
          f"z={tuple(bank['z'].shape)} logits={tuple(bank['logits'].shape)}", flush=True)
    print("bank alignment verified against the manifest", flush=True)

    # The soft labels KD will read. Validation accuracy cannot see memorisation,
    # so check the temperature here, where every training row's logits exist.
    train_mask = torch.tensor([r["split"] == config["dataset"]["train_split"] for r in rows])
    train_logits = bank["logits"][train_mask].float()
    train_accuracy = float((train_logits.argmax(1) == bank["y"][train_mask]).float().mean())
    print(f"\nteacher accuracy on TRAIN rows = {train_accuracy:.4f}", flush=True)
    if train_accuracy > 0.99:
        print("  The teacher has memorised the training set. That is expected with 12\n"
              "  trainable blocks, and it does not sink KD  -  measured on 2026-09-24 the\n"
              "  dark knowledge still matched the real validation confusion structure\n"
              "  (cosine 0.42, top-1 confusion agreement 26.5% against 2.9% by chance).\n"
              "  What it does mean is that temperature is doing the work, so check it:",
              flush=True)
    for rho in (1.0, 2.0, 4.0, 8.0):
        report = soft_label_report(train_logits, rho)
        share = report["entropy"] / report["entropy_max"]
        print(f"  rho={rho:>4}  max_prob={report['max_prob']:.4f}  "
              f"entropy={report['entropy']:.4f}/{report['entropy_max']:.4f} ({share:.0%})",
              flush=True)
    print("  Aim for 50-70% of maximum entropy. rho=4 measured 60%; rho=8 measured 94%,\n"
          "  which is nearly uniform and teaches noise. Higher temperature is NOT safer.",
          flush=True)


STAGES = ("teacher36", "bank36", "student36", "test36", "report36")


def resolve_stage(name: str):
    """Look the stage up lazily  -  ``student_36`` imports from this module, so binding
    its commands at import time would be a cycle."""
    if name == "teacher36":
        return command_teacher36
    if name == "bank36":
        return command_bank36
    if name == "report36":
        from tasks import report_36
        return report_36.command_report36
    from tasks import student_36
    return {"student36": student_36.command_student36,
            "test36": student_36.command_test36}[name]


# Which switches each run flips. Everything else comes from the config file.
#
# This lives here rather than in the config because the config is tracked in git:
# hand-editing it means every `git pull` fights the edit, and when the edit loses
# the run proceeds SILENTLY as the wrong variant. That happened once already -- a
# full student run completed as run 2 while its log said so on line 1 and nobody
# was reading line 1. Passing --run makes the choice explicit in the command, in
# the log, and in the checkpoint name, and it cannot be reverted by a pull.
#   run2_ce_only   the control. Isolates data + FiLM + mid-slice selection.
#   run3_kd_crd    the planned treatment. CRD repo calls this `-a 1 -b 0.8`.
#   run3b_crd_only CRD without KD. This is the CRD paper's PRIMARY setting
#                  (`-a 0 -b 0.8`); CRD+KD is its add-on. Worth running here for a
#                  specific reason: the teacher reached train accuracy 1.0000, so on
#                  every training row KD's target is a softened one-hot on the class
#                  CE already supplies. That part is closer to label smoothing than
#                  to knowledge transfer, and only the non-target ranking carries
#                  anything extra (measured: cosine 0.42 against the real confusion
#                  structure). Comparing this against run3 says whether the gain is
#                  representation transfer or smoothing.
#   run3c_kd_only  the complement, to finish the attribution.
#   run1_baseline  the BEATs recipe rebuilt inside this pipeline: CE only, no FiLM,
#                  checkpoint chosen on the WHOLE validation split. The published
#                  baseline's checkpoint is gone and it has no per-clip predictions,
#                  so its row can only carry the metrics its old file recorded and it
#                  cannot enter a paired test. This run fixes both: the same eight
#                  metrics on the same test clips, and McNemar against it becomes
#                  possible.
RUNS = {
    "run1_baseline": {"a_kd": 0.0, "b_crd": 0.0, "remix": False,
                      "film": False, "select_on": "full"},
    "run2_ce_only": {"a_kd": 0.0, "b_crd": 0.0, "remix": False,
                     "film": True, "select_on": "mid"},
    "run3_kd_crd": {"a_kd": 1.0, "b_crd": 0.8, "remix": False,
                    "film": True, "select_on": "mid"},
    "run3b_crd_only": {"a_kd": 0.0, "b_crd": 0.8, "remix": False,
                       "film": True, "select_on": "mid"},
    "run3c_kd_only": {"a_kd": 1.0, "b_crd": 0.0, "remix": False,
                      "film": True, "select_on": "mid"},
    "run4_remix": {"a_kd": 1.0, "b_crd": 0.8, "remix": True,
                   "film": True, "select_on": "mid"},
}
RUN_REQUIRED = ("student36", "test36")


def apply_run(config: dict, name: str) -> dict:
    switches = RUNS[name]
    config["run"] = name
    config["loss"]["a_kd"] = switches["a_kd"]
    config["loss"]["b_crd"] = switches["b_crd"]
    config.setdefault("remix", {})["enabled"] = switches["remix"]
    config["student"]["film"]["enabled"] = switches["film"]
    config["student"]["select_on"] = switches["select_on"]
    print(f"run={name} a_kd={switches['a_kd']} b_crd={switches['b_crd']} "
          f"remix={switches['remix']} film={switches['film']} "
          f"select_on={switches['select_on']}", flush=True)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--config", type=Path, default=HERE / "config/train_config.json")
    parser.add_argument("--run", choices=sorted(RUNS),
                        help="Required for student36 and test36. Sets a_kd, b_crd and remix.")
    arguments = parser.parse_args()

    if arguments.stage in RUN_REQUIRED and arguments.run is None:
        parser.error(
            f"{arguments.stage} needs --run. Choose one of: {', '.join(sorted(RUNS))}.\n"
            "It is required on purpose: the config file is tracked in git, so relying on\n"
            "a hand edit lets a `git pull` silently turn run 3 back into run 2."
        )
    if arguments.stage not in RUN_REQUIRED and arguments.run is not None:
        parser.error(f"{arguments.stage} ignores --run; drop it to avoid implying otherwise.")

    config = load_config(arguments.config)
    print(f"stage={arguments.stage} config={arguments.config}", flush=True)
    if arguments.run is not None:
        config = apply_run(config, arguments.run)
    resolve_stage(arguments.stage)(config)


if __name__ == "__main__":
    main()
