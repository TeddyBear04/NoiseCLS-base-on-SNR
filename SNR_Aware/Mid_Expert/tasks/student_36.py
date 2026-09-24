"""Task 6/7/8  -  train the student on mixtures across every SNR.

    python -u main.py student36 --config config/train_config.json
    python -u main.py test36    --config config/train_config.json

The student sees only ``mixture_path``. Everything the teacher knew reaches it
through the loss, never through its input, so nothing here leaks into inference.

    L = r * CE + a * rho^2 * KL(student || teacher) + b * CRD(z_s, z_t)

Run 2 (CE only), run 3 (+KD+CRD) and run 4 (+remix) all take THIS code path and
differ only by config. That is deliberate: if the control and the treatment ran
different code, a difference between them could come from the code rather than
the method, and the ablation would prove nothing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from mid_expert_lib import CRDLoss, FiLM, Projection, film_deviation, mid_slice_mask, sample_negatives
from tasks.run_36 import (
    HERE, Corpus, autocast, load_beats, metrics, resolve_pretrained, seed_everything,
)


class StudentModel(nn.Module):
    """BEATs over the mixture, FiLM-conditioned on SNR, then a 36-way head.

    ``forward`` returns the pooled embedding alongside the logits because the CRD
    term needs it. FiLM sits on the pooled embedding rather than inside the encoder
    blocks  -  see the deviation list in STATUS.md; the paper conditions at several
    layers, and doing that here would mean patching the vendored BEATs.
    """

    def __init__(self, beats, head: nn.Linear, film: FiLM | None):
        super().__init__()
        self.beats = beats
        self.head = head
        self.film = film

    def forward(self, waveform: torch.Tensor, snr_db: torch.Tensor):
        sequence, _ = self.beats.extract_features(waveform)
        pooled = sequence.mean(dim=1)
        if self.film is not None:
            pooled = self.film(pooled, snr_db)
        return self.head(pooled), pooled


def jitter_snr(snr_db: torch.Tensor, sigma_db: float, generator: torch.Generator) -> torch.Tensor:
    """Perturb the SNR the way the gate's error will at inference time.

    Training on the true SNR and testing on a regressed one would leave the student
    brittle in exactly the place it has to work. sigma is an ASSUMPTION until the
    gate exists and its RMSE is known  -  see STATUS.md.
    """
    if sigma_db <= 0:
        return snr_db
    noise = torch.randn(snr_db.shape, generator=generator) * sigma_db
    return snr_db + noise.to(snr_db.device)


def split_metrics(logits, targets, snrs, labels, band) -> dict:
    """Full-split metrics plus the mid-band slice, which is what we are judged on."""
    full = metrics(logits, targets, snrs.long(), labels)
    mask = mid_slice_mask(snrs.float(), band[0], band[1])
    mid = metrics(logits[mask], targets[mask], snrs[mask].long(), labels)
    mid["samples"] = int(mask.sum())
    return {"full": full, "mid": mid}


@torch.inference_mode()
def collect_predictions(model, loader, device) -> dict:
    """One inference pass. Metrics and per-clip predictions both come out of it."""
    model.eval()
    all_logits, all_targets, all_snrs = [], [], []
    for batch in loader:
        snr = batch["snr"].float().to(device, non_blocking=True)
        with autocast(device):
            logits, _ = model(batch["audio"].to(device, non_blocking=True), snr)
        all_logits.append(logits.float().cpu())
        all_targets.append(batch["target"])
        all_snrs.append(batch["snr"])
    return {"logits": torch.cat(all_logits), "target": torch.cat(all_targets),
            "snr": torch.cat(all_snrs)}


def evaluate_student(model, loader, device, labels, band) -> dict:
    out = collect_predictions(model, loader, device)
    return split_metrics(out["logits"], out["target"], out["snr"], labels, band)


class TeacherBank:
    """Frozen teacher embeddings and logits, indexed by manifest row.

    The teacher never trains again, so this bank is exact. ``g_t`` is frozen (the
    CRD reference learns both projections) because re-projecting 43,200 vectors with
    gradient every step is not affordable; a fixed projection still preserves
    relative distances. That is deviation #2 in STATUS.md.
    """

    def __init__(self, path: Path, device: torch.device, proj_dim: int, seed: int):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path} is missing, and KD or CRD needs it.\n"
                "Rebuild it with:  python -u main.py bank36 --config <config>\n"
                "That is one inference pass over the manifest, not a retrain. The teacher\n"
                "is frozen, so the rebuilt bank is identical to the original. It only needs\n"
                "artifacts/teacher_noise_best.pt to still exist. If that is gone too, the\n"
                "teacher has to be retrained with `main.py teacher36` first.\n"
                "Note `*.pt` is gitignored, so these files never travel with the repo."
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.z = payload["z"].to(device).float()
        self.logits = payload["logits"].to(device).float()
        self.y = payload["y"]
        self.size = self.z.shape[0]
        torch.manual_seed(seed)          # g_t is fixed, so its init must be reproducible
        self.g_t = Projection(self.z.shape[1], proj_dim).to(device)
        for parameter in self.g_t.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            self.projected = torch.cat([
                self.g_t(self.z[i:i + 8192]) for i in range(0, self.size, 8192)
            ])
        print(f"teacher bank: rows={self.size} z={tuple(self.z.shape)} "
              f"projected={tuple(self.projected.shape)}", flush=True)


def build_student(config, corpus, device):
    pretrained = resolve_pretrained(config["pretrained"]["candidates"])
    beats, raw_checkpoint = load_beats(device, pretrained)
    student_config = config["student"]
    blocks = student_config["trainable_blocks"]
    for parameter in beats.parameters():
        parameter.requires_grad = False
    for layer in beats.encoder.layers[-blocks:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True

    from train_beats_head import initialize_head
    head = initialize_head(raw_checkpoint, corpus.labels, corpus.label_to_mid, device)
    film = None
    if student_config["film"]["enabled"]:
        film = FiLM(768, student_config["film"]["hidden"]).to(device)
    return StudentModel(beats, head, film).to(device), blocks


def command_student36(config: dict) -> None:
    device = seed_everything(config["experiment"]["seed"])
    student_config = config["student"]
    loss_config = config["loss"]
    outputs = config["outputs"]
    out_dir = HERE / outputs["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    band = config["mid_band_db"]
    run_name = config["run"]

    corpus = Corpus(config, column="mixture_path")
    train_rows = corpus.by_split[config["dataset"]["train_split"]]
    validation_rows = corpus.by_split[config["dataset"]["validation_split"]]
    if config["runtime"]["smoke_test"]:
        train_rows = train_rows[:config["runtime"]["smoke_train_rows"]]
        validation_rows = validation_rows[:config["runtime"]["smoke_validation_rows"]]
        print("SMOKE TEST  -  results are not reportable", flush=True)

    use_kd = loss_config["a_kd"] > 0
    use_crd = loss_config["b_crd"] > 0
    print(f"run={run_name} device={device} train={len(train_rows)} "
          f"validation={len(validation_rows)} film={student_config['film']['enabled']} "
          f"kd={use_kd} crd={use_crd}", flush=True)

    model, blocks = build_student(config, corpus, device)

    bank = g_s = crd_criterion = None
    if use_kd or use_crd:
        bank = TeacherBank(out_dir / outputs["bank"], device,
                           loss_config["proj_dim"], config["experiment"]["seed"])
        if use_crd:
            g_s = Projection(768, loss_config["proj_dim"]).to(device)
            crd_criterion = CRDLoss(bank.size, loss_config["tau_nce"]).to(device)

    groups = [
        {"params": [p for p in model.beats.parameters() if p.requires_grad],
         "lr": student_config["encoder_lr"]},
        {"params": model.head.parameters(), "lr": student_config["head_lr"]},
    ]
    if model.film is not None:
        groups.append({"params": model.film.parameters(), "lr": student_config["film_lr"]})
    if g_s is not None:
        groups.append({"params": g_s.parameters(), "lr": student_config["film_lr"]})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    trainable = [p for group in groups for p in group["params"]]

    train_loader = corpus.loader(train_rows, student_config["batch_size"],
                                 student_config["workers"], True, device,
                                 config["experiment"]["seed"])
    validation_loader = corpus.loader(validation_rows,
                                      student_config["validation_batch_size"],
                                      student_config["workers"], False, device)
    generator = torch.Generator().manual_seed(config["experiment"]["seed"])
    sigma = student_config["film"]["snr_jitter_db"] if model.film is not None else 0.0

    best = evaluate_student(model, validation_loader, device, corpus.labels, band)
    best_state = snapshot(model, blocks)
    print(f"epoch=0 val_mid_acc={best['mid']['accuracy']:.4f} "
          f"val_mid_f1={best['mid']['macro_f1']:.4f} "
          f"val_full_acc={best['full']['accuracy']:.4f}", flush=True)

    history, stale = [], 0
    epochs = 1 if config["runtime"]["smoke_test"] else student_config["finetune_epochs"]
    for epoch in range(1, epochs + 1):
        model.beats.eval()
        for layer in model.beats.encoder.layers[-blocks:]:
            layer.train()
        model.head.train()
        if model.film is not None:
            model.film.train()
        optimizer.zero_grad(set_to_none=True)

        sums = {"ce": 0.0, "kd": 0.0, "crd": 0.0}
        correct = seen = 0
        started = time.perf_counter()
        for step, batch in enumerate(train_loader, start=1):
            waveform = batch["audio"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            true_snr = batch["snr"].float().to(device, non_blocking=True)
            rows = batch["row_index"].to(device, non_blocking=True)
            model_snr = jitter_snr(true_snr, sigma, generator)

            with autocast(device):
                logits, pooled = model(waveform, model_snr)
                ce = nn.functional.cross_entropy(logits, targets)
                loss = loss_config["r_ce"] * ce
                kd = crd = torch.zeros((), device=device)

                if use_kd:
                    rho = loss_config["rho_kd"]
                    teacher_logits = bank.logits[rows]
                    kd = nn.functional.kl_div(
                        nn.functional.log_softmax(logits / rho, dim=-1),
                        nn.functional.softmax(teacher_logits / rho, dim=-1),
                        reduction="batchmean",
                    ) * (rho ** 2)
                    loss = loss + loss_config["a_kd"] * kd

                if use_crd:
                    projected_student = g_s(pooled.float())
                    positives = bank.projected[rows]
                    negative_index = sample_negatives(
                        targets.cpu(), bank.y, loss_config["n_negatives"],
                        generator, loss_config["negative_mode"],
                    ).to(device)
                    negatives = bank.projected[negative_index]
                    crd = crd_criterion(projected_student, positives, negatives)
                    loss = loss + loss_config["b_crd"] * crd

            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss; refusing to save a NaN checkpoint")
            (loss / student_config["accumulation_steps"]).backward()
            if step % student_config["accumulation_steps"] == 0 or step == len(train_loader):
                nn.utils.clip_grad_norm_(trainable, student_config["gradient_clip_norm"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            batch_size = targets.shape[0]
            sums["ce"] += float(ce.detach()) * batch_size
            sums["kd"] += float(kd.detach()) * batch_size
            sums["crd"] += float(crd.detach()) * batch_size
            correct += int((logits.argmax(1) == targets).sum())
            seen += batch_size
            if step == 1 or step % 200 == 0 or step == len(train_loader):
                vram = (torch.cuda.max_memory_allocated() / 1024 ** 3
                        if device.type == "cuda" else 0.0)
                print(f"  epoch={epoch} batch={step}/{len(train_loader)} "
                      f"ce={sums['ce'] / seen:.4f} kd={sums['kd'] / seen:.4f} "
                      f"crd={sums['crd'] / seen:.4f} acc={correct / seen:.4f} "
                      f"max_vram_gb={vram:.2f}", flush=True)

        current = evaluate_student(model, validation_loader, device, corpus.labels, band)
        deviation = film_deviation(model.film) if model.film is not None else 0.0
        entry = {"epoch": epoch, "seconds": time.perf_counter() - started,
                 "train_accuracy": correct / seen,
                 "ce": sums["ce"] / seen, "kd": sums["kd"] / seen, "crd": sums["crd"] / seen,
                 "film_deviation": deviation,
                 "val_mid_accuracy": current["mid"]["accuracy"],
                 "val_mid_macro_f1": current["mid"]["macro_f1"],
                 "val_full_accuracy": current["full"]["accuracy"]}
        history.append(entry)
        print(f"epoch={epoch} ce={entry['ce']:.4f} kd={entry['kd']:.4f} "
              f"crd={entry['crd']:.4f} val_mid_acc={entry['val_mid_accuracy']:.4f} "
              f"val_mid_f1={entry['val_mid_macro_f1']:.4f} "
              f"film_dev={deviation:.3f}", flush=True)

        # Selection is on the MID slice, not the whole validation split. This is the
        # only place specialisation enters the main path, and it costs nothing.
        if current["mid"]["macro_f1"] > best["mid"]["macro_f1"] + 1e-4:
            best, stale = current, 0
            best_state = snapshot(model, blocks)
        else:
            stale += 1
            if stale >= student_config["patience"]:
                print(f"early_stop={epoch}", flush=True)
                break

    restore(model, best_state)
    checkpoint_path = out_dir / f"student_{run_name}.pt"
    torch.save({**best_state, "labels": corpus.labels, "run": run_name,
                "validation": best, "loss_config": loss_config}, checkpoint_path)
    (out_dir / f"student_{run_name}_history.json").write_text(
        json.dumps({"history": history, "best_validation": best,
                    "loss_config": loss_config, "student_config": student_config},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ncheckpoint -> {checkpoint_path}", flush=True)
    print(f"best val_mid_accuracy={best['mid']['accuracy']:.4f} "
          f"val_mid_macro_f1={best['mid']['macro_f1']:.4f}", flush=True)

    if model.film is not None and history and history[-1]["film_deviation"] < 1e-3:
        print("NOTE film_deviation stayed at zero  -  FiLM collapsed to identity and this "
              "run is effectively 'baseline retrained'. Still a valid control; say so "
              "in the report rather than claiming SNR conditioning did anything.",
              flush=True)


def snapshot(model, blocks: int) -> dict:
    first = len(model.beats.encoder.layers) - blocks
    prefixes = tuple(f"encoder.layers.{i}." for i in range(first, len(model.beats.encoder.layers)))
    state = {"encoder": {k: v.detach().cpu().clone()
                         for k, v in model.beats.state_dict().items() if k.startswith(prefixes)},
             "head": {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}}
    if model.film is not None:
        state["film"] = {k: v.detach().cpu().clone() for k, v in model.film.state_dict().items()}
    return state


def restore(model, state: dict) -> None:
    model.beats.load_state_dict(state["encoder"], strict=False)
    model.head.load_state_dict(state["head"])
    if model.film is not None and "film" in state:
        model.film.load_state_dict(state["film"])


def command_test36(config: dict) -> None:
    """Score the saved student on the test split and put it beside the baseline."""
    device = seed_everything(config["experiment"]["seed"])
    outputs = config["outputs"]
    out_dir = HERE / outputs["dir"]
    band = config["mid_band_db"]
    run_name = config["run"]

    corpus = Corpus(config, column="mixture_path")
    checkpoint_path = out_dir / f"student_{run_name}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{checkpoint_path} is missing  -  run `main.py student36` first.")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model, _ = build_student(config, corpus, device)
    restore(model, saved)
    test_rows = corpus.by_split[config["dataset"]["test_split"]]
    loader = corpus.loader(test_rows, config["student"]["validation_batch_size"],
                           config["student"]["workers"], False, device)
    collected = collect_predictions(model, loader, device)
    result = split_metrics(collected["logits"], collected["target"],
                           collected["snr"], corpus.labels, band)

    baseline = config["gates"]
    mid = result["mid"]
    per_snr = {k: round(v["accuracy"], 4) for k, v in mid.get("per_snr", {}).items()}
    print(f"\nrun={run_name} test mid slice ({mid['samples']} clips)", flush=True)
    print(f"  accuracy = {mid['accuracy']:.4f}  "
          f"(baseline {baseline['baseline_mid_accuracy']:.4f}, "
          f"delta {mid['accuracy'] - baseline['baseline_mid_accuracy']:+.4f})", flush=True)
    print(f"  macro_f1 = {mid['macro_f1']:.4f}  "
          f"(baseline {baseline['baseline_mid_macro_f1']:.4f}, "
          f"delta {mid['macro_f1'] - baseline['baseline_mid_macro_f1']:+.4f})", flush=True)
    print(f"  per_snr  = {per_snr}", flush=True)
    print("\nA delta under about 1.0 point is inside the standard error on 2,160 clips.\n"
          "Do not call it an improvement without the paired McNemar test in Task 9.",
          flush=True)

    payload = {"run": run_name, "test": result,
               "baseline": {"accuracy": baseline["baseline_mid_accuracy"],
                            "macro_f1": baseline["baseline_mid_macro_f1"]}}
    (out_dir / f"student_{run_name}_test.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Per-clip predictions, in manifest order over the test split, from the pass we
    # already ran. Task 9's McNemar test needs them: with 2,160 mid clips the standard
    # error on accuracy is ~1.0 point, so only a paired test resolves the 0.5-2 point
    # effect we expect.
    # Logits, not just argmax: mAP and macro-AUC need the scores, and without them
    # the CSV report can only carry the argmax-based half of the repo's metric set.
    np.savez_compressed(
        out_dir / f"student_{run_name}_predictions.npz",
        sample_id=np.array([r["sample_id"] for r in test_rows]),
        predicted=collected["logits"].argmax(1).numpy().astype(np.int16),
        target=collected["target"].numpy().astype(np.int16),
        snr=collected["snr"].numpy().astype(np.int16),
        logits=collected["logits"].numpy().astype(np.float16),
        labels=np.array(corpus.labels))
    print(f"predictions -> {out_dir / f'student_{run_name}_predictions.npz'}", flush=True)
