"""The three training/eval stages: teacher36, student36, test36.

The teacher is frozen and runs ONLINE beside the student, in the same step, on
the clean noise of the same clip. The BEATs project precomputed a bank instead;
that is not an option here because patch-level tokens for the whole manifest
would be about 13 GB, and running the teacher live also removes the staleness
that bank had to be caveated for.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mid_expert_lib import (
    CRDLoss, Projection, film_deviation, mid_slice_mask, sample_negatives,
)
from tasks.run_36 import (
    HERE, Classifier, Corpus, autocast, build_model, collect, metrics,
    seed_everything, snapshot, split_metrics, trainable_parameters,
)


def jitter_snr(snr_db, sigma_db, generator):
    """The gate's error, simulated. Training on true SNR and testing on an
    estimated one would leave the student brittle exactly where it has to work.
    sigma stays an assumption until a gate exists and its RMSE is measured."""
    if sigma_db <= 0:
        return snr_db
    noise = torch.randn(snr_db.shape, generator=generator) * sigma_db
    return snr_db + noise.to(snr_db.device)


def optimizer_for(model: Classifier, config: dict, section: str):
    settings = config[section]
    encoder = [p for p in model.encoder.parameters() if p.requires_grad]
    groups = [{"params": encoder, "lr": settings["encoder_lr"]},
              {"params": model.head.parameters(),
               "lr": settings.get("head_lr", settings.get("head_ft_lr"))}]
    if model.film is not None:
        groups.append({"params": model.film.parameters(),
                       "lr": settings.get("film_lr", 1e-3)})
    return groups


def run_epochs(model, config, section, corpus, train_rows, validation_rows,
               device, step_fn, select_on, extra_params=()):
    """Shared training loop. `step_fn(batch)` returns (loss, parts, logits, target)."""
    settings = config[section]
    band = config["mid_band_db"]
    groups = optimizer_for(model, config, section)
    for params in extra_params:
        groups.append({"params": params, "lr": settings.get("film_lr", 1e-3)})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    watched = [p for group in groups for p in group["params"]]

    train_loader = corpus.loader(train_rows, settings["batch_size"],
                                 settings["workers"], True, device,
                                 config["experiment"]["seed"])
    validation_loader = corpus.loader(validation_rows,
                                      settings["validation_batch_size"],
                                      settings["workers"], False, device)

    use_snr = model.film is not None
    best = split_metrics(collect(model, validation_loader, device, use_snr),
                         corpus.labels, band)
    best_state = snapshot(model)
    print(f"selecting on the '{select_on}' slice", flush=True)
    print(f"epoch=0 val_mid_acc={best['mid']['accuracy']:.4f} "
          f"val_full_acc={best['full']['accuracy']:.4f}", flush=True)

    epochs = 1 if config["runtime"]["smoke_test"] else settings["finetune_epochs"]
    history, stale = [], 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums, correct, seen = {}, 0, 0
        started = time.perf_counter()
        for step, batch in enumerate(train_loader, start=1):
            loss, parts, logits, target = step_fn(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss; refusing to save NaN weights")
            (loss / settings["accumulation_steps"]).backward()
            if step % settings["accumulation_steps"] == 0 or step == len(train_loader):
                nn.utils.clip_grad_norm_(watched, settings["gradient_clip_norm"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            size = target.shape[0]
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + float(value) * size
            correct += int((logits.argmax(1) == target).sum())
            seen += size
            if step == 1 or step % 200 == 0 or step == len(train_loader):
                shown = " ".join(f"{k}={v/seen:.4f}" for k, v in sums.items())
                vram = (torch.cuda.max_memory_allocated() / 1024**3
                        if device.type == "cuda" else 0.0)
                print(f"  epoch={epoch} batch={step}/{len(train_loader)} {shown} "
                      f"acc={correct/seen:.4f} vram={vram:.1f}GB", flush=True)

        current = split_metrics(collect(model, validation_loader, device, use_snr),
                                corpus.labels, band)
        entry = {"epoch": epoch, "seconds": time.perf_counter() - started,
                 "train_accuracy": correct / seen,
                 **{k: v / seen for k, v in sums.items()},
                 "val_mid_accuracy": current["mid"]["accuracy"],
                 "val_mid_macro_f1": current["mid"]["macro_f1"],
                 "val_full_accuracy": current["full"]["accuracy"],
                 "film_deviation": film_deviation(model.film) if model.film else 0.0}
        history.append(entry)
        shown = " ".join(f"{k}={v/seen:.4f}" for k, v in sums.items())
        print(f"epoch={epoch} {shown} val_mid_acc={entry['val_mid_accuracy']:.4f} "
              f"val_mid_f1={entry['val_mid_macro_f1']:.4f} "
              f"film_dev={entry['film_deviation']:.3f} "
              f"({entry['seconds']:.0f}s)", flush=True)

        if current[select_on]["macro_f1"] > best[select_on]["macro_f1"] + 1e-4:
            best, stale = current, 0
            best_state = snapshot(model)
        else:
            stale += 1
            if stale >= settings["patience"]:
                print(f"early_stop={epoch}", flush=True)
                break
    model.load_state_dict(best_state, strict=False)
    return best, history


# ------------------------------------------------------------------------ teacher


def command_teacher36(config: dict) -> None:
    device = seed_everything(config["experiment"]["seed"])
    out_dir = HERE / config["outputs"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime = config["runtime"]

    corpus = Corpus(config, column=config["teacher"]["source_column"])
    audit = corpus.audit()
    train_rows = corpus.by_split[config["dataset"]["train_split"]]
    validation_rows = corpus.by_split[config["dataset"]["validation_split"]]
    if runtime["smoke_test"]:
        train_rows = train_rows[:runtime["smoke_train_rows"]]
        validation_rows = validation_rows[:runtime["smoke_validation_rows"]]
        print("SMOKE TEST - results are not reportable", flush=True)

    model = build_model(config, device, film_enabled=False,
                        trainable_blocks=config["teacher"]["trainable_blocks"])
    criterion = nn.CrossEntropyLoss()

    def step(batch):
        audio = batch["audio"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        with autocast(device):
            logits, _, _ = model(audio)
            loss = criterion(logits, target)
        return loss, {"ce": loss.detach()}, logits, target

    best, history = run_epochs(model, config, "teacher", corpus, train_rows,
                               validation_rows, device, step, select_on="full")
    torch.save({"state": snapshot(model), "labels": corpus.labels,
                "validation": best}, out_dir / config["teacher"]["checkpoint"])
    (out_dir / config["outputs"]["history"]).write_text(
        json.dumps({"audit": audit, "history": history, "best": best}, indent=2),
        encoding="utf-8")

    accuracy = best["full"]["accuracy"]
    floor = config["gates"]["teacher_acc_floor"]
    baseline = config["gates"]["baseline_mid_accuracy"]
    print(f"\nteacher val_accuracy = {accuracy:.4f}", flush=True)
    print(f"baseline (mid slice) = {baseline:.4f}   headroom = "
          f"{accuracy - baseline:+.4f}", flush=True)
    gate = "PASS" if accuracy >= floor else "STOP"
    if gate == "STOP":
        print(f"GATE=STOP  below {floor:.2f}. The teacher sees CLEAN noise and still "
              "barely beats a baseline that sees the mixture, so the "
              "privileged-information premise has failed.", flush=True)
    else:
        print("GATE=PASS", flush=True)
    summary = {"teacher_val_accuracy": round(accuracy, 4),
               "teacher_val_macro_f1": round(best["full"]["macro_f1"], 4),
               "teacher_gate": gate, "audit": audit}
    (out_dir / config["outputs"]["summary"]).write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if gate == "STOP":
        raise SystemExit(2)


# ------------------------------------------------------------------------ student


def load_teacher(config: dict, device: torch.device) -> Classifier | None:
    path = HERE / config["outputs"]["dir"] / config["teacher"]["checkpoint"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing, and KD or CRD needs it.\n"
            "Train it with:  python -u main.py teacher36 --config <config>")
    teacher = build_model(config, device, film_enabled=False, trainable_blocks=0)
    teacher.load_state_dict(torch.load(path, map_location="cpu",
                                       weights_only=False)["state"], strict=False)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def command_student36(config: dict) -> None:
    device = seed_everything(config["experiment"]["seed"])
    settings, loss_config = config["student"], config["loss"]
    out_dir = HERE / config["outputs"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = config["run"]
    runtime = config["runtime"]

    use_kd = loss_config["a_kd"] > 0
    use_crd = loss_config["b_crd"] > 0
    # Only pay for loading the clean noise when a loss term actually reads it.
    paired = config["teacher"]["source_column"] if (use_kd or use_crd) else None
    corpus = Corpus(config, column=settings["source_column"], paired_column=paired)
    train_rows = corpus.by_split[config["dataset"]["train_split"]]
    validation_rows = corpus.by_split[config["dataset"]["validation_split"]]
    if runtime["smoke_test"]:
        train_rows = train_rows[:runtime["smoke_train_rows"]]
        validation_rows = validation_rows[:runtime["smoke_validation_rows"]]
        print("SMOKE TEST - results are not reportable", flush=True)

    model = build_model(config, device, settings["film"]["enabled"],
                        settings["trainable_blocks"])
    teacher = load_teacher(config, device) if (use_kd or use_crd) else None
    print(f"run={run_name} train={len(train_rows)} kd={use_kd} crd={use_crd} "
          f"teacher={'online' if teacher else 'none'}", flush=True)

    dim = config["backbone"]["embedding_dim"]
    g_s = g_t = crd = None
    extra = []
    if use_crd:
        g_s = Projection(dim, loss_config["proj_dim"]).to(device)
        # g_t frozen: the CRD reference learns both, but a fixed random projection
        # preserves relative distances and halves what has to be optimised here.
        torch.manual_seed(config["experiment"]["seed"])
        g_t = Projection(dim, loss_config["proj_dim"]).to(device)
        for parameter in g_t.parameters():
            parameter.requires_grad = False
        crd = CRDLoss(len(train_rows), loss_config["tau_nce"]).to(device)
        extra.append(list(g_s.parameters()))

    generator = torch.Generator().manual_seed(config["experiment"]["seed"])
    sigma = settings["film"]["snr_jitter_db"] if model.film is not None else 0.0
    patch_level = loss_config["crd_level"] == "patch"

    def step(batch):
        audio = batch["audio"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        true_snr = batch["snr"].float().to(device, non_blocking=True)
        snr_in = jitter_snr(true_snr, sigma, generator) if model.film else None

        with autocast(device):
            logits, pooled, patches = model(audio, snr_in)
            ce = nn.functional.cross_entropy(logits, target)
            loss = loss_config["r_ce"] * ce
            parts = {"ce": ce.detach()}

            if teacher is not None:
                with torch.no_grad():
                    t_logits, t_pooled, t_patches = teacher(
                        batch["paired"].to(device, non_blocking=True))

            if use_kd:
                rho = loss_config["rho_kd"]
                kd = nn.functional.kl_div(
                    nn.functional.log_softmax(logits / rho, dim=-1),
                    nn.functional.softmax(t_logits / rho, dim=-1),
                    reduction="batchmean") * (rho ** 2)
                loss = loss + loss_config["a_kd"] * kd
                parts["kd"] = kd.detach()

            if use_crd:
                if patch_level:
                    # Every patch is its own anchor. This is the change the whole
                    # project rests on: the BEATs run pooled first and threw the
                    # time-frequency structure away before the loss saw it.
                    b, n, _ = patches.shape
                    student_vec = g_s(patches.reshape(b * n, -1).float())
                    positive = g_t(t_patches.reshape(b * n, -1).float())
                    labels_flat = target.repeat_interleave(n)
                else:
                    student_vec = g_s(pooled.float())
                    positive = g_t(t_pooled.float())
                    labels_flat = target
                index = sample_negatives(
                    labels_flat.cpu(), labels_flat.cpu(),
                    loss_config["n_negatives"], generator,
                    loss_config["negative_mode"]).to(device)
                value = crd(student_vec, positive, positive[index])
                loss = loss + loss_config["b_crd"] * value
                parts["crd"] = value.detach()

        return loss, parts, logits, target

    best, history = run_epochs(model, config, "student", corpus, train_rows,
                               validation_rows, device, step,
                               select_on=settings.get("select_on", "mid"),
                               extra_params=extra)

    torch.save({"state": snapshot(model), "labels": corpus.labels, "run": run_name,
                "validation": best}, out_dir / f"student_{run_name}.pt")
    (out_dir / f"student_{run_name}_history.json").write_text(
        json.dumps({"history": history, "best_validation": best,
                    "loss_config": loss_config, "student_config": settings},
                   indent=2), encoding="utf-8")
    print(f"\ncheckpoint -> {out_dir / f'student_{run_name}.pt'}", flush=True)
    print(f"best val_mid_accuracy={best['mid']['accuracy']:.4f} "
          f"val_mid_macro_f1={best['mid']['macro_f1']:.4f}", flush=True)


# --------------------------------------------------------------------------- test


def noise_purity(student_embedding, teacher_embedding) -> dict:
    """Did the mixture's representation actually move toward the clean noise's.

    A different question from accuracy: the classification metrics say whether the
    label came out right, these say whether speech got removed in embedding space,
    which is what CRD optimises. A CRD run that does not move these on held-out
    data did not do its job, whatever accuracy did. They sit where SI-SDR sits for
    the separation branches - embedding space is the only place this one separates.
    """
    student = nn.functional.normalize(student_embedding.float(), dim=1)
    teacher = nn.functional.normalize(teacher_embedding.float(), dim=1)
    similarity = student @ teacher.T
    n = similarity.shape[0]
    positive = similarity.diagonal()
    negative = (similarity.sum() - positive.sum()) / (n * (n - 1))
    return {"emb_cos_pos": float(positive.mean()),
            "emb_cos_neg": float(negative),
            "emb_gap": float(positive.mean() - negative),
            "emb_retrieval_top1": float(
                (similarity.argmax(dim=1) == torch.arange(n)).float().mean())}


def command_test36(config: dict) -> None:
    device = seed_everything(config["experiment"]["seed"])
    out_dir = HERE / config["outputs"]["dir"]
    band = config["mid_band_db"]
    run_name = config["run"]
    settings = config["student"]

    checkpoint = out_dir / f"student_{run_name}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"{checkpoint} missing - run student36 first.")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)

    corpus = Corpus(config, column=settings["source_column"],
                    paired_column=config["teacher"]["source_column"])
    model = build_model(config, device, settings["film"]["enabled"], 0)
    model.load_state_dict(saved["state"], strict=False)

    test_rows = corpus.by_split[config["dataset"]["test_split"]]
    loader = corpus.loader(test_rows, settings["validation_batch_size"],
                           settings["workers"], False, device)
    out = collect(model, loader, device, use_snr=model.film is not None)
    result = split_metrics(out, corpus.labels, band)

    gates = config["gates"]
    mid = result["mid"]
    print(f"\nrun={run_name} test mid slice ({mid['samples']} clips)", flush=True)
    print(f"  accuracy = {mid['accuracy']:.4f}  (baseline "
          f"{gates['baseline_mid_accuracy']:.4f}, delta "
          f"{mid['accuracy'] - gates['baseline_mid_accuracy']:+.4f})", flush=True)
    print(f"  macro_f1 = {mid['macro_f1']:.4f}  (baseline "
          f"{gates['baseline_mid_macro_f1']:.4f}, delta "
          f"{mid['macro_f1'] - gates['baseline_mid_macro_f1']:+.4f})", flush=True)
    print("A delta under about 1.0 point is inside the standard error on 2,160 "
          "clips. Do not call it an improvement without the paired McNemar test.",
          flush=True)

    purity = {}
    try:
        teacher = load_teacher(config, device)
        teacher_out = collect(teacher, loader, device, use_snr=False)
        for name, mask in (("mid", mid_slice_mask(out["snr"].float(), *band)),
                           ("snr_5", out["snr"] == 5), ("snr_10", out["snr"] == 10)):
            if bool(mask.any()):
                purity[name] = noise_purity(out["pooled"][mask],
                                            teacher_out["pooled"][mask])
        print("\nembedding purity (mechanism check, not accuracy):", flush=True)
        for name, values in purity.items():
            print(f"  {name:<7} cos_pos={values['emb_cos_pos']:.4f} "
                  f"cos_neg={values['emb_cos_neg']:.4f} gap={values['emb_gap']:.4f} "
                  f"retrieval@1={values['emb_retrieval_top1']:.4f}", flush=True)
    except FileNotFoundError:
        print("\nno teacher checkpoint - skipping the embedding purity check.",
              flush=True)

    (out_dir / f"student_{run_name}_test.json").write_text(
        json.dumps({"run": run_name, "test": result, "purity": purity,
                    "baseline": {"accuracy": gates["baseline_mid_accuracy"],
                                 "macro_f1": gates["baseline_mid_macro_f1"]}},
                   indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / f"student_{run_name}_predictions.npz",
        sample_id=np.array([r["sample_id"] for r in test_rows]),
        predicted=out["logits"].argmax(1).numpy().astype(np.int16),
        target=out["target"].numpy().astype(np.int16),
        snr=out["snr"].numpy().astype(np.int16),
        logits=out["logits"].numpy().astype(np.float16),
        labels=np.array(corpus.labels))
    print(f"predictions -> {out_dir / f'student_{run_name}_predictions.npz'}",
          flush=True)
