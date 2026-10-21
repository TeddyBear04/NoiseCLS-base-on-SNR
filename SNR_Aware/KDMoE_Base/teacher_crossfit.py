"""Teacher BEATs cross-fit cho KDMoE_Base: huấn luyện 5 fold out-of-fold và xuất đặc trưng / logit / bản đồ
cho các phương pháp KD của thư mục này (KD logit, AT, AT rollout, FitNets, ReviewKD).

    python -u teacher_crossfit.py crossfit --config config/teacher.json
    python -u teacher_crossfit.py features --config config/teacher.json
    python -u teacher_crossfit.py atmaps   --config config/<...>.json   (bản đồ AT từ trọng số fold đã lưu)

Hạ tầng BEATs (Corpus, build_model, run_epochs, collect) ở teacher/ - bản chép từ SNR_Expert_SSLAM, chỉ giữ
backbone BEATs, mã nguồn BEATs ở teacher/beats. Chạy riêng khỏi main.py (gói `models`/`tasks` của teacher/ chỉ
nằm trên sys.path của tiến trình này). Đường dẫn tương đối trong config (outputs.dir,
crossfit.exclude_noise_ytids, crossfit.n_hat_cache) tính từ thư mục này; backbone.checkpoint_candidates tính từ
teacher/ (vd. "../pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
TEACHER = HERE / "teacher"
if str(TEACHER) not in sys.path:
    sys.path.insert(0, str(TEACHER))

from tasks.run_36 import (  # noqa: E402
    Corpus, autocast, build_model, collect, load_config, seed_everything, teacher_backbone, write_params,
)
from tasks.stages import run_epochs, smoke_subset  # noqa: E402


def rollout_hooks(blocks, layers: list[int], captured: list) -> list:
    """Attention rollout (Abnar & Zuidema, ACL 2020) up to each of `layers`.

    Per layer: head-averaged attention A, residual added as 0.5 A + 0.5 I, rows
    renormalised, multiplied onto the rollout of the layers below. BEATs has no CLS
    token and pools by the token mean, so the map is the mean row of the rollout:
    the share of each input patch in the pooled output. Appends [tokens, B].
    """
    state, hooks = {}, []

    def force_weights(module, args, kwargs):  # the encoder calls every layer with need_weights=False
        return args, {**kwargs, "need_weights": True, "need_head_weights": False}

    def make_hook(index):
        def hook(module, inputs, output):
            attn = output[1].detach().float()                              # [B, N, N], mean over heads
            attn = 0.5 * attn + 0.5 * torch.eye(attn.shape[-1], device=attn.device)
            attn = attn / attn.sum(-1, keepdim=True)
            state["r"] = attn if index == 0 else attn @ state["r"]
            if index + 1 in layers:
                captured.append(state["r"].mean(1).T)
        return hook

    for index, block in enumerate(blocks[:max(layers)]):
        hooks.append(block.self_attn.register_forward_pre_hook(force_weights, with_kwargs=True))
        hooks.append(block.self_attn.register_forward_hook(make_hook(index)))
    return hooks


def collect_with_attention(model, loader, device, layers: list[int], source: str = "energy") -> dict:
    """`collect` plus Attention Transfer maps of the given 1-based encoder layers.

    source "energy" (Zagoruyko & Komodakis, ICLR 2017, F_sum^2 up to a constant): mean
    over channels of the squared token activations. source "rollout": attention rollout,
    see `rollout_hooks`. Laid out on BEATs' patch grid (tokens are time-major, 8
    frequency patches per time step) -> [N, layers, T', 8].
    """
    layers = sorted(layers)
    blocks, captured, hooks = list(model.encoder.blocks()), [], []
    if source == "rollout":
        hooks = rollout_hooks(blocks, layers, captured)
    elif source != "energy":
        raise ValueError(f"at_source must be 'energy' or 'rollout', got {source!r}")
    for layer in layers if source == "energy" else []:
        hooks.append(blocks[layer - 1].register_forward_hook(
            lambda module, inputs, output: captured.append(output[0].detach().float().pow(2).mean(-1))))
    try:
        out = collect(model, loader, device, False)
    finally:
        for hook in hooks:
            hook.remove()
    per_batch = [torch.stack(captured[i:i + len(layers)], dim=0) for i in range(0, len(captured), len(layers))]
    maps = torch.cat([m.permute(2, 0, 1) for m in per_batch])          # [N, layers, tokens]
    out["at"] = maps.reshape(maps.shape[0], len(layers), -1, 8).half().cpu()
    return out


def crossfit_folds(train_rows: list[dict], folds: int, seed: int) -> dict[str, int]:
    """noise_ytid -> fold; every row of one noise clip lands in the same fold."""
    ytids = sorted({row["noise_ytid"] for row in train_rows})
    np.random.default_rng(seed).shuffle(ytids)
    return {ytid: index % folds for index, ytid in enumerate(ytids)}


class NHatClips(torch.utils.data.Dataset):
    """Rows read from the FRCRN n_hat cache (frcrn_{split}.npy [N, samples] + .json ids) instead of a manifest column."""

    def __init__(self, corpus: Corpus, rows: list, cache_dir: Path, split: str):
        self.corpus, self.rows = corpus, rows
        self.path = cache_dir / f"frcrn_{split}.npy"
        ids = json.loads(self.path.with_suffix(".json").read_text(encoding="utf-8"))
        self.position = {sample_id: i for i, sample_id in enumerate(ids)}
        self.array = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        if self.array is None:  # one memmap per worker
            self.array = np.load(self.path, mmap_mode="r")
        row = self.rows[index]
        return {"audio": torch.from_numpy(np.array(self.array[self.position[row["sample_id"]]], dtype=np.float32)),
                "target": self.corpus.label_to_index[row["label_names"]],
                "snr": int(float(row["target_snr_db"])),
                "row_index": self.corpus.index_of[id(row)]}


def command_features(config: dict) -> None:
    """Teacher features for feature distillation, from the cross-fit fold teachers.

    FitNets (Romero et al., ICLR 2015) takes one middle layer (`crossfit.hint_layer`);
    ReviewKD (Chen et al., CVPR 2021) takes several (`crossfit.hint_layers`, e.g. 3, 6,
    9, 12). Each layer's tokens are laid out on the patch grid [T', 8, 768]. Each train
    row takes the fold teacher that held it out, as the logits do; the feature losses
    are train-only, so validation features are not exported. Needs the fold weights
    (`crossfit.save_weights`). One fp16 .npy [train rows, T', 8, 768] per layer in
    sorted corpus order plus the matching mix ids, so the student reads it as a memmap.

    `crossfit.hint_source`: the column the teacher hears (default the teacher's own
    `source_column`, clean noise), or "n_hat" - the FRCRN estimate the student hears,
    read from `crossfit.n_hat_cache`. With n_hat the teacher's logits on n_hat are
    packed too (train out-of-fold, validation the mean of the folds), in the layout
    of the cross-fit pack, for logit KD next to the features.
    """
    device = seed_everything(config["experiment"]["seed"])
    out_dir = HERE / config["outputs"]["dir"]
    settings, crossfit, runtime = config["teacher"], config["crossfit"], config["runtime"]
    seed = config["experiment"]["seed"]
    layers = sorted(crossfit.get("hint_layers", [crossfit.get("hint_layer")]))
    source = crossfit.get("hint_source", settings["source_column"])
    suffix = "_nhat" if source == "n_hat" else ""
    corpus = Corpus(config, column=settings["source_column"] if source == "n_hat" else source)
    train_split, validation_split = config["dataset"]["train_split"], config["dataset"]["validation_split"]
    train_rows = corpus.by_split[train_split]
    validation_rows = corpus.by_split[validation_split]
    if runtime["smoke_test"]:
        train_rows = smoke_subset(train_rows, runtime["smoke_train_rows"], seed)
        validation_rows = smoke_subset(validation_rows, runtime["smoke_validation_rows"], seed)
    fold_of = crossfit_folds(train_rows, crossfit["folds"], seed)
    order = sorted(corpus.index_of[id(row)] for row in train_rows)
    position = {index: pos for pos, index in enumerate(order)}

    def loader(rows, split):
        if source != "n_hat":
            return corpus.loader(rows, settings["validation_batch_size"], settings["workers"], False, device)
        dataset = NHatClips(corpus, rows, (HERE / crossfit["n_hat_cache"]).resolve(), split)
        return torch.utils.data.DataLoader(dataset, batch_size=settings["validation_batch_size"],
                                           num_workers=settings["workers"], pin_memory=device.type == "cuda")

    paths = {layer: out_dir / f"teacher_hints_l{layer}{suffix}.npy" for layer in layers}
    hints = {}
    train_logits = torch.zeros(len(order), config["experiment"]["num_classes"])
    validation_sum, validation_order = None, None
    for fold in range(crossfit["folds"]):
        data = torch.load(out_dir / f"fold{fold}.pt", map_location="cpu", weights_only=False)
        if "state_dict" not in data:
            raise KeyError(f"fold{fold}.pt has no weights; rerun `teacher_crossfit.py crossfit` with crossfit.save_weights")
        model = build_model(config, device, film_enabled=False, trainable_blocks=settings["trainable_blocks"],
                            backbone=teacher_backbone(config))
        model.load_state_dict(data["state_dict"])
        model.eval()
        captured = {}
        blocks = list(model.encoder.blocks())
        hooks = [blocks[layer - 1].register_forward_hook(
            lambda module, inputs, output, layer=layer: captured.__setitem__(
                layer, output[0].detach().transpose(0, 1).half().cpu())) for layer in layers]
        held_out = [row for row in train_rows if fold_of[row["noise_ytid"]] == fold]
        with torch.inference_mode():
            for batch in loader(held_out, train_split):
                with autocast(device):
                    logits, _, _ = model(batch["audio"].to(device, non_blocking=True), None)
                rows = [position[index] for index in batch["row_index"].tolist()]
                train_logits[rows] = logits.float().cpu()
                for layer in layers:
                    tokens = captured.pop(layer)                           # [B, tokens, C]
                    if layer not in hints:
                        shape = (len(order), tokens.shape[1] // 8, 8, tokens.shape[2])
                        hints[layer] = np.lib.format.open_memmap(paths[layer], mode="w+", dtype=np.float16,
                                                                 shape=shape)
                    hints[layer][rows] = tokens.reshape(len(rows), *hints[layer].shape[1:]).numpy()
        for hook in hooks:
            hook.remove()
        if source == "n_hat":
            out = collect(model, loader(validation_rows, validation_split), device, False)
            logits = out["logits"].float()
            validation_sum = logits if validation_sum is None else validation_sum + logits
            validation_order = out["row_index"].tolist()
        print(f"fold {fold}: features of layers {layers} for {len(held_out)} held-out rows", flush=True)
        del model
        torch.cuda.empty_cache()
    ids = [corpus.rows[index]["sample_id"] for index in order]
    for layer in layers:
        hints[layer].flush()
        paths[layer].with_suffix(".ids.json").write_text(json.dumps(ids), encoding="utf-8")
        print(f"hints -> {paths[layer]} {hints[layer].shape}", flush=True)
    if source == "n_hat":
        labels_of = {id(row): int(corpus.label_to_index[row["label_names"]]) for row in corpus.rows}

        def pack(indices, logits):
            rows = [corpus.rows[i] for i in indices]
            target = np.array([labels_of[id(row)] for row in rows])
            return {"mix_id": [row["sample_id"] for row in rows],
                    "logits36": logits.numpy().astype(np.float16),
                    "snr_db": np.array([float(row["target_snr_db"]) for row in rows], dtype=np.float32),
                    "accuracy": float((logits.argmax(1).numpy() == target).mean())}

        packed = {"source": f"BEATs crossfit {crossfit['folds']} folds on FRCRN n_hat", "precision": "fp16 logits",
                  "label_order": "labels.txt", "train": pack(order, train_logits),
                  "validation": pack(validation_order, validation_sum / crossfit["folds"])}
        packed_path = out_dir / crossfit.get("hint_packed", "teacher_crossfit_nhat.pt")
        torch.save(packed, packed_path)
        print(f"logits on n_hat -> {packed_path}: train OOF acc {packed['train']['accuracy']:.4f}, "
              f"validation acc {packed['validation']['accuracy']:.4f}", flush=True)


def read_ytids(path: Path) -> set[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def command_crossfit(config: dict) -> None:
    """Teacher on clean noise, cross-fit over `crossfit.folds` folds of the train split.

    Folds group rows by noise_ytid, so every 4 s segment of one noise clip sits in one
    fold. Fold f trains on the other folds and predicts fold f: the train logits a
    student distils from are out-of-fold, where a teacher trained on all of train
    returns its own memorised labels (Mid_Expert: train accuracy 1.000). Validation
    logits are the mean of the fold teachers. Each fold selects its epoch on the
    validation rows whose noise clip is NOT in `crossfit.exclude_noise_ytids` (clips
    the AudioSet-labelled backbone was trained on). Fold outputs are saved as they
    finish, so a rerun resumes at the first missing fold.
    """
    device = seed_everything(config["experiment"]["seed"])
    out_dir = HERE / config["outputs"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    settings, crossfit, runtime = config["teacher"], config["crossfit"], config["runtime"]
    seed = config["experiment"]["seed"]
    write_params(config, out_dir, "teacher_crossfit")

    corpus = Corpus(config, column=settings["source_column"])
    train_rows = corpus.by_split[config["dataset"]["train_split"]]
    validation_rows = corpus.by_split[config["dataset"]["validation_split"]]
    leaked = read_ytids((HERE / crossfit["exclude_noise_ytids"]).resolve())
    if runtime["smoke_test"]:
        train_rows = smoke_subset(train_rows, runtime["smoke_train_rows"], seed)
        validation_rows = smoke_subset(validation_rows, runtime["smoke_validation_rows"], seed)
        print("SMOKE TEST - results are not reportable", flush=True)
    clean_validation = [row for row in validation_rows if row["noise_ytid"] not in leaked]

    folds = crossfit["folds"]
    fold_of = crossfit_folds(train_rows, folds, seed)
    print(f"crossfit: {folds} folds over {len(fold_of)} noise clips ({len(train_rows)} rows); "
          f"selection on {len(clean_validation)}/{len(validation_rows)} clean validation rows",
          flush=True)

    criterion = nn.CrossEntropyLoss()
    eval_batch, workers = settings["validation_batch_size"], settings["workers"]
    at_layers = list(crossfit.get("at_layers", []))  # export Attention Transfer maps of these layers
    at_source = crossfit.get("at_source", "energy")
    for fold in range(folds):
        fold_path = out_dir / f"fold{fold}.pt"
        if fold_path.is_file():
            print(f"fold {fold}: done earlier, {fold_path.name}", flush=True)
            continue
        fold_train = [row for row in train_rows if fold_of[row["noise_ytid"]] != fold]
        held_out = [row for row in train_rows if fold_of[row["noise_ytid"]] == fold]
        model = build_model(config, device, film_enabled=False,
                            trainable_blocks=settings["trainable_blocks"],
                            backbone=teacher_backbone(config))

        def step(batch, model=model):
            audio = batch["audio"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            with autocast(device):
                logits, _, _ = model(audio)
                loss = criterion(logits, target)
            return loss, {"ce": loss.detach()}, logits, target

        print(f"\n=== fold {fold}: train {len(fold_train)} rows, held out {len(held_out)}", flush=True)
        best, history = run_epochs(model, config, "teacher", corpus, fold_train,
                                   clean_validation, device, step, select_on="full")
        outputs = {}
        for name, rows in (("held_out", held_out), ("validation", validation_rows)):
            loader = corpus.loader(rows, eval_batch, workers, False, device)
            if at_layers:
                out = collect_with_attention(model, loader, device, at_layers, at_source)
            else:
                out = collect(model, loader, device, False)
            outputs[name] = {"row_index": out["row_index"], "logits": out["logits"].half()}
            if at_layers:
                outputs[name]["at"] = out["at"]
        if crossfit.get("save_weights"):  # later feature exports (FitNets hints) reuse the fold teachers
            outputs["state_dict"] = {k: v.half() if v.is_floating_point() else v
                                     for k, v in model.state_dict().items()}
        torch.save({"fold": fold, "best_epoch": best["epoch"],
                    "clean_val_accuracy": best["full"]["accuracy"],
                    "clean_val_macro_f1": best["full"]["macro_f1"], "history": history, **outputs}, fold_path)
        print(f"fold {fold}: best epoch {best['epoch']} clean-val acc {best['full']['accuracy']:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    # Out-of-fold logits for train, mean of the fold teachers for validation.
    labels_of = {id(row): int(corpus.label_to_index[row["label_names"]]) for row in corpus.rows}
    oof, oof_at, validation_sum, at_sum, fold_summary = {}, {}, None, None, []
    for fold in range(folds):
        data = torch.load(out_dir / f"fold{fold}.pt", map_location="cpu", weights_only=False)
        fold_summary.append({k: data[k] for k in ("fold", "best_epoch", "clean_val_accuracy", "clean_val_macro_f1")})
        for position, index in enumerate(data["held_out"]["row_index"].tolist()):
            oof[index] = data["held_out"]["logits"][position].float()
            if at_layers:
                oof_at[index] = data["held_out"]["at"][position]
        order = data["validation"]["row_index"].tolist()
        logits = data["validation"]["logits"].float()
        validation_sum = logits if validation_sum is None else validation_sum + logits
        if at_layers:
            maps = data["validation"]["at"].float()
            at_sum = maps if at_sum is None else at_sum + maps
    validation_logits = validation_sum / folds

    def pack(indices, logits):
        return pack_split(corpus, labels_of, leaked, indices, logits)

    train_indices = sorted(oof)
    packed = {"source": f"BEATs crossfit {folds} folds, {config['teacher']['_recipe']}",
              "precision": "fp16 logits", "label_order": "labels.txt",
              "train": pack(train_indices, torch.stack([oof[i] for i in train_indices])),
              "validation": pack(order, validation_logits)}
    if at_layers:  # out-of-fold maps for train, mean of the fold teachers for validation
        packed["at_layers"], packed["at_source"] = at_layers, at_source
        packed["train"]["at_maps"] = torch.stack([oof_at[i] for i in train_indices]).numpy()
        packed["validation"]["at_maps"] = (at_sum / folds).half().numpy()
    torch.save(packed, out_dir / crossfit["packed"])
    summary = {"folds": fold_summary,
               "train_oof_accuracy": round(packed["train"]["accuracy"], 4),
               "train_oof_accuracy_clean": round(packed["train"]["accuracy_clean"], 4),
               "validation_accuracy": round(packed["validation"]["accuracy"], 4),
               "validation_accuracy_clean": round(packed["validation"]["accuracy_clean"], 4)}
    (out_dir / config["outputs"]["summary"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def pack_split(corpus: Corpus, labels_of: dict, leaked: set[str], indices: list[int], logits: torch.Tensor) -> dict:
    """Một split của gói teacher: mix_id, logit fp16, SNR, accuracy (toàn bộ / phần sạch)."""
    rows = [corpus.rows[i] for i in indices]
    target = np.array([labels_of[id(row)] for row in rows])
    predicted = logits.argmax(1).numpy()
    clean = np.array([row["noise_ytid"] not in leaked for row in rows])
    return {"mix_id": [row["sample_id"] for row in rows],
            "logits36": logits.numpy().astype(np.float16),
            "snr_db": np.array([float(row["target_snr_db"]) for row in rows], dtype=np.float32),
            "accuracy": float((predicted == target).mean()),
            "accuracy_clean": float((predicted[clean] == target[clean]).mean()) if clean.any() else float("nan")}


def command_atmaps(config: dict) -> None:
    """Bản đồ Attention Transfer từ trọng số fold đã lưu (`crossfit.save_weights`), không train lại.

    `crossfit.at_layers`, `crossfit.at_source` ("energy" | "rollout", xem collect_with_attention). Teacher nghe
    cột `teacher.source_column` (nhiễu sạch) như lúc train. Train: fold không thấy mẫu (out-of-fold); validation:
    trung bình 5 fold. Ghi gói logit + at_maps (`crossfit.at_packed`) cùng định dạng gói của `crossfit`.
    """
    device = seed_everything(config["experiment"]["seed"])
    out_dir = HERE / config["outputs"]["dir"]
    settings, crossfit, runtime = config["teacher"], config["crossfit"], config["runtime"]
    seed, folds = config["experiment"]["seed"], crossfit["folds"]
    layers, source = list(crossfit["at_layers"]), crossfit["at_source"]
    corpus = Corpus(config, column=settings["source_column"])
    train_rows = corpus.by_split[config["dataset"]["train_split"]]
    validation_rows = corpus.by_split[config["dataset"]["validation_split"]]
    leaked = read_ytids((HERE / crossfit["exclude_noise_ytids"]).resolve())
    if runtime["smoke_test"]:
        train_rows = smoke_subset(train_rows, runtime["smoke_train_rows"], seed)
        validation_rows = smoke_subset(validation_rows, runtime["smoke_validation_rows"], seed)
    fold_of = crossfit_folds(train_rows, folds, seed)
    eval_batch, workers = settings["validation_batch_size"], settings["workers"]
    oof, oof_at, validation_sum, at_sum, order = {}, {}, None, None, None
    for fold in range(folds):
        data = torch.load(out_dir / f"fold{fold}.pt", map_location="cpu", weights_only=False)
        if "state_dict" not in data:
            raise KeyError(f"fold{fold}.pt has no weights; rerun `teacher_crossfit.py crossfit` with crossfit.save_weights")
        model = build_model(config, device, film_enabled=False, trainable_blocks=settings["trainable_blocks"],
                            backbone=teacher_backbone(config))
        model.load_state_dict(data["state_dict"])
        held_out = [row for row in train_rows if fold_of[row["noise_ytid"]] == fold]
        out = collect_with_attention(model, corpus.loader(held_out, eval_batch, workers, False, device), device,
                                     layers, source)
        for position, index in enumerate(out["row_index"].tolist()):
            oof[index], oof_at[index] = out["logits"][position].float(), out["at"][position]
        out = collect_with_attention(model, corpus.loader(validation_rows, eval_batch, workers, False, device), device,
                                     layers, source)
        order = out["row_index"].tolist()
        validation_sum = out["logits"].float() if validation_sum is None else validation_sum + out["logits"].float()
        at_sum = out["at"].float() if at_sum is None else at_sum + out["at"].float()
        print(f"fold {fold}: {source} maps of layers {layers} for {len(held_out)} held-out rows", flush=True)
        del model
        torch.cuda.empty_cache()
    labels_of = {id(row): int(corpus.label_to_index[row["label_names"]]) for row in corpus.rows}
    train_indices = sorted(oof)
    packed = {"source": f"BEATs crossfit {folds} folds, {source} maps", "precision": "fp16 logits",
              "label_order": "labels.txt", "at_layers": layers, "at_source": source,
              "train": pack_split(corpus, labels_of, leaked, train_indices, torch.stack([oof[i] for i in train_indices])),
              "validation": pack_split(corpus, labels_of, leaked, order, validation_sum / folds)}
    packed["train"]["at_maps"] = torch.stack([oof_at[i] for i in train_indices]).numpy()
    packed["validation"]["at_maps"] = (at_sum / folds).half().numpy()
    torch.save(packed, out_dir / crossfit["at_packed"])
    print(f"{source} maps -> {out_dir / crossfit['at_packed']}: train OOF acc {packed['train']['accuracy']:.4f}", flush=True)


COMMANDS = {"crossfit": command_crossfit, "features": command_features, "atmaps": command_atmaps}


def main() -> None:
    parser = argparse.ArgumentParser(description="BEATs cross-fit teacher: train folds, export features")
    parser.add_argument("command", choices=tuple(COMMANDS))
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    config_path = arguments.config if arguments.config.is_absolute() else (Path.cwd() / arguments.config)
    print(f"command={arguments.command} config={config_path}", flush=True)
    COMMANDS[arguments.command](load_config(config_path))


if __name__ == "__main__":
    main()
