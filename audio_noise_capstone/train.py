import argparse
import random
from pathlib import Path

import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from noise_pipeline.data import NoiseDataset
from noise_pipeline.model import NoiseNet


def make_logmel(device):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=16_000, n_fft=512, win_length=400, hop_length=160,
        n_mels=64, f_min=0, f_max=8_000, power=2.0,
    ).to(device)


def logmel(transform, waveform):
    return torch.log(transform(waveform).squeeze(1).clamp_min(1e-6))


def run_epoch(model, loader, transform, optimizer, device, separation_weight, pos_weight, train):
    model.train(train)
    bce, mse = nn.BCEWithLogitsLoss(pos_weight=pos_weight), nn.MSELoss()
    totals = torch.zeros(3, device=device)
    all_predictions, all_targets = [], []
    for batch in loader:
        mixture = batch["mixture"].to(device)
        oracle = batch["oracle_noise"].to(device)
        target = batch["target"].to(device)
        mixture_mel = logmel(transform, mixture)
        oracle_mel = logmel(transform, oracle)
        with torch.set_grad_enabled(train):
            logits, noise_hat, _ = model(mixture_mel)
            cls_loss = bce(logits, target)
            sep_loss = mse(noise_hat, oracle_mel)
            loss = cls_loss + separation_weight * sep_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        all_predictions.append((logits.sigmoid() >= 0.5).detach().cpu())
        all_targets.append(target.bool().detach().cpu())
        totals += torch.tensor([loss.item(), cls_loss.item(), sep_loss.item()], device=device)
    average = (totals / max(len(loader), 1)).tolist()
    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    tp = (predictions & targets).sum(dim=0).float()
    fp = (predictions & ~targets).sum(dim=0).float()
    fn = (~predictions & targets).sum(dim=0).float()
    eps = 1e-8
    class_precision = tp / (tp + fp + eps)
    class_recall = tp / (tp + fn + eps)
    class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall + eps)
    micro_precision = tp.sum() / (tp.sum() + fp.sum() + eps)
    micro_recall = tp.sum() / (tp.sum() + fn.sum() + eps)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + eps)
    return {
        "loss": average[0], "cls": average[1], "sep": average[2],
        "micro_precision": micro_precision.item(), "micro_recall": micro_recall.item(),
        "micro_f1": micro_f1.item(), "macro_f1": class_f1.mean().item(),
        "exact": (predictions == targets).all(dim=1).float().mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="mix-dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--separation-weight", type=float, default=0.3)
    parser.add_argument("--pos-weight-power", type=float, default=1.0)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit each split for a medium-sized experiment")
    parser.add_argument("--output", default="checkpoints/best.pt")
    args = parser.parse_args()

    random.seed(2026)
    torch.manual_seed(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    limit = 16 if args.smoke_test else args.limit
    train_set = NoiseDataset(args.data_root, "train_single", args.seconds, limit)
    val_set = NoiseDataset(args.data_root, "validation_single", args.seconds, limit)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_set, args.batch_size, num_workers=args.workers)
    model = NoiseNet(num_classes=train_set.num_classes).to(device)
    transform = make_logmel(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    positive = train_set.class_counts()
    # Counter sparse positives without allowing very rare classes to dominate.
    raw_pos_weight = (len(train_set) - positive) / positive.clamp_min(1)                                                                                          
    pos_weight = raw_pos_weight.pow(args.pos_weight_power).clamp(1, args.max_pos_weight).to(device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    epochs = 1 if args.smoke_test else args.epochs
    print(f"device={device} train={len(train_set)} validation={len(val_set)}")
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, transform, optimizer, device,
            args.separation_weight, pos_weight, True,
        )
        val_metrics = run_epoch(
            model, val_loader, transform, optimizer, device,
            args.separation_weight, pos_weight, False,
        )
        print(f"epoch={epoch} train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
              f"val_cls={val_metrics['cls']:.4f} val_sep={val_metrics['sep']:.4f} "
              f"micro_p={val_metrics['micro_precision']:.4f} micro_r={val_metrics['micro_recall']:.4f} "
              f"micro_f1={val_metrics['micro_f1']:.4f} macro_f1={val_metrics['macro_f1']:.4f} "
              f"exact={val_metrics['exact']:.4f}")
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            torch.save({"model": model.state_dict(), "args": vars(args)}, output)
    print(f"checkpoint={output}")


if __name__ == "__main__":
    main()
