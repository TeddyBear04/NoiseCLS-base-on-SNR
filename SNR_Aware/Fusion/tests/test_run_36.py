import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import dataset.embedding_cache as embedding_cache
from dataset.embedding_cache import BRANCH_DIMS, BRANCH_ORDER, EmbeddingCacheDataset
from models.fusion_head import FusionHead
from tasks.run_36 import (
    ABLATION_ROWS,
    evaluate_fusion,
    load_config,
    set_seed,
    slice_features,
    validate_config,
)
from utils.reporting import SNR_LEVELS

CONFIG = Path(__file__).resolve().parents[1] / "config" / "train_config.json"


def test_shipped_config_is_valid():
    config = load_config(CONFIG)
    validate_config(config)
    assert config["training"]["lambda_snr"] == 0.1


def test_ablation_covers_each_branch_alone_and_all_together():
    names = [name for name, _ in ABLATION_ROWS]
    assert names == ["beats_low", "beats_mid", "dpcrn_high", "fusion"]
    assert ABLATION_ROWS[-1][1] == ("beats_low", "beats_mid", "dpcrn_high")


def test_slice_features_extracts_the_right_columns_per_row():
    """Pins the column ranges every ablation row's numbers depend on.

    Each branch's slice must equal exactly its span of the 1792-wide cache
    row -- an off-by-one here would silently feed a single-branch row a
    scrambled mix of another branch's columns.
    """
    row = torch.arange(1792).unsqueeze(0).float()

    expected = {
        "beats_low": torch.arange(0, 768),
        "beats_mid": torch.arange(768, 1536),
        "dpcrn_high": torch.arange(1536, 1792),
        "fusion": torch.arange(0, 1792),
    }
    for name, selected in ABLATION_ROWS:
        result = slice_features(row, selected)
        assert torch.equal(result[0], expected[name].float())


def test_slice_features_is_order_independent():
    row = torch.arange(1792).unsqueeze(0).float()
    forward = slice_features(row, ("dpcrn_high", "beats_low"))
    backward = slice_features(row, ("beats_low", "dpcrn_high"))
    assert torch.equal(forward, backward)


def _write_probe_cache(tmp_path):
    """Cache a synthetic split with every label and every SNR level present."""
    labels = [f"label_{i}" for i in range(36)]
    levels = list(SNR_LEVELS)
    rows = 216  # lcm(36, 6): each label appears 6x, each SNR level appears 36x

    class FakeLoader:
        def __init__(self):
            self.batches = [{
                "mixture": torch.arange(rows).float().unsqueeze(1),
                "target": torch.arange(rows) % 36,
                "snr": torch.tensor([levels[i % 6] for i in range(rows)]),
            }]

        def __iter__(self):
            return iter(self.batches)

        def __len__(self):
            return len(self.batches)

    def make_embedder(width):
        def embed(_model, waveform):
            return torch.randn(waveform.shape[0], width)
        return embed

    original = embedding_cache.EMBEDDERS
    embedding_cache.EMBEDDERS = {
        name: make_embedder(embedding_cache.BRANCH_DIMS[name]) for name in BRANCH_ORDER
    }
    try:
        embedding_cache.write_cache(
            {name: None for name in BRANCH_ORDER},
            FakeLoader(), "probe", tmp_path, torch.device("cpu"),
        )
    finally:
        embedding_cache.EMBEDDERS = original

    (tmp_path / "labels.txt").write_text("\n".join(labels), encoding="utf-8")
    return labels, rows


def test_evaluate_fusion_returns_complete_report(tmp_path):
    """Covers the ~50 lines producing every published number with no prior coverage.

    Regression bait: dropping an SNR level from the iteration, forgetting a
    label in per_class, or miscounting samples would each slip through
    silently without this shape/completeness check.
    """
    labels, rows = _write_probe_cache(tmp_path)
    dataset = EmbeddingCacheDataset(tmp_path, "probe")
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    set_seed(2026)
    model = FusionHead(BRANCH_DIMS)
    report = evaluate_fusion(model, loader, torch.device("cpu"), labels)

    assert report["samples"] == rows
    assert set(report["per_snr"].keys()) == {f"{level:g}" for level in SNR_LEVELS}
    for values in report["per_snr"].values():
        assert "snr_mae" in values
        assert values["samples"] > 0
    assert set(report["per_class"].keys()) == set(labels)
    assert set(report["label_by_snr_f1"].keys()) == set(labels)
    for row in report["label_by_snr_f1"].values():
        assert set(row.keys()) == {f"{level:g}" for level in SNR_LEVELS}
    assert isinstance(report["snr_mae"], float)
    assert "macro_f1" in report["overall"]


def test_set_seed_makes_fusion_head_init_reproducible():
    """Pins the property ablation's per-row reseed depends on.

    Without a fresh set_seed() before each ABLATION_ROWS iteration, a row's
    weight init depends on how many previous rows already trained -- this
    checks that seeding alone is enough to make two builds identical.
    """
    set_seed(2026)
    first = FusionHead(BRANCH_DIMS)
    set_seed(2026)
    second = FusionHead(BRANCH_DIMS)
    for key in first.state_dict():
        assert torch.equal(first.state_dict()[key], second.state_dict()[key])
