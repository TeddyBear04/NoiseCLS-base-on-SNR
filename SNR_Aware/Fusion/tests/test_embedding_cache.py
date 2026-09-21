import csv

import numpy as np
import torch

from dataset.embedding_cache import BRANCH_DIMS, EmbeddingCacheDataset, embed_dpcrn

SAMPLES = 64000


def test_branch_dims_sum_to_fusion_input():
    assert sum(BRANCH_DIMS.values()) == 1792


def test_embed_dpcrn_returns_one_row_per_clip():
    from dataset.embedding_cache import DPCRN_ROOT, _load_module

    module = _load_module(DPCRN_ROOT / "models" / "dpcrn_noise.py", "fusion_dpcrn_noise")
    DPCRNNoiseClassifier = module.DPCRNNoiseClassifier

    model = DPCRNNoiseClassifier(classes_num=36).eval()
    assert embed_dpcrn(model, torch.randn(3, SAMPLES)).shape == (3, 256)


def test_both_branch_modules_load_in_one_process():
    """Guards the path-based loader that replaced package-name imports.

    Both sibling projects ship a ``models`` package; importing by package name
    cached the first and hid the second.
    """
    from pathlib import Path

    from dataset.embedding_cache import BEATS_ROOT, DPCRN_ROOT, _load_module

    beats = _load_module(BEATS_ROOT / "models" / "beats_loader.py", "fusion_beats_loader")
    dpcrn = _load_module(DPCRN_ROOT / "models" / "dpcrn_noise.py", "fusion_dpcrn_noise")
    assert hasattr(beats, "load_beats_classes")
    assert hasattr(dpcrn, "DPCRNNoiseClassifier")
    assert Path(beats.__file__).parents[1].name == "BEATs_Experts"
    assert Path(dpcrn.__file__).parents[1].name == "DPCRN_Noise_Target"


def _fake_cache(tmp_path, rows=7):
    for branch, dim in BRANCH_DIMS.items():
        np.save(tmp_path / f"test_{branch}.npy",
                np.random.randn(rows, dim).astype(np.float16))
    with (tmp_path / "test_meta.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index", "label_index", "snr_db"])
        for index in range(rows):
            writer.writerow([index, index % 36, [-5, 0, 5, 10, 15, 20][index % 6]])
    (tmp_path / "labels.txt").write_text(
        "\n".join(f"label_{i}" for i in range(36)), encoding="utf-8")


def test_dataset_concatenates_branches_in_a_fixed_order(tmp_path):
    _fake_cache(tmp_path)
    dataset = EmbeddingCacheDataset(tmp_path, "test")
    assert len(dataset) == 7
    assert dataset.feature_dim == 1792
    item = dataset[3]
    assert item["features"].shape == (1792,)
    assert item["features"].dtype.is_floating_point


def test_dataset_rejects_misaligned_branches(tmp_path):
    _fake_cache(tmp_path)
    np.save(tmp_path / "test_dpcrn_high.npy",
            np.random.randn(6, 256).astype(np.float16))   # one row short
    try:
        EmbeddingCacheDataset(tmp_path, "test")
    except ValueError as error:
        assert "row" in str(error).lower()
    else:
        raise AssertionError("misaligned cache must raise")


def test_write_cache_round_trips_in_loader_order(tmp_path):
    """Row alignment is the cache's contract, and nothing else checks it.

    A drift here would train the fusion head on mismatched labels without
    failing visibly, so the writer is exercised end to end with stand-in
    embedders that encode each row's index and branch into its vector.
    """
    import dataset.embedding_cache as embedding_cache

    batch_sizes = [3, 4, 2]
    codes = {"beats_low": 1, "beats_mid": 2, "dpcrn_high": 3}
    levels = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]

    class FakeLoader:
        """Irregular batch sizes, so a per-batch ordering bug cannot hide."""

        def __init__(self):
            self.batches, row = [], 0
            for size in batch_sizes:
                self.batches.append({
                    "mixture": torch.arange(row, row + size).float().unsqueeze(1),
                    "target": torch.arange(row, row + size) % 36,
                    "snr": torch.tensor([levels[(row + i) % 6] for i in range(size)]),
                })
                row += size

        def __iter__(self):
            return iter(self.batches)

        def __len__(self):
            return len(self.batches)

    def make_embedder(code, width):
        def embed(_model, waveform):
            return waveform[:, 0].unsqueeze(1).repeat(1, width) * 100 + code
        return embed

    original = embedding_cache.EMBEDDERS
    embedding_cache.EMBEDDERS = {
        name: make_embedder(codes[name], embedding_cache.BRANCH_DIMS[name])
        for name in embedding_cache.BRANCH_ORDER
    }
    try:
        embedding_cache.write_cache(
            {name: None for name in embedding_cache.BRANCH_ORDER},
            FakeLoader(), "probe", tmp_path, torch.device("cpu"),
        )
    finally:
        embedding_cache.EMBEDDERS = original

    loader = FakeLoader()
    expected_targets = torch.cat([batch["target"] for batch in loader.batches])
    expected_snrs = torch.cat([batch["snr"] for batch in loader.batches])

    cached = embedding_cache.EmbeddingCacheDataset(tmp_path, "probe")
    assert len(cached) == sum(batch_sizes)
    start = 0
    for name in embedding_cache.BRANCH_ORDER:
        width = embedding_cache.BRANCH_DIMS[name]
        for row in range(len(cached)):
            piece = cached[row]["features"][start:start + width]
            assert torch.allclose(piece, torch.full((width,), row * 100.0 + codes[name]), atol=0.5)
        start += width
    assert torch.equal(cached.targets, expected_targets)
    assert torch.allclose(cached.snrs, expected_snrs)
