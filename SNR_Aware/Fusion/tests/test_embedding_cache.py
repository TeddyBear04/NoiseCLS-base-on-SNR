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
