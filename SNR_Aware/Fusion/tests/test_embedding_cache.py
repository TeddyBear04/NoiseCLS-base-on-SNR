import torch

from dataset.embedding_cache import BRANCH_DIMS, embed_dpcrn

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
