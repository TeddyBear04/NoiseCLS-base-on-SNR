import torch

from dataset.embedding_cache import BRANCH_DIMS, embed_dpcrn

SAMPLES = 64000


def test_branch_dims_sum_to_fusion_input():
    assert sum(BRANCH_DIMS.values()) == 1792


def test_embed_dpcrn_returns_one_row_per_clip():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "DPCRN_Noise_Target"))
    from models.dpcrn_noise import DPCRNNoiseClassifier

    model = DPCRNNoiseClassifier(classes_num=36).eval()
    assert embed_dpcrn(model, torch.randn(3, SAMPLES)).shape == (3, 256)
