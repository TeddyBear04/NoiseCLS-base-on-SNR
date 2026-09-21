import torch

from dataset.embedding_cache import BRANCH_DIMS
from models.fusion_head import FusionHead


def test_forward_shapes():
    head = FusionHead(BRANCH_DIMS)
    output = head(torch.randn(4, 1792))
    assert output["logits"].shape == (4, 36)
    assert output["snr"].shape == (4,)


def test_each_branch_influences_the_output():
    """A branch whose slice is ignored would be dead weight in the cache.

    The perturbation must vary across the slice, not just shift it by a
    constant: per-branch LayerNorm normalizes each slice by its own mean and
    variance, so a *uniform* offset or positive rescale of a whole slice is
    mathematically invisible after normalization -- that would fail this test
    for every correctly-implemented branch, not just a dead one. Non-uniform
    noise changes the values LayerNorm actually preserves (the shape of the
    slice), so it genuinely distinguishes "this slice reaches the trunk" from
    "this slice is dropped".
    """
    head = FusionHead(BRANCH_DIMS).eval()
    features = torch.randn(1, 1792)
    with torch.no_grad():
        baseline = head(features)["logits"]
        start = 0
        for dim in BRANCH_DIMS.values():
            perturbed = features.clone()
            perturbed[:, start:start + dim] += 5.0 * torch.randn(1, dim)
            assert not torch.allclose(baseline, head(perturbed)["logits"], atol=1e-4)
            start += dim


def test_single_branch_mode_narrows_the_input():
    head = FusionHead({"dpcrn_high": 256})
    assert head(torch.randn(2, 256))["logits"].shape == (2, 36)
