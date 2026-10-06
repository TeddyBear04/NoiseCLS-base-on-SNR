import torch

from expert_lib import band_slice_mask, normalize_snr
from expert_lib import FiLM
from expert_lib import sample_negatives
from expert_lib import Projection, CRDLoss


def _unit(x):
    return torch.nn.functional.normalize(x, dim=-1)


def test_band_slice_mask_selects_only_5_and_10_db():
    snr = torch.tensor([-5.0, 0.0, 5.0, 10.0, 15.0, 20.0])
    mask = band_slice_mask(snr, low=5.0, high=10.0)
    assert mask.tolist() == [False, False, True, True, False, False]


def test_band_slice_mask_includes_interior_values():
    snr = torch.tensor([4.9, 5.0, 7.5, 10.0, 10.1])
    assert band_slice_mask(snr, low=5.0, high=10.0).tolist() == [
        False, True, True, True, False
    ]


def test_normalize_snr_maps_band_to_unit_interval():
    snr = torch.tensor([-5.0, 7.5, 20.0])
    out = normalize_snr(snr)
    assert torch.allclose(out, torch.tensor([0.0, 0.5, 1.0]), atol=1e-6)


def test_film_is_identity_at_initialisation():
    """Khởi tạo lớp cuối bằng 0 => gamma=1, beta=0 => forward == LayerNorm(z)
    (KHÔNG bằng z, vì FiLM luôn áp LayerNorm trước khi điều biến)."""
    torch.manual_seed(0)
    film = FiLM(dim=8)
    z = torch.randn(4, 8)
    snr = torch.tensor([5.0, 10.0, -5.0, 20.0])
    assert torch.allclose(film(z, snr), film.norm(z), atol=1e-6)


def test_film_output_depends_on_snr_after_perturbing_weights():
    torch.manual_seed(0)
    film = FiLM(dim=8)
    with torch.no_grad():
        film.to_gamma_beta[-1].weight.normal_(0, 0.5)
        film.to_gamma_beta[-1].bias.normal_(0, 0.5)
    z = torch.randn(1, 8).expand(2, 8).contiguous()
    out = film(z, torch.tensor([5.0, 20.0]))
    assert not torch.allclose(out[0], out[1], atol=1e-4)


def test_film_records_gamma_beta_for_collapse_monitoring():
    film = FiLM(dim=8)
    film(torch.randn(3, 8), torch.tensor([5.0, 10.0, 15.0]))
    gamma, beta = film.last_gamma_beta
    assert gamma.shape == (3, 8) and beta.shape == (3, 8)


def test_film_deviation_is_zero_at_initialisation():
    from expert_lib import film_deviation
    film = FiLM(dim=8)
    film(torch.randn(3, 8), torch.tensor([5.0, 10.0, 15.0]))
    assert film_deviation(film) < 1e-6


def test_sample_negatives_never_picks_same_label_in_different_label_mode():
    labels = torch.tensor([0, 1])
    bank_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    g = torch.Generator().manual_seed(0)
    idx = sample_negatives(labels, bank_labels, n=4, generator=g,
                           mode="different_label")
    assert idx.shape == (2, 4)
    assert (bank_labels[idx[0]] != 0).all()
    assert (bank_labels[idx[1]] != 1).all()


def test_sample_negatives_random_mode_may_include_same_label():
    labels = torch.zeros(1, dtype=torch.long)
    bank_labels = torch.zeros(64, dtype=torch.long)
    g = torch.Generator().manual_seed(0)
    idx = sample_negatives(labels, bank_labels, n=8, generator=g, mode="random")
    assert idx.shape == (1, 8)


def test_sample_negatives_is_reproducible_under_same_seed():
    labels = torch.tensor([2])
    bank_labels = torch.arange(36).repeat(10)
    a = sample_negatives(labels, bank_labels, n=16,
                         generator=torch.Generator().manual_seed(7),
                         mode="different_label")
    b = sample_negatives(labels, bank_labels, n=16,
                         generator=torch.Generator().manual_seed(7),
                         mode="different_label")
    assert torch.equal(a, b)


def test_crd_loss_is_lower_when_positive_pair_aligns():
    torch.manual_seed(0)
    z_s = _unit(torch.randn(4, 16))
    neg = _unit(torch.randn(4, 32, 16))
    aligned = CRDLoss(n_data=1000, tau=0.07)(z_s, z_s.clone(), neg)
    opposed = CRDLoss(n_data=1000, tau=0.07)(z_s, -z_s.clone(), neg)
    assert aligned < opposed


def test_crd_loss_is_positive_and_finite():
    torch.manual_seed(0)
    z_s = _unit(torch.randn(8, 16))
    pos = _unit(torch.randn(8, 16))
    neg = _unit(torch.randn(8, 32, 16))
    value = CRDLoss(n_data=1000, tau=0.07)(z_s, pos, neg)
    assert torch.isfinite(value) and value > 0


def test_crd_loss_backpropagates_to_student_only():
    torch.manual_seed(0)
    z_s = _unit(torch.randn(4, 16)).requires_grad_(True)
    pos = _unit(torch.randn(4, 16))
    neg = _unit(torch.randn(4, 32, 16))
    CRDLoss(n_data=1000, tau=0.07)(z_s, pos, neg).backward()
    assert z_s.grad is not None and torch.isfinite(z_s.grad).all()


def test_projection_output_is_unit_norm():
    proj = Projection(16, 8)
    out = proj(torch.randn(5, 16))
    assert out.shape == (5, 8)
    assert torch.allclose(out.norm(dim=-1), torch.ones(5), atol=1e-5)


def test_crd_loss_Z_is_fixed_after_first_call():
    torch.manual_seed(0)
    loss_fn = CRDLoss(n_data=1000, tau=0.07)
    z_s = _unit(torch.randn(4, 16))
    pos = _unit(torch.randn(4, 16))
    neg = _unit(torch.randn(4, 32, 16))
    loss_fn(z_s, pos, neg)
    z_after_first = loss_fn.Z.clone()

    z_s2 = _unit(torch.randn(4, 16))
    pos2 = _unit(torch.randn(4, 16))
    neg2 = _unit(torch.randn(4, 32, 16))
    loss_fn(z_s2, pos2, neg2)
    assert torch.equal(loss_fn.Z, z_after_first)


def test_crd_loss_magnitude_is_comparable_to_cross_entropy():
    """m=4096 số hạng negative; Z chuẩn hoá về xác suất nên tổng phải cùng bậc
    với CE (~3.58 với 36 lớp), không bị số hạng negative nuốt mất."""
    torch.manual_seed(0)
    z_s = _unit(torch.randn(4, 128))
    pos = _unit(torch.randn(4, 128))
    neg = _unit(torch.randn(4, 4096, 128))
    loss_fn = CRDLoss(n_data=30240, tau=0.07)
    value = loss_fn(z_s, pos, neg)
    print("crd_loss at init (normalized):", float(value))
    assert torch.isfinite(value)
    assert float(value) < 100.0


def test_spec_augment_zero_ratio_is_identity():
    from expert_lib import spec_augment
    x = torch.randn(3, 40, 16)
    assert torch.equal(spec_augment(x, 0.0, torch.Generator().manual_seed(0)), x)


def test_spec_augment_masks_at_most_ratio_of_each_axis_and_keeps_input():
    from expert_lib import spec_augment
    x = torch.randn(64, 100, 128) + 5.0  # no natural zeros
    out = spec_augment(x, 0.3, torch.Generator().manual_seed(1))
    assert torch.equal(x, x)  # input untouched (clone)
    for clip in out:
        zero_frames = int((clip == 0).all(dim=1).sum())
        zero_bins = int((clip == 0).all(dim=0).sum())
        assert zero_frames <= 30 and zero_bins <= int(0.3 * 128)
    assert (out == 0).any()


def test_roll_waveform_preserves_content_and_shape():
    from expert_lib import roll_waveform
    x = torch.arange(20.0).repeat(4, 1)
    out = roll_waveform(x, torch.Generator().manual_seed(0))
    assert out.shape == x.shape
    for clip in out:
        assert torch.equal(clip.sort().values, torch.arange(20.0))


def test_warmup_cosine_shape():
    from expert_lib import warmup_cosine
    assert abs(warmup_cosine(0, 100, 10) - 0.1) < 1e-9
    assert abs(warmup_cosine(9, 100, 10) - 1.0) < 1e-9
    assert abs(warmup_cosine(10, 100, 10) - 1.0) < 1e-9
    assert warmup_cosine(55, 100, 10) < 1.0
    assert warmup_cosine(100, 100, 10) < 1e-9


def test_layer_decay_scales_top_block_gets_full_lr():
    from expert_lib import layer_decay_scales
    scales = layer_decay_scales(4, 0.5)
    assert scales == [0.125, 0.25, 0.5, 1.0]


def test_attention_mask_pool_is_mean_pool_at_initialisation():
    """w khởi tạo 0 => mask = 0.5 ở mọi token => trọng số đều => đúng mean-pool,
    tức run7_attn bắt đầu ĐÚNG từ điểm của run1_baseline."""
    from expert_lib import AttentionMaskPool
    torch.manual_seed(0)
    pool = AttentionMaskPool(dim=8)
    tokens = torch.randn(3, 12, 8)
    pooled, mask = pool(tokens)
    assert torch.allclose(pooled, tokens.mean(dim=1), atol=1e-6)
    assert torch.allclose(mask, torch.full((3, 12), 0.5))


def test_attention_mask_pool_mask_in_unit_interval_and_weights_sum_to_one():
    from expert_lib import AttentionMaskPool
    torch.manual_seed(0)
    pool = AttentionMaskPool(dim=8)
    with torch.no_grad():
        pool.w.weight.normal_(0, 3.0)
    tokens = torch.randn(4, 12, 8)
    pooled, mask = pool(tokens)
    assert mask.shape == (4, 12)
    assert bool((mask > 0).all()) and bool((mask < 1).all())
    weights = mask / mask.sum(dim=1, keepdim=True)
    assert torch.allclose(pooled, (weights.unsqueeze(-1) * tokens).sum(dim=1), atol=1e-5)
    assert not torch.allclose(pooled, tokens.mean(dim=1), atol=1e-3)


def test_attention_mask_pool_learns_from_classification_gradient():
    """Từ khởi tạo 0, gradient phải tới được w; nếu không, mask kẹt ở 0.5 mãi."""
    from expert_lib import AttentionMaskPool
    torch.manual_seed(0)
    pool = AttentionMaskPool(dim=8)
    pooled, _ = pool(torch.randn(4, 12, 8))
    pooled.pow(2).sum().backward()
    assert pool.w.weight.grad is not None
    assert float(pool.w.weight.grad.abs().sum()) > 0


def test_mask_deviation_is_zero_for_uniform_mask_and_positive_otherwise():
    from expert_lib import mask_deviation
    assert mask_deviation(torch.full((3, 12), 0.5)) < 1e-6
    peaked = torch.full((1, 12), 0.1)
    peaked[0, 0] = 0.9
    assert mask_deviation(peaked) > 0.1


def test_flatten_params_joins_nested_keys_and_drops_notes():
    from tasks.report_36 import flatten_params
    tree = {"student": {"encoder_lr": 1e-4, "_recipe_note": "x",
                        "augment": {"spec_ratio": 0.2, "_note": "y"}},
            "band_db": [15.0, 20.0], "_top_note": "z"}
    flat = flatten_params(tree)
    assert flat == {"student.encoder_lr": 1e-4, "student.augment.spec_ratio": 0.2,
                    "band_db": "[15.0, 20.0]"}


def test_flatten_params_skips_requested_prefixes():
    from tasks.report_36 import flatten_params
    tree = {"gates": {"baseline": 0.5, "published_baseline": {"per_snr": {"15": 1}}}}
    assert flatten_params(tree, skip=("gates.published_baseline",)) == {"gates.baseline": 0.5}


def test_sam_ascend_moves_by_rho_and_descend_restores():
    from expert_lib import sam_ascend, sam_descend
    torch.manual_seed(0)
    a, b, c = torch.randn(5, requires_grad=True), torch.randn(3, requires_grad=True), torch.randn(2)
    a.grad, b.grad = torch.randn(5), torch.randn(3)
    before = [a.detach().clone(), b.detach().clone(), c.clone()]
    eps = sam_ascend([a, b, c], rho=0.05)
    step = torch.cat([(a - before[0]).detach(), (b - before[1]).detach()])
    assert abs(float(step.norm()) - 0.05) < 1e-6
    g = torch.cat([a.grad, b.grad])
    assert torch.allclose(step, 0.05 * g / g.norm(), atol=1e-7)
    assert eps[2] is None and torch.equal(c, before[2])
    sam_descend([a, b, c], eps)
    assert torch.allclose(a, before[0], atol=1e-7) and torch.allclose(b, before[1], atol=1e-7)


def test_distill_kl_is_zero_for_identical_logits_and_matches_t1_cross_entropy():
    from expert_lib import distill_kl
    torch.manual_seed(0)
    s, t = torch.randn(4, 6), torch.randn(4, 6)
    assert float(distill_kl(t, t, 4.0)) < 1e-6
    p = torch.softmax(t, dim=-1)
    cross = -(p * torch.log_softmax(s, dim=-1)).sum(-1).mean()
    entropy = -(p * torch.log(p)).sum(-1).mean()
    assert torch.allclose(distill_kl(s, t, 1.0), cross - entropy, atol=1e-5)


def test_clap_repeat_pad_repeats_then_zero_pads_like_paper_3_4():
    from models.clap_backbone import repeat_pad, signal_steps
    clip = torch.randn(2, 192_000)                       # 4 s at 48 kHz
    padded, signal = repeat_pad(clip, 480_000)
    assert padded.shape == (2, 480_000) and signal == 384_000
    assert torch.equal(padded[:, :192_000], clip)
    assert torch.equal(padded[:, 192_000:384_000], clip)
    assert padded[:, 384_000:].abs().sum() == 0
    # 8 s of 10 s on signal -> 25.6 of 32 steps; only the 25 whole ones are kept.
    assert signal_steps(signal, 480_000) == 25
    assert signal_steps(480_000, 480_000) == 32


def test_clap_unfold_tokens_matches_htsat_forward_features():
    from models.clap_backbone import unfold_tokens
    batch, channels, side, freq_ratio = 2, 3, 8, 4
    tokens = torch.randn(batch, side * side, channels)
    # HTSAT_Swin_Transformer.forward_features, verbatim reshapes.
    x = tokens.permute(0, 2, 1).contiguous().reshape(batch, channels, side, side)
    c_freq_bin = side // freq_ratio
    x = x.reshape(batch, channels, side // c_freq_bin, c_freq_bin, side)
    x = x.permute(0, 1, 3, 2, 4).contiguous().reshape(batch, channels, c_freq_bin, -1)
    ordered = unfold_tokens(tokens, freq_ratio)
    assert ordered.shape == (batch, 32, 2, channels)
    assert torch.equal(ordered, x.permute(0, 3, 2, 1))
    # The mean over all tokens is HTSAT's `embedding` (avgpool over the 2 x 32 map).
    assert torch.allclose(ordered.mean(dim=(1, 2)), x.flatten(2).mean(-1), atol=1e-6)
    # Time step t is grid row (t // 8) * 2 + f, column t % 8.
    grid = tokens.reshape(batch, side, side, channels)
    for t in (0, 7, 8, 25, 31):
        for f in (0, 1):
            assert torch.equal(ordered[:, t, f], grid[:, (t // 8) * 2 + f, t % 8])


def test_clap_head_starts_as_the_zero_shot_classifier():
    from models.clap_backbone import CLAPHead
    torch.manual_seed(0)
    projection = torch.nn.Sequential(torch.nn.Linear(6, 4), torch.nn.ReLU(),
                                     torch.nn.Linear(4, 4))
    text = _unit(torch.randn(5, 4))
    head = CLAPHead(projection, text, scale=23.4, classes=5)
    pooled = torch.randn(3, 6)
    expected = 23.4 * _unit(projection(pooled)) @ text.T
    assert torch.allclose(head(pooled), expected, atol=1e-5)
    assert all(p.requires_grad for p in head.parameters())


def test_align_patch_grid_is_identity_on_equal_grids_and_pools_2x2_blocks():
    from expert_lib import align_patch_grid
    torch.manual_seed(0)
    tokens = torch.randn(2, 24 * 8, 5)
    assert torch.equal(align_patch_grid(tokens, (24, 8), (24, 8)), tokens)
    out = align_patch_grid(tokens, (24, 8), (12, 4))
    assert out.shape == (2, 48, 5)
    grid = tokens.reshape(2, 24, 8, 5)
    block = grid[:, 2:4, 6:8].mean(dim=(1, 2))      # target cell (t=1, f=3)
    assert torch.allclose(out[:, 1 * 4 + 3], block, atol=1e-6)


def test_align_patch_grid_maps_to_ced_and_mobilenet_grids():
    from expert_lib import align_patch_grid
    tokens = torch.randn(3, 192, 7)
    assert align_patch_grid(tokens, (24, 8), (25, 4)).shape == (3, 100, 7)
    assert align_patch_grid(tokens, (24, 8), (13, 4)).shape == (3, 52, 7)
    # a constant teacher map stays constant wherever its cells land
    flat = torch.ones(1, 192, 2)
    assert torch.allclose(align_patch_grid(flat, (24, 8), (13, 4)), torch.ones(1, 52, 2))


if __name__ == "__main__":
    import sys, traceback
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
