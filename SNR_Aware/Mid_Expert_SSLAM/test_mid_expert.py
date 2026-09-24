import torch

from mid_expert_lib import mid_slice_mask, normalize_snr
from mid_expert_lib import FiLM
from mid_expert_lib import sample_negatives
from mid_expert_lib import Projection, CRDLoss


def _unit(x):
    return torch.nn.functional.normalize(x, dim=-1)


def test_mid_slice_mask_selects_only_5_and_10_db():
    snr = torch.tensor([-5.0, 0.0, 5.0, 10.0, 15.0, 20.0])
    mask = mid_slice_mask(snr, low=5.0, high=10.0)
    assert mask.tolist() == [False, False, True, True, False, False]


def test_mid_slice_mask_includes_interior_values():
    snr = torch.tensor([4.9, 5.0, 7.5, 10.0, 10.1])
    assert mid_slice_mask(snr, low=5.0, high=10.0).tolist() == [
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
    from mid_expert_lib import film_deviation
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
