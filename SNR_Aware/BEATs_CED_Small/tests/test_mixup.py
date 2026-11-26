"""Mixup stage 2 cho KD-MoE (ablation final-report.xlsx, 2026-11-20)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from relationkd import CRDHead  # noqa: E402
from src.loss import KDMoELoss, KDMoELossConfig  # noqa: E402
from src.train.kdmoe_trainer import KDMoETrainer  # noqa: E402


def _fake_trainer(prob=1.0, alpha=0.4, stage=2):
    config = SimpleNamespace(mixup_prob=prob, mixup_alpha=alpha)
    return SimpleNamespace(config=config, stage=stage, _mix_rng=np.random.default_rng(2027))


def test_mixup_permutes_whole_noise_pairs():
    trainer = _fake_trainer()
    ytids = ["a", "a", "b", "b", "c", "c", "d", "d"]
    batch = {"noise_ytid": ytids, "snr_est": torch.arange(8.0)}
    wave, noise = torch.randn(8, 100), torch.randn(8, 100)
    mixed, mixed_noise = KDMoETrainer._mixup(trainer, batch, wave, noise)
    lam, perm = batch["mix"]
    assert 0.0 <= lam <= 1.0
    for i in range(0, 8, 2):  # hai clip của một cặp trộn với hai clip của CÙNG một cặp khác
        assert ytids[perm[i]] == ytids[perm[i + 1]]
        assert perm[i] != perm[i + 1]
    assert torch.allclose(mixed, lam * wave + (1 - lam) * wave[perm])
    assert torch.allclose(mixed_noise, lam * noise + (1 - lam) * noise[perm])
    assert torch.allclose(batch["snr_est"], lam * torch.arange(8.0) + (1 - lam) * torch.arange(8.0)[perm])


def test_mixup_off_leaves_batch_untouched():
    for trainer in (_fake_trainer(prob=0.0), _fake_trainer(alpha=0.0), _fake_trainer(stage=1)):
        batch = {"noise_ytid": ["a", "a"]}
        wave = torch.randn(2, 10)
        out, _ = KDMoETrainer._mixup(trainer, batch, wave, None)
        assert out is wave and "mix" not in batch


def _loss_and_batch():
    membership = torch.zeros(36, 3)
    membership[torch.arange(36), torch.arange(36) % 3] = 1.0
    config = KDMoELossConfig(kd_temperature=4.0, kd_weight=1.0, kd_target="logits", route_ce_weight=1.0,
                             cons_weight=0.1, front_weight=1.0)
    torch.manual_seed(0)
    output = SimpleNamespace(logits=torch.randn(4, 36), router_logits=torch.randn(4, 3), noise_mel=None,
                             noise_target=None, e=torch.randn(4, 16))
    batch = {"label": torch.tensor([0, 1, 2, 3]), "group": torch.tensor([0, 1, 2, 0]),
             "teacher_logits36": torch.randn(4, 36), "noise_ytid": ["a", "a", "b", "b"], "snr_db": torch.zeros(4)}
    return KDMoELoss(config, membership), output, batch


def test_loss_with_identity_mix_equals_clean_loss():
    loss, output, batch = _loss_and_batch()
    clean = loss(output, batch)
    mixed = loss(output, {**batch, "mix": (1.0, torch.arange(4))})
    for key in ("ce", "kd", "route_ce", "total"):
        assert torch.allclose(clean[key], mixed[key], atol=1e-6), key


def test_loss_mixes_labels_and_teacher_targets():
    loss, output, batch = _loss_and_batch()
    perm = torch.tensor([2, 3, 0, 1])
    lam = 0.3
    terms = loss(output, {**batch, "mix": (lam, perm)})
    ce = loss.ce  # CE của loss (kèm label smoothing của config)
    expected = lam * ce(output.logits, batch["label"]) + (1 - lam) * ce(output.logits, batch["label"][perm])
    assert torch.allclose(terms["ce"], expected, atol=1e-6)
    p_t = (batch["teacher_logits36"] / 4.0).softmax(-1)
    p_t = lam * p_t + (1 - lam) * p_t[perm]
    kl = (p_t * (p_t.log() - (output.logits / 4.0).log_softmax(-1))).sum(-1).mean() * 16.0
    assert torch.allclose(terms["kd"], kl, atol=1e-5)


def test_crd_negatives_exclude_both_mixed_classes():
    head = CRDHead(16, 8, 4, 0.07, 64)
    head.set_bank(torch.randn(360, 8), torch.arange(360) % 36)
    labels, labels2 = torch.tensor([0, 5]), torch.tensor([7, 9])
    index = head.sample_negatives(labels, labels2)
    bank = head.bank_labels[index]
    assert not (bank == labels[:, None]).any() and not (bank == labels2[:, None]).any()
