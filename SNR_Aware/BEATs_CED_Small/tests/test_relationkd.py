from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from relationkd import (  # noqa: E402
    balanced_aux_scale,
    CRDHead,
    RelationKDConfig,
    RelationKDLoss,
    linear_cka,
    noise_energy_mask,
    relation_mask,
    sample_relation_loss,
    spatial_cka_loss,
    spatial_similarity_loss,
)
from reviewkd import to_teacher_grid  # noqa: E402
from src.loss import KDMoELossConfig  # noqa: E402
from src.loss.kdmoe import load_membership  # noqa: E402

YAMLS = {name: HERE / "config" / "unified" / f"kdmoe_mn10_{name}.yaml"
         for name in ("spkd", "cka", "relation", "spkd_mask", "snrroute_relation", "crd")}


def test_spatial_loss_zero_on_same_relations_and_ignores_width():
    teacher = torch.randn(2, 192, 768)
    student = teacher @ torch.linalg.qr(torch.randn(768, 768))[0][:, :112]  # khác số kênh
    assert spatial_similarity_loss(teacher.clone(), teacher) == 0
    assert spatial_similarity_loss(student, teacher) > 0  # chiếu xuống 112 chiều làm đổi cosine
    scaled = teacher * torch.rand(2, 192, 1) * 5           # co giãn từng vị trí: cosine giữ nguyên
    assert spatial_similarity_loss(scaled, teacher) < 1e-10


def test_spatial_loss_excludes_diagonal():
    teacher = torch.randn(1, 16, 8)
    student = torch.randn(1, 16, 4)
    s, t = (torch.nn.functional.normalize(x, dim=-1) for x in (student, teacher))
    g_s, g_t = s @ s.transpose(1, 2), t @ t.transpose(1, 2)
    off = ~torch.eye(16, dtype=torch.bool)
    expected = torch.nn.functional.smooth_l1_loss(g_s[:, off], g_t[:, off])
    assert torch.allclose(spatial_similarity_loss(student, teacher, True), expected)
    with_diagonal = torch.nn.functional.smooth_l1_loss(g_s, g_t)  # đường chéo = 0 lỗi, kéo trung bình xuống
    assert torch.allclose(spatial_similarity_loss(student, teacher, False), with_diagonal)
    assert with_diagonal < expected


def test_cka_properties():
    x = torch.randn(3, 192, 112, dtype=torch.float64)
    rotation = torch.linalg.qr(torch.randn(112, 112, dtype=torch.float64))[0]
    assert torch.allclose(linear_cka(x, x).double(), torch.ones(3, dtype=torch.float64), atol=1e-5)
    # bất biến với quay trực giao, co giãn đều và tịnh tiến (CKA trừ trung bình theo vị trí)
    assert torch.allclose(linear_cka(x, 3.0 * x @ rotation + 1.0).double(), torch.ones(3, dtype=torch.float64),
                          atol=1e-5)
    y = torch.randn(3, 192, 768)
    assert 0 < spatial_cka_loss(x.float(), y) <= 1


def test_spatial_losses_detach_teacher_and_train_student():
    for loss in (lambda s, t: spatial_similarity_loss(s, t), spatial_cka_loss):
        student = torch.randn(2, 192, 112, requires_grad=True)
        teacher = torch.randn(2, 192, 768, requires_grad=True)
        loss(student, teacher).backward()
        assert teacher.grad is None
        assert student.grad is not None and student.grad.abs().sum() > 0


def test_grid_flatten_matches_teacher_token_order():
    """Student (B, C, F, T) -> (B, T * F, C) phải trùng thứ tự token BEATs (B, T, F, C).flatten(1, 2)."""
    f, t = 8, 24
    index = torch.arange(t * f, dtype=torch.float32).view(t, f)          # token BEATs: chỉ số t * 8 + f
    student_map = index.T.reshape(1, 1, f, t)                            # cùng nội dung, bố cục (F, T)
    grid = to_teacher_grid(student_map, (f, t), 1.0, 1.0)
    flat = grid.permute(0, 3, 2, 1).flatten(1, 2)[0, :, 0]
    assert torch.allclose(flat, index.flatten(), atol=1e-4)  # grid_sample tại tâm ô: sai số ~4e-6


def test_relation_mask_drops_diagonal_duplicates_and_same_noise():
    ids = ["a", "b", "c", "c"]
    ytids = ["n1", "n1", "n2", "n3"]
    mask = relation_mask(ids, ytids, torch.device("cpu"))
    assert not mask.diagonal().any()
    assert not mask[0, 1] and not mask[1, 0]      # cùng noise_ytid
    assert not mask[2, 3] and not mask[3, 2]      # cùng sample_id
    assert mask[0, 2] and mask[1, 3]
    assert torch.equal(mask, mask.T)


def test_sample_relation_loss_masks_and_gradients():
    student = torch.randn(6, 1280, requires_grad=True)
    teacher = torch.randn(6, 768, requires_grad=True)
    mask = relation_mask([str(i) for i in range(6)], ["x", "x", "y", "y", "z", "z"], torch.device("cpu"))
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    loss, stats = sample_relation_loss(student, teacher, mask, labels)
    assert stats["rel_pairs"] == 24 and "rel_same" in stats and "rel_diff" in stats
    loss.backward()
    assert teacher.grad is None and student.grad.abs().sum() > 0
    empty, stats = sample_relation_loss(student, teacher, torch.zeros(6, 6, dtype=torch.bool))
    assert empty == 0 and stats["rel_pairs"] == 0


def test_configs_one_loss_per_run():
    spkd, cka, relation = (RelationKDConfig.from_yaml(YAMLS[n]) for n in ("spkd", "cka", "relation"))
    assert spkd.spatial_kd_type == "smooth_l1" and spkd.spatial_kd_weight > 0 and not spkd.relation_kd_weight
    assert cka.spatial_kd_type == "cka" and cka.spatial_kd_weight > 0 and not cka.relation_kd_weight
    assert relation.relation_kd_weight > 0 and not relation.spatial_kd_weight
    try:
        RelationKDConfig(spatial_kd_weight=0.1, relation_kd_weight=0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("two relation losses in one run must be rejected")


def test_loss_adds_weighted_aux_to_base_total():
    loss_config = KDMoELossConfig.from_yaml(YAMLS["spkd"])
    relation = RelationKDConfig(spatial_kd_weight=0.5)
    criterion = RelationKDLoss(loss_config, load_membership(HERE / "config" / "membership.npy"), relation)
    batch_size = 4
    output = SimpleNamespace(logits=torch.randn(batch_size, 36), router_logits=torch.randn(batch_size, 3),
                             e=torch.randn(batch_size, 1280), gate=torch.ones(1), noise_mel=None, noise_target=None,
                             spatial=torch.randn(batch_size, 192, 112))
    batch = {"label": torch.randint(0, 36, (batch_size,)), "group": torch.randint(0, 3, (batch_size,)),
             "teacher_logits36": torch.randn(batch_size, 36), "noise_ytid": ["a", "a", "b", "b"],
             "teacher_hint": torch.randn(batch_size, 24, 8, 768)}
    terms = criterion(output, batch)
    base = terms["ce"] + 0.1 * terms["cons"] + terms["kd"] + terms["route_ce"]
    assert torch.allclose(terms["aux"], 0.5 * terms["spatial"])
    assert torch.allclose(terms["total"], base + terms["aux"])
    assert criterion.last_terms is terms


def test_noise_energy_mask_follows_loud_cells_with_floor_and_mean_one():
    logmel = torch.full((2, 1, 128, 384), -1.0)
    logmel[0, :, :16, :16] = 1.0                       # góc tần số thấp, đầu clip: ô (t=0, f=0) to nhất
    mask = noise_energy_mask(logmel, (8, 24), 0.81, 1.0, temperature=0.5, floor=0.1)
    assert mask.shape == (2, 192) and not mask.requires_grad
    assert torch.allclose(mask.mean(dim=1), torch.ones(2), atol=1e-5)
    assert mask[0].argmax() == 0 and mask[0, 0] > 5 * mask[0, -1]
    assert mask.min() >= 0.1 / mask.max(dim=1).values.max()  # chặn dưới trước khi chuẩn hoá
    assert torch.allclose(mask[1], torch.ones(192), atol=1e-5)  # nhiễu đều -> mặt nạ phẳng


def test_weighted_spatial_loss_reduces_to_unweighted_and_detaches_weight():
    student, teacher = torch.randn(2, 192, 112, requires_grad=True), torch.randn(2, 192, 768)
    flat = spatial_similarity_loss(student, teacher, True, torch.ones(2, 192))
    assert torch.allclose(flat, spatial_similarity_loss(student, teacher, True))
    leaf = torch.rand(2, 192, requires_grad=True)
    spatial_similarity_loss(student, teacher, True, leaf + 0.1).backward()
    assert leaf.grad is None and student.grad.abs().sum() > 0


def test_crd_head_negatives_and_gradients():
    head = CRDHead(1280, 768, 256, 0.07, 64)
    bank_labels = torch.arange(36).repeat(10)
    head.set_bank(torch.randn(len(bank_labels), 768), bank_labels)
    labels = torch.tensor([0, 1, 2, 3])
    index = head.sample_negatives(labels)
    assert index.shape == (4, 64) and not (bank_labels[index] == labels[:, None]).any()
    student = torch.randn(4, 1280, requires_grad=True)
    teacher = torch.randn(4, 768, requires_grad=True)
    loss = head(student, teacher, labels)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert teacher.grad is None and student.grad.abs().sum() > 0
    assert head.embed_s.weight.grad is not None and head.embed_t.weight.grad is not None
    assert "bank" not in head.state_dict()  # bank không lưu vào checkpoint


def test_r4_r6_configs():
    mask = RelationKDConfig.from_yaml(YAMLS["spkd_mask"])
    assert mask.spatial_mask and mask.spatial_kd_type == "smooth_l1" and mask.spatial_kd_weight > 0
    assert RelationKDConfig.from_yaml(YAMLS["snrroute_relation"]).relation_kd_weight > 0
    crd = RelationKDConfig.from_yaml(YAMLS["crd"])
    assert crd.crd_weight > 0 and not crd.spatial_kd_weight and not crd.relation_kd_weight
    for bad in ({"spatial_kd_weight": 1.0, "crd_weight": 1.0}, {"spatial_kd_weight": 1.0, "spatial_kd_type": "cka",
                                                                 "spatial_mask": True}):
        try:
            RelationKDConfig(**bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} must be rejected")

def test_adaptive_aux_scale_multiplies_aux_and_keeps_raw():
    """aux_target_ratio > 0: loss phụ nhân aux_scale (bắt đầu 0 = chưa hiệu chỉnh), giữ aux_raw cho phép đo gradient."""
    loss_config = KDMoELossConfig.from_yaml(YAMLS["spkd"])
    relation = RelationKDConfig(spatial_kd_weight=0.5, aux_target_ratio=0.2)
    criterion = RelationKDLoss(loss_config, load_membership(HERE / "config" / "membership.npy"), relation)
    assert criterion.aux_scale == 0.0
    criterion.aux_scale = 3.0
    batch_size = 4
    output = SimpleNamespace(logits=torch.randn(batch_size, 36), router_logits=torch.randn(batch_size, 3),
                             e=torch.randn(batch_size, 1280), gate=torch.ones(1), noise_mel=None, noise_target=None,
                             spatial=torch.randn(batch_size, 192, 112))
    batch = {"label": torch.randint(0, 36, (batch_size,)), "group": torch.randint(0, 3, (batch_size,)),
             "teacher_logits36": torch.randn(batch_size, 36), "noise_ytid": ["a", "a", "b", "b"],
             "teacher_hint": torch.randn(batch_size, 24, 8, 768)}
    terms = criterion(output, batch)
    base = terms["ce"] + 0.1 * terms["cons"] + terms["kd"] + terms["route_ce"]
    assert torch.allclose(terms["aux_raw"], 0.5 * terms["spatial"])
    assert torch.allclose(terms["aux"], 3.0 * 0.5 * terms["spatial"])
    assert torch.allclose(terms["total"], base + terms["aux"])


def test_adaptive_aux_scale_off_by_default():
    loss_config = KDMoELossConfig.from_yaml(YAMLS["spkd"])
    criterion = RelationKDLoss(loss_config, load_membership(HERE / "config" / "membership.npy"),
                               RelationKDConfig(spatial_kd_weight=0.5))
    assert criterion.aux_scale == 1.0


def test_balanced_aux_scale_matches_target_ratio_then_smooths_and_clamps():
    # lần đầu (scale 0): đặt thẳng để |grad aux| = ratio * |grad CE|
    assert abs(balanced_aux_scale(0.0, grad_ce=2.0, grad_raw=0.5, ratio=0.2, ema=0.9) - 0.8) < 1e-9
    # sau đó: trung bình trượt trong không gian log, mục tiêu bị kẹp trong [scale / 10, scale * 10]
    s = balanced_aux_scale(1.0, grad_ce=1.0, grad_raw=1.0, ratio=0.2, ema=0.0)
    assert abs(s - 0.2) < 1e-9
    s = balanced_aux_scale(1.0, grad_ce=1.0, grad_raw=1e-9, ratio=0.2, ema=0.0)
    assert abs(s - 10.0) < 1e-9
    s = balanced_aux_scale(4.0, grad_ce=1.0, grad_raw=0.05, ratio=0.2, ema=0.5)   # mục tiêu 4 -> giữ 4
    assert abs(s - 4.0) < 1e-9
    # gradient loss phụ bằng 0 / không hữu hạn: giữ nguyên
    assert balanced_aux_scale(2.0, grad_ce=1.0, grad_raw=0.0, ratio=0.2, ema=0.9) == 2.0
    assert balanced_aux_scale(2.0, grad_ce=float("nan"), grad_raw=1.0, ratio=0.2, ema=0.9) == 2.0
