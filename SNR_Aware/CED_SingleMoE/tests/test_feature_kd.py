"""GFD-Cosine: loss, nạp vector gộp teacher theo sample_id, mixup nhất quán và một bước train qua KDTrainer."""

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.dataset import load_teacher_features
from src.feature_kd import GlobalCosineFeatureKD
from src.kd_trainer import KDTrainer, apply_mixup_with_teachers


def test_loss_is_one_minus_cosine_after_projector():
    torch.manual_seed(0)
    fkd = GlobalCosineFeatureKD(8, 16)
    s, t = torch.randn(4, 8), torch.randn(4, 16)
    expected = 1.0 - F.cosine_similarity(fkd.projectors[0](s), t, dim=-1).mean()
    assert torch.allclose(fkd([s], t[:, None]), expected, atol=1e-6)
    # cùng hướng với teacher -> 0, ngược hướng -> 2
    z = fkd.projectors[0](s).detach()[:, None]
    assert fkd([s], 3.0 * z).item() < 1e-5
    assert abs(fkd([s], -z).item() - 2.0) < 1e-5


def test_multi_tap_is_mean_of_per_tap_losses():
    torch.manual_seed(0)
    fkd = GlobalCosineFeatureKD(8, 16, num_taps=3)
    s, t = [torch.randn(4, 8) for _ in range(3)], torch.randn(4, 3, 16)
    per_tap = [1.0 - F.cosine_similarity(fkd.projectors[k](s[k]), t[:, k], dim=-1).mean() for k in range(3)]
    assert torch.allclose(fkd(s, t), torch.stack(per_tap).mean(), atol=1e-6)


def test_mask_drops_rows_without_teacher_features():
    torch.manual_seed(0)
    fkd = GlobalCosineFeatureKD(8, 16)
    s, t = torch.randn(4, 8), torch.randn(4, 16)
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    assert torch.allclose(fkd([s], t[:, None], mask), fkd([s[:2]], t[:2, None]), atol=1e-6)


def test_gradient_reaches_student_and_projector():
    fkd = GlobalCosineFeatureKD(8, 16)
    s = torch.randn(4, 8, requires_grad=True)
    fkd([s], torch.randn(4, 1, 16)).backward()
    assert s.grad is not None and s.grad.abs().sum() > 0
    assert fkd.projectors[0][0].weight.grad is not None


def test_dataset_stacks_one_feature_file_per_layer(tmp_path):
    from src.dataset import NHatDataset

    ids = ["a", "b", "c"]
    for layer in (6, 12):
        path = tmp_path / f"teacher_hints_l{layer}_g1x1.npy"
        np.save(path, np.full((3, 1, 1, 4), layer, dtype=np.float16) + np.arange(3, dtype=np.float16)[:, None, None, None])
        path.with_suffix(".ids.json").write_text(json.dumps(ids), encoding="utf-8")
    np.save(tmp_path / "nhat.npy", np.zeros((3, 8), dtype=np.float32))
    (tmp_path / "nhat.json").write_text(json.dumps(ids), encoding="utf-8")

    class Base:
        rows = [{"sample_id": "c"}, {"sample_id": "a"}]

        def __len__(self):
            return 2

        def __getitem__(self, i):
            return {}

    dataset = NHatDataset(Base(), tmp_path / "nhat.npy",
                          teacher_features=[tmp_path / "teacher_hints_l6_g1x1.npy", tmp_path / "teacher_hints_l12_g1x1.npy"])
    item = dataset[0]  # sample "c" = hàng 2
    assert item["teacher_feat"].shape == (2, 4)
    assert item["teacher_feat"][:, 0].tolist() == [8.0, 14.0]
    assert item["teacher_feat_mask"].item() == 1.0


def test_teacher_features_matched_by_sample_id(tmp_path):
    path = tmp_path / "teacher_pooled.npy"
    np.save(path, np.arange(12, dtype=np.float16).reshape(3, 4))
    path.with_suffix(".ids.json").write_text(json.dumps(["a", "b", "c"]), encoding="utf-8")
    rows = [{"sample_id": "c"}, {"sample_id": "x"}, {"sample_id": "a"}]
    position, mask = load_teacher_features(path, rows)
    assert position.tolist() == [2, -1, 0]
    assert mask.tolist() == [1.0, 0.0, 1.0]


def test_mixup_mixes_teacher_features_with_same_permutation():
    torch.manual_seed(0)
    np.random.seed(0)
    b = 6
    wave = torch.randn(b, 10)
    feat = wave[:, None, :4].clone()  # đặc trưng = 4 mẫu đầu của waveform -> trộn đúng thì vẫn khớp
    mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    out = apply_mixup_with_teachers(
        wave, torch.eye(b), torch.zeros(b), torch.randn(b, 1, 3), torch.ones(b, 1), torch.ones(b, 1),
        alpha=0.4, prob=1.0, teacher_feat=feat, feat_mask=mask)
    wave_m, feat_m, mask_m = out[0], out[6], out[7]
    assert torch.allclose(feat_m, wave_m[:, None, :4], atol=1e-6)
    # clip trộn với clip thiếu đặc trưng thì bị loại
    assert mask_m.sum() <= mask.sum() and mask_m[5] == 0


def test_one_training_step_with_feature_kd(tmp_path):
    pytest.importorskip("timm")
    from src.models.split_moe import CEDSingleMoESplitFFN

    torch.manual_seed(0)
    model = CEDSingleMoESplitFFN(expert_names=("low", "high"), patch_stride=(16, 8))
    b, c = 2, 36
    target = F.one_hot(torch.tensor([3, 7]), c).float()
    batch = {
        "waveform": torch.randn(b, 64000) * 0.05,
        "target": target,
        "target_idx": target.argmax(-1),
        "snr_db": torch.tensor([0.0, 20.0]),
        "teacher_logits": torch.randn(b, 2, c),
        "teacher_feat": torch.randn(b, 3, 768),
        "teacher_feat_mask": torch.ones(b),
    }
    trainer = KDTrainer(
        model, [batch], [batch], torch.device("cpu"), epochs=1, mixup_prob=0.0, out_dir=tmp_path,
        feature_kd={"enabled": True, "weight": 0.1, "grad_ratio_every": 1,
                    "taps": [{"student_block": b} for b in (6, 9, 12)]},
    )
    before = trainer.feat_kd.projectors[0][0].weight.detach().clone()
    result = trainer.train_epoch(0)
    assert 0.0 <= result["loss_feat"] <= 2.0
    assert result["grad_ratio_feat"] > 0.0
    assert not torch.equal(before, trainer.feat_kd.projectors[0][0].weight)
    # projector không nằm trong checkpoint của student
    assert not any("projector" in k for k in model.state_dict())
