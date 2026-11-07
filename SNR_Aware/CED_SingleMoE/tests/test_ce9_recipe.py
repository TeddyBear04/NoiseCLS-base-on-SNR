"""Recipe ce9 (MLNC vuong_mix_dataset_v2, user 2026-11-07) cho SplitFFN: chia lr theo nguồn khởi tạo, tắt SpecAugment."""

import json
from pathlib import Path

from src.kd_trainer import KDTrainer
from src.models.split_moe import CEDSingleMoESplitFFN

CONFIG = Path(__file__).resolve().parent.parent / "config"


def _model_with_fake_checkpoint() -> CEDSingleMoESplitFFN:
    """Không tải HF: giả lập load_ced_tiny bằng danh sách khoá mà nó nạp (blocks., patch_embed., norm., pos_embed)."""
    model = CEDSingleMoESplitFFN(num_classes=36, specaug={"freq_masks": 0, "freq_width": 8, "time_masks": 0, "time_width": 24})
    vit_keys = [k.replace(".mlp.ffn_shared.", ".mlp.") for k in model.vit_base.vit.state_dict()
                if k.startswith(("blocks.", "patch_embed.", "norm.")) and ".ffn_low." not in k and ".ffn_high." not in k
                and "_rel_bias" not in k and "grep" not in k and "cls_to" not in k and "to_cls" not in k]
    model._loaded_vit_keys = frozenset(vit_keys + ["pos_embed"])
    return model


def test_pretrained_names_cover_split_ffn_copies_but_not_new_modules():
    model = _model_with_fake_checkpoint()
    pretrained = model.pretrained_parameter_names()
    for block in (10, 11):
        for ffn in ("ffn_shared", "ffn_low", "ffn_high"):
            assert f"vit_base.vit.blocks.{block}.mlp.{ffn}.fc1.weight" in pretrained
    assert "vit_base.vit.blocks.0.attn.qkv.weight" in pretrained
    assert "vit_base.vit.pos_embed" in pretrained
    for name, _ in model.named_parameters():
        if name.startswith(("head_", "router.", "acoustic_extractor.")) or name == "vit_base.vit.cls_token":
            assert name not in pretrained, name


def test_lr_head_makes_two_groups_by_initialisation():
    model = _model_with_fake_checkpoint()
    trainer = object.__new__(KDTrainer)
    trainer.model, trainer.lr_head = model, 3e-4
    groups = trainer._build_param_groups(3e-5, 1.2e-4, 2e-4, 0.01)
    assert [g["lr"] for g in groups] == [3e-5, 3e-4]
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert sum(p.numel() for g in groups for p in g["params"]) == total
    head_ids = {id(p) for p in groups[1]["params"]}
    assert id(model.head_low.weight) in head_ids and id(model.vit_base.vit.blocks[10].mlp.ffn_low.fc1.weight) not in head_ids


def test_specaug_zero_is_identity_in_training():
    model = _model_with_fake_checkpoint().train()
    x = __import__("torch").randn(2, 1, 64, 398)
    assert (model.spec_augment(x) == x).all()


def test_ce9_configs_change_only_the_recipe():
    for base, ce9 in (("train_config.json", "train_config_ce9.json"), ("train_config_overlap.json", "train_config_overlap_ce9.json")):
        a = json.loads((CONFIG / base).read_text(encoding="utf-8"))
        b = json.loads((CONFIG / ce9).read_text(encoding="utf-8"))
        assert b["experiment"]["seed"] == 2026 and b["loss"]["label_smoothing"] == 0.0
        assert b["train"]["mixup_prob"] == 0.0 and b["model"]["specaug"]["freq_masks"] == b["model"]["specaug"]["time_masks"] == 0
        assert (b["train"]["lr_backbone"], b["train"]["lr_head"], b["train"]["weight_decay"]) == (3e-5, 3e-4, 0.01)
        assert (b["train"]["epochs"], b["train"]["batch_size"], b["train"]["warmup_epochs"], b["train"]["grad_clip"]) == (15, 64, 2, 1.0)
        for section in ("teacher", "routing"):
            assert a[section] == b[section], section
        assert {k: v for k, v in a["loss"].items() if k != "label_smoothing"} == {k: v for k, v in b["loss"].items() if k != "label_smoothing"}
        assert {k: v for k, v in a["model"].items() if k != "specaug"} == {k: v for k, v in b["model"].items() if k != "specaug"}


def test_ced_mini_variant_widths_and_forward(sample_waveform):
    """ced_variant mini: 256 chiều / 4 head (head_dim 64 như tiny), router và head theo chiều rộng mới."""
    import torch

    model = CEDSingleMoESplitFFN(num_classes=36, ced_variant="mini", patch_stride=(16, 8)).eval()
    vit = model.vit_base.vit
    assert model.vit_base.embed_dim == 256 and vit.pos_embed.shape == (1, 1 + 4 * 48, 256)
    assert vit.blocks[0].attn.num_heads == 4 and vit.blocks[0].attn.head_dim == 64
    assert model.head_low.in_features == 256 and model.vit_base.vit.blocks[10].mlp.ffn_low.fc1.out_features == 1024
    with torch.no_grad():
        out = model(sample_waveform, return_dict=True)
    assert out["logits"].shape == (2, 36)


def test_mini_config_only_changes_variant():
    a = json.loads((CONFIG / "train_config_overlap_ce9.json").read_text(encoding="utf-8"))
    b = json.loads((CONFIG / "train_config_overlap_ce9_mini.json").read_text(encoding="utf-8"))
    assert b["model"].pop("ced_variant") == "mini"
    b["model"].pop("_note_variant")
    for section in ("teacher", "model", "routing", "loss", "train"):
        assert a[section] == b[section], section


def test_mgd_feature_kd_shapes_mask_and_grid():
    """MGD: token student lưới (4, 48) f-major, teacher (25, 4, 768) làm phẳng -> loss hữu hạn, có gradient, clip thiếu bị loại."""
    import torch

    from src.feature_kd import MaskedGenerativeFeatureKD

    torch.manual_seed(0)
    mgd = MaskedGenerativeFeatureKD(192, 768, 0.5, (25, 4))
    tokens = torch.randn(3, 4 * 48, 192, requires_grad=True)
    teacher = torch.randn(3, 1, 25 * 4 * 768)
    loss = mgd(tokens, (4, 48), teacher, torch.tensor([1.0, 1.0, 0.0]))
    loss.backward()
    assert torch.isfinite(loss) and tokens.grad[:2].abs().sum() > 0 and tokens.grad[2].abs().sum() == 0
    # teacher cùng lưới student thì pool là đồng nhất: teacher = 0 và student = 0 -> chỉ còn bias của khối sinh
    zero = mgd(torch.zeros(1, 4 * 25, 192), (4, 25), torch.zeros(1, 1, 25 * 4 * 768))
    assert zero >= 0


def test_mgd_config_and_last_tokens(sample_waveform):
    import torch

    cfg = json.loads((CONFIG / "train_config_overlap_ce9_mgd.json").read_text(encoding="utf-8"))
    base = json.loads((CONFIG / "train_config_overlap_ce9.json").read_text(encoding="utf-8"))
    fkd = cfg["feature_kd"]
    assert (fkd["method"], fkd["weight"], fkd["mask_ratio"], fkd["teacher_grid"]) == ("mgd", 7e-05, 0.5, [25, 4])
    for section in ("teacher", "model", "routing", "loss", "train"):
        assert cfg[section] == base[section], section
    model = CEDSingleMoESplitFFN(num_classes=36, patch_stride=(16, 8)).eval()
    with torch.no_grad():
        out = model(sample_waveform, return_dict=True)
    assert out["last_tokens"].shape == (2, 4 * 48, 192)


def test_kdmoe_style_kd_matches_hinton_kl_and_configs():
    """kd_style kdmoe: loss = CE + T^2 KL(softmax(t/T) || softmax(s/T)) như logits_kl của FRCRN_KD_MoE (T 4)."""
    import torch
    import torch.nn.functional as F

    for name, base, mgd in (("train_config_overlap_ce9_kdmoe_mgd.json", "train_config_overlap_ce9_mgd.json", True),
                            ("train_config_overlap_ce9_kdmoe.json", "train_config_overlap_ce9.json", False)):
        a = json.loads((CONFIG / base).read_text(encoding="utf-8"))
        b = json.loads((CONFIG / name).read_text(encoding="utf-8"))
        loss = b["loss"]
        assert (loss["kd_style"], loss["kd_temperature"], loss["kd_weight"], loss["ce_weight"], loss["label_smoothing"]) == ("kdmoe", 4.0, 1.0, 1.0, 0.0)
        assert loss["lambda_route"] == a["loss"]["lambda_route"] and b["teacher"]["mode"] == "clean5fold"
        for section in ("teacher", "model", "routing", "train"):
            assert a[section] == b[section], section
        assert bool(b.get("feature_kd", {}).get("enabled")) == mgd and b.get("feature_kd") == a.get("feature_kd")
    torch.manual_seed(0)
    s, t, T = torch.randn(5, 36), torch.randn(5, 36) * 3, 4.0
    p_t = F.softmax(t / T, dim=-1)
    ours = (T ** 2) * (p_t * (p_t.clamp_min(1e-8).log() - F.log_softmax(s / T, dim=-1))).sum(-1).mean()
    ref = (T ** 2) * F.kl_div(F.log_softmax(s / T, dim=-1), p_t, reduction="batchmean")
    assert torch.allclose(ours, ref, atol=1e-5)


def test_lr1e5_configs_only_lower_the_learning_rates():
    for base, low in (("train_config_overlap_ce9.json", "train_config_overlap_ce9_lr1e5.json"),
                      ("train_config_overlap_ce9_mini.json", "train_config_overlap_ce9_mini_lr1e5.json")):
        a = json.loads((CONFIG / base).read_text(encoding="utf-8"))
        b = json.loads((CONFIG / low).read_text(encoding="utf-8"))
        assert (b["train"].pop("lr_backbone"), b["train"].pop("lr_head")) == (1e-5, 1e-4)
        b["train"].pop("_note_lr"); a["train"].pop("lr_backbone"); a["train"].pop("lr_head")
        for section in ("teacher", "model", "routing", "loss", "train"):
            assert a[section] == b[section], section


def test_long_schedule_configs_and_early_stop_rule():
    for base, new in (("train_config_overlap_ce9_lr1e5.json", "train_config_overlap_ce9_lr1e5_e45p3.json"),
                      ("train_config_overlap_ce9_kdmoe.json", "train_config_overlap_ce9_kdmoe_lr1e5_e45p3.json"),
                      ("train_config_ce9.json", "train_config_ce9_lr1e5_e45p3.json")):
        a = json.loads((CONFIG / base).read_text(encoding="utf-8"))
        b = json.loads((CONFIG / new).read_text(encoding="utf-8"))
        assert (b["train"].pop("epochs"), b["train"].pop("patience")) == (45, 3)
        assert (b["train"].pop("lr_backbone"), b["train"].pop("lr_head")) == (1e-5, 1e-4)
        b["train"].pop("_note_epochs"); b["train"].pop("_note_lr", None)
        for key in ("epochs", "lr_backbone", "lr_head", "_note_lr"):
            a["train"].pop(key, None)
        for section in ("teacher", "model", "routing", "loss", "train"):
            assert a[section] == b[section], (new, section)
    # quy tắc dừng: val top-1 tăng strict mới reset bộ đếm
    curve, best, since, stopped = [0.50, 0.55, 0.56, 0.56, 0.555, 0.559, 0.57], 0.0, 0, None
    for epoch, acc in enumerate(curve, 1):
        since = 0 if acc > best else since + 1
        best = max(best, acc)
        if since >= 3:
            stopped = epoch
            break
    assert stopped == 6


def test_patience8_configs_only_change_patience():
    for name in ("train_config_overlap_ce9_lr1e5_e45p3.json", "train_config_overlap_ce9_kdmoe_lr1e5_e45p3.json", "train_config_ce9_lr1e5_e45p3.json"):
        a = json.loads((CONFIG / name).read_text(encoding="utf-8"))
        b = json.loads((CONFIG / name.replace("_e45p3", "_e45p8")).read_text(encoding="utf-8"))
        assert (a["train"]["patience"], b["train"].pop("patience")) == (3, 8)
        b["train"].pop("_note_patience"); a["train"].pop("patience")
        for section in ("teacher", "model", "routing", "loss", "train"):
            assert a[section] == b[section], (name, section)
        assert b["paths"]["output_dir"] == a["paths"]["output_dir"].replace("_e45p3", "_e45p8")
