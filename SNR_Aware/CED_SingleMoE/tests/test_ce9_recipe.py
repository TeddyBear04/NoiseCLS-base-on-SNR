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
