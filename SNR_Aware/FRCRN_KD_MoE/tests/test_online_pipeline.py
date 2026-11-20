from __future__ import annotations

import ast
import json

import torch
from torch import nn

from SNR_Aware.FRCRN_KD_MoE.online_pipeline import (
    HERE,
    load_config,
    loss_config,
    validate_config,
)
from SNR_Aware.FRCRN_KD_MoE.src.train.kdmoe_trainer import KDMoETrainer
from SNR_Aware.FRCRN_KD_MoE.src.utils.checkpoint import save_checkpoint
from SNR_Aware.FRCRN_KD_MoE.src.utils.ema import EMA

OTHER_FOLDERS = ("MLNC", "SNR_Expert_SSLAM", "BEATs_Experts", "Mid_Expert", "noise_pipeline")


def test_default_config_resolves_local_frcrn_checkpoint_and_teacher():
    config = load_config(HERE / "config" / "train_config.json")
    paths = validate_config(config, require_dataset=False)
    assert config["runtime"]["seed"] == 2026
    assert paths["frcrn_checkpoint"] == (HERE / "model" / "frcrn_best.pt").resolve()
    assert paths["frcrn_checkpoint"].is_file()
    assert paths["teacher_cache"] == (HERE / "model" / "teacher_midexpert.pt").resolve()


def test_configs_only_point_inside_this_folder():
    """Mọi đường dẫn trong config nằm trong FRCRN_KD_MoE (dataset và dữ liệu ngoài code thì được phép)."""
    for cfg in (HERE / "config").glob("*.json"):
        raw = json.loads(cfg.read_text(encoding="utf-8"))
        values = list(raw.get("paths", {}).values())
        values += [raw.get("crossfit", {}).get(key) for key in ("exclude_noise_ytids", "n_hat_cache")]
        values += raw.get("backbone", {}).get("checkpoint_candidates", [])
        for value in values:
            for item in value if isinstance(value, list) else [value]:
                if item and not str(item).startswith(("/marimo", "../../.agents")):
                    assert not any(name in str(item) for name in OTHER_FOLDERS), (cfg.name, item)


def test_code_imports_no_other_method_folder():
    """Không file .py nào của thư mục import hay trỏ sang MLNC / SNR_Expert_SSLAM / BEATs_Experts."""
    for path in HERE.rglob("*.py"):
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and "\n" not in node.value:
                assert not any(f"/{name}" in node.value or f'"{name}"' == f'"{node.value}"'
                               for name in OTHER_FOLDERS), (path.name, node.value)


def test_stage2_snapshot_uses_ema_parameters():
    model = nn.Linear(3, 2)
    ema = EMA(model, decay=0.9)
    expected = {name: value.clone() for name, value in ema.shadow.items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)

    trainer = object.__new__(KDMoETrainer)
    trainer.model = model
    trainer.stage = 2
    trainer.ema = ema
    trainer._resumed_best_state = None

    snapshot = trainer.snapshot()

    for name, value in expected.items():
        assert torch.equal(snapshot[name], value)


def test_save_checkpoint_accepts_selected_state(tmp_path):
    model = nn.Linear(3, 2)
    selected = {name: torch.full_like(value, 7.0) for name, value in model.state_dict().items()}
    path = save_checkpoint(tmp_path / "best.pt", model, state_dict=selected, epoch=4)

    saved = torch.load(path, map_location="cpu", weights_only=True)

    assert saved["epoch"] == 4
    for value in saved["model"].values():
        assert torch.all(value == 7.0)


def test_json_can_override_loss_hyperparameters():
    config = loss_config(
        HERE / "config" / "unified" / "kdmoe_mn10_snrroute.yaml",
        {"kd_temperature": 4.0, "kd_weight": 0.5},
    )

    assert config.kd_temperature == 4.0
    assert config.kd_weight == 0.5


def test_unknown_loss_override_is_rejected():
    try:
        loss_config(
            HERE / "config" / "unified" / "kdmoe_mn10_snrroute.yaml",
            {"not_a_loss_field": 1},
        )
    except KeyError as error:
        assert "not_a_loss_field" in str(error)
    else:
        raise AssertionError("unknown loss override was accepted")

CE9_YAML = HERE / "config" / "unified" / "kdmoe_ced_tiny_meanpool_ce9_kd.yaml"


def test_ema_decay_zero_tracks_the_live_weights():
    """ema_decay 0 = không EMA (recipe ce9): shadow luôn bằng trọng số hiện tại."""
    model = nn.Linear(3, 2)
    ema = EMA(model, decay=0.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(5.0)
    ema.update(model)
    for name, parameter in model.named_parameters():
        assert torch.equal(ema.shadow[name], parameter)


def test_ce9_recipe_yaml():
    """Recipe ce9 (MLNC vuong_mix_dataset_v2 ce9_original_run.json, ced_tiny) áp vào KD-MoE CED-Tiny."""
    import yaml

    from SNR_Aware.FRCRN_KD_MoE.src.train.config import KDMoETrainConfig

    train = KDMoETrainConfig.from_yaml(CE9_YAML)
    assert (train.lr, train.weight_decay, train.epochs, train.stage1_epochs) == (3e-5, 0.01, 15, 0)
    assert (train.ema_decay, train.min_lr, train.selection_metric, train.batch_size) == (0.0, 0.0, "acc", 64)
    assert abs(train.warmup_ratio - 2 / 15) < 1e-9 and train.patience >= train.epochs
    assert loss_config(CE9_YAML).label_smoothing == 0.0
    raw = yaml.safe_load(CE9_YAML.read_text(encoding="utf-8"))
    assert raw["model"]["specaug_freq_masks"] == 0 and raw["model"]["specaug_time_masks"] == 0
    assert raw["student"]["layer_decay"] == 1.0 and raw["student"]["drop_path"] == 0.0
    assert raw["student"]["head_lr_scale"] == 10.0          # head 3e-4 = 10 x backbone 3e-5


def test_ced_head_lr_scale_applies_to_new_modules_only():
    from SNR_Aware.FRCRN_KD_MoE.ced import CEDKDMoE

    model = object.__new__(CEDKDMoE)
    model._student = {"layer_decay": 1.0, "head_lr_scale": 10.0}
    model._num_blocks = 12
    for name in ("frontend.init_bn.weight", "encoder.blocks.3.attn.qkv.weight", "encoder.patch_embed.proj.weight",
                 "encoder.norm.weight", "post_fc.norm.weight"):
        assert model.lr_scale(name) == 1.0, name
    for name in ("gate.proj.weight", "noise_head.proj.weight", "router.net.0.weight", "experts.0.fc.weight",
                 "classifier.weight"):
        assert model.lr_scale(name) == 10.0, name


def test_ce9sa_differs_from_ce9_only_by_specaugment():
    """Recipe 'ce9 + SpecAugment' (user 2026-11-05): yaml/JSON ce9sa = ce9 + 1 mask tần số + 1 mask thời gian."""
    import yaml

    unified = HERE / "config" / "unified"
    for size in ("tiny", "mini", "small"):
        for kind in ("ce", "kd", "crd", "mgd"):
            base = yaml.safe_load((unified / f"kdmoe_ced_{size}_meanpool_ce9_{kind}.yaml").read_text(encoding="utf-8"))
            sa = yaml.safe_load((unified / f"kdmoe_ced_{size}_meanpool_ce9sa_{kind}.yaml").read_text(encoding="utf-8"))
            assert (sa["model"]["specaug_freq_masks"], sa["model"]["specaug_time_masks"]) == (1, 1)
            assert (sa["model"]["specaug_freq_width"], sa["model"]["specaug_time_width"]) == (12, 80)
            sa["model"]["specaug_freq_masks"] = sa["model"]["specaug_time_masks"] = 0
            assert sa == base, (size, kind)
        for method in ("1_ce", "2_logit", "3_mgd", "4_crd", "5_logit_mgd", "6_logit_crd"):
            base = json.loads((unified / f"train_ced_{size}_ce9_{method}.json").read_text(encoding="utf-8"))
            sa = json.loads((unified / f"train_ced_{size}_ce9sa_{method}.json").read_text(encoding="utf-8"))
            assert sa["paths"].pop("student_config") == base["paths"].pop("student_config").replace("_ce9_", "_ce9sa_")
            assert sa["paths"].pop("output").endswith(f"ced_{size}_ce9sa_{method}")
            base["paths"].pop("output"), sa.pop("_note"), base.pop("_note")
            assert sa == base, (size, method)


def test_ce9sa_lr1e5_only_lowers_the_learning_rate():
    """ce9sa_lr1e5 (user 2026-11-07) = ce9sa với lr backbone 1e-5 (head 1e-4 qua head_lr_scale 10)."""
    import yaml

    unified = HERE / "config" / "unified"
    for size in ("tiny", "mini"):
        for kind in ("ce", "kd", "crd", "mgd"):
            base = yaml.safe_load((unified / f"kdmoe_ced_{size}_meanpool_ce9sa_{kind}.yaml").read_text(encoding="utf-8"))
            low = yaml.safe_load((unified / f"kdmoe_ced_{size}_meanpool_ce9sa_lr1e5_{kind}.yaml").read_text(encoding="utf-8"))
            assert low["train"]["lr"] == 1e-5 and low["student"]["head_lr_scale"] == 10.0
            low["train"]["lr"] = base["train"]["lr"]
            assert low == base, (size, kind)
        for method in ("1_ce", "2_logit", "3_mgd", "4_crd", "5_logit_mgd", "6_logit_crd"):
            base = json.loads((unified / f"train_ced_{size}_ce9sa_{method}.json").read_text(encoding="utf-8"))
            low = json.loads((unified / f"train_ced_{size}_ce9sa_lr1e5_{method}.json").read_text(encoding="utf-8"))
            assert low["paths"].pop("student_config") == base["paths"].pop("student_config").replace("_ce9sa_", "_ce9sa_lr1e5_")
            assert low["paths"].pop("output").endswith(f"ced_{size}_ce9sa_lr1e5_{method}")
            base["paths"].pop("output"), low.pop("_note"), base.pop("_note")
            assert low == base, (size, method)


def test_long_schedule_configs_only_change_epochs_patience_warmup():
    """e45p3 (user 2026-11-07): run 'hội tụ muộn' chạy lại với lr 1e-5 (head 1e-4), 45 epoch, patience 3, warm-up giữ 2 epoch.
    Bản ce9 không SpecAugment chưa từng có yaml lr 1e-5 nên so với yaml ce9 gốc (lr 3e-5) sau khi trả lr về."""
    import yaml

    unified = HERE / "config" / "unified"
    yamls = sorted(unified.glob("kdmoe_ced_tiny_meanpool_*_e45p3_*.yaml"))
    assert len(yamls) == 5
    for path in yamls:
        base_name = path.name.replace("_e45p3_", "_")
        if not (unified / base_name).is_file():
            base_name = base_name.replace("_lr1e5_", "_")
        base = yaml.safe_load((unified / base_name).read_text(encoding="utf-8"))
        new = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert (new["train"]["epochs"], new["train"]["patience"], new["train"]["lr"]) == (45, 3, 1e-5), path.name
        assert abs(new["train"]["warmup_ratio"] * 45 - 2.0) < 1e-9 and new["student"]["head_lr_scale"] == 10.0, path.name
        for key in ("epochs", "patience", "warmup_ratio", "lr"):
            new["train"][key] = base["train"][key]
        assert new == base, path.name
    # e45p8 (user 2026-11-08) = e45p3 với patience 8, không khác gì khác
    for path in sorted(unified.glob("kdmoe_ced_tiny_meanpool_*_e45p8_*.yaml")):
        p3 = yaml.safe_load((unified / path.name.replace("_e45p8_", "_e45p3_")).read_text(encoding="utf-8"))
        p8 = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert (p3["train"]["patience"], p8["train"]["patience"]) == (3, 8), path.name
        p8["train"]["patience"] = 3
        assert p8 == p3, path.name
    assert len(sorted(unified.glob("train_ced_tiny_*_e45p8_*.json"))) == 6
    jsons = sorted(unified.glob("train_ced_tiny_*_e45p3_*.json"))
    assert len(jsons) == 6
    for path in jsons:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        assert (HERE / cfg["paths"]["student_config"]).is_file()
        assert cfg["paths"]["output"].endswith(path.stem.removeprefix("train_"))
