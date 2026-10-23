from __future__ import annotations

import ast
import json

import pytest

import torch
from torch import nn

from SNR_Aware.KDMoE_Base.online_pipeline import (
    HERE,
    load_config,
    loss_config,
    validate_config,
)
from SNR_Aware.KDMoE_Base.src.train.kdmoe_trainer import KDMoETrainer
from SNR_Aware.KDMoE_Base.src.utils.checkpoint import save_checkpoint
from SNR_Aware.KDMoE_Base.src.utils.ema import EMA

OTHER_FOLDERS = ("FRCRN_KD_MoE", "MLNC", "SNR_Expert_SSLAM", "BEATs_Experts", "Mid_Expert", "noise_pipeline")


def test_default_config_resolves_local_frcrn_checkpoint_and_teacher():
    config = load_config(HERE / "config" / "train_vit_ce.json")
    assert config["runtime"]["seed"] == 2026
    if not (HERE / "model" / "frcrn_best.pt").is_file():
        pytest.skip("model/frcrn_best.pt chưa có (xin trực tiếp, xem README)")
    if not (HERE / "output" / "teacher_unified" / "teacher_unified.pt").is_file():
        pytest.skip("logit teacher chưa có (run.sh bước teacher)")
    paths = validate_config(config, require_dataset=False)
    assert paths["frcrn_checkpoint"] == (HERE / "model" / "frcrn_best.pt").resolve()
    assert paths["teacher_cache"] == (HERE / "output" / "teacher_unified" / "teacher_unified.pt").resolve()


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
        HERE / "config" / "kdmoe_s_ce.yaml",
        {"kd_temperature": 4.0, "kd_weight": 0.5},
    )

    assert config.kd_temperature == 4.0
    assert config.kd_weight == 0.5


def test_unknown_loss_override_is_rejected():
    try:
        loss_config(
            HERE / "config" / "kdmoe_s_ce.yaml",
            {"not_a_loss_field": 1},
        )
    except KeyError as error:
        assert "not_a_loss_field" in str(error)
    else:
        raise AssertionError("unknown loss override was accepted")
