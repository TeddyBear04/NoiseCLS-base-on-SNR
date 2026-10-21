from __future__ import annotations

import ast
import json

import pytest

from SNR_Aware.KDMoE_AT.online_pipeline import HERE, load_config, validate_config

OTHER_FOLDERS = ("FRCRN_KD_MoE", "MLNC", "SNR_Expert_SSLAM", "BEATs_Experts", "Mid_Expert", "noise_pipeline")


def test_default_config_resolves_local_frcrn_checkpoint_and_teacher():
    config = load_config(HERE / "config" / "train_panns_at_energy.json")
    assert config["runtime"]["seed"] == 2026
    if not (HERE / "model" / "frcrn_best.pt").is_file():
        pytest.skip("model/frcrn_best.pt chưa có (xin trực tiếp, xem README)")
    if not (HERE / "output" / "teacher_unified" / "beats_teacher_logits_clean.pt").is_file():
        pytest.skip("logit teacher chưa có (run.sh bước teacher)")
    paths = validate_config(config, require_dataset=False)
    assert paths["frcrn_checkpoint"] == (HERE / "model" / "frcrn_best.pt").resolve()
    assert paths["teacher_cache"] == (HERE / "output" / "teacher_unified" / "beats_teacher_logits_clean.pt").resolve()


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
