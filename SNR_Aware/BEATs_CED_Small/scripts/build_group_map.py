from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS_JSON = ROOT / "partition_labels_36.json"
CONFIG_DIR = ROOT / "config"
GROUP_MAP_FILE = "group_map.json"
MEMBERSHIP_FILE = "membership.npy"
NUM_CLASSES = 36
EXPECTED_GROUP_SIZES = (5, 12, 9, 10)

EXPERT_GROUPS: dict[str, tuple[str, ...]] = {
    "E0": (
        "/m/0f8s22",
        "/m/0239kh",
        "/m/0dwtp",
        "/m/0mbct",
        "/m/03qtq",
    ),
    "E1": (
        "/m/01bjv",
        "/m/01m2v",
        "/m/0fqfqc",
        "/m/0g6b5",
        "/m/032s66",
        "/m/03v3yw",
        "/m/07r4wb8",
        "/m/0fx9l",
        "/m/01lsmm",
        "/m/07rn7sz",
        "/m/07qcx4z",
        "/m/081rb",
    ),
    "E2": (
        "/m/042v_gx",
        "/m/01wy6",
        "/m/01s0ps",
        "/m/0l14j_",
        "/m/03qjg",
        "/m/06ncr",
        "/m/07brj",
        "/m/07gql",
        "/m/07y_7",
    ),
    "E3": (
        "/m/028ght",
        "/m/05tny_",
        "/m/03q5_w",
        "/m/01b_21",
        "/m/02_nn",
        "/m/025_jnm",
        "/m/01j3sz",
        "/m/07qrkrw",
        "/m/07q6cd_",
        "/m/07cx4",
    ),
}


def build_group_map(labels_json: Path = LABELS_JSON, out_dir: Path = CONFIG_DIR) -> dict:
    with open(labels_json, encoding="utf-8") as f:
        labels = json.load(f)
    if len(labels) != NUM_CLASSES or len({row["mid"] for row in labels}) != NUM_CLASSES:
        raise ValueError(f"{labels_json} must contain {NUM_CLASSES} unique classes")
    group_index = {group: index for index, group in enumerate(EXPERT_GROUPS)}
    mid_to_group = {mid: group_index[group] for group, mids in EXPERT_GROUPS.items() for mid in mids}
    unknown = sorted(set(mid_to_group) - {row["mid"] for row in labels})
    if unknown:
        raise KeyError(f"mids not present in {labels_json}: {unknown}")
    sizes = tuple(len(mids) for mids in EXPERT_GROUPS.values())
    if sizes != EXPECTED_GROUP_SIZES:
        raise ValueError(f"group sizes {sizes} do not match {EXPECTED_GROUP_SIZES}")
    group_map = {
        "groups": {group: list(mids) for group, mids in EXPERT_GROUPS.items()},
        "mid_to_group": {row["mid"]: mid_to_group[row["mid"]] for row in labels},
        "classes": [
            {
                "mid": row["mid"],
                "name": row["name"],
                "label_index": int(row["label_index"]),
                "group": mid_to_group[row["mid"]],
            }
            for row in sorted(labels, key=lambda r: int(r["label_index"]))
        ],
    }
    indices = [row["label_index"] for row in group_map["classes"]]
    if indices != list(range(NUM_CLASSES)):
        raise ValueError(f"label_index must cover 0..{NUM_CLASSES - 1}, got {indices}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / GROUP_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(group_map, f, indent=2)
        f.write("\n")
    np.save(out_dir / MEMBERSHIP_FILE, membership_matrix(group_map))
    return group_map


def membership_matrix(group_map: dict) -> np.ndarray:
    num_groups = len(group_map["groups"])
    membership = np.zeros((len(group_map["classes"]), num_groups), dtype=np.float32)
    for row in group_map["classes"]:
        membership[row["label_index"], row["group"]] = 1.0
    return membership


def load_group_map(path: Path = CONFIG_DIR / GROUP_MAP_FILE) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"group map not found, run build_group_map() first: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_membership(path: Path = CONFIG_DIR / MEMBERSHIP_FILE) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"membership matrix not found, run build_group_map() first: {path}")
    membership = np.load(path)
    if membership.shape != (NUM_CLASSES, len(EXPERT_GROUPS)):
        raise ValueError(f"membership must have shape {(NUM_CLASSES, len(EXPERT_GROUPS))}, got {membership.shape}")
    return membership
