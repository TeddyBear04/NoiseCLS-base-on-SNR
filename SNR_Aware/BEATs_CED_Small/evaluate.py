"""Đánh giá các run KD-MoE của thư mục này trên validation + test, toàn bộ và theo từng mức SNR.

    python -u evaluate.py --dataset /marimo/dataset/mix-dataset --out output/eval/<đợt> [--only mn10]

Hai bộ trọng số mỗi run, vì trainer lưu trọng số thường ở best.pt nhưng chấm (và chọn epoch) bằng EMA:
  best_raw  stage2/best.pt            trọng số thường của epoch được chọn
  ema_last  stage2/resume.pt["ema"]   EMA của epoch cuối (số log in ở dòng "validation" cuối)
Mỗi run giữ đầu vào lúc train:
  fp32_cache   n_hat = FRCRN(mixture) fp32, không TF32 (output/frcrn_cache_fp32; cache test dựng ở đây)
  bf16_online  FRCRN chạy trong autocast bf16 của student (bản FRCRN_KD_MoE đời đầu)
Tập con: all / clean (clip nhiễu không nằm trong AudioSet train) / leaked. Ghi metrics_by_snr.csv và
predictions_<run>_<weights>_<split>.csv (cho McNemar) vào --out.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from online_pipeline import HERE, load_frcrn, student_type
from scripts.frcrn_cache import FRCRN_CACHE_DIR, build_frcrn_cache, build_frcrn_snr
from scripts.load_dataset import KDMoEMixDataset, build_dataloader
from src.model.kdmoe import KDMoEModel, frcrn_snr_db

OUTPUT, CONFIG = HERE / "output", HERE / "config"
RUNS = {  # run: (thư mục, pipeline đầu vào, yaml model)
    "FRCRN_KD_MoE old (ViT, online bf16)": (OUTPUT / "train", "bf16_online", CONFIG / "kdmoe_s.yaml"),
    "FRCRN_KD_MoE PANNs-CNN10 (3 SNR experts)": (OUTPUT / "panns_cnn10" / "train", "fp32_cache",
                                                 CONFIG / "kdmoe_panns_cnn10.yaml"),
    "FRCRN_KD_MoE PANNs-CNN10 clean-val selection": (OUTPUT / "panns_cnn10_cleanval" / "train", "fp32_cache",
                                                     CONFIG / "kdmoe_panns_cnn10.yaml"),
    "FRCRN_KD_MoE PANNs-CNN10 teacher-crossfit e50": (OUTPUT / "panns_cnn10_teachercf_e50" / "train", "fp32_cache",
                                                      CONFIG / "kdmoe_panns_cnn10.yaml"),
    "FRCRN_KD_MoE PANNs-CNN10 teacher-crossfit + AT": (OUTPUT / "panns_cnn10_teachercf_at" / "train", "fp32_cache",
                                                       CONFIG / "kdmoe_panns_cnn10_at.yaml"),
    "FRCRN_KD_MoE PANNs-CNN10 teacher-crossfit + AT rollout": (OUTPUT / "panns_cnn10_teachercf_rollout" / "train",
                                                               "fp32_cache", CONFIG / "kdmoe_panns_cnn10_at.yaml"),
    "FRCRN_KD_MoE PANNs-CNN10 teacher-crossfit + FitNets": (OUTPUT / "panns_cnn10_teachercf_fitnets" / "train",
                                                            "fp32_cache", CONFIG / "kdmoe_panns_cnn10_fitnets.yaml"),
    "FRCRN_KD_MoE PANNs-CNN10 teacher-crossfit + AT rollout Pearson": (
        OUTPUT / "panns_cnn10_teachercf_atpearson" / "train", "fp32_cache", CONFIG / "kdmoe_panns_cnn10_at_pearson.yaml"),
    **{f"FRCRN_KD_MoE PANNs-CNN10 ReviewKD {tag}": (OUTPUT / f"panns_cnn10_reviewkd_{tag.lower()}" / "train",
                                                   "fp32_cache", CONFIG / "kdmoe_panns_cnn10_reviewkd.yaml")
       for tag in ("R0", "R1")},
    "FRCRN_KD_MoE EfficientAT mn10 teacher-crossfit": (OUTPUT / "mn10_teachercf" / "train", "fp32_cache",
                                                       CONFIG / "kdmoe_mn10.yaml"),
    "FRCRN_KD_MoE EfficientAT mn10 teacher-crossfit + RMS": (OUTPUT / "mn10_teachercf_rms" / "train", "fp32_cache",
                                                             CONFIG / "kdmoe_mn10_rms.yaml"),
}
# Giao thức thống nhất (config/unified/): một teacher, RMS bật, val sạch, 50 epoch / patience 8 cho mọi phương pháp
UNIFIED = {"vit_ce": "kdmoe_s_ce", "vit_kd": "kdmoe_s_kd", "panns": "kdmoe_panns_cnn10",
           "panns_at_energy": "kdmoe_panns_cnn10_at", "panns_at_rollout": "kdmoe_panns_cnn10_at",
           "panns_at_pearson": "kdmoe_panns_cnn10_at_pearson", "panns_fitnets": "kdmoe_panns_cnn10_fitnets",
           "panns_reviewkd": "kdmoe_panns_cnn10_reviewkd", "mn10": "kdmoe_mn10",
           "mn10_at_energy": "kdmoe_mn10_at", "mn10_at_rollout": "kdmoe_mn10_at", "mn10_at_pearson": "kdmoe_mn10_at_pearson",
           "mn10_fitnets": "kdmoe_mn10_fitnets", "mn10_reviewkd": "kdmoe_mn10_reviewkd",
           "mn10_snrroute": "kdmoe_mn10_snrroute",
           "mn10_snrroute_t4": "kdmoe_mn10_snrroute",
           "mn10_reviewkd_t4_soft": "kdmoe_mn10_reviewkd",
           "mn10_crd_t4": "kdmoe_mn10_crd",
           "ced_crd_t4": "kdmoe_ced_crd",
           "ced_tiny_crd_t4": "kdmoe_ced_tiny_crd",
           "ced_mini_crd_t4": "kdmoe_ced_mini_crd",
           "ced_tiny_meanpool_crd_t4": "kdmoe_ced_tiny_meanpool_crd",
           "ced_mini_meanpool_crd_t4": "kdmoe_ced_mini_meanpool_crd",
           "ced_meanpool_crd_t4": "kdmoe_ced_meanpool_crd",
           "ced_tiny_meanpool_ce": "kdmoe_ced_tiny_meanpool_ce",
           "ced_mini_meanpool_ce": "kdmoe_ced_mini_meanpool_ce",
           "ced_meanpool_ce": "kdmoe_ced_meanpool_ce",
           "ced_tiny_meanpool_kd_t4": "kdmoe_ced_tiny_meanpool_kd",
           "ced_tiny_meanpool_kd_fitnets_b100": "kdmoe_ced_tiny_meanpool_kd_fitnets",
           "ced_tiny_meanpool_kd_fitnets_b10": "kdmoe_ced_tiny_meanpool_kd_fitnets",
           "ced_tiny_meanpool_kd_fitnets_blk12_hw100": "kdmoe_ced_tiny_meanpool_kd_fitnets_b12",
           "ced_tiny_meanpool_kd_fitnets_blk12_hw010": "kdmoe_ced_tiny_meanpool_kd_fitnets_b12",
           "ced_tiny_meanpool_patchcrd_t4": "kdmoe_ced_tiny_meanpool_patchcrd",
           "ced_mini_meanpool_patchcrd_t4": "kdmoe_ced_mini_meanpool_patchcrd",
           "ced_meanpool_patchcrd_t4": "kdmoe_ced_meanpool_patchcrd",
           "mn10_patchcrd_t4": "kdmoe_mn10_patchcrd",
           "ced_tiny_meanpool_mgd_t4": "kdmoe_ced_tiny_meanpool_mgd",
           "ced_tiny_meanpool_vitkd_t4": "kdmoe_ced_tiny_meanpool_vitkd",
           "ced_tiny_meanpool_qkv_t4": "kdmoe_ced_tiny_meanpool_qkv",
           "ced_tiny_meanpool_spkd_t4": "kdmoe_ced_tiny_meanpool_spkd",
           "ced_tiny_meanpool_kd_spkd_t4": "kdmoe_ced_tiny_meanpool_kd_spkd",
           "ced_tiny_meanpool_spkd12_t4": "kdmoe_ced_tiny_meanpool_spkd12",
           "ced_tiny_meanpool_kd_spkd12_t4": "kdmoe_ced_tiny_meanpool_kd_spkd12",
           "ced_tiny_meanpool_kd_mgd_spkd12_t4": "kdmoe_ced_tiny_meanpool_kd_mgd_spkd12",
           "ced_tiny_meanpool_mgd_hf_t4": "kdmoe_ced_tiny_meanpool_mgd",
           "ced_tiny_meanpool_vitkd_s_t4": "kdmoe_ced_tiny_meanpool_vitkd",
           "ced_tiny_meanpool_qkv_w01_t4": "kdmoe_ced_tiny_meanpool_qkv",
           "ced_tiny_meanpool_mgd_x10_t4": "kdmoe_ced_tiny_meanpool_mgd",
           "ced_mini_meanpool_kd_t4": "kdmoe_ced_mini_meanpool_kd",
           "ced_mini_meanpool_mgd_t4": "kdmoe_ced_mini_meanpool_mgd",
           "ced_meanpool_kd_t4": "kdmoe_ced_meanpool_kd",
           "ced_meanpool_mgd_t4": "kdmoe_ced_meanpool_mgd",
           # đợt KD CED-Tiny ở T=4 (user 2026-11-02): logit, CRD, MGD, logit+CRD, logit+MGD + biến thể tinh chỉnh + bản đồ tin cậy + feature KD khác
           "ced_tiny_kd8_10_logit_fitnets12_t4": "kdmoe_ced_tiny_meanpool_kd_fitnets_b12",
           "ced_tiny_kd8_11_logit_spkd12_t4": "kdmoe_ced_tiny_meanpool_kd_spkd12",
           "ced_tiny_kd8_12_logit_vitkd_t4": "kdmoe_ced_tiny_meanpool_vitkd",
           "ced_tiny_kd8_13_logit_qkv_t4": "kdmoe_ced_tiny_meanpool_qkv",
           "ced_tiny_kd8_14_logit_patchcrd_t4": "kdmoe_ced_tiny_meanpool_patchcrd",
           "ced_tiny_kd8_1_logit_t4": "kdmoe_ced_tiny_meanpool_kd",
           "ced_tiny_kd8_2_crd_t4": "kdmoe_ced_tiny_meanpool_crd",
           "ced_tiny_kd8_3_mgd_t4": "kdmoe_ced_tiny_meanpool_mgd",
           "ced_tiny_kd8_4_logit_crd_t4": "kdmoe_ced_tiny_meanpool_crd",
           "ced_tiny_kd8_5_logit_mgd_t4": "kdmoe_ced_tiny_meanpool_mgd",
           "ced_tiny_kd8_6_logit_mgdpkd_a1_t4": "kdmoe_ced_tiny_meanpool_mgd",
           "ced_tiny_kd8_7_logit_mgdpkd_a3_t4": "kdmoe_ced_tiny_meanpool_mgd",
           "ced_tiny_kd8_8_logit_crd_l02_t4": "kdmoe_ced_tiny_meanpool_crd",
           "ced_tiny_kd8_9_logit_rel_t4": "kdmoe_ced_tiny_meanpool_kd_rel",
           # recipe ce9 của MLNC (vuong_mix_dataset_v2, user 2026-11-04) trên cùng 6 phương pháp kd8
           "ced_tiny_ce9_1_ce": "kdmoe_ced_tiny_meanpool_ce9_ce",
           "ced_tiny_ce9_2_logit": "kdmoe_ced_tiny_meanpool_ce9_kd",
           "ced_tiny_ce9_3_mgd": "kdmoe_ced_tiny_meanpool_ce9_mgd",
           "ced_tiny_ce9_4_crd": "kdmoe_ced_tiny_meanpool_ce9_crd",
           "ced_tiny_ce9_5_logit_mgd": "kdmoe_ced_tiny_meanpool_ce9_mgd",
           "ced_tiny_ce9_6_logit_crd": "kdmoe_ced_tiny_meanpool_ce9_crd",
           "ced_mini_ce9_1_ce": "kdmoe_ced_mini_meanpool_ce9_ce",
           "ced_mini_ce9_2_logit": "kdmoe_ced_mini_meanpool_ce9_kd",
           "ced_mini_ce9_3_mgd": "kdmoe_ced_mini_meanpool_ce9_mgd",
           "ced_mini_ce9_4_crd": "kdmoe_ced_mini_meanpool_ce9_crd",
           "ced_mini_ce9_5_logit_mgd": "kdmoe_ced_mini_meanpool_ce9_mgd",
           "ced_mini_ce9_6_logit_crd": "kdmoe_ced_mini_meanpool_ce9_crd",
           "ced_small_ce9_1_ce": "kdmoe_ced_small_meanpool_ce9_ce",
           "ced_small_ce9_2_logit": "kdmoe_ced_small_meanpool_ce9_kd",
           "ced_small_ce9_3_mgd": "kdmoe_ced_small_meanpool_ce9_mgd",
           "ced_small_ce9_4_crd": "kdmoe_ced_small_meanpool_ce9_crd",
           "ced_small_ce9_5_logit_mgd": "kdmoe_ced_small_meanpool_ce9_mgd",
           "ced_small_ce9_6_logit_crd": "kdmoe_ced_small_meanpool_ce9_crd",
           # logit + MGD, config cu (kd8) cho CED-Mini / CED-Small (user 2026-11-05)
           "ced_mini_kd8_5_logit_mgd_t4": "kdmoe_ced_mini_meanpool_mgd",
           "ced_small_kd8_5_logit_mgd_t4": "kdmoe_ced_meanpool_mgd",
           "ced_mini_kd8_1_logit_t4": "kdmoe_ced_mini_meanpool_kd",
           "ced_mini_kd8_4_logit_crd_t4": "kdmoe_ced_mini_meanpool_crd",
           "ced_mini_kd8_2_crd_t4": "kdmoe_ced_mini_meanpool_crd",
           "ced_mini_kd8_3_mgd_t4": "kdmoe_ced_mini_meanpool_mgd",
           "ced_mini_kd8_0_ce_t4": "kdmoe_ced_mini_meanpool_ce",
           "ced_small_kd8_1_logit_t4": "kdmoe_ced_meanpool_kd",
           "ced_small_kd8_4_logit_crd_t4": "kdmoe_ced_meanpool_crd",
           "ced_small_kd8_2_crd_t4": "kdmoe_ced_meanpool_crd",
           "ced_small_kd8_3_mgd_t4": "kdmoe_ced_meanpool_mgd",
           "ced_small_kd8_0_ce_t4": "kdmoe_ced_meanpool_ce",
           # logit + MGD can bang gradient (aux_target_ratio 0.2), config cu + ce9, Mini/Small (user 2026-11-05)
           "ced_mini_kd8_15_logit_mgdbal_t4": "kdmoe_ced_mini_meanpool_mgd",
           "ced_small_kd8_15_logit_mgdbal_t4": "kdmoe_ced_meanpool_mgd",
           "ced_mini_ce9_7_logit_mgdbal": "kdmoe_ced_mini_meanpool_ce9_mgd",
           "ced_small_ce9_7_logit_mgdbal": "kdmoe_ced_small_meanpool_ce9_mgd",
           # recipe ce9, KD logit / logit+MGD, seed 42 / 7 / 1234 (user 2026-11-05)
           "ced_tiny_ce9_2_logit_s42": "kdmoe_ced_tiny_meanpool_ce9_kd",
           "ced_tiny_ce9_2_logit_s7": "kdmoe_ced_tiny_meanpool_ce9_kd",
           "ced_tiny_ce9_2_logit_s1234": "kdmoe_ced_tiny_meanpool_ce9_kd",
           "ced_tiny_ce9_5_logit_mgd_s42": "kdmoe_ced_tiny_meanpool_ce9_mgd",
           "ced_tiny_ce9_5_logit_mgd_s7": "kdmoe_ced_tiny_meanpool_ce9_mgd",
           "ced_tiny_ce9_5_logit_mgd_s1234": "kdmoe_ced_tiny_meanpool_ce9_mgd",
           "ced_mini_ce9_2_logit_s42": "kdmoe_ced_mini_meanpool_ce9_kd",
           "ced_mini_ce9_2_logit_s7": "kdmoe_ced_mini_meanpool_ce9_kd",
           "ced_mini_ce9_2_logit_s1234": "kdmoe_ced_mini_meanpool_ce9_kd",
           "ced_mini_ce9_5_logit_mgd_s42": "kdmoe_ced_mini_meanpool_ce9_mgd",
           "ced_mini_ce9_5_logit_mgd_s7": "kdmoe_ced_mini_meanpool_ce9_mgd",
           "ced_mini_ce9_5_logit_mgd_s1234": "kdmoe_ced_mini_meanpool_ce9_mgd",
           "ced_small_ce9_2_logit_s42": "kdmoe_ced_small_meanpool_ce9_kd",
           "ced_small_ce9_2_logit_s7": "kdmoe_ced_small_meanpool_ce9_kd",
           "ced_small_ce9_2_logit_s1234": "kdmoe_ced_small_meanpool_ce9_kd",
           "ced_small_ce9_5_logit_mgd_s42": "kdmoe_ced_small_meanpool_ce9_mgd",
           "ced_small_ce9_5_logit_mgd_s7": "kdmoe_ced_small_meanpool_ce9_mgd",
           "ced_small_ce9_5_logit_mgd_s1234": "kdmoe_ced_small_meanpool_ce9_mgd",
           # recipe ce9 + SpecAugment (1x12 mel + 1x80 frame) cho moi run, 3 co (user 2026-11-05)
           "ced_tiny_ce9sa_1_ce": "kdmoe_ced_tiny_meanpool_ce9sa_ce",
           "ced_tiny_ce9sa_2_logit": "kdmoe_ced_tiny_meanpool_ce9sa_kd",
           "ced_tiny_ce9sa_3_mgd": "kdmoe_ced_tiny_meanpool_ce9sa_mgd",
           "ced_tiny_ce9sa_4_crd": "kdmoe_ced_tiny_meanpool_ce9sa_crd",
           "ced_tiny_ce9sa_5_logit_mgd": "kdmoe_ced_tiny_meanpool_ce9sa_mgd",
           "ced_tiny_ce9sa_6_logit_crd": "kdmoe_ced_tiny_meanpool_ce9sa_crd",
           "ced_mini_ce9sa_1_ce": "kdmoe_ced_mini_meanpool_ce9sa_ce",
           "ced_mini_ce9sa_2_logit": "kdmoe_ced_mini_meanpool_ce9sa_kd",
           "ced_mini_ce9sa_3_mgd": "kdmoe_ced_mini_meanpool_ce9sa_mgd",
           "ced_mini_ce9sa_4_crd": "kdmoe_ced_mini_meanpool_ce9sa_crd",
           "ced_mini_ce9sa_5_logit_mgd": "kdmoe_ced_mini_meanpool_ce9sa_mgd",
           "ced_mini_ce9sa_6_logit_crd": "kdmoe_ced_mini_meanpool_ce9sa_crd",
           "ced_small_ce9sa_1_ce": "kdmoe_ced_small_meanpool_ce9sa_ce",
           "ced_small_ce9sa_2_logit": "kdmoe_ced_small_meanpool_ce9sa_kd",
           "ced_small_ce9sa_3_mgd": "kdmoe_ced_small_meanpool_ce9sa_mgd",
           "ced_small_ce9sa_4_crd": "kdmoe_ced_small_meanpool_ce9sa_crd",
           "ced_small_ce9sa_5_logit_mgd": "kdmoe_ced_small_meanpool_ce9sa_mgd",
           "ced_small_ce9sa_6_logit_crd": "kdmoe_ced_small_meanpool_ce9sa_crd",
           # ce9 + SpecAugment, lr 1e-5 / 1e-4 (muc beats12 cua ce9), Tiny + Mini (user 2026-11-07)
           "ced_tiny_ce9sa_lr1e5_1_ce": "kdmoe_ced_tiny_meanpool_ce9sa_lr1e5_ce",
           "ced_tiny_ce9sa_lr1e5_2_logit": "kdmoe_ced_tiny_meanpool_ce9sa_lr1e5_kd",
           "ced_tiny_ce9sa_lr1e5_3_mgd": "kdmoe_ced_tiny_meanpool_ce9sa_lr1e5_mgd",
           "ced_tiny_ce9sa_lr1e5_4_crd": "kdmoe_ced_tiny_meanpool_ce9sa_lr1e5_crd",
           "ced_tiny_ce9sa_lr1e5_5_logit_mgd": "kdmoe_ced_tiny_meanpool_ce9sa_lr1e5_mgd",
           "ced_tiny_ce9sa_lr1e5_6_logit_crd": "kdmoe_ced_tiny_meanpool_ce9sa_lr1e5_crd",
           "ced_mini_ce9sa_lr1e5_1_ce": "kdmoe_ced_mini_meanpool_ce9sa_lr1e5_ce",
           "ced_mini_ce9sa_lr1e5_2_logit": "kdmoe_ced_mini_meanpool_ce9sa_lr1e5_kd",
           "ced_mini_ce9sa_lr1e5_3_mgd": "kdmoe_ced_mini_meanpool_ce9sa_lr1e5_mgd",
           "ced_mini_ce9sa_lr1e5_4_crd": "kdmoe_ced_mini_meanpool_ce9sa_lr1e5_crd",
           "ced_mini_ce9sa_lr1e5_5_logit_mgd": "kdmoe_ced_mini_meanpool_ce9sa_lr1e5_mgd",
           "ced_mini_ce9sa_lr1e5_6_logit_crd": "kdmoe_ced_mini_meanpool_ce9sa_lr1e5_crd",
           # chay lai run hoi tu muon: lr 1e-5 -> 45 epoch, lr 3e-5 -> 30 epoch, early stop patience 3 (user 2026-11-07)
           "ced_tiny_ce9sa_lr1e5_e45p3_2_logit": "kdmoe_ced_tiny_meanpool_ce9sa_lr1e5_e45p3_kd",
           "ced_tiny_ce9sa_lr1e5_e45p3_6_logit_crd": "kdmoe_ced_tiny_meanpool_ce9sa_lr1e5_e45p3_crd",
           "ced_tiny_ce9sa_lr1e5_e45p3_1_ce": "kdmoe_ced_tiny_meanpool_ce9sa_lr1e5_e45p3_ce",
           "ced_tiny_ce9sa_lr1e5_e45p3_3_mgd": "kdmoe_ced_tiny_meanpool_ce9sa_lr1e5_e45p3_mgd",
           "ced_tiny_ce9sa_lr1e5_e45p3_5_logit_mgd": "kdmoe_ced_tiny_meanpool_ce9sa_lr1e5_e45p3_mgd",
           "ced_tiny_ce9_lr1e5_e45p3_2_logit_s7": "kdmoe_ced_tiny_meanpool_ce9_lr1e5_e45p3_kd",
           # lr 1e-5, 45 epoch, early stop patience 8 (user 2026-11-08)
           "ced_tiny_ce9_lr1e5_e45p8_2_logit_s7": "kdmoe_ced_tiny_meanpool_ce9_lr1e5_e45p8_kd",
           "ced_tiny_ce9sa_lr1e5_e45p8_1_ce": "kdmoe_ced_tiny_meanpool_ce9sa_lr1e5_e45p8_ce",
           "ced_tiny_ce9sa_lr1e5_e45p8_2_logit": "kdmoe_ced_tiny_meanpool_ce9sa_lr1e5_e45p8_kd",
           "ced_tiny_ce9sa_lr1e5_e45p8_3_mgd": "kdmoe_ced_tiny_meanpool_ce9sa_lr1e5_e45p8_mgd",
           "ced_tiny_ce9sa_lr1e5_e45p8_5_logit_mgd": "kdmoe_ced_tiny_meanpool_ce9sa_lr1e5_e45p8_mgd",
           "ced_tiny_ce9sa_lr1e5_e45p8_6_logit_crd": "kdmoe_ced_tiny_meanpool_ce9sa_lr1e5_e45p8_crd",
           # GFD-Cosine (user 2026-11-03): mean token block -> Linear+LN -> 1 - cos với mean token BEATs cùng lớp
           "ced_tiny_gfd2_b12_w005_t4": "kdmoe_ced_tiny_meanpool_gfd_b12",
           "ced_tiny_gfd3_b12_w010_t4": "kdmoe_ced_tiny_meanpool_gfd_b12",
           "ced_tiny_gfd4_b12_w020_t4": "kdmoe_ced_tiny_meanpool_gfd_b12",
           "ced_tiny_gfd5_b6912_w010_t4": "kdmoe_ced_tiny_meanpool_gfd_b6912",
           "mn10_snrroute_crd_t4": "kdmoe_mn10_snrroute_relation",
           "mn10_spkd": "kdmoe_mn10_spkd", "mn10_cka": "kdmoe_mn10_cka", "mn10_relation": "kdmoe_mn10_relation",
           "mn10_spkd_mask": "kdmoe_mn10_spkd_mask", "mn10_snrroute_relation": "kdmoe_mn10_snrroute_relation",
           "mn10_crd": "kdmoe_mn10_crd",
           "panns_snrroute": "kdmoe_panns_cnn10_snrroute"}
RUNS.update({f"Unified {name}": (OUTPUT / "unified" / name / "train", "fp32_cache", CONFIG / "unified" / f"{stem}.yaml")
             for name, stem in UNIFIED.items()})
# Ablation không FRCRN: student nghe thẳng mixture
RUNS["Unified mn10_crd_t4_mixture"] = (OUTPUT / "unified" / "mn10_crd_t4_mixture" / "train", "mixture",
                                       CONFIG / "unified" / "kdmoe_mn10_crd.yaml")
RUNS["Unified ced_tiny_meanpool_ce_mixture"] = (OUTPUT / "unified" / "ced_tiny_meanpool_ce_mixture" / "train", "mixture",
                                             CONFIG / "unified" / "kdmoe_ced_tiny_meanpool_ce.yaml")
RUNS["Unified ced_tiny_meanpool_kd_t4_mixture"] = (OUTPUT / "unified" / "ced_tiny_meanpool_kd_t4_mixture" / "train", "mixture",
                                                CONFIG / "unified" / "kdmoe_ced_tiny_meanpool_kd.yaml")
# mix-dataset-v2 (không nhiễu sạch): teacher BEATs nghe n_hat, student không L_front / stage 1
RUNS["Unified v2nhat_ced_tiny_meanpool_kd_t4"] = (OUTPUT / "unified" / "v2nhat_ced_tiny_meanpool_kd_t4" / "train",
                                                "fp32_cache", CONFIG / "unified" / "kdmoe_ced_tiny_meanpool_kd.yaml")
for _name, _stem in (("v2nhat_ced_tiny_meanpool_crd_t4", "kdmoe_ced_tiny_meanpool_crd"),
                     ("v2nhat_ced_tiny_meanpool_mgd_t4", "kdmoe_ced_tiny_meanpool_mgd"),
                     ("v2nhat_ced_tiny_meanpool_kd_spkd12_t4", "kdmoe_ced_tiny_meanpool_kd_spkd12")):
    RUNS[f"Unified {_name}"] = (OUTPUT / "unified" / _name / "train", "fp32_cache", CONFIG / "unified" / f"{_stem}.yaml")
for _name in ("ced_tiny_meanpool_kd_ld050", "ced_tiny_meanpool_kd_ld075", "ced_tiny_meanpool_kd_dual"):
    RUNS[f"Unified v2nhat_{_name}_t4"] = (OUTPUT / "unified" / f"v2nhat_{_name}_t4" / "train", "fp32_cache",
                                          CONFIG / "unified" / f"kdmoe_{_name}.yaml")
# noise_ytid nằm trong AudioSet balanced+unbalanced train: backbone/teacher pretrain có nhãn AudioSet đã thấy
LEAKED = {line.strip() for line in (CONFIG / "audioset_train_noise_ytids.txt").read_text().splitlines()
          if line.strip() and not line.startswith("#")}
SPLITS = ("validation", "test")
SNRS = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)


def build_model(yaml_path: Path) -> KDMoEModel:
    """KDMoEModel; ReviewKDMoE khi yaml có mục reviewkd; EfficientATKDMoE khi student.type = efficientat."""
    import yaml

    if "reviewkd" in yaml.safe_load(yaml_path.read_text(encoding="utf-8")):
        if student_type(yaml_path) == "efficientat":
            from efficientat import EfficientATReviewKDMoE

            return EfficientATReviewKDMoE.from_yaml(yaml_path)
        from reviewkd import ReviewKDMoE

        return ReviewKDMoE.from_yaml(yaml_path)
    if student_type(yaml_path) == "ced":
        from ced import CEDKDMoE

        return CEDKDMoE.from_yaml(yaml_path)
    if student_type(yaml_path) == "efficientat":
        from efficientat import EfficientATKDMoE

        return EfficientATKDMoE.from_yaml(yaml_path)
    return KDMoEModel.from_yaml(yaml_path)


def weight_sets(run_dir: Path) -> dict[str, dict]:
    stage2 = run_dir / "stage2"
    sets = {"best_raw": torch.load(stage2 / "best.pt", map_location="cpu", weights_only=False)["model"]}
    resume = stage2 / "resume.pt"
    if resume.is_file():
        data = torch.load(resume, map_location="cpu", weights_only=False)
        state = dict(data["model"])
        state.update(data["ema"])
        sets["ema_last"] = state
    return sets


@torch.no_grad()
def predict(model: KDMoEModel, split: str, pipeline: str, separator, device) -> dict[str, np.ndarray]:
    cache = FRCRN_CACHE_DIR / f"frcrn_{split}.npy" if pipeline == "fp32_cache" else None
    edges = model.config.snr_band_edges if model.config.route_by == "snr" else None
    dataset = KDMoEMixDataset(split=split, load_teacher=False, frcrn_cache=cache, snr_band_edges=edges,
                              load_noise=False, dual_input=getattr(model, "dual_input", False))  # không dùng nhiễu sạch
    probs, routers, labels, groups, snrs, ids, ytids = [], [], [], [], [], [], []
    for batch in build_dataloader(dataset, 64, False, 4):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            waveform = batch["waveform"].to(device)
            snr_est = batch["snr_est"].to(device) if "snr_est" in batch else None
            if pipeline == "bf16_online":
                mixture, waveform = waveform, separator(waveform)
                snr_est = frcrn_snr_db(mixture, waveform)
            output = model(waveform, None, snr_est)
        probs.append(output.logits.float().softmax(-1).cpu().numpy())
        routers.append(output.router_logits.argmax(-1).cpu().numpy())
        labels.append(batch["label"].numpy())
        groups.append(batch["group"].numpy())
        snrs.append(batch["snr_db"].numpy())
        ids += list(batch["sample_id"])
        ytids += list(batch["noise_ytid"])
    return {"prob": np.concatenate(probs), "router": np.concatenate(routers), "label": np.concatenate(labels),
            "group": np.concatenate(groups), "snr": np.concatenate(snrs), "id": np.array(ids),
            "leaked": np.array([y in LEAKED for y in ytids])}


def metrics(p: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    if not mask.any():  # tập con rỗng (vd. "leaked" của mix-dataset-v2, toàn clip sạch)
        return {"n": 0, **{k: float("nan") for k in ("accuracy", "top3", "macro_f1", "precision_macro",
                                                      "recall_macro", "mAP", "router_acc")}}
    prob, label = p["prob"][mask], p["label"][mask]
    pred = prob.argmax(1)
    classes = np.arange(prob.shape[1])
    precision, recall, f1, _ = precision_recall_fscore_support(label, pred, labels=classes, zero_division=0)
    onehot = np.eye(prob.shape[1])[label]
    present = onehot.sum(0) > 0
    ap = [average_precision_score(onehot[:, c], prob[:, c]) for c in classes[present]]
    top3 = (np.argsort(-prob, 1)[:, :3] == label[:, None]).any(1).mean()
    return {"n": int(mask.sum()), "accuracy": float((pred == label).mean()), "top3": float(top3),
            "macro_f1": float(f1.mean()), "precision_macro": float(precision.mean()),
            "recall_macro": float(recall.mean()), "mAP": float(np.mean(ap)),
            "router_acc": float((p["router"][mask] == p["group"][mask]).mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate finished FRCRN_KD_MoE runs on validation and test")
    parser.add_argument("--dataset", required=True, help="Mixture-36l root (manifest.csv, labels.txt, wav)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", default=None, help="chỉ các run có tên chứa chuỗi này")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    os.environ["MLNC_MIXTURE_36L"] = arguments.dataset
    out = arguments.out if arguments.out.is_absolute() else HERE / arguments.out
    out.mkdir(parents=True, exist_ok=True)
    exact = {run: spec for run, spec in RUNS.items() if arguments.only in (run, f"Unified {run}")} if arguments.only else {}
    runs = exact or {run: spec for run, spec in RUNS.items() if not arguments.only or arguments.only in run}

    device = torch.device("cuda")
    separator = load_frcrn(CONFIG / "frcrn.yaml", HERE / "model" / "frcrn_best.pt", device)
    build_frcrn_cache(separator, device, logging.getLogger("eval"), splits=SPLITS)
    build_frcrn_snr(logging.getLogger("eval"), splits=SPLITS)

    rows = []
    for run, (run_dir, pipeline, yaml_path) in runs.items():
        if not (run_dir / "stage2" / "best.pt").is_file():
            print(f"skip {run}: no stage2/best.pt", flush=True)
            continue
        finished = (run_dir / "stage2" / "result.json").is_file()
        for weights, state in weight_sets(run_dir).items():
            model = build_model(yaml_path).to(device)
            model.load_state_dict(state)
            model.eval()
            for split in SPLITS:
                p = predict(model, split, pipeline, separator, device)
                tag = f"{run} | {weights} | {split}"
                subsets = {"all": np.ones(len(p["label"]), bool), "clean": ~p["leaked"], "leaked": p["leaked"]}
                for subset, base in subsets.items():
                    for snr in ("all",) + SNRS:
                        mask = base if snr == "all" else base & (p["snr"] == snr)
                        rows.append({"run": run, "weights": weights, "pipeline": pipeline, "finished": finished,
                                     "router_groups": f"{model.config.route_by}{model.config.num_groups}",
                                     "split": split, "subset": subset, "snr_db": snr, **metrics(p, mask)})
                for row in (rows[-21], rows[-14]):  # (all, toàn bộ SNR) và (clean, toàn bộ SNR)
                    print(tag, row["subset"], {k: round(v, 4) for k, v in row.items() if isinstance(v, float)},
                          flush=True)
                safe = run.split(" (")[0].replace(" ", "_")
                with open(out / f"predictions_{safe}_{weights}_{split}.csv", "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["sample_id", "snr_db", "label", "pred", "pred_prob", "group", "router_pred",
                                "audioset_train_leaked"])
                    for i in range(len(p["label"])):
                        w.writerow([p["id"][i], p["snr"][i], p["label"][i], int(p["prob"][i].argmax()),
                                    round(float(p["prob"][i].max()), 5), p["group"][i], p["router"][i],
                                    int(p["leaked"][i])])

    with open(out / "metrics_by_snr.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out / "metrics_by_snr.csv", len(rows), "rows")


if __name__ == "__main__":
    main()
