"""CED-Small (Dinkel, Wang, Yan, Zhang, Wang - ICASSP 2024, "CED: Consistent Ensemble Distillation for Audio
Tagging"; github.com/RicherMans/CED) làm encoder cho KD-MoE, pooling bằng soft attention mask A(t, f).

ced_small: ViT 12 block, 384 chiều, 6 head, ~22M tham số, mAP AudioSet 49.6. Lấy từ code gốc
(`models/audiotransformer.py`), giống SNR_Expert_SSLAM/models/ced_backbone.py:

  * mel      FrontEnd 16 kHz (đúng tần số n_hat, không resample), n_fft 512, cửa sổ 512, bước 160, 64 mel 0-8 kHz,
             AmplitudeToDB(top_db 120); 4 s -> 401 frame, cắt còn 400 (frame cuối vốn rơi khỏi conv patch).
  * init_bn  BatchNorm2d trên 64 mel; SpecAugment của KD-MoE tác dụng SAU nó (dải bị che đọc như năng lượng trung bình).
  * patches  16 x 16, bước 16 -> lưới 4 tần số x 25 thời gian = 100 token; time_pos_embed cắt còn 25.

Pooling (`AttentionMaskPool`): gated attention (Ilse, Tomczak, Welling - ICML 2018) s_k = w'(tanh(V h_k) *
sigmoid(U h_k)), mặt nạ A_k = sigmoid(s_k) trong [0, 1], z = sum A_k h_k / sum A_k; w khởi tạo 0 nên ban đầu
đúng bằng mean-pool của CED (bản của run7_attn / run8_attn_kd_crd). Sau đó LayerNorm pretrained của
`outputlayer` CED. Cổng mặt nạ, NoiseHead, router 3 khoảng SNR + 3 expert giữ nguyên KD-MoE.

Layer-wise lr decay như SNR_Expert_SSLAM (BEATs, Chen et al., ICML 2023, bảng 4): block i nhận
lr * decay^(11 - i), block trên cùng + norm cuối lr đầy đủ, phần encoder ngoài block (patch, pos, init_bn) lr
đầy đủ, attention pool lr * attention_lr_scale.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn

from external_repo import import_isolated
from src.model.kdmoe import KDMoEConfig, KDMoEModel
from src.model.kdmoe.gate import MaskGate, NoiseHead
from src.model.kdmoe.panns import HintRegressor

NUM_FRAMES = 400
SECOND_CHANNELS = ("mixture", "reliability")
RELIABILITY_FLOOR_DB = 40.0


def reliability_map(mel_nhat: Tensor, mel_mixture: Tensor) -> Tensor:
    """Bản đồ tin cậy của n_hat trên từng ô mel (B, 1, 64, T): 10 log10(P_nhat / P_mixture) - đúng mặt nạ nhiễu FRCRN đã áp,
    cắt trong [-40, 0] dB rồi đưa về [-1, 1] (1 = ô giữ nguyên mixture, -1 = ô bị FRCRN xoá ~ ô thuộc tiếng nói).
    Hai log-mel phải cùng hệ số RMS (model.shared_rms). Chẩn đoán 2026-11-02: ở 20 dB n_hat mất ~1/3 biên độ nhiễu và méo
    ngang nhiễu; mặt nạ cho student biết ô nào là 'lỗ' (dữ liệu thiếu) thay vì đọc nó như năng lượng thật - như mặt nạ M
    đưa vào mạng cùng dữ liệu trong GAIN (Yoon et al., ICML 2018)."""
    delta = (mel_nhat - mel_mixture).clamp(-RELIABILITY_FLOOR_DB, 0.0)
    return delta / (RELIABILITY_FLOOR_DB / 2) + 1.0


def build_ced(student: dict[str, Any]) -> nn.Module:
    """AudioTransformer CED chưa nạp trọng số."""
    (module,) = import_isolated(student["repo_dir"], student["repo_url"], student["commit"], "models.audiotransformer")
    return getattr(module, student["variant"])(pretrained=False, drop_path_rate=student.get("drop_path", 0.0))


def ced_state(url: str) -> dict[str, Tensor]:
    """State dict CED theo tên của repo gốc. Nhận .pt của Zenodo hoặc model.safetensors của bản HF chính chủ
    (mispeech/ced-small): bản HF thêm tiền tố "encoder." và đặt BatchNorm của init_bn ở "init_bn." thay cho
    "init_bn.1."; không có buffer mel của front_end (cố định, dựng lại từ tham số)."""
    if not url.split("?")[0].endswith(".safetensors"):
        source = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        return source.get("model", source)
    import hashlib

    from safetensors.torch import load_file

    path = Path(torch.hub.get_dir()) / "checkpoints" / f"ced_{hashlib.sha1(url.encode()).hexdigest()[:12]}.safetensors"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(path))
    state = {}
    for key, value in load_file(str(path)).items():
        key = key.removeprefix("encoder.")
        state["init_bn.1." + key.removeprefix("init_bn.") if key.startswith("init_bn.") else key] = value
    return state


class CEDFrontend(nn.Module):
    """Waveform 16 kHz -> (log-mel sau init_bn (B, 1, 64, 400), log-mel nhiễu cùng chuẩn hoá hoặc None).

    Đích L_front chuẩn hoá bằng thống kê đang chạy của init_bn (eval, no_grad) để không trộn nhiễu sạch vào
    thống kê BatchNorm của mixture."""

    def __init__(self, front_end: nn.Module, init_bn: nn.Module, num_frames: int = NUM_FRAMES,
                 dual: bool = False, second: str = "mixture") -> None:
        super().__init__()
        if second not in SECOND_CHANNELS:
            raise ValueError(f"student.second_channel must be one of {SECOND_CHANNELS}, got {second!r}")
        self.front_end = front_end
        self.init_bn = init_bn
        self.num_frames = num_frames
        self.dual = dual
        self.second = second
        # dual: waveform (B, 2, T) = [n_hat, mixture]; mixture có BatchNorm riêng (thống kê khác n_hat), sao từ init_bn.
        # second "reliability": kênh 2 = bản đồ tin cậy, không cần BatchNorm
        self.init_bn_mix = copy.deepcopy(init_bn) if dual and second == "mixture" else None

    def _mel(self, audio: Tensor) -> Tensor:
        return self.front_end(audio.float())[..., : self.num_frames].unsqueeze(1)   # (B, 1, 64, 400)

    def forward(self, waveform: Tensor, noise: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            if self.dual and self.second == "reliability":
                mel_nhat = self._mel(waveform[:, 0])
                spec = torch.cat([self.init_bn(mel_nhat), reliability_map(mel_nhat, self._mel(waveform[:, 1]))], dim=1)
            elif self.dual:
                spec = torch.cat([self.init_bn(self._mel(waveform[:, 0])), self.init_bn_mix(self._mel(waveform[:, 1]))],
                                 dim=1)                                                      # (B, 2, 64, 400)
            else:
                spec = self.init_bn(self._mel(waveform))
            target = None
            if noise is not None:
                with torch.no_grad():
                    training = self.init_bn.training
                    self.init_bn.eval()
                    target = self.init_bn(self._mel(noise))
                    self.init_bn.train(training)
        return spec, target


class CEDEncoder(nn.Module):
    """patch_embed + pos + 12 block + norm của CED; (B, 1, 64, T) -> token (B, F'=4, T'=25, 384)."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.patch_embed = net.patch_embed
        self.time_pos_embed = net.time_pos_embed
        self.freq_pos_embed = net.freq_pos_embed
        self.pos_drop = net.pos_drop
        self.blocks = net.blocks
        self.norm = net.norm

    qkv: tuple[Tensor, Tensor, Tensor] | None = None

    def capture_qkv(self, block: int) -> None:
        """Ghi Q/K/V (B, N, C) của block `block` (1-based) mỗi lần forward lúc train (MiniLMv2)."""
        def hook(module, inputs, output):
            if self.training:
                self.qkv = tuple(output.chunk(3, dim=-1))
        self.blocks[block - 1].attn.qkv.register_forward_hook(hook)

    def forward(self, x: Tensor, taps: tuple[int, ...] = ()) -> tuple[Tensor, dict[int, Tensor]]:
        """-> (token cuối sau norm, {block (1-based): token sau block}), mọi token (B, F', T', C)."""
        x = self.patch_embed(x)                                              # (B, C, F', T')
        batch, channels, freq, time = x.shape
        x = x + self.time_pos_embed[:, :, :, :time] + self.freq_pos_embed
        x = self.pos_drop(x.flatten(2).transpose(1, 2))                     # 'b c f t -> b (f t) c' như CED
        tapped = {}
        for index, block in enumerate(self.blocks, 1):
            x = block(x)
            if index in taps:
                tapped[index] = x.reshape(batch, freq, time, channels)
        return self.norm(x).reshape(batch, freq, time, channels), tapped


class AttentionMaskPool(nn.Module):
    """Soft attention mask A(t, f) trên token rồi pool có trọng số (bản chép của SNR_Expert_SSLAM/expert_lib.py)."""

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)
        nn.init.zeros_(self.w.weight)
        nn.init.zeros_(self.w.bias)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        score = self.w(torch.tanh(self.V(tokens)) * torch.sigmoid(self.U(tokens)))   # (B, N, 1)
        mask = torch.sigmoid(score.squeeze(-1))
        weights = mask / mask.sum(dim=1, keepdim=True)
        return (weights.unsqueeze(-1) * tokens).sum(dim=1), mask


class CEDPool(nn.Module):
    """Token (B, C, F', T') -> embedding: AttentionMaskPool (hoặc mean-pool gốc của CED khi attention=False) rồi
    LayerNorm pretrained của outputlayer CED."""

    def __init__(self, dim: int, norm: nn.LayerNorm, hidden: int, attention: bool = True) -> None:
        super().__init__()
        self.attn = AttentionMaskPool(dim, hidden) if attention else None
        self.norm = norm
        self.last_mask: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        tokens = x.flatten(2).transpose(1, 2)
        if self.attn is None:
            return self.norm(tokens.mean(dim=1))
        pooled, mask = self.attn(tokens)
        self.last_mask = mask.detach()
        return self.norm(pooled)


class CEDKDMoE(KDMoEModel):
    """KD-MoE với encoder CED (mục `student` trong yaml), đi đường encode CNN của KD-MoE (encoder: panns_cnn10):
    token -> cổng -> NoiseHead / soft attention pool -> router + expert."""

    TOP_LEVEL = {"front_end": "frontend.front_end", "init_bn": "frontend.init_bn", "patch_embed": "encoder.patch_embed",
                 "time_pos_embed": "encoder.time_pos_embed", "freq_pos_embed": "encoder.freq_pos_embed",
                 "blocks": "encoder.blocks", "norm": "encoder.norm"}

    def __init__(self, config: KDMoEConfig, frontend_config: dict[str, Any] | None = None,
                 student: dict[str, Any] | None = None) -> None:
        if config.encoder != "panns_cnn10":
            raise ValueError(f"CEDKDMoE uses the CNN encode path (encoder: panns_cnn10), got {config.encoder!r}")
        self._student = dict(student or {})
        self.dual_input = bool(self._student.get("dual_input", False))   # waveform (B, 2, T) = [n_hat, mixture]
        self.second_channel = str(self._student.get("second_channel", "mixture"))
        self.shared_rms = self.dual_input and self.second_channel == "reliability"  # giữ tỉ lệ n_hat / mixture
        super().__init__(config, frontend_config)

    def _build_panns(self, config: KDMoEConfig, frontend_config: dict[str, Any]) -> int:
        net = build_ced(self._student)
        dim = net.embed_dim
        if self.dual_input:  # patch_embed nhận 2 kênh; kênh mixture khởi tạo 0 -> bắt đầu giống hệt student một kênh
            old = net.patch_embed.proj
            proj = nn.Conv2d(2, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None)
            nn.init.zeros_(proj.weight)
            net.patch_embed.proj = proj
        self.frontend = CEDFrontend(net.front_end, net.init_bn, config.num_frames, dual=self.dual_input,
                                    second=self.second_channel)
        self.encoder = CEDEncoder(net)
        self.gate = MaskGate(dim, config.gate_eps, config.gate_bias_init) if config.use_gate else None
        self.noise_head = NoiseHead(dim, config.patch_size, config.freq_bins, config.num_frames)
        pooling = self._student.get("pooling", "attention")
        if pooling not in ("attention", "mean"):
            raise ValueError(f"student.pooling must be 'attention' or 'mean', got {pooling!r}")
        self.post_fc = CEDPool(dim, net.outputlayer[0], int(self._student.get("attention_hidden", 128)),
                               attention=pooling == "attention")
        self._num_blocks = len(net.blocks)
        self._patch_grid = (config.num_frames // 16, config.freq_bins // 16)   # (T', F') = (25, 4)
        self._relation_patch = bool(self._student.get("patch_crd", False))
        self._tokens: Tensor | None = None
        # FitNets (Romero et al., ICLR 2015): guided layer = token sau block `student.hint_block` -> regressor tuyến tính
        # -> lưới token BEATs (at_grid, at_*_fraction), so MSE với hint teacher (token BEATs layer 6)
        self._hint_block = int(self._student.get("hint_block", 0))
        # feature-based KD (featurekd.py): student.fkd = {taps: [1, 2, 12], qkv_block: 12}
        fkd = self._student.get("fkd") or {}
        self._fkd_taps = tuple(int(t) for t in fkd.get("taps", ()))
        self._qkv_block = int(fkd.get("qkv_block", 0))
        self._num_shallow = max(len(self._fkd_taps) - 1, 0)
        self._extra: dict | None = None
        if self._qkv_block:
            self.encoder.capture_qkv(self._qkv_block)
        if self._hint_block:
            self.hint_regressor = HintRegressor(dim, config.hint_dim)
        return dim

    def _encode_panns(self, x: Tensor, with_noise: bool) -> tuple:
        taps = tuple(set(self._fkd_taps) | ({self._hint_block} - {0}))
        tokens, tapped = self.encoder(x, taps)                              # (B, F', T', C)
        hint = None
        if self._hint_block:
            cfg = self.config
            hint = self.hint_regressor(tapped[self._hint_block].permute(0, 3, 2, 1), cfg.at_grid, cfg.at_time_fraction,
                                       cfg.at_freq_fraction)                # (B, C, T', F') -> (B, T, F, hint_dim)
        self._extra = None
        if self.training and self._fkd_taps:
            grid = lambda t: t.permute(0, 3, 1, 2)                          # (B, F', T', C) -> (B, C, F', T')
            last = max(self._fkd_taps)
            self._extra = {"last": grid(tapped[last]), "shallow": [grid(tapped[i]) for i in sorted(self._fkd_taps) if i != last]}
            if self._qkv_block:  # token student f-major -> t-major như teacher đã pool
                reorder = lambda t: t.reshape(t.shape[0], tokens.shape[1], tokens.shape[2], -1).permute(0, 2, 1, 3).flatten(1, 2)
                self._extra["qkv"] = tuple(reorder(t) for t in self.encoder.qkv)
        tokens, gate = self._gate(tokens)
        noise_mel = self.noise_head(tokens) if with_noise else None
        self._tokens = tokens                                               # token sau cổng, cho CRD theo patch
        return self.post_fc(tokens.permute(0, 3, 1, 2)), gate, noise_mel, None, hint

    def forward(self, waveform: Tensor, noise: Tensor | None = None, snr_est: Tensor | None = None):
        """Khi train và student.patch_crd: thêm `spatial` = token sau cổng (B, T' * F', C), chỉ số t * F' + f (relationkd.py)."""
        from relationkd import RelationKDOutput

        output = super().forward(waveform, noise, snr_est)
        tokens, self._tokens = self._tokens, None
        spatial = tokens.permute(0, 2, 1, 3).flatten(1, 2) if self.training and self._relation_patch else None
        extra, self._extra = self._extra, None
        return RelationKDOutput(**output._asdict(), spatial=spatial, extra=extra)

    PRETRAINED_PREFIXES = ("frontend.", "encoder.", "post_fc.norm.")  # nạp từ checkpoint AudioSet của CED

    def lr_scale(self, name: str) -> float:
        """Hệ số lr theo tên tham số: layer decay trên block CED, attention pool nhân attention_lr_scale, mọi module
        mới (cổng, NoiseHead, router, expert, head...) nhân head_lr_scale (recipe ce9: head 3e-4 = 10 x backbone 3e-5)."""
        if name.startswith("encoder.blocks."):
            index = int(name.split(".")[2])
            return float(self._student.get("layer_decay", 1.0)) ** (self._num_blocks - 1 - index)
        if name.startswith("post_fc.attn."):
            return float(self._student.get("attention_lr_scale", 1.0))
        if name.startswith(self.PRETRAINED_PREFIXES):
            return 1.0
        return self.head_lr_scale

    @property
    def head_lr_scale(self) -> float:
        return float(self._student.get("head_lr_scale", 1.0))

    def load_ced_pretrained(self) -> int:
        """Checkpoint AudioSet của CED (student.checkpoint_url): front_end/init_bn -> frontend, patch/pos/block/norm ->
        encoder, outputlayer.0 (LayerNorm) -> post_fc.norm; outputlayer.1 (527 lớp) bỏ."""
        url = self._student["checkpoint_url"]
        source = ced_state(url)
        state: dict[str, Tensor] = {}
        for key, value in source.items():
            top, _, rest = key.partition(".")
            if top in self.TOP_LEVEL:
                state[self.TOP_LEVEL[top] + ("." + rest if rest else "")] = value
            elif key.startswith("outputlayer.0."):
                state["post_fc.norm." + key.removeprefix("outputlayer.0.")] = value
        if self.dual_input:  # checkpoint một kênh -> hai kênh: kênh 1 (mixture) = 0, init_bn_mix = bản sao init_bn
            weight = state["encoder.patch_embed.proj.weight"]
            wide = torch.zeros(weight.shape[0], 2, *weight.shape[2:], dtype=weight.dtype)
            wide[:, :1] = weight
            state["encoder.patch_embed.proj.weight"] = wide
            if self.second_channel == "mixture":
                state.update({key.replace("frontend.init_bn.", "frontend.init_bn_mix.", 1): value
                              for key, value in list(state.items()) if key.startswith("frontend.init_bn.")})
        own = self.state_dict()
        bad = [key for key, value in state.items() if key not in own or own[key].shape != value.shape]
        missing = [key for key in own if key.startswith(("encoder.", "frontend.init_bn.", "post_fc.norm."))
                   and key not in state]
        if bad or missing:
            raise ValueError(f"{url}: CED weights do not fit: bad {bad[:3]}, missing {missing[:3]}")
        self.load_state_dict(state, strict=False)
        return len(state)

    @classmethod
    def from_yaml(cls, path: str | Path) -> CEDKDMoE:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(KDMoEConfig.from_dict(raw[KDMoEConfig.section]), raw.get("frontend"), raw["student"])
