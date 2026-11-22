"""PANNs CNN10 (Kong et al., 2020) làm encoder cho KD-MoE.

Kiến trúc và front-end theo bản phát hành qiuqiangkong/audioset_tagging_cnn (pytorch/models.py, Cnn10):
32 kHz, cửa sổ hann 1024, bước 320, 64 mel 50-14000 Hz, log10 với amin 1e-10 (ref 1, không top_db),
bn0 theo trục mel, 4 ConvBlock (3x3 x2, BN, ReLU, avg-pool 2x2, dropout 0.2), gộp mean tần số rồi
max + mean thời gian, dropout 0.5, fc1 512 + ReLU, dropout 0.5. Trọng số pretrain AudioSet:
zenodo.org/record/3987831 Cnn10_mAP=0.380.pth. melW lấy thẳng từ checkpoint nên log-mel khớp lúc pretrain.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor, nn

PANNS_SAMPLE_RATE = 32000
PANNS_WINDOW = 1024
PANNS_HOP = 320
PANNS_MELS = 64
PANNS_FMIN = 50.0
PANNS_FMAX = 14000.0
PANNS_AMIN = 1e-10


class PannsLogMel(nn.Module):
    """Waveform 16 kHz (B, S) -> log-mel PANNs (B, 1, 64, num_frames), cắt giữa cố định.

    Trả thêm đích L_front: log-mel của nhiễu sạch, chuẩn hoá bằng mean/std của log-mel đầu vào
    như LogMel của KD-MoE ViT.
    """

    def __init__(self, sample_rate: int = 16000, num_frames: int = 400, norm_eps: float = 1e-5) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.num_frames = num_frames
        self.norm_eps = norm_eps
        self.register_buffer("window", torch.hann_window(PANNS_WINDOW, periodic=True), persistent=False)
        mel = torchaudio.functional.melscale_fbanks(
            PANNS_WINDOW // 2 + 1, PANNS_FMIN, PANNS_FMAX, PANNS_MELS, PANNS_SAMPLE_RATE, norm="slaney",
            mel_scale="slaney",
        )
        self.register_buffer("melW", mel)  # (513, 64); ghi đè bằng logmel_extractor.melW của checkpoint

    def logmel(self, waveform: Tensor) -> Tensor:
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            audio = torchaudio.functional.resample(waveform.float(), self.sample_rate, PANNS_SAMPLE_RATE)
            spec = torch.stft(audio, PANNS_WINDOW, PANNS_HOP, PANNS_WINDOW, window=self.window, center=True,
                              pad_mode="reflect", return_complex=True)
            power = spec.real.square() + spec.imag.square()  # (B, 513, T)
            mel = torch.matmul(power.transpose(1, 2), self.melW).transpose(1, 2)  # (B, 64, T)
            return 10.0 * torch.log10(mel.clamp_min(PANNS_AMIN))

    def _crop(self, x: Tensor) -> Tensor:
        start = max(x.shape[-1] - self.num_frames, 0) // 2
        return x[..., start : start + self.num_frames]

    def forward(self, waveform: Tensor, noise: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        x = self._crop(self.logmel(waveform))
        if noise is None:
            return x.unsqueeze(1), None
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)
        target = (self._crop(self.logmel(noise)) - mean) / (std + self.norm_eps)
        return x.unsqueeze(1), target.unsqueeze(1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=2)


BLOCK_CHANNELS = (64, 128, 256, 512)


class HintRegressor(nn.Module):
    """FitNets (Romero et al., ICLR 2015): regressor conv trên guided layer của student -> kích thước hint teacher.

    Feature map block (B, C, T', mel') đưa về lưới patch teacher (như AT) rồi conv 1x1 tuyến tính C -> hint_dim:
    hint BEATs là token sau LayerNorm, không qua phi tuyến, nên regressor cũng không có phi tuyến
    (paper: u_h và r phải cùng phi tuyến). 1x1 giao hoán với nội suy song tuyến, thứ tự hai bước không đổi kết quả.
    -> (B, T, F, hint_dim), cùng layout với hint teacher (T', 8, 768).
    """

    def __init__(self, channels: int, hint_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(channels, hint_dim, kernel_size=1)

    def forward(self, block: Tensor, grid: tuple[int, int], time_fraction: float, freq_fraction: float) -> Tensor:
        return self.proj(to_grid(block, grid, time_fraction, freq_fraction)).permute(0, 2, 3, 1)


class PannsCnn10(nn.Module):
    """log-mel (B, 1, 64, T) -> feature map (B, 512, 64/16, T/16).

    Bên trong chạy đúng layout của bản gốc (B, 1, T, mel): kernel 3x3 pretrain có trục thời gian/tần
    số riêng, đảo layout là áp kernel sai trục.
    """

    out_channels = 512

    def __init__(self) -> None:
        super().__init__()
        self.bn0 = nn.BatchNorm2d(PANNS_MELS)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)

    def forward(self, x: Tensor, return_blocks: bool = False) -> Tensor | tuple[Tensor, list[Tensor]]:
        x = x.transpose(2, 3)  # (B, 1, T, mel)
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)  # BN theo trục mel, y như bản gốc
        blocks = []
        for block in (self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4):
            x = block(x)
            blocks.append(x)  # (B, C, T', mel'), trước dropout
            x = F.dropout(x, p=0.2, training=self.training)
        x = x.transpose(2, 3)  # (B, 512, mel', T')
        return (x, blocks) if return_blocks else x


def to_grid(x: Tensor, grid: tuple[int, int], time_fraction: float, freq_fraction: float) -> Tensor:
    """(B, C, T', mel') -> (B, C, T, F): cắt [0, time_fraction] x [0, freq_fraction] rồi nội suy về lưới."""
    rows, cols = grid
    t = (torch.arange(rows, device=x.device) + 0.5) / rows * time_fraction * 2 - 1
    f = (torch.arange(cols, device=x.device) + 0.5) / cols * freq_fraction * 2 - 1
    mesh_t, mesh_f = torch.meshgrid(t, f, indexing="ij")
    sample = torch.stack((mesh_f, mesh_t), dim=-1)  # grid_sample: (x = mel, y = thời gian)
    # border: điểm lấy mẫu ở mép có thể rơi ngoài tâm ô đầu tiên; đệm 0 sẽ kéo bản đồ xuống ở mép
    return F.grid_sample(x, sample.to(x.dtype).expand(x.shape[0], -1, -1, -1), align_corners=False,
                         padding_mode="border")


def attention_maps(blocks: list[Tensor], grid: tuple[int, int], time_fraction: float,
                   freq_fraction: float) -> Tensor:
    """Bản đồ Attention Transfer (Zagoruyko & Komodakis, ICLR 2017; F_sum^2 lấy trung bình kênh) của các block
    CNN10 [(B, C, T', mel')] trên lưới (T, F) -> (B, L, T, F)."""
    maps = [to_grid(block.float().pow(2).mean(dim=1, keepdim=True), grid, time_fraction, freq_fraction).squeeze(1)
            for block in blocks]
    return torch.stack(maps, dim=1)


class PannsPool(nn.Module):
    """Feature map (B, C, F', T') -> embedding (B, 512): mean tần số, max + mean thời gian, fc1."""

    def __init__(self, channels: int = 512) -> None:
        super().__init__()
        self.fc1 = nn.Linear(channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        x = x.mean(dim=2)
        x = x.amax(dim=2) + x.mean(dim=2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        return F.dropout(x, p=0.5, training=self.training)


def panns_state(path: str | Path) -> dict[str, Tensor]:
    """Checkpoint Cnn10 gốc -> khoá của KD-MoE (frontend.melW, encoder.*, post_fc.fc1.*)."""
    source = torch.load(path, map_location="cpu", weights_only=False)
    source = source.get("model", source)
    state: dict[str, Tensor] = {"frontend.melW": source["logmel_extractor.melW"]}
    for key, value in source.items():
        if key.startswith(("bn0.", "conv_block")):
            state[f"encoder.{key}"] = value
        elif key.startswith("fc1."):
            state[f"post_fc.{key}"] = value
    return state
