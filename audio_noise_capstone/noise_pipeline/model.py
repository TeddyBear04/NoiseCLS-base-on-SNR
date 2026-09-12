import torch
from torch import nn


class BandEncoder(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(),
        )
        self.rnn = nn.LSTM(hidden, hidden, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        # [B, F, T] -> a time sequence after pooling frequency.
        x = self.cnn(x.unsqueeze(1)).mean(dim=2).transpose(1, 2)
        sequence, _ = self.rnn(x)
        weights = torch.softmax(self.attention(sequence), dim=1)
        embedding = (sequence * weights).sum(dim=1)
        return sequence, embedding


class NoiseNet(nn.Module):
    def __init__(self, n_mels: int = 64, num_classes: int = 21, hidden: int = 32):
        super().__init__()
        # Approximate 0-2, 2-4 and 4-8 kHz regions for a 64-bin mel spectrum.
        self.band_slices = (slice(0, 36), slice(36, 50), slice(50, 64))
        self.separator_encoders = nn.ModuleList(BandEncoder(hidden) for _ in range(3))
        self.separator_gate = nn.Linear(hidden * 2, 1)
        self.mask_decoder = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden * 2, 3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden * 2, n_mels, 1), nn.Sigmoid(),
        )
        self.classifier_encoders = nn.ModuleList(BandEncoder(hidden) for _ in range(3))
        self.classifier_gate = nn.Linear(hidden * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden * 2, num_classes),
        )

    def _encode_bands(self, features, encoders):
        return [encoder(features[:, band, :]) for encoder, band in zip(encoders, self.band_slices)]

    def forward(self, mixture_logmel):
        encoded = self._encode_bands(mixture_logmel, self.separator_encoders)
        sequences = torch.stack([item[0] for item in encoded], dim=2)  # [B,T,3,H]
        embeddings = torch.stack([item[1] for item in encoded], dim=1)
        gates = torch.softmax(self.separator_gate(embeddings), dim=1)
        fused_sequence = (sequences * gates.unsqueeze(1)).sum(dim=2)
        mask = self.mask_decoder(fused_sequence.transpose(1, 2))
        # A ratio mask is meaningful in the non-negative power domain, not on
        # signed log values. Convert back to Mel power, apply it, then return to
        # log space for both the separation loss and the classifier.
        noise_power_hat = mask * mixture_logmel.exp()
        noise_hat = torch.log(noise_power_hat.clamp_min(1e-6))

        classified = self._encode_bands(noise_hat, self.classifier_encoders)
        class_embeddings = torch.stack([item[1] for item in classified], dim=1)
        class_gates = torch.softmax(self.classifier_gate(class_embeddings), dim=1)
        fused_embedding = (class_embeddings * class_gates).sum(dim=1)
        return self.classifier(fused_embedding), noise_hat, mask
