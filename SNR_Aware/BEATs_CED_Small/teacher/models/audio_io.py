"""Manifest and audio loading.

Copied from `BEATs_Experts/noise_pipeline/mix_data.py` rather than imported, and
that is deliberate. Putting `BEATs_Experts` on `sys.path` drags its `config` and
`tasks` packages along, both of which collide with this project's own directories
of those names. The collision is silent until an import resolves to the wrong
package, which has already cost this project one debugging round. Two short
functions are cheaper than that dependency.
"""

from __future__ import annotations

import csv
from pathlib import Path

import soundfile
import torch


def load_float_audio(path: str | Path, sample_rate: int = 16_000,
                     num_samples: int | None = None) -> torch.Tensor:
    """Mono float tensor shaped [samples], padded or truncated to `num_samples`."""
    waveform, source_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    tensor = torch.from_numpy(waveform.copy()).mean(dim=1)
    if source_rate != sample_rate:
        import torchaudio

        tensor = torchaudio.functional.resample(tensor, source_rate, sample_rate)
    if num_samples is not None:
        tensor = tensor[:num_samples]
        if tensor.shape[0] < num_samples:
            tensor = torch.nn.functional.pad(tensor, (0, num_samples - tensor.shape[0]))
    return tensor


def load_mix_manifest(root: str | Path, manifest_file: str = "manifest.csv"):
    with (Path(root) / manifest_file).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))
