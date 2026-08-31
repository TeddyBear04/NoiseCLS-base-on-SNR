import json
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class AudioFeaturesConfig(BaseModel):
    """Waveform and log-mel settings for the 16 kHz noise dataset."""

    sample_rate: int = Field(default=16_000, gt=0)
    clip_seconds: float = Field(default=4.0, gt=0.0)
    inference_hop_seconds: float = Field(default=2.0, gt=0.0)
    window_size: int = Field(default=512, gt=0)
    hop_size: int = Field(default=160, gt=0)
    mel_bins: int = Field(default=128, gt=0)
    fmin: int = Field(default=20, ge=0)
    fmax: int = Field(default=8_000, gt=0)
    time_drop_width: int = Field(default=64, ge=0)
    time_stripes_num: int = Field(default=2, ge=0)
    freq_drop_width: int = Field(default=8, ge=0)
    freq_stripes_num: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def validate_audio_settings(self) -> "AudioFeaturesConfig":
        if self.fmax > self.sample_rate // 2:
            raise ValueError("audio_features.fmax must not exceed the Nyquist frequency")
        if self.inference_hop_seconds > self.clip_seconds:
            raise ValueError("inference_hop_seconds must be <= clip_seconds")
        return self


class SnrBandConfig(BaseModel):
    """One inclusive SNR interval used to break evaluation down by noise level."""

    name: str
    min_db: float
    max_db: float

    @model_validator(mode="after")
    def validate_band(self) -> "SnrBandConfig":
        if self.min_db > self.max_db:
            raise ValueError(f"snr band {self.name!r}: min_db must not exceed max_db")
        return self


# The 36-label dataset draws target SNR from {-5, 0, 5, 10, 15, 20} dB, so these
# six one-value bands cover every clip exactly once and keep each level separate.
DEFAULT_SNR_BANDS = [
    SnrBandConfig(name="-5dB", min_db=-5.0, max_db=-5.0),
    SnrBandConfig(name="0dB", min_db=0.0, max_db=0.0),
    SnrBandConfig(name="5dB", min_db=5.0, max_db=5.0),
    SnrBandConfig(name="10dB", min_db=10.0, max_db=10.0),
    SnrBandConfig(name="15dB", min_db=15.0, max_db=15.0),
    SnrBandConfig(name="20dB", min_db=20.0, max_db=20.0),
]


class ModelConfig(BaseModel):
    backbone: str = "Cnn14MobileV2LocalSNR"
    pretrained: bool = False
    classes_num: int = Field(default=36, gt=0)


class LocalSNRConfig(BaseModel):
    """Time-local SNR targets, regression head, and masked-loss settings."""

    enabled: bool = True
    segment_seconds: float = Field(default=0.5, gt=0.0)
    segment_hop_seconds: float = Field(default=0.25, gt=0.0)
    speech_activity_db_below_peak: float = Field(default=-40.0, lt=0.0)
    min_target_db: float = -20.0
    max_target_db: float = 30.0
    target_offset_db: float = 5.0
    target_scale_db: float = Field(default=15.0, gt=0.0)
    hidden_dim: int = Field(default=128, gt=0)
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    loss: Literal["mse", "huber"] = "huber"
    loss_weight: float = Field(default=0.1, ge=0.0)
    inference_smoothing_points: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def validate_local_snr(self) -> "LocalSNRConfig":
        if self.segment_hop_seconds > self.segment_seconds:
            raise ValueError("local_snr.segment_hop_seconds must be <= segment_seconds")
        if self.min_target_db >= self.max_target_db:
            raise ValueError("local_snr.min_target_db must be smaller than max_target_db")
        if self.inference_smoothing_points % 2 == 0:
            raise ValueError("local_snr.inference_smoothing_points must be odd")
        return self


class SplitterConfig(BaseModel):
    """Dataset settings; the supplied train/validation/test manifests are authoritative."""

    dataset_path: str = "/marimo/36_labels"
    signal_type: Literal["mixture", "clean", "noise", "oracle_noise"] = "mixture"
    train_directory: str = "train"
    validation_directory: str = "validation"
    test_directory: str = "test"
    # Sub-directories holding the two stems used by dynamic-SNR augmentation.
    clean_directory: str = "clean"
    noise_directory: str = "noise"
    # labels.txt (one name per line) or the older selected_labels.csv catalog.
    selected_labels_file: str = "labels.txt"
    use_predefined_splits: bool = True
    include_video: bool = False
    save_results: bool = False

    # Kept for compatibility with copied utilities. The manifest splits are not recomputed.
    seed: int = Field(default=2026, ge=0)
    test_sample_per_class: int = Field(default=1, gt=0)
    split_strategy: str = "predefined_manifest"
    evaluation_mode: str = "holdout"
    num_folds: int = Field(default=5, gt=1)
    fold_index: Optional[int] = None
    cv_val_ratio: float = Field(default=0.2, gt=0.0, lt=1.0)

    # Optional controlled time-varying SNR augmentation on train samples.
    dynamic_snr_enabled: bool = True
    dynamic_snr_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    dynamic_snr_min_db: float = -5.0
    dynamic_snr_max_db: float = 20.0
    dynamic_snr_control_seconds: float = Field(default=0.5, gt=0.0)

    # Train items are remixed from this record's noise stem and a clean utterance drawn
    # from anywhere in the split, breaking the fixed clean/noise pairing baked into the
    # manifest. Every train item is remixed when this is on, so dynamic_snr_probability
    # only applies when it is off. Labels always follow the noise stem.
    cross_pairing_enabled: bool = True
    # Roll the noise stem against the clean stem so onset positions stop lining up.
    noise_time_shift_enabled: bool = True

    @model_validator(mode="after")
    def validate_dataset_settings(self) -> "SplitterConfig":
        if not self.use_predefined_splits:
            raise ValueError("the dataset must use its predefined manifest splits")
        if self.dynamic_snr_min_db >= self.dynamic_snr_max_db:
            raise ValueError("dynamic_snr_min_db must be smaller than dynamic_snr_max_db")
        return self


class TrainConfig(BaseModel):
    epochs: int = Field(default=100, gt=0)
    batch_size: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    monitor: Literal[
        "macro_f1", "mAP", "hamming_accuracy", "subset_accuracy", "loss"
    ] = "macro_f1"
    early_stopping: bool = True
    patience: int = Field(default=15, gt=0)
    delta: float = Field(default=0.0, ge=0.0)
    cache_audio: bool = False
    num_workers: int = Field(default=-1, ge=-1)
    pin_memory: bool = True
    threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    use_pos_weight: bool = True
    max_pos_weight: float = Field(default=20.0, ge=1.0)
    random_seed: int = Field(default=2026, ge=0)
    # Runs on the 36-label dataset write here. The 21-label results in
    # "checkpoint" are kept as-is and are never overwritten.
    ckpt_dir: str = "checkpoint_36_labels_local_snr"
    profile_model: bool = False
    snr_bands: List[SnrBandConfig] = Field(default_factory=lambda: list(DEFAULT_SNR_BANDS))
    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset_splitter: SplitterConfig = Field(default_factory=SplitterConfig)
    audio_features: AudioFeaturesConfig = Field(default_factory=AudioFeaturesConfig)
    local_snr: LocalSNRConfig = Field(default_factory=LocalSNRConfig)

    @model_validator(mode="after")
    def validate_multitask_settings(self) -> "TrainConfig":
        if self.local_snr.enabled and self.dataset_splitter.signal_type != "mixture":
            raise ValueError("local SNR training requires dataset_splitter.signal_type='mixture'")
        if self.local_snr.segment_seconds > self.audio_features.clip_seconds:
            raise ValueError("local_snr.segment_seconds must be <= audio_features.clip_seconds")
        return self

    @classmethod
    def from_json(cls, path: str = "config/train_config.json") -> "TrainConfig":
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            config = cls(**json.load(handle))

        # Relative dataset/checkpoint paths are resolved from the project directory,
        # not from the notebook's current working directory.
        project_root = config_path.parent.parent
        dataset_path = Path(config.dataset_splitter.dataset_path).expanduser()
        if not dataset_path.is_absolute():
            config.dataset_splitter.dataset_path = str((project_root / dataset_path).resolve())
        checkpoint_path = Path(config.ckpt_dir).expanduser()
        if not checkpoint_path.is_absolute():
            config.ckpt_dir = str((project_root / checkpoint_path).resolve())
        return config
