"""Compatibility wrapper around the authoritative manifest splits.

The original fish-dataset splitter was intentionally removed: recomputing a
random split would break the noise_ytid leakage protection in this dataset.
"""

from pathlib import Path

from config import SplitterConfig

from .dataloader_melspectrogram import read_label_catalog, read_manifest


class PredefinedManifestSplitter:
    def __init__(self, config: SplitterConfig) -> None:
        self.config = config
        self.root = Path(config.dataset_path)
        self.labels = read_label_catalog(self.root, config.selected_labels_file)

    def split_data(self):
        directories = (
            self.config.train_directory,
            self.config.test_directory,
            self.config.validation_directory,
        )
        return tuple(
            read_manifest(
                self.root,
                directory,
                self.config.signal_type,
                self.labels,
                clean_directory=self.config.clean_directory,
                noise_directory=self.config.noise_directory,
            )
            for directory in directories
        )


# Old notebooks can import the old symbol without silently using the old policy.
FishDataSplitter = PredefinedManifestSplitter
