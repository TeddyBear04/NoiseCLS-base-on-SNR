"""Safe loader for the vendored BEATs implementation."""

from __future__ import annotations

import sys
from pathlib import Path


def load_beats_classes():
    """Return ``BEATs`` and ``BEATsConfig`` from the local vendor source."""
    source = Path(__file__).resolve().parent / "beats"
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from BEATs import BEATs, BEATsConfig

    return BEATs, BEATsConfig
