"""Resolution of packaged data directories.

Data files live outside the package directory in the source tree but are shipped
alongside it when installed. This resolves both layouts so the same code works
from a checkout and from an installed wheel.
"""

from __future__ import annotations

import os
from pathlib import Path

_PKG = Path(__file__).resolve().parent


def _resolve_data_dir() -> Path:
    override = os.getenv("LEGISWATCH_DATA_DIR")
    if override:
        return Path(override)
    for candidate in (_PKG / "data", _PKG.parents[1] / "data"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate the data directory. Set LEGISWATCH_DATA_DIR to override."
    )


DATA_DIR = _resolve_data_dir()
CORPUS_PATH = DATA_DIR / "corpus" / "corpus.json"
GOLD_PATH = DATA_DIR / "gold" / "gold_set.json"
REPLAY_DIR = DATA_DIR / "replay"
