"""Corpus loading."""

from __future__ import annotations

import json
from pathlib import Path

from .paths import CORPUS_PATH


def load_corpus(path: Path | None = None, include_negative_controls: bool = True) -> list[dict]:
    data = json.loads((path or CORPUS_PATH).read_text())
    docs = data["documents"]
    if not include_negative_controls:
        docs = [d for d in docs if not d.get("is_negative_control")]
    return docs


def corpus_metadata(path: Path | None = None) -> dict:
    data = json.loads((path or CORPUS_PATH).read_text())
    return {k: v for k, v in data.items() if k != "documents"}
