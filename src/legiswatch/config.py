"""Runtime configuration.

All tunable behaviour is centralised here and overridable by environment
variable, so deployments differ by configuration rather than by patch.

The routing thresholds are the values most likely to change after a deployment
measures real reviewer agreement. They are exposed rather than buried as module
constants precisely because they are expected to be tuned.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Immutable settings snapshot, resolved once at import."""

    # -- domain ---------------------------------------------------------
    institution_name: str = field(
        default_factory=lambda: os.getenv("LEGISWATCH_INSTITUTION", "Indiana University")
    )
    institution_class: str = field(
        default_factory=lambda: os.getenv(
            "LEGISWATCH_INSTITUTION_CLASS", "state educational institution"
        )
    )

    # -- inference ------------------------------------------------------
    provider: str = field(default_factory=lambda: os.getenv("LEGISWATCH_PROVIDER", "replay"))
    temperature: float = field(default_factory=lambda: _env_float("LEGISWATCH_TEMPERATURE", 0.0))
    max_retries: int = field(default_factory=lambda: int(os.getenv("LEGISWATCH_MAX_RETRIES", "2")))

    # -- guardrails -----------------------------------------------------
    # Minimum similarity for a quote to count as present in its source. Set high
    # on purpose: the tolerance absorbs whitespace and punctuation drift, not
    # paraphrase.
    fuzzy_threshold: float = field(
        default_factory=lambda: _env_float("LEGISWATCH_FUZZY_THRESHOLD", 92.0)
    )

    # -- routing --------------------------------------------------------
    # Deliberately conservative. A first deployment should over-fill the review
    # queue; raise the bar only after measuring where reviewers disagree with
    # the pipeline.
    auto_file_threshold: float = field(
        default_factory=lambda: _env_float("LEGISWATCH_AUTO_FILE_THRESHOLD", 0.85)
    )
    min_verifier_confidence: float = field(
        default_factory=lambda: _env_float("LEGISWATCH_MIN_VERIFIER_CONFIDENCE", 0.75)
    )

    # Confidence blend weights. The extractor's assessment of its own output is
    # the least reliable of the three signals and is weighted accordingly.
    weight_extractor: float = field(
        default_factory=lambda: _env_float("LEGISWATCH_WEIGHT_EXTRACTOR", 0.25)
    )
    weight_verifier: float = field(
        default_factory=lambda: _env_float("LEGISWATCH_WEIGHT_VERIFIER", 0.45)
    )
    weight_groundedness: float = field(
        default_factory=lambda: _env_float("LEGISWATCH_WEIGHT_GROUNDEDNESS", 0.30)
    )

    # -- review ---------------------------------------------------------
    require_review_comment: bool = field(
        default_factory=lambda: _env_bool("LEGISWATCH_REQUIRE_REVIEW_COMMENT", True)
    )

    def __post_init__(self) -> None:
        total = self.weight_extractor + self.weight_verifier + self.weight_groundedness
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Confidence weights must sum to 1.0, got {total:.4f}")
        for name in ("auto_file_threshold", "min_verifier_confidence"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if not 0.0 <= self.fuzzy_threshold <= 100.0:
            raise ValueError("fuzzy_threshold must be in [0, 100]")


settings = Settings()

# Obligation types that always require human confirmation regardless of model
# confidence. This is a consequence judgement, not a confidence judgement: a
# missed external report is a regulatory finding and a missed prohibition is
# legal exposure, so neither is delegated to a score.
ALWAYS_REVIEW_TYPES: frozenset[str] = frozenset({"report", "prohibition"})
