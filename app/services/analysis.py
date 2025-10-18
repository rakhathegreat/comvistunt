"""Services responsible for anthropometric analysis."""

from __future__ import annotations

from dataclasses import dataclass

from config_manager import get_config
from model.comvistunt import get_haz, get_height, get_landmarks, get_weight

from app.core.storage import CAPTURE_IMAGE_PATH

from .exceptions import NotFoundError


@dataclass(slots=True)
class AnalysisOutcome:
    """Represents calculated anthropometric measurements."""

    height: float
    haz: float
    weight: float


def analyze_growth(gender: str, age: int) -> AnalysisOutcome:
    """Compute height, HAZ, and weight metrics from the stored capture image."""

    file_path = CAPTURE_IMAGE_PATH
    landmarks = get_landmarks(str(file_path))
    height = get_height(landmarks, get_config("CM_PER_PX"))
    haz, _ = get_haz(height, gender, age)
    weight = get_weight(height)

    if height is None or haz is None or weight is None:
        raise NotFoundError("Can't analyze.")

    return AnalysisOutcome(height=height, haz=haz, weight=weight)


__all__ = ["AnalysisOutcome", "analyze_growth"]
