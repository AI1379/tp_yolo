from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .config import TriggerConfig


@dataclass(frozen=True)
class FrameStats:
    total_boxes: int
    low_conf_ratio: float
    tiny_box_count: int
    mean_conf: float


@dataclass(frozen=True)
class RouteDecision:
    trigger: bool
    selected_experts: tuple[str, ...]
    reason: str


def compute_frame_stats(detections: Iterable[dict[str, Any]], tiny_area_ratio: float, low_conf_threshold: float) -> FrameStats:
    confs: list[float] = []
    tiny_count: int = 0
    total: int = 0
    low_conf: int = 0

    for det in detections:
        total += 1
        conf: float = float(det["conf"])
        area: float = float(det["area_ratio"])
        confs.append(conf)
        if conf < low_conf_threshold:
            low_conf += 1
        if area < tiny_area_ratio:
            tiny_count += 1

    mean_conf: float = float(np.mean(confs)) if confs else 0.0
    low_ratio: float = float(low_conf / total) if total else 0.0
    return FrameStats(total_boxes=total, low_conf_ratio=low_ratio, tiny_box_count=tiny_count, mean_conf=mean_conf)


class RuleBasedRouter:
    """Simple and stable routing policy for initial deployment."""

    def __init__(self, expert_names: Iterable[str], config: TriggerConfig) -> None:
        self.expert_names: tuple[str, ...] = tuple(expert_names)
        self.config: TriggerConfig = config

    def decide(self, detections: Iterable[dict[str, Any]]) -> RouteDecision:
        stats: FrameStats = compute_frame_stats(
            detections,
            tiny_area_ratio=self.config.tiny_area_ratio,
            low_conf_threshold=self.config.low_conf_threshold,
        )

        cond_low_conf: bool = stats.low_conf_ratio > self.config.low_conf_ratio_trigger
        cond_tiny: bool = stats.tiny_box_count >= self.config.tiny_count_trigger
        cond_sparse: bool = stats.total_boxes <= self.config.min_boxes_trigger
        cond_dense: bool = stats.total_boxes >= self.config.max_boxes_trigger

        trigger: bool = cond_low_conf or cond_tiny or cond_sparse or cond_dense
        if not trigger:
            return RouteDecision(False, tuple(), "base_model_sufficient")

        # Initial policy: when uncertain, run top-k experts by fixed priority.
        k: int = min(self.config.max_experts_per_frame, len(self.expert_names))
        selected: tuple[str, ...] = self.expert_names[:k]

        reasons: list[str] = []
        if cond_low_conf:
            reasons.append("low_confidence_ratio")
        if cond_tiny:
            reasons.append("many_tiny_objects")
        if cond_sparse:
            reasons.append("too_few_boxes")
        if cond_dense:
            reasons.append("too_many_boxes")

        return RouteDecision(True, selected, "+".join(reasons))
