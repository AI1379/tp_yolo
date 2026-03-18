from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TriggerConfig:
    """Thresholds for deciding whether expert models should run."""

    low_conf_threshold: float = 0.35
    low_conf_ratio_trigger: float = 0.50
    tiny_area_ratio: float = 0.01
    tiny_count_trigger: int = 4
    min_boxes_trigger: int = 1
    max_boxes_trigger: int = 200
    max_experts_per_frame: int = 2


@dataclass(frozen=True)
class ExpertConfig:
    """Configuration for one expert model."""

    name: str
    model_path: str
    focus_classes: tuple[int, ...] = field(default_factory=tuple)
    conf: float = 0.20
    iou: float = 0.60


@dataclass(frozen=True)
class CascadeConfig:
    """Top-level cascade inference configuration."""

    base_model_path: str
    class_names: tuple[str, ...]
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    base_conf: float = 0.25
    base_iou: float = 0.60
    merge_iou: float = 0.55
