from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes

from .config import CascadeConfig, ExpertConfig
from .router import RouteDecision, RuleBasedRouter


@dataclass(frozen=True)
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    # Normalized coordinates [0.0, 1.0] relative to image size.
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area_ratio(self) -> float:
        return max(0.0, (self.x2 - self.x1) * (self.y2 - self.y1))
    
    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class InferenceOutput:
    detections: tuple[Detection, ...]
    route: RouteDecision


def iou_xyxy(a: Detection, b: Detection) -> float:
    inter_x1: float = max(a.x1, b.x1)
    inter_y1: float = max(a.y1, b.y1)
    inter_x2: float = min(a.x2, b.x2)
    inter_y2: float = min(a.y2, b.y2)

    inter_w: float = max(0.0, inter_x2 - inter_x1)
    inter_h: float = max(0.0, inter_y2 - inter_y1)
    inter: float = inter_w * inter_h

    if inter <= 0.0:
        return 0.0

    area_a: float = max(0.0, (a.x2 - a.x1) * (a.y2 - a.y1))
    area_b: float = max(0.0, (b.x2 - b.x1) * (b.y2 - b.y1))
    union: float = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def nms_classwise(
    dets: Iterable[Detection], iou_thresh: float
) -> tuple[Detection, ...]:
    buckets: dict[int, list[Detection]] = {}
    for d in dets:
        buckets.setdefault(d.cls_id, []).append(d)

    keep: list[Detection] = []
    for cls_id in sorted(buckets):
        candidates: list[Detection] = sorted(
            buckets[cls_id], key=lambda x: x.conf, reverse=True
        )
        while candidates:
            best: Detection = candidates.pop(0)
            keep.append(best)
            survivors: list[Detection] = []
            for c in candidates:
                if iou_xyxy(best, c) < iou_thresh:
                    survivors.append(c)
            candidates = survivors

    return tuple(keep)


class CascadeMoEPipeline:
    """Initial cascade implementation based on Ultralytics YOLO models."""

    def __init__(self, config: CascadeConfig, experts: Iterable[ExpertConfig]) -> None:
        self.config: CascadeConfig = config
        self.base_model: YOLO = YOLO(config.base_model_path)
        self.expert_cfgs: dict[str, ExpertConfig] = {e.name: e for e in experts}
        self.expert_models: dict[str, YOLO] = {
            e.name: YOLO(e.model_path) for e in experts
        }
        self.router: RuleBasedRouter = RuleBasedRouter(
            self.expert_cfgs.keys(), config.trigger
        )

    def _predict(
        self, model: YOLO, image: Any, conf: float, iou: float
    ) -> tuple[Detection, ...]:
        results: list[Results] = model.predict(
            source=image, conf=conf, iou=iou, verbose=False
        )
        if not results:
            return tuple()

        result: Results = results[0]
        boxes: Boxes | None = result.boxes
        if boxes is None or boxes.xyxy is None:
            return tuple()

        h, w = result.orig_shape
        w_f: float = float(w)
        h_f: float = float(h)

        out: list[Detection] = []
        for i in range(len(boxes)):
            cls_id: int = int(boxes.cls[i].item())
            conf_i: float = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()

            # Store normalized coordinates to keep model-agnostic logic later.
            x1n: float = max(0.0, min(1.0, x1 / w_f))
            y1n: float = max(0.0, min(1.0, y1 / h_f))
            x2n: float = max(0.0, min(1.0, x2 / w_f))
            y2n: float = max(0.0, min(1.0, y2 / h_f))

            cls_name: str = (
                self.config.class_names[cls_id]
                if cls_id < len(self.config.class_names)
                else str(cls_id)
            )
            out.append(
                Detection(
                    cls_id=cls_id,
                    cls_name=cls_name,
                    conf=conf_i,
                    x1=x1n,
                    y1=y1n,
                    x2=x2n,
                    y2=y2n,
                )
            )

        return tuple(out)

    def _to_router_features(self, dets: Iterable[Detection]) -> list[dict[str, Any]]:
        return [
            {
                "cls_id": d.cls_id,
                "conf": d.conf,
                "area_ratio": d.area_ratio,
            }
            for d in dets
        ]

    def infer(self, image: Any) -> InferenceOutput:
        base_dets: tuple[Detection, ...] = self._predict(
            self.base_model, image, conf=self.config.base_conf, iou=self.config.base_iou
        )
        decision: RouteDecision = self.router.decide(
            self._to_router_features(base_dets)
        )

        all_dets: list[Detection] = list(base_dets)
        if decision.trigger:
            for expert_name in decision.selected_experts:
                cfg: ExpertConfig = self.expert_cfgs[expert_name]
                model: YOLO = self.expert_models[expert_name]
                expert_dets: tuple[Detection, ...] = self._predict(
                    model, image, conf=cfg.conf, iou=cfg.iou
                )

                if cfg.focus_classes:
                    focus: set[int] = set(cfg.focus_classes)
                    expert_dets = tuple(d for d in expert_dets if d.cls_id in focus)

                all_dets.extend(expert_dets)

        merged: tuple[Detection, ...] = nms_classwise(
            all_dets, iou_thresh=self.config.merge_iou
        )
        return InferenceOutput(detections=merged, route=decision)

    def infer_as_dict(self, image: Any) -> dict[str, Any]:
        out: InferenceOutput = self.infer(image)
        return {
            "route": asdict(out.route),
            "detections": [asdict(d) for d in out.detections],
        }
