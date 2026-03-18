from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from moe_yolo.cascade import CascadeMoEPipeline
from moe_yolo.config import CascadeConfig, ExpertConfig, TriggerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run preliminary cascade MoEYOLO inference.")
    parser.add_argument("--image", type=Path, required=True, help="Input image path.")
    parser.add_argument("--base-model", type=str, default="yolo11m.pt")
    parser.add_argument("--tiny-expert", type=str, default="yolo11m.pt")
    parser.add_argument("--ground-expert", type=str, default="yolo11m.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {args.image}")

    class_names = (
        "blind_road",
        "crosswalk",
        "cone",
        "tubular_marker",
        "drum",
        "barricade",
        "barrier",
        "fence",
    )

    config = CascadeConfig(
        base_model_path=args.base_model,
        class_names=class_names,
        trigger=TriggerConfig(
            low_conf_threshold=0.35,
            low_conf_ratio_trigger=0.50,
            tiny_area_ratio=0.01,
            tiny_count_trigger=4,
            max_experts_per_frame=2,
        ),
    )

    experts = [
        ExpertConfig(
            name="ground_expert",
            model_path=args.ground_expert,
            focus_classes=(0, 1),
            conf=0.20,
            iou=0.60,
        ),
        ExpertConfig(
            name="tiny_obstacle_expert",
            model_path=args.tiny_expert,
            focus_classes=(2, 3, 4, 5, 6, 7),
            conf=0.20,
            iou=0.60,
        ),
    ]

    pipeline = CascadeMoEPipeline(config=config, experts=experts)
    out = pipeline.infer_as_dict(image)

    print("route:", out["route"])
    print("detections:", len(out["detections"]))
    for det in out["detections"][:20]:
        print(det)


if __name__ == "__main__":
    main()
