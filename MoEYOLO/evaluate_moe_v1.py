from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

from moe_yolo.cascade import CascadeMoEPipeline
from moe_yolo.config import CascadeConfig, ExpertConfig, TriggerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automatic evaluation for MoEYOLO v1 checkpoints.")

    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--merged-data-yaml", type=Path, default=root / "data" / "merged_yolo_detect" / "data.yaml")

    parser.add_argument("--base-ckpt", type=Path, required=True)
    parser.add_argument("--ground-expert-ckpt", type=Path)
    parser.add_argument("--tiny-expert-ckpt", type=Path)
    parser.add_argument("--ground-expert-data-yaml", type=Path, help="Dataset yaml for ground expert evaluation.")
    parser.add_argument("--tiny-expert-data-yaml", type=Path, help="Dataset yaml for tiny expert evaluation.")

    parser.add_argument("--device", type=str, default="1")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--split", type=str, default="val")

    parser.add_argument("--sample-images", type=int, default=300, help="Number of images for cascade latency/trigger stats.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "reports" / "eval_moe_v1.json",
        help="Path to output report JSON.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing yaml: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Invalid yaml content: {path}")
    return data


def parse_names(data_yaml: dict) -> list[str]:
    names = data_yaml.get("names")
    if isinstance(names, dict):
        out = []
        for i in sorted(int(k) for k in names.keys()):
            out.append(str(names[i]))
        return out
    if isinstance(names, list):
        return [str(n) for n in names]
    raise ValueError("Missing names in merged data yaml")


def default_expert_data_yaml(expert_name: str) -> Path:
    return Path(__file__).resolve().parent / "artifacts" / "expert_subsets" / expert_name / "data.yaml"


def resolve_expert_data_yaml(explicit_path: Path | None, expert_name: str) -> Path:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Expert data yaml not found: {explicit_path}")
        return explicit_path

    inferred = default_expert_data_yaml(expert_name)
    if not inferred.exists():
        raise FileNotFoundError(
            f"Missing data yaml for {expert_name}: {inferred}. "
            f"Provide --{expert_name.replace('_', '-')}-data-yaml explicitly or build expert subsets first."
        )
    return inferred


def to_float_metric(value) -> float:
    """Convert metric values from scalar/tensor/array/list to a Python float."""
    if isinstance(value, (int, float)):
        return float(value)

    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            pass

    if hasattr(value, "mean"):
        try:
            return float(value.mean())
        except (TypeError, ValueError):
            pass

    try:
        seq = list(value)
    except TypeError as exc:
        raise TypeError(f"Unsupported metric type: {type(value)}") from exc

    if not seq:
        return 0.0

    return float(sum(float(v) for v in seq) / len(seq))


def metric_dict(results, names: list[str]) -> dict:
    box = results.box

    # Ultralytics metrics can vary by version: prefer mean metrics when available.
    precision = getattr(box, "mp", box.p)
    recall = getattr(box, "mr", box.r)

    report: dict[str, object] = {
        "precision": to_float_metric(precision),
        "recall": to_float_metric(recall),
        "map50": to_float_metric(box.map50),
        "map50_95": to_float_metric(box.map),
    }

    class_maps = {}
    if hasattr(box, "maps") and box.maps is not None:
        maps = list(box.maps)
        for i, m in enumerate(maps):
            cls_name = names[i] if i < len(names) else str(i)
            class_maps[cls_name] = float(m)
    report["per_class_map50_95"] = class_maps
    return report


def evaluate_detector(
    model_path: Path,
    data_yaml: Path,
    split: str,
    device: str,
    imgsz: int,
    batch: int,
    names: list[str],
) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    model = YOLO(str(model_path))
    results = model.val(
        data=str(data_yaml),
        split=split,
        device=device,
        imgsz=imgsz,
        batch=batch,
        verbose=False,
    )
    return metric_dict(results, names)


def resolve_split_images(merged_yaml: dict, split: str) -> list[Path]:
    data_root = Path(str(merged_yaml["path"]))
    split_rel = str(merged_yaml.get(split, f"images/{split}"))
    image_dir = data_root / split_rel
    if not image_dir.exists():
        raise FileNotFoundError(f"Split image directory not found: {image_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in sorted(image_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    if not images:
        raise ValueError(f"No images found in split: {image_dir}")
    return images


def evaluate_cascade_runtime(
    merged_yaml: dict,
    class_names: list[str],
    split: str,
    sample_images: int,
    seed: int,
    base_ckpt: Path,
    ground_expert_ckpt: Path,
    tiny_expert_ckpt: Path,
) -> dict:
    images = resolve_split_images(merged_yaml, split)
    if sample_images > 0 and sample_images < len(images):
        rng = random.Random(seed)
        images = rng.sample(images, sample_images)

    config = CascadeConfig(
        base_model_path=str(base_ckpt),
        class_names=tuple(class_names),
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
            model_path=str(ground_expert_ckpt),
            focus_classes=(0, 1),
            conf=0.20,
            iou=0.60,
        ),
        ExpertConfig(
            name="tiny_obstacle_expert",
            model_path=str(tiny_expert_ckpt),
            focus_classes=(2, 3, 4, 5, 6, 7),
            conf=0.20,
            iou=0.60,
        ),
    ]

    pipeline = CascadeMoEPipeline(config=config, experts=experts)

    triggered = 0
    reason_counter: Counter[str] = Counter()
    det_count = 0
    elapsed_ms: list[float] = []

    for image_path in images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        t0 = time.perf_counter()
        out = pipeline.infer(frame)
        dt = (time.perf_counter() - t0) * 1000.0

        elapsed_ms.append(dt)
        det_count += len(out.detections)
        if out.route.trigger:
            triggered += 1
            reason_counter[out.route.reason] += 1

    n = len(elapsed_ms)
    if n == 0:
        raise ValueError("No valid images were processed for cascade evaluation")

    elapsed_ms.sort()
    p95_idx = min(n - 1, int(0.95 * n))

    return {
        "num_images": n,
        "trigger_rate": float(triggered / n),
        "avg_detections_per_image": float(det_count / n),
        "latency_ms_avg": float(sum(elapsed_ms) / n),
        "latency_ms_p95": float(elapsed_ms[p95_idx]),
        "trigger_reason_counts": dict(reason_counter),
    }


def main() -> None:
    args = parse_args()

    merged = load_yaml(args.merged_data_yaml)
    class_names = parse_names(merged)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_yaml": str(args.merged_data_yaml.resolve()),
        "split": args.split,
        "device": args.device,
        "models": {
            "base": str(args.base_ckpt),
            "ground_expert": str(args.ground_expert_ckpt) if args.ground_expert_ckpt else None,
            "tiny_expert": str(args.tiny_expert_ckpt) if args.tiny_expert_ckpt else None,
        },
        "data": {
            "base": str(args.merged_data_yaml.resolve()),
            "ground_expert": None,
            "tiny_expert": None,
        },
        "metrics": {},
    }

    print("[eval] base model")
    report["metrics"]["base"] = evaluate_detector(
        model_path=args.base_ckpt,
        data_yaml=args.merged_data_yaml,
        split=args.split,
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
        names=class_names,
    )

    if args.ground_expert_ckpt:
        ground_yaml = resolve_expert_data_yaml(args.ground_expert_data_yaml, "ground_expert")
        ground_names = parse_names(load_yaml(ground_yaml))
        print("[eval] ground expert")
        report["metrics"]["ground_expert"] = evaluate_detector(
            model_path=args.ground_expert_ckpt,
            data_yaml=ground_yaml,
            split=args.split,
            device=args.device,
            imgsz=args.imgsz,
            batch=args.batch,
            names=ground_names,
        )
        report["data"]["ground_expert"] = str(ground_yaml.resolve())

    if args.tiny_expert_ckpt:
        tiny_yaml = resolve_expert_data_yaml(args.tiny_expert_data_yaml, "tiny_obstacle_expert")
        tiny_names = parse_names(load_yaml(tiny_yaml))
        print("[eval] tiny obstacle expert")
        report["metrics"]["tiny_obstacle_expert"] = evaluate_detector(
            model_path=args.tiny_expert_ckpt,
            data_yaml=tiny_yaml,
            split=args.split,
            device=args.device,
            imgsz=args.imgsz,
            batch=args.batch,
            names=tiny_names,
        )
        report["data"]["tiny_expert"] = str(tiny_yaml.resolve())

    if args.ground_expert_ckpt and args.tiny_expert_ckpt:
        print("[eval] cascade runtime and trigger stats")
        report["metrics"]["cascade_runtime"] = evaluate_cascade_runtime(
            merged_yaml=merged,
            class_names=class_names,
            split=args.split,
            sample_images=args.sample_images,
            seed=args.seed,
            base_ckpt=args.base_ckpt,
            ground_expert_ckpt=args.ground_expert_ckpt,
            tiny_expert_ckpt=args.tiny_expert_ckpt,
        )

    ensure_dir(args.report.parent)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n")
    print(f"[done] report saved: {args.report}")


if __name__ == "__main__":
    main()
