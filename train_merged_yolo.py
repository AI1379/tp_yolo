"""Merge three datasets and train a unified YOLO detection model.

Datasets:
- TP-Dataset boxes_fixed (detection)
- Crosswalk Detection (detection)
- ROADWorks_yolo_seg (segmentation polygons converted to boxes)
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml
from ultralytics import YOLO


@dataclass(frozen=True)
class SourceDataset:
    key: str
    root: Path
    split_map: dict[str, str]
    class_names: list[str]
    is_segmentation: bool
    split_first_layout: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge three datasets and train a YOLO model.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Root directory containing prepared datasets.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data") / "merged_yolo_detect",
        help="Output directory for merged dataset and YAML.",
    )
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="Ultralytics model name or path.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=96, help="Global batch size.")
    parser.add_argument("--workers", type=int, default=24, help="Data loader workers.")
    parser.add_argument("--device", type=str, default="1,2,3", help="CUDA devices, default uses the last three 3090 cards.")
    parser.add_argument("--project", type=str, default="runs/merged", help="Ultralytics project directory.")
    parser.add_argument("--name", type=str, default="merged_yolo_det", help="Ultralytics run name.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting an existing run directory.")
    parser.add_argument("--merge-only", action="store_true", help="Only build merged dataset, do not start training.")
    return parser.parse_args()


def canonical_class_name(name: str) -> str:
    alias_map = {
        "crosswalk": "crosswalk",
        "blind_road": "blind_road",
        "cone": "cone",
        "tubular_marker": "tubular_marker",
        "drum": "drum",
        "barricade": "barricade",
        "barrier": "barrier",
        "fence": "fence",
    }
    key = name.strip().lower()
    return alias_map.get(key, key)


def build_source_datasets(data_root: Path) -> list[SourceDataset]:
    return [
        SourceDataset(
            key="tp",
            root=data_root / "TP-Dataset" / "YOLO_Data" / "boxes_fixed",
            split_map={"train": "train", "val": "val", "test": "test"},
            class_names=["blind_road"],
            is_segmentation=False,
            split_first_layout=True,
        ),
        SourceDataset(
            key="crosswalk",
            root=data_root / "Crosswalk Detection.v5i.yolo26",
            split_map={"train": "train", "val": "valid", "test": "test"},
            class_names=["Crosswalk"],
            is_segmentation=False,
            split_first_layout=True,
        ),
        SourceDataset(
            key="roadworks",
            root=data_root / "ROADWorks_yolo_seg",
            split_map={"train": "train", "val": "val"},
            class_names=["cone", "tubular_marker", "drum", "barricade", "barrier", "fence"],
            is_segmentation=True,
            split_first_layout=False,
        ),
    ]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def parse_floats(values: Iterable[str]) -> list[float]:
    parsed: list[float] = []
    for value in values:
        clean = value.strip().replace("%", "")
        if not clean:
            continue
        parsed.append(float(clean))
    return parsed


def seg_to_det_bbox(coords: list[float]) -> tuple[float, float, float, float] | None:
    if len(coords) < 6 or len(coords) % 2 != 0:
        return None
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min
    h = y_max - y_min
    if w <= 0.0 or h <= 0.0:
        return None
    x_c = x_min + w / 2.0
    y_c = y_min + h / 2.0
    return clamp01(x_c), clamp01(y_c), clamp01(w), clamp01(h)


def remap_label_line(
    line: str,
    source: SourceDataset,
    class_to_index: dict[str, int],
) -> str | None:
    parts = line.strip().split()
    if not parts:
        return None

    try:
        src_cls = int(parts[0])
    except ValueError:
        return None

    if src_cls < 0 or src_cls >= len(source.class_names):
        return None

    canonical_name = canonical_class_name(source.class_names[src_cls])
    dst_cls = class_to_index[canonical_name]

    if source.is_segmentation:
        coords = parse_floats(parts[1:])
        bbox = seg_to_det_bbox(coords)
        if bbox is None:
            return None
        x_c, y_c, w, h = bbox
        return f"{dst_cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"

    nums = parse_floats(parts[1:5])
    if len(nums) != 4:
        return None
    x_c, y_c, w, h = (clamp01(v) for v in nums)
    return f"{dst_cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"


def collect_unified_classes(sources: list[SourceDataset]) -> list[str]:
    preferred_order = [
        "blind_road",
        "crosswalk",
        "cone",
        "tubular_marker",
        "drum",
        "barricade",
        "barrier",
        "fence",
    ]
    discovered = {canonical_class_name(name) for src in sources for name in src.class_names}
    ordered = [name for name in preferred_order if name in discovered]
    for name in sorted(discovered):
        if name not in ordered:
            ordered.append(name)
    return ordered


def merge_datasets(data_root: Path, output_root: Path) -> Path:
    sources = build_source_datasets(data_root)
    for source in sources:
        if not source.root.exists():
            raise FileNotFoundError(f"Missing dataset root: {source.root}")

    unified_classes = collect_unified_classes(sources)
    class_to_index = {name: i for i, name in enumerate(unified_classes)}

    if output_root.exists():
        shutil.rmtree(output_root)

    for split in ("train", "val", "test"):
        ensure_dir(output_root / "images" / split)
        ensure_dir(output_root / "labels" / split)

    stats = {"train": 0, "val": 0, "test": 0}
    skipped = 0

    def split_dirs(source: SourceDataset, src_split: str) -> tuple[Path, Path]:
        if source.split_first_layout:
            return source.root / src_split / "images", source.root / src_split / "labels"
        return source.root / "images" / src_split, source.root / "labels" / src_split

    for source in sources:
        for dst_split, src_split in source.split_map.items():
            image_dir, label_dir = split_dirs(source, src_split)
            if not image_dir.exists() or not label_dir.exists():
                continue

            for image_path in sorted(image_dir.iterdir()):
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue

                label_path = label_dir / f"{image_path.stem}.txt"
                if not label_path.exists():
                    continue

                out_lines: list[str] = []
                for line in label_path.read_text().splitlines():
                    remapped = remap_label_line(line, source, class_to_index)
                    if remapped is not None:
                        out_lines.append(remapped)

                if not out_lines:
                    skipped += 1
                    continue

                merged_stem = f"{source.key}_{src_split}_{image_path.stem}"
                dst_image = output_root / "images" / dst_split / f"{merged_stem}{image_path.suffix.lower()}"
                dst_label = output_root / "labels" / dst_split / f"{merged_stem}.txt"
                suffix_idx = 1
                while dst_image.exists() or dst_label.exists():
                    merged_stem_i = f"{merged_stem}_{suffix_idx}"
                    dst_image = output_root / "images" / dst_split / f"{merged_stem_i}{image_path.suffix.lower()}"
                    dst_label = output_root / "labels" / dst_split / f"{merged_stem_i}.txt"
                    suffix_idx += 1

                link_or_copy(image_path, dst_image)
                dst_label.write_text("\n".join(out_lines) + "\n")
                stats[dst_split] += 1

    yaml_path = output_root / "data.yaml"
    yaml_data = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(unified_classes),
        "names": {i: name for i, name in enumerate(unified_classes)},
    }
    yaml_path.write_text(yaml.safe_dump(yaml_data, sort_keys=False, allow_unicode=False))

    print("Merged dataset complete.")
    print(f"  train images: {stats['train']}")
    print(f"  val images:   {stats['val']}")
    print(f"  test images:  {stats['test']}")
    print(f"  skipped labels after remap: {skipped}")
    print(f"  classes: {unified_classes}")
    print(f"  yaml: {yaml_path}")

    return yaml_path


def train_model(args: argparse.Namespace, yaml_path: Path) -> None:
    model = YOLO(args.model)
    model.train(
        task="detect",
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
    )


def main() -> None:
    args = parse_args()
    yaml_path = merge_datasets(args.data_root, args.output_root)
    if args.merge_only:
        print("Merge-only mode enabled; training skipped.")
        return
    train_model(args, yaml_path)


if __name__ == "__main__":
    main()
