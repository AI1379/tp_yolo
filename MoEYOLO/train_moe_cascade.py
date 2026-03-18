from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml
from ultralytics import YOLO


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    focus_classes: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline and expert YOLO models for MoEYOLO cascade.")

    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--merged-data-yaml", type=Path, default=root / "data" / "merged_yolo_detect" / "data.yaml")
    parser.add_argument("--base-model", type=str, default=str(root / "yolo11m.pt"))

    parser.add_argument("--mode", choices=("base", "experts", "all"), default="all")
    parser.add_argument("--out-root", type=Path, default=Path(__file__).resolve().parent / "artifacts")

    parser.add_argument("--device", type=str, default="1,2,3", help="Use last three 3090 cards by default.")
    parser.add_argument("--epochs-base", type=int, default=30)
    parser.add_argument("--epochs-expert", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--fraction", type=float, default=1.0, help="Use a data fraction for quick trials.")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--amp", action="store_true", help="Enable AMP mixed precision. Default is off for offline stability.")

    parser.add_argument("--project", type=str, default="runs/moeyolo")
    parser.add_argument("--name-prefix", type=str, default="moe")

    parser.add_argument("--rebuild-expert-data", action="store_true", help="Rebuild expert subsets before training experts.")
    return parser.parse_args()


def load_merged_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing merged dataset yaml: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Invalid data.yaml format")
    return data


def parse_class_names(yaml_data: dict) -> list[str]:
    names = yaml_data.get("names")
    if isinstance(names, dict):
        out = []
        for i in sorted(int(k) for k in names.keys()):
            out.append(str(names[i]))
        return out
    if isinstance(names, list):
        return [str(n) for n in names]
    raise ValueError("data.yaml must contain 'names' as dict or list")


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


def write_yaml(path: Path, data: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False))


def build_expert_subset(
    merged_data_root: Path,
    merged_names: list[str],
    expert: ExpertSpec,
    output_root: Path,
    force_rebuild: bool,
) -> Path:
    name_to_id = {name: i for i, name in enumerate(merged_names)}
    focus_src_ids = [name_to_id[c] for c in expert.focus_classes]
    src_to_dst = {src_id: i for i, src_id in enumerate(focus_src_ids)}

    subset_root = output_root / "expert_subsets" / expert.name
    if subset_root.exists() and force_rebuild:
        shutil.rmtree(subset_root)

    cached_yaml = subset_root / "data.yaml"
    if cached_yaml.exists() and not force_rebuild:
        return cached_yaml

    for split in ("train", "val", "test"):
        img_in = merged_data_root / "images" / split
        lbl_in = merged_data_root / "labels" / split
        if not img_in.exists() or not lbl_in.exists():
            continue

        img_out = subset_root / "images" / split
        lbl_out = subset_root / "labels" / split
        ensure_dir(img_out)
        ensure_dir(lbl_out)

        for image_path in sorted(img_in.iterdir()):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue

            label_in = lbl_in / f"{image_path.stem}.txt"
            label_out = lbl_out / f"{image_path.stem}.txt"

            out_lines: list[str] = []
            if label_in.exists():
                for line in label_in.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        src_cls = int(parts[0])
                    except ValueError:
                        continue
                    if src_cls not in src_to_dst:
                        continue
                    dst_cls = src_to_dst[src_cls]
                    out_lines.append(" ".join([str(dst_cls)] + parts[1:5]))

            # Keep all images to preserve negatives for expert classes.
            link_or_copy(image_path, img_out / image_path.name)
            label_out.write_text(("\n".join(out_lines) + "\n") if out_lines else "")

    data_yaml = subset_root / "data.yaml"
    yaml_data = {
        "path": str(subset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(expert.focus_classes),
        "names": {i: n for i, n in enumerate(expert.focus_classes)},
    }
    write_yaml(data_yaml, yaml_data)
    return data_yaml


def train_one(
    model_path: str,
    data_yaml: Path,
    device: str,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int,
    fraction: float,
    patience: int,
    lr0: float,
    amp: bool,
    project: str,
    name: str,
) -> Path:
    model = YOLO(model_path)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
        fraction=fraction,
        patience=patience,
        lr0=lr0,
        amp=amp,
        project=project,
        name=name,
        exist_ok=True,
    )

    save_dir = None
    if results is not None and hasattr(results, "save_dir"):
        save_dir = Path(getattr(results, "save_dir"))
    elif getattr(model, "trainer", None) is not None and hasattr(model.trainer, "save_dir"):
        save_dir = Path(getattr(model.trainer, "save_dir"))

    if save_dir is None:
        raise RuntimeError("Could not infer save_dir from Ultralytics training run")

    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Expected trained checkpoint not found: {best}")
    return best


def main() -> None:
    args = parse_args()

    merged_yaml = load_merged_yaml(args.merged_data_yaml)
    merged_data_root = Path(merged_yaml["path"]).resolve()
    class_names = parse_class_names(merged_yaml)

    ensure_dir(args.out_root)

    experts = [
        ExpertSpec(name="ground_expert", focus_classes=("blind_road", "crosswalk")),
        ExpertSpec(name="tiny_obstacle_expert", focus_classes=("cone", "tubular_marker", "drum", "barricade", "barrier", "fence")),
    ]

    base_ckpt = Path(args.base_model)

    if args.mode in {"base", "all"}:
        print(f"[train] base model on devices {args.device}")
        base_ckpt = train_one(
            model_path=str(base_ckpt),
            data_yaml=args.merged_data_yaml,
            device=args.device,
            epochs=args.epochs_base,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            fraction=args.fraction,
            patience=args.patience,
            lr0=args.lr0,
            amp=args.amp,
            project=args.project,
            name=f"{args.name_prefix}_base",
        )
        print(f"[done] base checkpoint: {base_ckpt}")

    if args.mode in {"experts", "all"}:
        print("[prep] building expert subsets")
        subset_yaml_by_expert: dict[str, Path] = {}
        for expert in experts:
            subset_yaml = build_expert_subset(
                merged_data_root=merged_data_root,
                merged_names=class_names,
                expert=expert,
                output_root=args.out_root,
                force_rebuild=args.rebuild_expert_data,
            )
            subset_yaml_by_expert[expert.name] = subset_yaml
            print(f"[prep] {expert.name}: {subset_yaml}")

        for expert in experts:
            print(f"[train] {expert.name} on devices {args.device}")
            expert_best = train_one(
                model_path=str(base_ckpt),
                data_yaml=subset_yaml_by_expert[expert.name],
                device=args.device,
                epochs=args.epochs_expert,
                imgsz=args.imgsz,
                batch=args.batch,
                workers=args.workers,
                fraction=args.fraction,
                patience=args.patience,
                lr0=args.lr0,
                amp=args.amp,
                project=args.project,
                name=f"{args.name_prefix}_{expert.name}",
            )
            print(f"[done] {expert.name} checkpoint: {expert_best}")

    print("[complete] training pipeline finished")


if __name__ == "__main__":
    main()
