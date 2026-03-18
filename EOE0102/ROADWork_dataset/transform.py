import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ──────────────────────────────────────────────
# 1️⃣ 6个目标类别 颜色定义（从 labelColors.png 观察得到）
# ──────────────────────────────────────────────
CLASS_CONFIG = {
    #  类别名          YOLO ID    RGB颜色              颜色容差
    "cone":           (0, np.array([30,  119, 179])),  # #1E77B3
    "tubular_marker": (1, np.array([170, 118, 213])),  # #AA76D5
    "drum":           (2, np.array([44,   79, 206])),  # #2C4FCE
    "barricade":      (3, np.array([248, 135, 182])),  # #F887B6
    "barrier":        (4, np.array([246, 116, 185])),  # #F674B9
    "fence":          (5, np.array([251, 172, 187])),  # #FBACBB
}

COLOR_TOLERANCE = 15   # 颜色匹配容差（像素颜色允许的偏差范围）
MIN_AREA        = 150  # 最小实例面积（过滤噪声）
MIN_POINTS      = 6    # 多边形最少顶点数

# ──────────────────────────────────────────────
# 2️⃣ 路径配置（根据你的实际目录修改）
# ──────────────────────────────────────────────
DATASET_ROOT = Path(__file__).parent
DEFAULT_IMAGES_DIR = DATASET_ROOT / "images"
DEFAULT_ANN_ROOT = DATASET_ROOT
DEFAULT_OUTPUT_ROOT = DATASET_ROOT / "yolo_roadwork_v2"

# ──────────────────────────────────────────────
# 3️⃣ 颜色匹配函数
# ──────────────────────────────────────────────
def get_class_mask(color_img_rgb, target_rgb, tolerance=COLOR_TOLERANCE):
    lower = np.clip(target_rgb - tolerance, 0, 255).astype(np.uint8)
    upper = np.clip(target_rgb + tolerance, 0, 255).astype(np.uint8)
    return cv2.inRange(color_img_rgb, lower, upper) > 0

# ──────────────────────────────────────────────
# 4️⃣ 轮廓提取函数（🔧 修复：返回所有有效轮廓）
# ──────────────────────────────────────────────
def extract_polygon(binary_mask):
    """
    从二值 mask 提取轮廓多边形
    返回所有有效轮廓的列表
    """
    h, w = binary_mask.shape
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []

    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:
            continue

        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        pts = approx.reshape(-1, 2)

        if len(pts) < MIN_POINTS:
            continue

        coords = []
        for x, y in pts:
            coords.extend([
                f"{min(max(x / w, 0.0), 1.0):.6f}",
                f"{min(max(y / h, 0.0), 1.0):.6f}"
            ])
        polygons.append(" ".join(coords))
    
    return polygons

# ──────────────────────────────────────────────
# 5️⃣ 单张图片转换（🔧 修复：处理多轮廓 + IDs 图通道）
# ──────────────────────────────────────────────
def convert_one_image(ids_path, color_path, out_txt_path, debug=False):
    ids_img = cv2.imread(str(ids_path), cv2.IMREAD_UNCHANGED)
    color_bgr = cv2.imread(str(color_path))
    if ids_img is None or color_bgr is None:
        return 0, []

    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    h, w = ids_img.shape[:2]

    # 🔧 修复：检查 IDs 图是否为多通道
    if len(ids_img.shape) == 3:
        ids_img = cv2.cvtColor(ids_img, cv2.COLOR_RGB2GRAY)
    
    if debug:
        print(f"   IDs 图形状: {ids_img.shape}, 唯一值: {len(np.unique(ids_img))}")

    lines = []
    class_ids = []

    for cls_name, (yolo_id, target_rgb) in CLASS_CONFIG.items():
        class_mask = get_class_mask(color_rgb, target_rgb)

        if not class_mask.any():
            continue

        instance_ids_in_class = np.unique(ids_img[class_mask])
        instance_ids_in_class = instance_ids_in_class[instance_ids_in_class != 0]

        if debug:
            print(f"   {cls_name}: 实例ID数 = {len(instance_ids_in_class)}, IDs = {instance_ids_in_class[:10]}")

        if len(instance_ids_in_class) == 0:
            instance_mask = class_mask.astype(np.uint8) * 255
            polys = extract_polygon(instance_mask)
            for poly in polys:
                lines.append(f"{yolo_id} {poly}")
                class_ids.append(yolo_id)
        else:
            for inst_id in instance_ids_in_class:
                instance_mask = (class_mask & (ids_img == inst_id)).astype(np.uint8) * 255
                polys = extract_polygon(instance_mask)
                for poly in polys:
                    lines.append(f"{yolo_id} {poly}")
                    class_ids.append(yolo_id)

    if lines:
        with open(out_txt_path, "w") as f:
            f.write("\n".join(lines))
        return len(lines), class_ids
    return 0, []

# ──────────────────────────────────────────────
# 6️⃣ 建立图片名 → 标注文件 的对应关系
# ──────────────────────────────────────────────
def build_mapping(images_dir, ann_dir):
    mapping = {}
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        stem = image_path.stem
        ids_file = ann_dir / f"{stem}_Ids.png"
        color_file = ann_dir / f"{stem}_labelColors.png"

        if ids_file.exists() and color_file.exists():
            mapping[image_path] = (ids_file, color_file)

    return mapping

# ──────────────────────────────────────────────
# 7️⃣ 处理一个分割（train 或 val）
# ──────────────────────────────────────────────
def process_split(split_name, images_dir, ann_dir, output_root, debug_first_n=5):
    print(f"\n{'='*55}")
    print(f"处理 {split_name} 集...")

    out_img_dir = output_root / "images" / split_name
    out_lbl_dir = output_root / "labels" / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    mapping = build_mapping(images_dir, ann_dir)
    print(f"找到匹配图片：{len(mapping)} 张")

    valid_images = 0
    total_instances = 0
    failed_images = 0
    class_counter = {name: 0 for name in CLASS_CONFIG}
    processed = 0

    for jpg_path, (ids_path, color_path) in tqdm(mapping.items(), desc=split_name):
        out_txt = out_lbl_dir / (jpg_path.stem + ".txt")
        debug = processed < debug_first_n  # 只调试前几张

        try:
            n, class_ids = convert_one_image(ids_path, color_path, out_txt, debug=debug)

            if n > 0:
                shutil.copy(jpg_path, out_img_dir / jpg_path.name)
                valid_images += 1
                total_instances += n

                for cls_id in class_ids:
                    for cls_name, (yolo_id, _) in CLASS_CONFIG.items():
                        if yolo_id == cls_id:
                            class_counter[cls_name] += 1
        except Exception as exc:
            failed_images += 1
            print(f"跳过 {jpg_path.name}: {exc}")
        
        processed += 1

    print(f"\n✅ {split_name} 完成：{valid_images} 张图片，{total_instances} 个实例，{failed_images} 张失败")
    print("各类别实例数：")
    for cls_name, count in class_counter.items():
        bar = "█" * min(count // 50, 40)
        print(f"  {cls_name:<16} {count:>5}  {bar}")

    return valid_images


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ROADWorks semantic annotations to YOLO segmentation format.")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR, help="Directory containing original images.")
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=DEFAULT_ANN_ROOT,
        help="Directory containing train/ and val/ annotation folders, or gtFine/ as extracted from the zip.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Directory where YOLO output will be written.")
    parser.add_argument("--debug-first-n", type=int, default=5, help="Print debug info for the first N images in each split.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip sample color verification.")
    return parser.parse_args()


def resolve_annotation_dirs(annotations_root: Path) -> tuple[Path, Path]:
    if (annotations_root / "train").exists() and (annotations_root / "val").exists():
        return annotations_root / "train", annotations_root / "val"

    gt_fine_root = annotations_root / "gtFine"
    if (gt_fine_root / "train").exists() and (gt_fine_root / "val").exists():
        return gt_fine_root / "train", gt_fine_root / "val"

    raise FileNotFoundError(f"Could not find train/val annotations under {annotations_root}")

# ──────────────────────────────────────────────
# 8️⃣ 验证颜色匹配
# ──────────────────────────────────────────────
def verify_colors(sample_color_path):
    print(f"\n🔍 验证颜色匹配...")
    color_bgr = cv2.imread(str(sample_color_path))
    if color_bgr is None:
        print("❌ 文件读取失败")
        return

    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    total_pixels = color_rgb.shape[0] * color_rgb.shape[1]

    for cls_name, (yolo_id, target_rgb) in CLASS_CONFIG.items():
        mask = get_class_mask(color_rgb, target_rgb)
        count = mask.sum()
        pct = count / total_pixels * 100
        status = "✅" if count > 0 else "⚠️  未找到"
        print(f"  {cls_name:<16} 匹配像素: {count:>6}  ({pct:.2f}%)  {status}")

# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    train_ann_dir, val_ann_dir = resolve_annotation_dirs(args.annotations_root)

    if not args.skip_verify:
        sample = train_ann_dir / "boston_2d8e13b1a8304d8395dcf6479ca61814_000000_05100_labelColors.png"
        if sample.exists():
            verify_colors(sample)

            sample_ids = train_ann_dir / "boston_2d8e13b1a8304d8395dcf6479ca61814_000000_05100_Ids.png"
            ids_img = cv2.imread(str(sample_ids), cv2.IMREAD_UNCHANGED)
            if ids_img is not None:
                print(f"\n🔍 IDs 图信息:")
                print(f"   形状: {ids_img.shape}")
                print(f"   类型: {ids_img.dtype}")
                print(f"   唯一值数量: {len(np.unique(ids_img))}")
                print(f"   唯一值示例: {np.unique(ids_img)[:20]}")

    train_count = process_split("train", args.images_dir, train_ann_dir, args.output_root, debug_first_n=args.debug_first_n)
    val_count = process_split("val", args.images_dir, val_ann_dir, args.output_root, debug_first_n=args.debug_first_n)

    yaml_lines = [
        f"path: {args.output_root.absolute()}",
        "train: images/train",
        "val:   images/val",
        "",
        f"nc: {len(CLASS_CONFIG)}",
        "names:",
    ]
    for cls_name in CLASS_CONFIG:
        yolo_id = CLASS_CONFIG[cls_name][0]
        yaml_lines.append(f"  {yolo_id}: {cls_name}")

    yaml_path = args.output_root / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write("\n".join(yaml_lines))

    print(f"\n{'='*55}")
    print(f"🎉 全部完成！")
    print(f"   训练集：{train_count} 张")
    print(f"   验证集：{val_count} 张")