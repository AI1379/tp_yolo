import cv2
import numpy as np
from pathlib import Path
import shutil
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
DATASET_ROOT  = Path(__file__).parent
IMAGES_DIR    = DATASET_ROOT / "images"
TRAIN_ANN_DIR = DATASET_ROOT / "train"
VAL_ANN_DIR   = DATASET_ROOT / "val"
OUTPUT_ROOT   = Path("./yolo_roadwork_v2")

# ──────────────────────────────────────────────
# 3️⃣ 颜色匹配函数
# ──────────────────────────────────────────────
def get_class_mask(color_img_rgb, target_rgb, tolerance=COLOR_TOLERANCE):
    diff = color_img_rgb.astype(np.int32) - target_rgb.astype(np.int32)
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    return dist < tolerance

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
        return 0

    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    h, w = ids_img.shape[:2]

    # 🔧 修复：检查 IDs 图是否为多通道
    if len(ids_img.shape) == 3:
        ids_img = cv2.cvtColor(ids_img, cv2.COLOR_RGB2GRAY)
    
    if debug:
        print(f"   IDs 图形状: {ids_img.shape}, 唯一值: {len(np.unique(ids_img))}")

    lines = []

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
        else:
            for inst_id in instance_ids_in_class:
                instance_mask = (class_mask & (ids_img == inst_id)).astype(np.uint8) * 255
                polys = extract_polygon(instance_mask)
                for poly in polys:
                    lines.append(f"{yolo_id} {poly}")

    if lines:
        with open(out_txt_path, "w") as f:
            f.write("\n".join(lines))
        return len(lines)
    return 0

# ──────────────────────────────────────────────
# 6️⃣ 建立图片名 → 标注文件 的对应关系
# ──────────────────────────────────────────────
def build_mapping(images_dir, ann_dir):
    mapping = {}
    for jpg in images_dir.glob("*.jpg"):
        stem = jpg.stem
        ids_file = ann_dir / f"{stem}_Ids.png"
        color_file = ann_dir / f"{stem}_labelColors.png"

        if ids_file.exists() and color_file.exists():
            mapping[jpg] = (ids_file, color_file)

    return mapping

# ──────────────────────────────────────────────
# 7️⃣ 处理一个分割（train 或 val）
# ──────────────────────────────────────────────
def process_split(split_name, images_dir, ann_dir, debug_first_n=5):
    print(f"\n{'='*55}")
    print(f"处理 {split_name} 集...")

    out_img_dir = OUTPUT_ROOT / "images" / split_name
    out_lbl_dir = OUTPUT_ROOT / "labels" / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    mapping = build_mapping(images_dir, ann_dir)
    print(f"找到匹配图片：{len(mapping)} 张")

    valid_images = 0
    total_instances = 0
    class_counter = {name: 0 for name in CLASS_CONFIG}
    processed = 0

    for jpg_path, (ids_path, color_path) in tqdm(mapping.items(), desc=split_name):
        out_txt = out_lbl_dir / (jpg_path.stem + ".txt")
        debug = processed < debug_first_n  # 只调试前几张
        n = convert_one_image(ids_path, color_path, out_txt, debug=debug)

        if n > 0:
            shutil.copy(jpg_path, out_img_dir / jpg_path.name)
            valid_images += 1
            total_instances += n

            with open(out_txt) as f:
                for line in f:
                    cls_id = int(line.split()[0])
                    for cls_name, (yolo_id, _) in CLASS_CONFIG.items():
                        if yolo_id == cls_id:
                            class_counter[cls_name] += 1
        
        processed += 1

    print(f"\n✅ {split_name} 完成：{valid_images} 张图片，{total_instances} 个实例")
    print("各类别实例数：")
    for cls_name, count in class_counter.items():
        bar = "█" * min(count // 50, 40)
        print(f"  {cls_name:<16} {count:>5}  {bar}")

    return valid_images

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
    sample = TRAIN_ANN_DIR / "boston_2d8e13b1a8304d8395dcf6479ca61814_000000_05100_labelColors.png"
    verify_colors(sample)

    # 🔧 添加：验证 IDs 图
    sample_ids = TRAIN_ANN_DIR / "boston_2d8e13b1a8304d8395dcf6479ca61814_000000_05100_Ids.png"
    ids_img = cv2.imread(str(sample_ids), cv2.IMREAD_UNCHANGED)
    if ids_img is not None:
        print(f"\n🔍 IDs 图信息:")
        print(f"   形状: {ids_img.shape}")
        print(f"   类型: {ids_img.dtype}")
        print(f"   唯一值数量: {len(np.unique(ids_img))}")
        print(f"   唯一值示例: {np.unique(ids_img)[:20]}")

    print("\n按 Enter 开始转换...")
    input()

    train_count = process_split("train", IMAGES_DIR, TRAIN_ANN_DIR, debug_first_n=5)
    val_count = process_split("val", IMAGES_DIR, VAL_ANN_DIR, debug_first_n=5)

    yaml_lines = [
        f"path: {OUTPUT_ROOT.absolute()}",
        "train: images/train",
        "val:   images/val",
        "",
        f"nc: {len(CLASS_CONFIG)}",
        "names:",
    ]
    for cls_name in CLASS_CONFIG:
        yolo_id = CLASS_CONFIG[cls_name][0]
        yaml_lines.append(f"  {yolo_id}: {cls_name}")

    yaml_path = OUTPUT_ROOT / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write("\n".join(yaml_lines))

    print(f"\n{'='*55}")
    print(f"🎉 全部完成！")
    print(f"   训练集：{train_count} 张")
    print(f"   验证集：{val_count} 张")