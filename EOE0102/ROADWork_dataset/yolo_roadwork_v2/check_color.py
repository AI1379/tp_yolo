import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_yolo_segmentation(split='train', num_samples=5):
    """
    可视化YOLO分割标注（适配yolo_roadwork_v2文件夹结构）
    :param split: 检查的数据集划分，'train' 或 'val'
    :param num_samples: 随机检查的样本数量
    """
    # -------------------------- 核心配置：6种物品与对应RGB颜色 --------------------------
    CLASS_NAMES = [
        "Cone",          # 类别ID 0
        "TubularMarker", # 类别ID 1
        "Drum",          # 类别ID 2
        "Barricade",     # 类别ID 3
        "Barrier",       # 类别ID 4
        "Fence"          # 类别ID 5
    ]
    # 对应RGB颜色（十六进制转RGB元组）
    CLASS_COLORS = [
        (30, 119, 179),   # Cone: 1E77B3
        (170, 118, 213),  # TubularMarker: AA76D5
        (44, 79, 206),     # Drum: 2C4FCE
        (248, 135, 182),   # Barricade: F887B6
        (246, 116, 185),   # Barrier: F674B9
        (251, 172, 187)    # Fence: FBACBB
    ]

    # -------------------------- 自动适配文件夹路径 --------------------------
    # 获取当前脚本所在目录（即yolo_roadwork_v2根目录）
    root_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建图片和标注文件夹路径
    img_dir = os.path.join(root_dir, 'images', split)
    label_dir = os.path.join(root_dir, 'labels', split)

    # 检查路径是否存在
    if not os.path.exists(img_dir):
        print(f"❌ 错误：图片文件夹不存在：{img_dir}")
        return
    if not os.path.exists(label_dir):
        print(f"❌ 错误：标注文件夹不存在：{label_dir}")
        return
    print(f"✅ 正在检查数据集：{split}")
    print(f"   图片路径：{img_dir}")
    print(f"   标注路径：{label_dir}")

    # 获取所有图片文件
    img_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(img_exts)]
    if not img_files:
        print(f"❌ 错误：图片文件夹中未找到任何图片文件")
        return
    # 限制样本数量
    if len(img_files) > num_samples:
        img_files = img_files[:num_samples]
    print(f"   共找到 {len(img_files)} 张图片用于检查\n")

    for img_file in img_files:
        # 1. 读取图片
        img_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️  警告：无法读取图片 {img_file}，跳过")
            continue
        img_h, img_w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 用于绘制半透明遮罩
        mask_overlay = img_rgb.copy()

        # 2. 读取对应标注文件
        label_file = f"{img_name}.txt"
        label_path = os.path.join(label_dir, label_file)
        if not os.path.exists(label_path):
            print(f"⚠️  警告：未找到标注文件 {label_file}，跳过此图片")
            continue

        # 3. 解析标注并绘制分割轮廓+遮罩
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        for line in lines:
            items = list(map(float, line.split()))
            class_id = int(items[0])
            # 校验类别ID是否在合法范围内
            if class_id < 0 or class_id >= len(CLASS_NAMES):
                print(f"⚠️  警告：图片 {img_file} 中存在未知类别ID {class_id}，跳过该实例")
                continue
            
            # 提取成对的坐标，转为像素坐标
            coords = np.array(items[1:]).reshape(-1, 2)
            coords[:, 0] *= img_w  # x转为像素坐标
            coords[:, 1] *= img_h  # y转为像素坐标
            coords = coords.astype(np.int32)

            # 获取当前类别的名称和颜色
            class_name = CLASS_NAMES[class_id]
            color = CLASS_COLORS[class_id]
            
            # 绘制半透明填充遮罩
            cv2.fillPoly(mask_overlay, [coords], color)
            # 绘制轮廓线（更清晰的边界）
            cv2.polylines(img_rgb, [coords], isClosed=True, color=color, thickness=2)

            # 绘制类别标签
            x_min, y_min = coords.min(axis=0)
            cv2.putText(img_rgb, class_name, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 4. 合并原图、遮罩，实现半透明效果
        alpha = 0.4  # 遮罩透明度，0=完全透明，1=完全不透明
        img_result = cv2.addWeighted(img_rgb, 1 - alpha, mask_overlay, alpha, 0)

        # 5. 显示结果
        plt.figure(figsize=(12, 12))
        plt.imshow(img_result)
        plt.axis('off')
        plt.title(f"[{split}] 分割标注校验: {img_file}")
        plt.show()

# -------------------------- 运行配置 --------------------------
if __name__ == "__main__":
    # 在这里修改参数：
    # split: 检查 'train' 训练集 或 'val' 验证集
    # num_samples: 检查的图片数量
    visualize_yolo_segmentation(split='train', num_samples=10)