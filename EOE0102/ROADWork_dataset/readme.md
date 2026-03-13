将语义分割标注数据转换为 YOLO 分割格式 的自动化工具。

支持从彩色语义图（_labelColors.png）和实例 ID 图（_Ids.png）提取多边形坐标，生成 YOLO 训练所需的 .txt 标注文件。


|功能	| 说明 |
|:--:|:--:|
|🎨 颜色匹配	| 根据预设 RGB 颜色值，识别 6 类目标物体 |
| 🔢 实例分离	| 利用实例 ID 图区分同一类别中的多个独立物体 |
|📐 轮廓提取	| 从二值 mask 提取多边形顶点坐标|
|📝 格式转换	|将坐标归一化并输出为 YOLO 分割格式|
|📊 数据集划分	|自动处理 train 和 val 两个子集|
|✅ 质量验证|	输出各类别实例数量统计|

📁 文件夹结构

#### 输入结构（转换前）

```html
yolo_roadwork_dataset/
├── transform.py              # 转换脚本
├── images/
│   ├── boston_2bdb5a72602342a5991b402beb8b7ab4_000000_02610.jpg
│   └── ...
├── train/                    # 训练集标注
│   ├── boston_2d8e13b1a8304d8395dcf6479ca61814_000000_05100_Ids.png
│   ├── boston_2d8e13b1a8304d8395dcf6479ca61814_000000_05100_labelColors.png
│   ├── boston_2d8e13b1a8304d8395dcf6479ca61814_000000_05100_labelIds.png
│   └── ...
└── val/                      # 验证集标注
    ├── boston_2bdb5a72602342a5991b402beb8b7ab4_000000_02610_Ids.png
    ├── boston_2bdb5a72602342a5991b402beb8b7ab4_000000_02610_labelColors.png
    ├── boston_2bdb5a72602342a5991b402beb8b7ab4_000000_02610_labelIds.png
    └── ...
```
#### 输出结构（转换后）
```html
yolo_roadwork_v2/
├── data.yaml                 # YOLO 数据集配置文件
├── images/
│   ├── train/                # 训练集图片（复制）
│   └── val/                  # 验证集图片（复制）
├── labels/
│   ├── train/                # 训练集标注（新生成）
│   └── val/                  # 验证集标注（新生成）
└── output_check/             # 可视化验证结果
    ├── train/
    └── val/
```
⚠️ 注意：transform.py 应与 images/、train/、val/ 文件夹放在同一层级。

```html
CLASS_CONFIG = {
    "cone":           (0, np.array([30,  119, 179])),  # 交通锥
    "tubular_marker": (1, np.array([170, 118, 213])),  # 柱形标记
    "drum":           (2, np.array([44,   79, 206])),  # 路障桶
    "barricade":      (3, np.array([248, 135, 182])),  # 栅栏式路障
    "barrier":        (4, np.array([246, 116, 185])),  # 隔离栏
    "fence":          (5, np.array([251, 172, 187])),  # 围栏
}
```