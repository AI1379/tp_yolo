# MoEYOLO 架构与流程详细可视化

## 目录
1. [系统总体架构](#系统总体架构)
2. [推理流程详解](#推理流程详解)
3. [训练流程详解](#训练流程详解)
4. [数据流与转换](#数据流与转换)
5. [显存与计算分配](#显存与计算分配)
6. [优化决策树](#优化决策树)

---

## 系统总体架构

### 模块关系图

```
┌──────────────────────────────────────────────────────────────────┐
│                         MoEYOLO 系统                               │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────────┐
│   输入图像 (JPG)     │
│  640×640 分辨率     │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────────────────────────┐
│    CascadeMoEPipeline                   │
│  (moe_yolo/cascade.py)                  │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ 1. 加载基础模型                  │   │
│  │    model_base = YOLO(base.pt)   │   │
│  └──────────────┬──────────────────┘   │
│                 │                       │
│  ┌──────────────▼──────────────────┐   │
│  │ 2. 前向推理（基础模型）          │   │
│  │    det_base = model_base(img)   │   │
│  │    发出 N 个框，8 个类别         │   │
│  └──────────────┬──────────────────┘   │
│                 │                       │
│  ┌──────────────▼──────────────────┐   │
│  │ 3. 路由决策 RuleBasedRouter      │   │
│  │    if 框少 → trigger_ground     │   │
│  │    if 小物多 → trigger_tiny      │   │
│  │    else → 返回基础结果          │   │
│  └──────────────┬──────────────────┘   │
│                 │                       │
│    ┌────────────┴────────────┐          │
│    │ 激活哪个专家？          │          │
│    │                         │          │
│    ▼                         ▼          │
│ ┌─────────────┐         ┌──────────┐   │
│ │地面专家模型 │         │微小物体  │   │
│ │(expert1)    │         │专家      │   │
│ │2 类输出     │         │(expert2) │   │
│ │             │         │6 类输出  │   │
│ └──────┬──────┘         └────┬─────┘   │
│        │                     │         │
│    ┌───────────────────────────┐       │
│    │ 4. 融合多个输出            │       │
│    │    merge_detections()     │       │
│    │    ⤳ NMS 去重             │       │
│    │    ⤳ 去除重复框           │       │
│    └───────────┬───────────────┘       │
│                │                       │
│    ┌───────────▼────────────┐          │
│    │ 5. 最终结果            │          │
│    │  - final_boxes         │          │
│    │  - final_classes       │          │
│    │  - final_confidence    │          │
│    └───────────┬────────────┘          │
└─────────────────────────────────────────┘
                 │
                 ↓
        ┌─────────────────┐
        │  可视化输出图   │
        │  标注框和标签   │
        └─────────────────┘
```

### 文件模块对应关系

```
MoEYOLO/
│
├── train_moe_cascade.py
│   ├─ parse_args()              ← 解析训练参数
│   ├─ build_expert_subset()     ← 创建专家子数据集
│   ├─ train_one()               ← 单个模型训练
│   └─ main()                    ← 协调基础和专家训练
│
├── moe_yolo/
│   │
│   ├── cascade.py
│   │   ├─ CascadeMoEPipeline    ← 级联推理管道
│   │   ├─ predict()             ← 单张图推理
│   │   └─ postprocess()         ← 后处理和NMS
│   │
│   ├── router.py
│   │   ├─ RuleBasedRouter       ← 决策路由
│   │   ├─ should_activate_ground_expert()  ← 规则1
│   │   └─ should_activate_tiny_expert()    ← 规则2
│   │
│   ├── config.py
│   │   └─ MoEConfig             ← 配置类
│   │
│   └── train_router_stub.py
│       └─ 辅助函数
│
├── evaluate_moe_v1.py
│   ├─ load_models()             ← 加载 3 个模型
│   ├─ evaluate_detector()       ← 单个模型评估
│   ├─ evaluate_cascade()        ← 级联系统评估
│   └─ main()                    ← 生成 JSON 报告
│
└── runs/detect/runs/moeyolo/
    ├── moe_v1_base/
    │   └── weights/best.pt      ← 基础模型权重
    ├── moe_v1_ground_expert/
    │   └── weights/best.pt      ← 地面专家权重
    └── moe_v1_tiny_obstacle_expert/
        └── weights/best.pt      ← 微小物体专家权重
```

---

## 推理流程详解

### 完整推理演示

假设输入一张含有多个物体的图像：

```
【输入图像场景说明】
一个道路场景，包含：
- 2 个人行横道（crosswalk）- 大对象
- 5 个锥形物体（cone）- 小对象
- 3 个栅栏段（fence）- 小对象
- 1 个路障（barricade）- 小对象

【推理步骤1：基础模型推理】
input: 640×640 RGB 图像
       ↓
YOLO11m 神经网络
       ↓
output: [
  [x1, y1, w1, h1, conf=0.92, class=1],   # crosswalk ✓大且清晰
  [x2, y2, w2, h2, conf=0.88, class=1],   # crosswalk ✓大且清晰
  [x3, y3, w3, h3, conf=0.45, class=2],   # cone (置信度低)
  [x4, y4, w4, h4, conf=0.42, class=2],   # cone (置信度低)
  [x5, y5, w5, h5, conf=0.38, class=7]    # fence (很不确定)
]
总检测 5 个框，其中 3 个是小物体且置信度都很低

【决策路由分析】
检测框数量: 5 个
  - 规则1检查："框 < 3 个?" → 否（5 > 3），不激活地面专家
  
小物体数量: 3 个（cone, fence）
  - 规则2检查："小物 > 5 个?" → 否（3 < 5），不激活微小物体专家？
  
等等！但这三个小物体置信度都很低(< 0.5)
  - 实际规则："小物 > 5 个 OR 小物置信度都很低" → 是（需要帮助）
  - 激活微小物体专家！

【推理步骤2：微小物体专家推理】
input: 同一张 640×640 图像
       ↓
YOLO11m (训练在 cone/tubular_marker/drum/barricade/barrier/fence 上)
       ↓
output: [
  [x3, y3, w3, h3, conf=0.78, class=0],   # cone (0 = cone in this model)
  [x4, y4, w4, h4, conf=0.72, class=0],   # cone 
  [x5, y5, w5, h5, conf=0.85, class=5],   # fence (5 = fence in this model)
  [x6, y6, w6, h6, conf=0.68, class=3]    # barricade (这是新发现！)
]
小物体专家发现了 4 个框，其中有一个基础模型漏掉的！

【推理步骤3：融合结果】
来源1（基础模型）:
  ✓ crosswalk #1 (0.92)
  ✓ crosswalk #2 (0.88)
  ✗ cone (0.45)
  ✗ cone (0.42)
  ✗ fence (0.38)

来源2（微小物体专家）:
  ✓ cone #1 (0.78)
  ✓ cone #2 (0.72)
  ✓ fence (0.85)
  ✓ barricade (0.68)   ← 新发现

融合策略：
  1. 收集所有框
  2. 根据类别和位置进行 NMS（去重）
  3. 冲突时保留置信度高的

最终输出：
  ✓ crosswalk #1 (来自基础, 0.92)
  ✓ crosswalk #2 (来自基础, 0.88)
  ✓ cone #1 (来自专家, 0.78)
  ✓ cone #2 (来自专家, 0.72)
  ✓ fence (来自专家, 0.85)
  ✓ barricade (来自专家, 0.68)

【性能对比】
基础模型单独：5 个框，其中 3 个置信度不够
专家融合后：6 个框，其中 5 个置信度 ≥ 0.7

Recall 从 83% 提升到 100% ✓
Precision 从 60% 提升到 83% ✓
```

### 关键代码实现

```python
# cascade.py 核心逻辑

class CascadeMoEPipeline:
    def predict(self, image, conf=0.5):
        # 步骤1：基础模型
        img_resized = preprocess(image)
        base_detections = self.base_model(img_resized)
        
        # 步骤2：路由决策
        router = RuleBasedRouter()
        activate_ground = router.should_activate_ground_expert(base_detections)
        activate_tiny = router.should_activate_tiny_expert(base_detections)
        
        expert_detections = []
        if activate_ground:
            expert_det = self.ground_expert_model(img_resized)
            expert_detections.extend(expert_det)
            # → 返回 crosswalk/blind_road 更精准的检测
        
        if activate_tiny:
            expert_det = self.tiny_expert_model(img_resized)
            expert_detections.extend(expert_det)
            # → 返回小物体更精准的分类
        
        # 步骤3：融合
        if expert_detections:
            all_detections = np.vstack([
                base_detections,
                expert_detections
            ])
        else:
            all_detections = base_detections
        
        # 步骤4：NMS 去重
        final_detections = nms(all_detections, conf_threshold=conf)
        
        return {
            'detections': final_detections,
            'triggered_experts': [
                'ground_expert' if activate_ground else None,
                'tiny_obstacle_expert' if activate_tiny else None
            ]
        }
```

---

## 训练流程详解

### 训练三阶段流程

```
【=== 第1阶段：数据准备 ===】
input: data/merged_yolo_detect/（8类，6537 train + 2425 val）

执行 build_expert_subset():
  │
  ├─ 过滤 ground_expert 类别
  │  ├─ 找到所有有 blind_road 或 crosswalk 的图
  │  ├─ 复制这些图到 artifacts/expert_subsets/ground_expert/
  │  ├─ 生成 data.yaml（2 类定义）
  │  └─ 结果：~2500 张图有这两类
  │
  └─ 过滤 tiny_obstacle_expert 类别
     ├─ 找到所有有 cone/tubular_marker/drum/barricade/barrier/fence 的图
     ├─ 复制这些图到 artifacts/expert_subsets/tiny_obstacle_expert/
     ├─ 生成 data.yaml（6 类定义）
     └─ 结果：~5500 张图有这六类


【=== 第2阶段：基础模型训练 ===】
配置：
  数据集：merged_yolo_detect（8 类）
  模型：YOLO11m（2000 万参数）
  超参数：
    ├─ batch_size = 96
    ├─ img_size = 640
    ├─ epochs = 30
    └─ device = [GPU1, GPU2, GPU3]（3 块 GPU 分布式）

训练流程：
  epoch 1:  训练 6537 图 ÷ 96 = 68 批次 → 验证 2425 图
  epoch 2:  同上
  ...
  epoch 30: 同上
  
  每个 epoch：
    ┌─ 前向传播：图 → 模型 → 8 类预测
    ├─ 计算损失：预测 vs 真实
    ├─ 反向传播：求梯度
    ├─ 参数更新：梯度下降
    └─ 验证：每个 epoch 末尾在验证集测试
  
  输出：best.pt（在验证集 mAP 最高时保存）
  时间：~30 分钟（GPU）


【=== 第3阶段：专家模型训练 ===】

子任务1：地面专家训练
  配置：
    数据集：expert_subsets/ground_expert（2 类）
    模型：YOLO11m（2000 万参数，但只需要预测 2 类）
    超参数：
      ├─ batch_size = 96
      ├─ img_size = 640
      ├─ epochs = 20  ← 比基础少，因为数据少
      └─ device = [GPU1, GPU2, GPU3]
  
  训练：
    ├─ 优势：数据集 100% 都包含这两个类别
    ├─ 效果：模型能专注学习这两个类的特征
    ├─ 结果：mAP 更高（0.69 vs基础的 0.46）
    └─ 时间：~15 分钟

子任务2：微小物体专家训练
  配置：
    数据集：expert_subsets/tiny_obstacle_expert（6 类）
    模型：YOLO11m
    超参数：
      ├─ batch_size = 96
      ├─ img_size = 640
      ├─ epochs = 20
      └─ device = [GPU1, GPU2, GPU3]
  
  训练：
    ├─ 优势：只处理小物体，减少大物体干扰
    ├─ 效果：小物体分类更精准
    ├─ 结果：mAP < 基础（0.36），但分类准（precision 0.68）
    └─ 时间：~15 分钟


【=== 总耗时 ===】
基础：30 分钟
地面：15 分钟
微小：15 分钟
总计：60 分钟（3 GPU 并行可优化）
```

### DDP 分布式训练原理

```
【单 GPU 训练】
数据集（100 张图）
    ↓
┌─ GPU 0 ────────────────────┐
│  batch = 96 张图            │
│  前向推理 → 反向传播 → 更新 │
│  处理时间：1 个 epoch = 2s  │
└────────────────────────────┘

【3 GPU 分布式训练 (DDP)】
数据集（100 张图）
    ↓ 【分割成 3 份】
    ├─ 33 张 → GPU 1 ─┐
    ├─ 33 张 → GPU 2  ├─ 并行处理
    └─ 33 张 → GPU 3 ─┘
    
每个 GPU：
  ├─ GPU 1: batch=32(96/3) 前向 + 反向
  ├─ GPU 2: batch=32 前向 + 反向
  └─ GPU 3: batch=32 前向 + 反向
    
然后【同步梯度】:
  ├─ GPU 1 的梯度 + GPU 2 的梯度 + GPU 3 的梯度
  └─ 平均 → 更新所有模型参数
  
结果：1 个 epoch = 2s ÷ 3 ≈ 1.3s（加速但不完全线性）
优势：
  ✓ 速度提升 ~2-2.5 倍
  ✗ 需要同步通信（网络开销）
  ✗ 显存仍然多，反而是 batch_size 大导致 OOM
```

---

## 数据流与转换

### 从输入到输出的数据变换

```
【输入图像】
file: image.jpg
size: 1920×1080 (任意分辨率)
dtype: uint8 [0-255]


【预处理步骤1：读取和尺寸调整】
┌───────────────────────────┐
│ 使用 PIL/cv2 读取         │
│ image = Image.open(jpg)   │
│ shape: (1080, 1920, 3)    │  [H, W, C]
│ dtype: uint8              │
└───────────┬───────────────┘
            ↓
┌───────────────────────────┐
│ 调整到 640×640            │
│ 填充或缩放                │
│ shape: (640, 640, 3)      │
│ dtype: uint8 [0-255]      │
└───────────┬───────────────┘


【预处理步骤2：归一化】
┌───────────────────────────┐
│ 除以 255 化为浮点数       │
│ image / 255               │
│ shape: (640, 640, 3)      │
│ dtype: float32 [0-1]      │
└───────────┬───────────────┘


【预处理步骤3：通道排序】
┌───────────────────────────┐
│ PIL 使用 RGB               │
│ PyTorch 使用 CHW 格式     │
│ transpose & reshape       │
│ shape: (1, 3, 640, 640)   │  [batch, channels, H, W]
│ dtype: float32            │
└───────────┬───────────────┘


【YOLO 神经网络】
┌─────────────────────────── ─────┐
│ Input: (1, 3, 640, 640)          │
│                                 │
│ [Backbone]                      │
│ 预先训练的特征提取器            │
│ 多尺度特征：                    │
│   - Layer1: (1, 128, 160, 160)  │
│   - Layer2: (1, 256, 80, 80)    │
│   - Layer3: (1, 512, 40, 40)    │
│                                 │
│ [Neck]                          │
│ 特征融合与金字塔                │
│                                 │
│ [Head]                          │
│ 预测层                          │
│   ├─ Box regression             │
│   │  输出: (1, N, 4) [x,y,w,h]  │
│   ├─ Confidence                 │
│   │  输出: (1, N, 1) [conf]     │
│   └─ Class probability          │
│      输出: (1, N, 8) [class]    │
│                                 │
│ Output: (1, N×13, 8)            │
│ N ≈ 8400 (所有可能框位置)       │
│ 13 = 4(box) + 1(conf) + 8(cls)  │
└─────────────────────────── ─────┘
           ↓


【后处理步骤1：阈值过滤】
┌───────────────────────────┐
│ 删除 conf < 0.5 的预测    │
│ 从 8400 个框 → ~300 个    │
└───────────┬───────────────┘


【后处理步骤2：NMS】
┌───────────────────────────┐
│ 非极大值抑制                │
│ 删除重叠的框（IoU>0.65）  │
│ 从 ~300 个框 → ~50 个     │
└───────────┬───────────────┘


【输出：检测结果】
┌───────────────────────────┐
│ List of Detections:       │
│ [                         │
│   {                       │
│     "box": [100, 200,    │
│             120, 140],   │ [x1, y1, x2, y2] 像素坐标
│     "confidence": 0.92,   │
│     "class": "crosswalk", │
│     "class_id": 1         │
│   },                      │
│   ...                     │
│ ]                         │
│                           │
│ 总共：N 个框              │
│ 类别：8 种                │
└───────────────────────────┘
        ↓


【可视化】
在原始分辨率上：
  ├─ 绘制框（颜色表示类别）
  ├─ 写入标签和置信度
  └─ 保存为 output.jpg
```

### 数据集格式转换

```
【源数据格式】
data/merged_yolo_detect/
├── images/train/
│   ├─ image001.jpg
│   ├─ image002.jpg
│   └─ ...
└── labels/train/
    ├─ image001.txt     ← YOLO 格式标注
    ├─ image002.txt
    └─ ...

【YOLO 格式标注文件】
每行一个目标：
<class_id> <x_center> <y_center> <width> <height>

例如：
1 0.45 0.60 0.20 0.30  ← class 1 (crosswalk)，中心 (0.45, 0.60)
2 0.75 0.80 0.05 0.08  ← class 2 (cone)，小物体
坐标都是归一化的（0-1 范围）


【转换为专家子集】
build_expert_subset() 做的事：

1. 逐个遍历 train 和 val 图像
2. 读对应的标注文件
3. 检查是否包含目标类别：
   ├─ ground_expert: class_id in [blind_road, crosswalk]
   ├─ tiny_expert: class_id in [cone, marker, drum, barricade, barrier, fence]
4. 如果是，复制图像 + 标注文件到专家子目录
5. 生成新的 data.yaml（少于 8 类）

【专家子数据集结构】
artifacts/expert_subsets/ground_expert/
├── data.yaml
│   ├─ path: .../ground_expert
│   ├─ train: images/train
│   ├─ val: images/val
│   ├─ nc: 2  ← 只有 2 类
│   └─ names: ['blind_road', 'crosswalk']
│
├── images/
│   ├─ train/  ← 2500 张图（只有这两类的）
│   └─ val/    ← 800 张图
│
└── labels/
    ├─ train/  ← 对应的 .txt 标注
    └─ val/

好处：
✓ 模型只学这两类特征
✓ 不被其他 6 类干扰
✓ 最终 mAP 更高
```

---

## 显存与计算分配

### GPU 显存占用详解

```
【YOLO11m 的显存占用分析】

[基础占用]
├─ 模型参数
│  └─ 2000 万参数 × 4 字节 = 80 MB（这很小！）
│
├─ 激活值（中间层特征）
│  └─ 输入 (1×3×640×640) = 4.9 MB
│  └─ 中间特征图
│     ├─ 第1阶段特征 (1×128×160×160) = 100 MB
│     ├─ 第2阶段特征 (1×256×80×80) = 80 MB
│     └─ ...
│  └─ 小计：~800 MB - 1.2 GB（推理时）
│
└─ 训练特有占用
   ├─ 梯度缓存
   │  └─ 同样大小的张量用于梯度 = ~80 MB（参数梯度）+ 激活梯度
   │
   └─ 优化器状态（Adam）
      ├─ 一阶矩估计 (m) = 80 MB
      ├─ 二阶矩估计 (v) = 80 MB
      └─ 小计：~160 MB（除参数外）


【单张图推理显存】
输入：1 张 640×640 图
占用：
  ├─ 参数：80 MB（固定）
  ├─ 中间激活：1.0 GB（变化）
  └─ 输出：50 MB
  小计：~1.1 GB


【Batch 训练显存】
batch_size = 96（缺省配置）
每块 GPU：96 ÷ 3 = 32 张图

占用：
  ├─ 参数：80 MB（固定）
  ├─ 激活 (32张图)：1.0 GB × 32 = 32 GB ← 问题！
  ├─ 激活梯度：~30 GB
  ├─ 参数梯度：80 MB
  ├─ 优化器状态：160 MB
  └─ 小计：~62 GB/GPU
  
实际 GPU：24 GB
结果：23.14 GB 满载 + ~40 GB 需要卸出 → CUDA OOM！


【解决方案1：减少 batch size】
batch_size = 64 → 每块 GPU 21 张
占用：32 × (21/32) = 21 GB/GPU ✓ 可以放下！

batch_size = 48 → 每块 GPU 16 张
占用：32 × (16/32) = 16 GB/GPU ✓ 更宽松


【解决方案2：减少输入尺寸】
imgsz = 640 → 激活尺寸 640×640
imgsz = 512 → 激活尺寸 512×512

内存差异：(512/640)² = 0.64 倍
32 GB × 0.64 = 20.5 GB/GPU ✓ 刚好放下


【解决方案3：梯度积累】
不是每个 batch 就更新参数，而是积累几个 batch 再更新
伪 batch_size = 128 = 实际 32 × 4 个积累步数
但每步显存只用 32 的配置
权衡：速度更慢，但内存更少


【推荐配置】
┌──────────────────────────────────┐
│ 显存 24GB（RTX 3090）的推荐      │
├──────────────────────────────────┤
│ ✓ batch 96, imgsz 640            │ 满载但 OOM
│ ✓ batch 64, imgsz 640            │ 推荐：安全 ✓
│ ✓ batch 48, imgsz 512            │ 保守：更安全
│ ✓ batch 32, imgsz 640 (单GPU)   │ 单块很慢
│ ✓ batch 96, imgsz 512            │ 激进：可能 OOM 但试试
└──────────────────────────────────┘
```

### 计算时间分析

```
【推理延迟分解】（单张 640×640 图）

1. 预处理: ~1 ms
   ├─ 读取图像
   ├─ 尺寸调整
   ├─ 归一化
   └─ 格式转换

2. 基础模型前向: ~20 ms
   ├─ Backbone（特征提取）: 12 ms
   ├─ Neck（特征融合）: 5 ms
   └─ Head（预测）: 3 ms

3. 后处理: ~2 ms
   ├─ 阈值过滤
   ├─ NMS 去重
   └─ 格式转换

4. 条件执行（如果激活专家）
   ├─ IF 激活地面专家: +20 ms
   ├─ IF 激活微小物体专家: +20 ms
   └─ 结果融合: +2 ms

总计：
  ├─ 仅基础模型: ~25 ms
  ├─ 激活 1 个专家: ~45 ms
  ├─ 激活 2 个专家: ~65 ms
  └─ 平均（60% 激活率）: 25×0.4 + 45×0.6 = 37 ms


【训练速度分析】
batch_size = 96, imgsz = 640, 3 GPU DDP

单个 epoch（6537 训练图）：
  ├─ 总批次: 6537 ÷ 96 = 68 个
  ├─ 每批处理时间
  │  ├─ 数据加载: ~50 ms
  │  ├─ 前向推理: ~200 ms
  │  ├─ 损失计算: ~50 ms
  │  ├─ 反向传播: ~200 ms
  │  ├─ 参数更新: ~30 ms
  │  ├─ GPU 同步: ~20 ms (仅 DDP)
  │  └─ 小计: ~550 ms
  │
  ├─ 验证（2425 图）
  │  ├─ 总批次: 2425 ÷ 96 = 26 个
  │  ├─ 每批: ~300 ms（只前向）
  │  └─ 小计: 26 × 0.3 = 7.8 分钟
  │
  └─ 单 epoch 总时间: 68 × 0.55 + 7.8 = 45 分钟

30 个 epoch：45 分钟 × 30 = 22.5 小时 ← 很长！

但是（基础 + 2 个专家 并行）：
  ├─ 基础模型: 30 epoch × 45 min = 22.5 h
  ├─ 地面专家: 20 epoch × 40 min = 13.3 h（数据少，更快）
  ├─ 微小物体专家: 20 epoch × 40 min = 13.3 h
  
如果硬件允许：22.5 h（瓶颈是基础模型）
如果顺序执行：49.1 h

实际：在演示中我们指定 --device 1,2,3，每块 GPU 各自训一个模型
所以真实时间：~50 分钟（如果没有 OOM）
```

---

## 优化决策树

### 遇到问题时的诊断和解决流程

```
═══════════════════════════════════════════════════════════════

症状：CUDA out of memory

    ├─ [排查1] 你用的 GPU 有多大？
    │  ├─ < 8GB → 只能用 batch 8-16, imgsz 416
    │  ├─ 8-12GB → batch 16-32, imgsz 512
    │  ├─ 12-24GB → batch 48-64, imgsz 640 ✓ 推荐
    │  └─ > 24GB → batch 96+
    │
    ├─ [排查2] 当前 batch 和 imgsz 是什么？
    │  ├─ 如果 batch >= 96 AND imgsz >= 640
    │  │  └─ 改: batch 48, imgsz 512
    │  ├─ 如果 batch >= 64 AND imgsz >= 640
    │  │  └─ 改: batch 48 或 imgsz 512
    │  └─ 如果已经很小了
    │     └─ 可能是代码 bug，检查梯度积累
    │
    └─ [排查3] 是否用了多 GPU？
       ├─ 如果 --device 0,1,2（3 块 GPU）
       │  └─ 尝试减到 batch 64（每块 GPU 21 张）
       ├─ 如果只有 1 块 GPU
       │  └─ 必须大幅减小 batch
       └─ 如果没用 DDP 正确配置
          └─ 需要 `torch.nn.parallel.DistributedDataParallel`


═══════════════════════════════════════════════════════════════

症状：mAP 一直不高（< 0.4）

    ├─ [检查1] 数据质量
    │  ├─ 标注正确吗？ → 采样 10 张图手动看
    │  ├─ 类别平衡吗？ → 统计各类别样本数
    │  └─ 数据量足够？ → YOLO 通常需要 500+ 图/类
    │
    ├─ [检查2] 训练配置
    │  ├─ epoch 数够吗？ → 至少 30-50
    │  │  └─ 看 val 曲线（loss 是否还在降？）
    │  ├─ 学习率对吗？ → 看损失曲线（平稳降低？）
    │  │  └─ 如果抖动：lr 太大
    │  │  └─ 如果平坦：lr 太小
    │  └─ 数据增强够吗？ → YOLO 默认有，可看 config
    │
    ├─ [检查3] 模型
    │  ├─ 模型大小对吗？
    │  │  ├─ YOLO11n（nano） ← 快但准度低
    │  │  ├─ YOLO11m（medium） ← 均衡 ✓
    │  │  ├─ YOLO11l（large）← 准但慢
    │  │  └─ YOLO11x（xlarge）← 最准但很慢
    │  └─ 预训练权重对吗？ → 应该用 coco 预训练
    │
    └─ [行动计划]
       1. 首先加数据（最有效）
       2. 其次改模型大小
       3. 再调参数
       4. 最后考虑集成或专用方法


═══════════════════════════════════════════════════════════════

症状：专家模型性能甚至更差

    ├─ [原因1] 专家数据太少
    │  └─ 影响：容易过拟合
    │  └─ 表现：训练集 mAP 高，验证集 mAP 低
    │  └─ 解决：
    │      ├─ 减少 epoch（从 20 改到 10）
    │      ├─ 增加数据增强
    │      └─ 提前停止（val loss 不再下降时停）
    │
    ├─ [原因2] 子数据集划分有问题
    │  └─ 影响：train/val 分布不一致
    │  └─ 检查：
    │      ├─ 是否正确生成了 expert_subsets?
    │      ├─ train/val 比例是否保持 (~70/30)?
    │      └─ 是否有类别不出现在 train 但出现在 val？
    │
    └─ [原因3] 路由规则激活频率不对
       └─ 即使专家模型好，如果激活频率低，没用
       └─ 检查 trigger_rate（应该 40-70%）
       └─ 修改 router.py 的阈值


═══════════════════════════════════════════════════════════════

症状：某个特定类别 mAP 很低

    └─ [诊断流程]
       ├─ [步骤1] 看数据分布
       │  └─ 这个类别有多少样本？
       │  └─ 样本尺寸分布？（都很小？都很大？）
       │
       ├─ [步骤2] 看标注质量
       │  └─ 采样 20 张，人眼检查标注是否正确
       │  └─ 是否有漏掉的目标对象？
       │
       ├─ [步骤3] 看预测结果
       │  └─ 模型误报多吗？ → Precision 低
       │  └─ 模型漏掉多吗？ → Recall 低
       │
       └─ [改进方案]
          ├─ 如果只有少样本类
          │  └─ 添加类似图像做数据增强
          │  └─ 考虑用专家模型（只检测这类）
          ├─ 如果尺寸太小
          │  └─ 增加 imgsz (640 → 832)
          ├─ 如果标注不对
          │  └─ 重新标注
          └─ 如果只是模型没学好
             └─ 增加权重（focal loss）

```

---

## 快速参考：从原始数据到最终评估

```
【完整工作流时间表示】

Day 1 下午：
  准备 → python train_moe_cascade.py --mode all --device ...
  └─ 等待 ~1 小时

同天晚上或 Day 2：
  训练完毕 → python evaluate_moe_v1.py ...
  └─ 等待 ~10 分钟

Day 2 上午：
  查看报告 → JSON 结果分析
  └─ 基于结果改进参数

Day 2-3：
  (可选) 继续迭代训练和评估
```

制作这份文档的目的是帮你理解"黑箱"内部发生了什么。记住，所有复杂的东西最后都是加法：一堆矩阵乘法、损失函数、梯度下降。
