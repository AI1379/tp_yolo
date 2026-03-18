# MoEYOLO 级联多专家模型完整指南

## 目录
1. [项目概述](#项目概述)
2. [核心概念讲解](#核心概念讲解)
3. [模型架构详解](#模型架构详解)
4. [设计思想](#设计思想)
5. [模型使用流程](#模型使用流程)
6. [训练详细说明](#训练详细说明)
7. [评估脚本原理](#评估脚本原理)
8. [参数含义详解](#参数含义详解)
9. [常见问题](#常见问题)

---

## 项目概述

### 这个项目在做什么？

我们要训练一个**道路工作检测模型**。任务是：给定一张道路照片，找出其中的各种工作对象，例如：
- 锥形物体（cone）
- 盲道标志（blind_road）
- 人行横道（crosswalk）
- 栅栏（fence）
- 其他工作相关标志

### 为什么用 MoEYOLO?

直接用一个 YOLO 模型检测所有 8 种对象的问题：
1. **难以同时优化**：有些对象很大（crosswalk），有些很小（cone、tubular_marker）
2. **干扰**：当有很多微小物体时，模型预测大对象会变差
3. **计算浪费**：不是所有图片都需要用全部模型容量

**MoEYOLO 的解决方案**：使用"级联多专家"架构
- **基础模型**：快速检测主要对象（8 类）
- **专家模型**：在特定条件下激活，优化特定难题
  - **地面专家**：专门处理 crosswalk 和 blind_road 这两个难的大对象
  - **微小物体专家**：专门处理 cone、tubular_marker 等小对象

---

## 核心概念讲解

### 1. 什么是目标检测（Object Detection）？

给定一张图像，模型需要输出：
- **位置**：物体在图中的位置（通常用矩形框表示）
- **类别**：这是什么物体（例如 "cone" 或 "crosswalk"）
- **置信度**：模型对这个预测有多确定（0-1，1 表示 100% 确定）

```
输入图像：
┌─────────────────────┐
│  ┌─────────┐        │  (锥形物体)
│  │ cone    │        │  置信度: 0.95
│  └─────────┘        │
│     ┌──────────────┐ │  (人行横道)
│     │ crosswalk    │ │  置信度: 0.87
│     └──────────────┘ │
└─────────────────────┘
```

### 2. YOLO 是什么？

YOLO（You Only Look Once）是一种实时目标检测算法。"一次性看"意思是：
- 模型只需要看一遍图像就能做出所有预测（相比其他方法需要多次扫描）
- 速度快，能实时处理视频或直播

我们用的是 **YOLO11m**（11 是版本号，m 是 medium 中等大小）。

### 3. 什么是"级联"？

级联（Cascade）意思是**按顺序处理**，像多个守卫一样：

```
输入图像
   ↓
[守卫1: 基础模型] → 做初步检测
   ↓
[守卫2: 决策路由] → 根据图像特点决定激活哪个专家
   ↓
[守卫3: 专家模型们] → 如果条件满足，运行专家进行细化
   ↓
输出最终检测结果
```

### 4. 什么是"多专家"（MoE = Mixture of Experts）？

不同的专家模型各有所长，像医疗团队：
- **全科医生**（基础模型）：能处理常见问题
- **心脏专家**（地面专家）：特别擅长处理 crosswalk、blind_road
- **眼科专家**（微小物体专家）：特别擅长处理微小物体

这样设计的好处：
- **准确率更高**：专家在自己的领域更准
- **计算更高效**：不是所有图都需要运行所有模型
- **可扩展性好**：可以轻松添加新的专家

---

## 模型架构详解

### 总体架构流程图

```
输入图像（H×W×3）
        ↓
┌─────────────────────────────────────┐
│   基础模型（YOLO11m）                 │
│   - 输入：640×640 图像                │
│   - 输出：检测框、类别、置信度          │
│   - 类别数：8 种                     │
│   - 配置：yolo11m.pt                 │
└─────────────────────────────────────┘
        ↓
    决策路由（RuleBasedRouter）
    ↓           ↓
条件1: 检测到的条件2:
小物体太多？ 检测框太少？
   ↓              ↓
激活          激活
地面专家  微小物体专家
   ↓              ↓
┌──────────────┐  ┌──────────────────────┐
│ 地面专家     │  │ 微小物体专家         │
│ - 2 种类别   │  │ - 6 种类别           │
│ - 优化crosswalk│ - 优化 cone 等       │
└──────────────┘  └──────────────────────┘
        ↓              ↓
    融合所有检测框
        ↓
输出：最终的检测结果（融合后）
```

### 模型细节

#### 基础模型
```python
名称: YOLO11m
结构: 
- 输入：640×640 的 RGB 图像
- 输出：(N, 6) 张量
  其中 N 是检测到的框数，6 表示 [x_center, y_center, width, height, confidence, class_id]

参数量: ~20M（medium 尺寸）
速度: ~30-50ms(单张)
训练数据: merged_yolo_detect, 8 种类别
- blind_road（盲道）
- crosswalk（人行横道）
- cone（锥形物体）
- tubular_marker（管状标记）
- drum（圆桶）
- barricade（路障）
- barrier（屏障）
- fence（栅栏）
```

#### 地面专家（ground_expert）
```python
名称: YOLO11m (for ground_expert)
特化: 专门检测 2 种类别
- blind_road
- crosswalk

训练数据: 从合并数据集中过滤出只包含这两类的图像
何时激活: 
  1. 基础模型检测到的框数太少（< 3 个）
     原因：这两个对象很大，图中应该有明显目标
  2. 帮助基础模型提高这两类的准确率

预期效果: 更高的准确率（通过专注减少干扰）
```

#### 微小物体专家（tiny_obstacle_expert）
```python
名称: YOLO11m (for tiny_obstacle_expert)
特化: 专门检测 6 种微小类别
- cone
- tubular_marker
- drum
- barricade
- barrier
- fence

训练数据: 从合并数据集中过滤出包含这些微小类别的图像
何时激活:
  1. 基础模型检测到的小物体太多（> 5 个且大多是微小类）
     原因：需要更好地处理这些复杂场景
  2. 帮助基础模型在微小物体上分类

预期效果: 对微小物体的更精确分类
```

### 决策路由规则（RuleBasedRouter）

路由器使用以下规则决定是否激活专家：

```python
def should_trigger_experts(detections_from_base_model):
    """
    判断是否需要激活专家模型
    """
    # 规则1：检测框太少 → 激活地面专家
    if len(detections) < 3:
        return "activate_ground_expert"
    
    # 规则2：很多微小物体 → 激活微小物体专家
    num_small_objects = count_objects_with_area_less_than_1000_pixels()
    if num_small_objects > 5:
        return "activate_tiny_expert"
    
    # 默认：不激活
    return "use_base_only"
```

---

## 设计思想

### 为什么这样设计？

#### 1. 为什么用基础模型 + 专家而不是一个大模型？

**对比：**

| 方案 | 优点 | 缺点 |
|------|------|------|
| 一个大 YOLO 模型 | 简单，统一 | 速度慢，对所有类别同样的优化，容易互相干扰 |
| 级联多专家 | 快速（条件搜索），准确（特化优化），省计算 | 稍复杂，需要路由决策 |

**性能对比假设：**
```
一个模型检测所有8类:
- 平均延迟: 50ms
- mAP50-95: 0.50
- 每张图都需要大计算

级联多专家:
- 基础模型: 30ms（所有图）
- 60% 的图激活专家: +20-30ms
- 平均延迟: 30 + 0.6×25 = 45ms（还快了！）
- mAP50-95: 0.58（准确率高了！）
```

#### 2. 为什么选择这 2 个专家？

**分析过程：**

观察数据集，发现有两类问题：
1. **大对象检测困难**：crosswalk（人行横道）很大很明显，但基础模型容易漏掉
   - 原因：训练时这个类别样本比例小
   - 解决：用一个专门的模型，数据集中 100% 是这两类，强制模型学

2. **小对象检测混乱**：cone、tubular_marker 等很小，容易互相混淆
   - 原因：小物体难以识别特征
   - 解决：用一个专门的微小物体检测模型，减少大对象的干扰

#### 3. 为什么这样激活专家？

**路由规则的逻辑：**

```
规则1：if 检测框 < 3 个 → 激活地面专家
理由：
- 道路工作场景，通常至少有几个明显的对象
- 如果检测很少，可能漏掉了大类别对象（crosswalk/blind_road）
- 送给地面专家，让它重新看一遍

规则2：if 小物体很多 → 激活微小物体专家
理由：
- 基础模型在微小物体上效果差
- 当图中微小物体很多（> 5个），说明是复杂场景
- 需要专家来细化这些微小物体的分类
```

---

## 模型使用流程

### 快速开始

#### 方式1：使用评估脚本（推荐新手）

```bash
cd MoEYOLO
python evaluate_moe_v1.py \
  --base-ckpt runs/detect/runs/moeyolo/moe_v1_base/weights/best.pt \
  --ground-expert-ckpt runs/detect/runs/moeyolo/moe_v1_ground_expert/weights/best.pt \
  --tiny-expert-ckpt runs/detect/runs/moeyolo/moe_v1_tiny_obstacle_expert/weights/best.pt \
  --device 1 \
  --batch 32 \
  --report artifacts/reports/my_eval.json
```

**输出：** `artifacts/reports/my_eval.json`，包含所有模型的 mAP、precision、recall 等指标

#### 方式2：直接在代码中使用

```python
from moe_yolo.cascade import CascadeMoEPipeline

# 初始化流程
pipeline = CascadeMoEPipeline(
    base_model_path="path/to/base/best.pt",
    ground_expert_path="path/to/ground_expert/best.pt",
    tiny_expert_path="path/to/tiny_expert/best.pt",
    device=0  # 使用哪块 GPU
)

# 在一张图上运行
image_path = "test_image.jpg"
results = pipeline.predict(image_path, conf=0.5)

# results 包含：
# {
#   "base_detections": [...],
#   "expert_detections": [...],
#   "final_detections": [...],  # 融合后的
#   "triggered_experts": ["ground_expert"],
#   "processing_time_ms": 45.2
# }

# 可视化
results.visualize_and_save("output.jpg")
```

### 完整工作流

```
1. 准备数据
   └─ 放在 data/merged_yolo_detect/

2. 训练
   └─ python train_moe_cascade.py --mode all --device 1,2,3

3. 评估
   └─ python evaluate_moe_v1.py [参数] --report results.json

4. 分析结果
   └─ 查看 results.json 中的指标
```

---

## 训练详细说明

### 训练流程

```
启动训练脚本
    ↓
[第1步] 构建专家子数据集
    - 从 8 类数据集中筛选出 2 类数据（地面专家）
    - 从 8 类数据集中筛选出 6 类数据（微小物体专家）
    - 生成对应的 data.yaml 文件
    ↓
[第2步] 训练基础模型
    - 输入：full 8 类数据
    - 输出：moe_v1_base/weights/best.pt
    - 时间：~20-30 分钟（30 个 epoch）
    ↓
[第3步] 训练地面专家
    - 输入：2 类子数据集
    - 输出：moe_v1_ground_expert/weights/best.pt
    - 时间：~10-15 分钟（20 个 epoch）
    ↓
[第4步] 训练微小物体专家  
    - 输入：6 类子数据集
    - 输出：moe_v1_tiny_obstacle_expert/weights/best.pt
    - 时间：~10-15 分钟（20 个 epoch）
    ↓
完成！生成 3 个训练好的模型
```

### 训练参数详解

```bash
python train_moe_cascade.py \
  --mode all \                      # 训练模式：base/experts/all
  --device 1,2,3 \                  # 使用哪块 GPU（用逗号分隔）
  --batch 96 \                      # 批大小：每次处理 96 张图
  --imgsz 640 \                     # 输入图像尺寸：640×640
  --epochs-base 30 \                # 基础模型训练轮数
  --epochs-expert 20 \              # 专家模型训练轮数
  --name-prefix moe_v1 \            # 结果文件夹前缀
  --resume path/to/checkpoint.pt    # 恢复中断的训练
```

**参数含义：**

| 参数 | 含义 | 影响 |
|------|------|------|
| `mode` | 训练什么模型 | `base` 只训基础模型；`experts` 只训专家；`all` 全部 |
| `device` | GPU 编号 | `1,2,3` 使用 3 块 GPU（多 GPU 分布式训练），单块用 `0` |
| `batch` | 批大小 | 越大速度快但需要更多显存；96用完 24GB，改成 48/64 可能稳定 |
| `imgsz` | 输入尺寸 | 640 检测效果好但显存多；512 更省显存；更小的物体需要更大的尺寸 |
| `epochs-base` | 基础轮数 | 通常 20-50，越多越可能过拟合；30 平衡点 |
| `epochs-expert` | 专家轮数 | 通常 15-30，数据少所以轮数可以少 |

### 数据准备

```
data/
  merged_yolo_detect/            # 原始完整数据
    data.yaml                    # 8 类定义
    images/
      train/                     # 6537 张训练图
      val/                       # 2425 张验证图
    labels/
      train/                     # YOLO 格式标注
      val/

[训练脚本自动生成]
  expert_subsets/
    ground_expert/
      data.yaml                  # 2 类定义（blind_road, crosswalk）
      images/
        train/                   # 过滤后的训练图
        val/
      labels/
        train/                   # 过滤后的标注
        val/
    tiny_obstacle_expert/
      data.yaml                  # 6 类定义
      ... [同样结构]
```

### 显存占用分析

**当前设置 (batch=96, imgsz=640, device 1,2,3)：**

```
每块 GPU 的分配：
- 批大小：96 / 3 = 32 张/GPU
- 每张图内存：3×640×640×4 bytes = 4.9 MB（输入）
- 模型参数：~20M 参数 × 4 bytes = 80 MB
- 中间特征图：~3-4 GB（推理时最大）
- 梯度缓存：~等于模型大小 = 80 MB
- 优化器状态：~等于模型大小 = 80 MB
- 总计：~3.5-4.5 GB/GPU

问题分析：
✓ 要求：24 GB GPU
✓ 现在：用满 23+ GB
✗ 结果：CUDA OOM（显存溢出）
```

**解决方案：**

```python
# 方案1：减少批大小（推荐）
--batch 48         # 每 GPU 16 张，总共 48 张，显存需求 ~12 GB/GPU
--batch 64         # 每 GPU 21 张，总共 64 张，显存需求 ~16 GB/GPU

# 方案2：减少输入尺寸
--imgsz 512        # 减小输入图尺寸，显存减少 (640/512)^2 = 1.57x
                   # 512 仍能检测中等物体，对微小物体有影响

# 方案3：组合
--batch 64 --imgsz 512

# 方案4：单 GPU 训练（慢但稳定）
--device 0 --batch 32
```

---

## 评估脚本原理

### 评估脚本做什么？

评估脚本的目的是**自动计算模型性能指标**，回答这些问题：
- 基础模型准不准？
- 专家模型准不准？
- 级联pipeline 工作好不好？

### 评估流程

```
输入：3 个 checkpoint(.pt 文件)
    ↓
[步骤1] 加载模型
    - 加载基础 YOLO11m
    - 加载地面专家 YOLO11m
    - 加载微小物体专家 YOLO11m
    - 加载验证集
    ↓
[步骤2] 基础模型评估
    - 在验证集上测试基础模型
    - 计算：precision, recall, mAP50, mAP50-95, 等等
    - 输出：每个类别的准确率
    ↓
[步骤3] 专家模型评估
    - 在专家子数据集上测试每个专家
    - 地面专家：只在 2 类上计算指标
    - 微小物体专家：只在 6 类上计算指标
    ↓
[步骤4] 级联 pipeline 评估
    - 加载完整的 CascadeMoEPipeline
    - 在样本图像上运行 pipeline
    - 记录：
      * 有多少图激活了专家
      * 运行时间
      * 检测结果
    ↓
[步骤5] 生成报告
    - 输出 JSON 文件
    - 包含所有指标
    - 便于对比和分析
```

### 关键概念：精度指标

#### Precision（精准率）

```python
定义：预测对了 / 所有预测
意义：模型说"这是锥形物体"的话，有多少百分比真的是锥形物体

例子：
预测 100 个"锥形物体"
其中 85 个真的是锥形物体
其中 15 个是其他东西（误报）
Precision = 85/100 = 0.85 = 85%

含义：模型不容易过度预测
```

#### Recall（召回率）

```python
定义：预测对了 / 所有真实的
意义：图里真的有的锥形物体，有多少百分比被模型找到

例子：
图里真的有 100 个锥形物体
模型找到了 78 个
模型漏掉了 22 个
Recall = 78/100 = 0.78 = 78%

含义：模型不容易漏掉东西
```

#### mAP（平均精度）

```python
定义：综合考虑 Precision 和 Recall 的指标
mAP50：IoU 阈值 = 0.5 时的平均精度
mAP50-95：IoU 阈值从 0.5 到 0.95 的平均值（更严格）

通俗解释：
- mAP50 粗糙定义"对不对"：预测框和真实框重叠 50% 以上就算对
- mAP50-95 精细定义"对不对"：要求重叠从 50% 逐步到 95%

一般来说 mAP50-95 < mAP50，因为要求更严格
```

### 级联性能指标

```json
"cascade_runtime": {
    "num_images": 5,                    # 测试了多少张图
    "trigger_rate": 0.6,               # 有 60% 的图激活了专家
    "avg_detections_per_image": 4.8,   # 平均每张图检测 4.8 个对象
    "latency_ms_avg": 241.88,          # 平均处理时间 241ms
    "latency_ms_p95": 1080.95,         # 95% 的图在 1080ms 内完成
    "trigger_reason_counts": {
        "many_tiny_objects": 2,         # 2 张图因为小物体太多激活
        "too_few_boxes": 1              # 1 张图因为检测框太少激活
    }
}
```

**这些指标的含义：**

| 指标 | 含义 | 期望值 |
|------|------|--------|
| trigger_rate | 有多少图需要专家 | 40-70% 合理（太少说明基础模型失效；太多说明路由太敏感） |
| latency_avg | 平均处理速度 | < 300ms 可以实时 |
| latency_p95 | 95% 的图都能在多快处理完 | 有些复杂图可以慢，但 p95 很重要 |
| avg_detections | 平均检测数 | 根据场景定；太少可能漏掉，太多可能误报 |

---

## 参数含义详解

### 训练参数

```python
# 数据相关
--batch 96          # 批大小
                    # 含义：每次用多少张图一起训练
                    # 影响：更大 = 更稳定但更慢；更小 = 更快但更不稳定
                    # 单位：张图
                    # 建议：使用 GPU 显存 60-80%

--imgsz 640         # 输入图像尺寸
                    # 含义：将所有图放大/缩小到 640×640
                    # 影响：更大 = 能检测更小的物体但更慢；更小 = 快但检测大物体时退化
                    # 典型值：[320, 416, 512, 640, 896]
                    # 建议：首选 640；内存不足用 512

--epochs-base 30    # 基础模型训练多少轮
                    # 含义：把整个数据集看 30 遍
                    # 影响：更多轮次 = 可能过拟合；更少轮次 = 可能欠拟合
                    # 建议：开始用 30；根据验证集指标调整

--epochs-expert 20  # 专家模型训练多少轮
                    # 含义：把专家数据集看 20 遍
                    # 影响：同上
                    # 建议：< epochs-base，因为数据少

# 硬件相关
--device 1,2,3      # 使用哪块 GPU
                    # 含义：CUDA_VISIBLE_DEVICES 编号
                    # 例子：--device 0,1 用第一和第二块 GPU
                    #       --device 0 只用第一块 GPU
                    # 建议：使用多块 GPU 加速（DDP 分布式）

# 模式相关
--mode all          # 训练什么
                    # "base"：只训基础模型
                    # "experts"：只训 2 个专家模型
                    # "all"：全部（基础 → 专家）
                    # 建议：完整流程用 "all"

--name-prefix moe_v1  # 结果文件夹前缀
                      # 会生成：moe_v1_base, moe_v1_ground_expert 等
```

### 评估参数

```python
# Checkpoint 相关
--base-ckpt path/to/best.pt        # 基础模型权重文件路径
--ground-expert-ckpt path/to/...   # 地面专家权重文件路径
--tiny-expert-ckpt path/to/...     # 微小物体专家权重文件路径
                                    # 这些是 .pt 文件，包含了训练好的参数

# 数据相关
--batch 32                          # 评估时的批大小
                                    # 评估时用更小的值（如 32-64）通常更稳定
                                    # 可以小于训练时的 batch，不影响结果

--sample-images 100                 # 级联评估时样本多少张图
                                    # 用来测试 pipeline 的实际运行效果
                                    # 更多 = 更准但更慢

# 输出相关
--report artifacts/reports/my_eval.json  # 输出 JSON 文件位置
                                         # 包含所有指标和分析结果

# 硬件相关
--device 1                          # 评估用哪块 GPU
                                    # 通常用 1 块就够，可以是训练设备外的 GPU
```

### 指标参数详解

这些指标由 Ultralytics 自动计算，你在 JSON 输出中会看到：

```json
{
  "base": {
    "precision": 0.739,           # 精准率（0-1，越高越好）
    "recall": 0.648,              # 召回率（0-1，越高越好）
    "map50": 0.675,               # mAP @ IoU=0.50
    "map50_95": 0.460,            # mAP @ IoU=0.50:0.95（最常用的指标）
    "per_class_map50_95": {
      "blind_road": 0.744,        # 每个类别的专项指标
      "crosswalk": 0.801,
      ...
    }
  }
}
```

**衡量什么？**

| 指标 | 衡量方面 | 改进方法 |
|------|---------|---------|
| precision | 误报率（假阳性） | 增加数据，调整置信度阈值 |
| recall | 漏检率（假阴性） | 增加数据，提高模型容量 |
| mAP50-95 | 综合准确率 | 是最重要的指标 |
| per_class mAP | 特定类别准确率 | 针对性改进差的类别 |

---

## 常见问题

### Q1：为什么基础模型精准率 0.739，我觉得不够好？

**A：** 这是正常的。解释：
- YOLO 在 COCO 数据集（1000+ 类）上 mAP 约 0.50
- 我们有 8 类自定义数据，0.74 precision 很不错
- mAP50-95 = 0.46 说明小物体检测还有改进空间
- 可以尝试：
  1. 增加训练数据
  2. 增加模型大小（换 YOLO11l 或更大）
  3. 调整训练参数（学习率、数据增强）

### Q2：为什么专家的性能还不如基础模型？

**A：** 这取决于数据分布：
- 地面专家 mAP=0.68（比基础模型好，这很好！）
- 微小物体专家 mAP=0.36（比基础模型差，这很正常）

原因：
- 微小物体本来就难检测（整个视觉领域都承认这点）
- 微小物体专家的任务是分类这些微小物体，不是检测它们
- 可以改进：使用更高分辨率（--imgsz 832）或更大的模型

### Q3：OOM（显存溢出）怎么办？

**A：** 按优先级尝试：
```bash
# 1. 减少批大小（最有效）
python train_moe_cascade.py --batch 48 --imgsz 640  # 代替 --batch 96

# 2. 减少输入尺寸
python train_moe_cascade.py --batch 96 --imgsz 512

# 3. 组合
python train_moe_cascade.py --batch 64 --imgsz 512

# 4. 单 GPU
python train_moe_cascade.py --device 0 --batch 32 --imgsz 640
```

### Q4：如何改进检测性能？

**排序优先级：**

```
1. [最有效] 增加数据
   - 更多标注图像 > 更多参数
   
2. 数据增强
   - 混合、旋转、亮度调整等
   - YOLO 自带，检查 augmentation 配置
   
3. 增加模型大小
   - YOLO11m → YOLO11l → YOLO11x
   - 需要更多显存，速度变慢
   
4. 调整超参数
   - 学习率、weight decay、warmup
   - 效果可能 ±2-3% mAP
   
5. 后处理
   - NMS 阈值调整
   - 置信度阈值调整
```

### Q5：级联的 trigger_rate 为什么这么高/低？

**说明什么：**
```
trigger_rate 过高（>80%）:
  - 说明路由规则太敏感
  - 基础模型效果不够好
  - 改进：调整路由规则，或改进基础模型

trigger_rate 过低（<20%）:
  - 说明基础模型已经很好
  - 专家模型用处有限
  - 可能是好现象，性能已经够好
```

### Q6：能同时运行多个训练吗？

**A：** 可以，但要指定不同的 GPU：
```bash
# Terminal 1：训练 v1 版本
python train_moe_cascade.py --device 0 --name-prefix v1

# Terminal 2：训练 v2 版本（用不同 GPU）
python train_moe_cascade.py --device 1 --name-prefix v2

# 不能用：--device 0,1 （会竞争）
```

---

## 总结

### 快速回顾

| 方面 | 要点 |
|------|------|
| **架构** | 基础模型 + 2 个专家 + 路由器 |
| **为什么** | 快速、准确、计算高效 |
| **主要优势** | 对特定问题（大对象、小对象）有针对性 |
| **关键指标** | mAP50-95（最重要）、precision、recall |
| **常见问题** | OOM 用减小 batch；性能不够加数据 |
| **评估方法** | 用 evaluate_moe_v1.py 自动生成 JSON 报告 |

### 学习路径建议

```
初级（理解概念）:
1. 读这份文档第 1-3 节
2. 理解什么是检测、什么是级联
3. 跑一遍评估脚本看结果

中级（能改参数）:
1. 尝试改变 --batch, --imgsz
2. 观察 JSON 输出中的指标变化
3. 做一个小规模的调试训练（少轮数）

高级（能改代码）:
1. 修改路由规则（moe_yolo/router.py）
2. 添加第三个专家
3. 尝试不同的模型聚合方法
```

### 进阶阅读

如果你想深入学习：
- YOLO 论文：https://arxiv.org/abs/1612.08242
- Mixture of Experts 论文：https://arxiv.org/abs/1701.06538
- Ultralytics 官方文档：https://docs.ultralytics.com/

---

**问题？** 回到相关章节查找，或再次运行评估脚本查看实际数字。
