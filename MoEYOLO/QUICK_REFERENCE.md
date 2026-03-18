# MoEYOLO 快速参考与速查表

## 📋 快速参考卡片

### 模型结构一句话总结
```
基础模型（8类通用） → 路由决策 → 专家模型优化
   ↓
  快速初筛        ↓          ↓ 精细化
  30-50ms   根据图像特点   +20-30ms
           激活专家       （条件触发）
```

### 三个模型用途

| 模型 | 输入 | 输出 | 何时用 |
|------|------|------|--------|
| 基础模型 | 任意图 | 8类检测 | 快速筛选 |
| 地面专家 | 有crosswalk/blind_road的图 | 2类精准检测 | 大对象识别差时 |
| 微小物体专家 | 微小物体很多的图 | 6类分类 | 小物体混乱时 |

---

## 🚀 常用命令速查

### 1. 训练命令

```bash
# 完整训练（推荐新手）
cd MoEYOLO
python train_moe_cascade.py \
  --mode all \
  --device 1,2,3 \
  --batch 96 \
  --imgsz 640 \
  --epochs-base 30 \
  --epochs-expert 20 \
  --name-prefix moe_v1

# 如果 OOM（显存溢出）
python train_moe_cascade.py \
  --mode all \
  --device 1,2,3 \
  --batch 48 \
  --imgsz 512 \
  --epochs-base 30 \
  --epochs-expert 20 \
  --name-prefix moe_v1_reduced

# 单 GPU 训练（慢但稳定）
python train_moe_cascade.py \
  --mode all \
  --device 0 \
  --batch 32 \
  --imgsz 640 \
  --epochs-base 30 \
  --epochs-expert 20 \
  --name-prefix moe_v1_single_gpu

# 只训基础模型（快速测试）
python train_moe_cascade.py \
  --mode base \
  --device 1,2,3 \
  --batch 96 \
  --imgsz 640 \
  --epochs-base 5 \
  --name-prefix smoke_test

# 恢复中断的训练
python train_moe_cascade.py \
  --mode all \
  --resume runs/detect/runs/moeyolo/moe_v1_base/weights/last.pt
```

### 2. 评估命令

```bash
# 基础评估（推荐开始用这个）
cd MoEYOLO
python evaluate_moe_v1.py \
  --base-ckpt runs/detect/runs/moeyolo/moe_v1_base/weights/best.pt \
  --ground-expert-ckpt runs/detect/runs/moeyolo/moe_v1_ground_expert/weights/best.pt \
  --tiny-expert-ckpt runs/detect/runs/moeyolo/moe_v1_tiny_obstacle_expert/weights/best.pt \
  --device 1 \
  --batch 32 \
  --report artifacts/reports/eval_v1.json

# 快速测试（少样本）
python evaluate_moe_v1.py \
  --base-ckpt ... \
  --ground-expert-ckpt ... \
  --tiny-expert-ckpt ... \
  --device 1 \
  --batch 32 \
  --sample-images 5 \
  --report artifacts/reports/smoke_eval.json

# 完整评估（所有验证数据）
python evaluate_moe_v1.py \
  --base-ckpt ... \
  --ground-expert-ckpt ... \
  --tiny-expert-ckpt ... \
  --device 1 \
  --batch 64 \
  --sample-images 300 \
  --report artifacts/reports/full_eval.json
```

### 3. 推理命令（在代码中）

```python
from moe_yolo.cascade import CascadeMoEPipeline
from PIL import Image

# 初始化
pipeline = CascadeMoEPipeline(
    base_model_path="runs/detect/runs/moeyolo/moe_v1_base/weights/best.pt",
    ground_expert_path="runs/detect/runs/moeyolo/moe_v1_ground_expert/weights/best.pt",
    tiny_expert_path="runs/detect/runs/moeyolo/moe_v1_tiny_obstacle_expert/weights/best.pt",
    device=0  # 使用 GPU 0
)

# 推理
image = Image.open("test.jpg")
results = pipeline.predict(image, conf=0.5)  # conf=0.5 表示置信度阈值

# 查看结果
print(f"总检测数: {len(results['final_detections'])}")
print(f"激活专家: {results['triggered_experts']}")
print(f"处理时间: {results['processing_time_ms']:.1f}ms")

# 绘制和保存
results.visualize().save("output.jpg")
```

---

## 📊 指标速查表

### 关键指标含义

| 指标 | 范围 | 含义 | 怎样算很好 |
|------|------|------|----------|
| **Precision** | 0-1 | 模型说对的概率 | > 0.8 |
| **Recall** | 0-1 | 能找到的比例 | > 0.8 |
| **mAP50** | 0-1 | 宽松的精度指标 | > 0.7 |
| **mAP50-95** | 0-1 | 严格的精度指标（**最重要**） | > 0.5 |
| **trigger_rate** | 0-1 | 多少图激活专家 | 0.4-0.7 理想 |
| **latency_ms** | ms | 处理一张图用时 | < 300ms 可实时 |

### 从指标反推问题

```
指标低 ← 可能原因 → 解决方案

mAP50-95 < 0.4
  ← 数据不足或不够好 → 增加数据或改进标注
  ← 模型太小 → 换 YOLO11l/x
  ← 参数选择不对 → 尝试不同的 batch/imgsz

Recall 很低（如 0.5）但 Precision 中等（如 0.8）
  ← 漏检太多 → 增加数据，或用更大模型
  ← 置信度阈值太高 → 降低 conf 参数

Precision 很低但 Recall 高
  ← 误报太多 → 增加数据，或提高置信度
  ← 模型容量不足 → 换更大模型

trigger_rate 很低（< 0.2）
  ← 基础模型已经很好 → 不用改，性能已到位
  ← 路由规则不适合 → 修改 router.py

trigger_rate 很高（> 0.9）
  ← 基础模型太差 → 改进基础模型
  ← 路由规则太敏感 → 调整阈值
```

---

## 🔍 文件和对应关系

### 最重要的文件

```
MoEYOLO/
├── train_moe_cascade.py           # ← 用这个训练
├── evaluate_moe_v1.py             # ← 用这个评估
├── moe_yolo/
│   ├── cascade.py                 # 级联 pipeline（推理逻辑）
│   ├── router.py                  # 路由规则（什么时候激活专家）
│   ├── config.py                  # 配置文件
│   └── train_router_stub.py        # 训练辅助
├── runs/detect/runs/moeyolo/
│   ├── moe_v1_base/weights/best.pt     # ← 基础模型（用这个）
│   ├── moe_v1_ground_expert/.../best.pt  # ← 地面专家（用这个）
│   └── moe_v1_tiny_obstacle_expert/... # ← 微小物体专家（用这个）
└── artifacts/
    ├── reports/                    # 评估报告 JSON 文件
    └── expert_subsets/            # 专家子数据集（自动生成）
```

### 数据对应关系

```
data/
└── merged_yolo_detect/           # 原始完整数据（8类）
    ├── data.yaml                  # 原始类别定义
    ├── images/train/              # 6537 张训练图
    ├── images/val/                # 2425 张验证图
    └── labels/                    # YOLO 格式标注

[自动生成 - 来自上面的数据]
├── expert_subsets/
│   ├── ground_expert/data.yaml    # 2 类：blind_road, crosswalk
│   ├── ground_expert/images/train/  # 过滤后的图
│   └── ...
│   ├── tiny_obstacle_expert/data.yaml  # 6 类
│   └── ...
```

---

## ⚙️ 参数调优指南

### 根据你的需求选择参数

#### 情况1：GPU 内存有限（如 8GB）

```bash
python train_moe_cascade.py \
  --batch 16 \
  --imgsz 416 \
  --device 0
```
**权衡**：速度慢，精度可能略低

#### 情况2：想快速验证想法（测试）

```bash
python train_moe_cascade.py \
  --mode base \
  --batch 64 \
  --imgsz 512 \
  --epochs-base 3 \
  --epochs-expert 2
```
**时间**：~5-10 分钟，快速检查代码是否工作

#### 情况3：要求最好精度（没有时间限制）

```bash
python train_moe_cascade.py \
  --batch 128 \
  --imgsz 832 \
  --device 0,1,2,3 \
  --epochs-base 50 \
  --epochs-expert 30
```
**时间**：~2-3 小时，质量最高

#### 情况4：多 GPU、想要平衡

```bash
python train_moe_cascade.py \
  --batch 96 \
  --imgsz 640 \
  --device 0,1,2 \
  --epochs-base 30 \
  --epochs-expert 20
```
**时间**：~1.5 小时，推荐配置

---

## 📈 预期性能参考

### 当前模型预期指标（基于实验）

```
基础模型（8类）:
  Precision: ~0.74
  Recall:    ~0.65
  mAP50:     ~0.68
  mAP50-95:  ~0.46

地面专家（2类）:
  Precision: ~0.89
  Recall:    ~0.83
  mAP50:     ~0.91
  mAP50-95:  ~0.69

微小物体专家（6类）:
  Precision: ~0.68
  Recall:    ~0.59
  mAP50:     ~0.59
  mAP50-95:  ~0.36

系统性能:
  平均延迟:      ~180-250ms
  Max 延迟:      ~1500ms
  触发率:        ~60%
```

**如何改进？**
1. **+5-10% mAP** → 增加数据
2. **+3-5% mAP** → 用更大模型（YOLO11l）
3. **+2-3% mAP** → 调参（lr, augmentation）
4. **+1-2% mAP** → 集成多个模型

---

## 🐛 常见错误和解决

### 错误1：CUDA out of memory

```
症状：RuntimeError: CUDA out of memory
原因：batch 太大或图像太大
解决：
  python train_moe_cascade.py \
    --batch 48 \           # ← 从 96 改to 48
    --imgsz 512 \          # ← 从 640 改to 512
    --device 1,2,3
```

### 错误2：找不到 checkpoint

```
症状：FileNotFoundError: .../best.pt
原因：路径写错了
解决：
  1. 确认输入命令中的路径对不对
  2. 确认训练已完成并生成了 best.pt
  3. 用绝对路径而不是相对路径：
     ./runs/detect/runs/...
```

### 错误3：IndexError in expert evaluation

```
症状：IndexError: index 4 is out of bounds for axis 1
原因：专家用了错误的 data.yaml
解决：
  # 检查文件是否存在
  ls artifacts/expert_subsets/ground_expert/data.yaml
  ls artifacts/expert_subsets/tiny_obstacle_expert/data.yaml
  # 如果不存在，先运行完整训练（mode=all）
```

---

## 📚 进阶话题

### 想添加第三个专家？

编辑 `train_moe_cascade.py` 的 `build_expert_subset()` 函数：

```python
# 添加代码到 build_expert_subset() 中
elif expert_name == "color_expert":
    # 专门检测有特定颜色的物体
    subset_classes = ["blind_road", "barricade"]  # 定义这个专家的类别
    subset_name = "color_expert"
```

编辑 `moe_yolo/cascade.py` 来整合新专家。

### 想修改路由规则？

编辑 `moe_yolo/router.py`：

```python
def should_activate_ground_expert(detections):
    # 现在的规则
    return len(detections) < 3
    
    # 改成：只有在非常少的情况下才激活
    # return len(detections) < 1

def should_activate_tiny_expert(detections):
    # 现在的规则
    return count_small_objects(detections) > 5
    
    # 改成：更敏感
    # return count_small_objects(detections) > 3
```

### 想用不同加权方式融合结果？

编辑 `moe_yolo/cascade.py` 的融合部分：

```python
# 现在：直接合并和 NMS
def merge_results(base_det, expert_det):
    all_det = np.vstack([base_det, expert_det])
    return nms(all_det)  # 非极大值抑制
    
# 可以改成：加权平均
def merge_results(base_det, expert_det):
    # 基础的检测框保留 80%，专家的检测框改进 20%
    base_det[:, 4] *= 0.8  # 降低基础置信度
    expert_det[:, 4] *= 0.9  # 提高专家置信度
    return nms(np.vstack([base_det, expert_det]))
```

---

## 📞 需要更多帮助？

- 名词不懂？→ 查看 `COMPREHENSIVE_GUIDE.md` 第 2 节（核心概念）
- 参数调不对？→ 查看上面的"参数调优指南"
- 想改代码？→ 查看"进阶话题"
- 其他技术问题？→ 看 Ultralytics 官方文档

---

**最后提示**：保存这份文件，遇到问题先来这里查，能解决 90% 的常见问题！
