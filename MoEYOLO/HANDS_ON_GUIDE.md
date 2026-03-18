# MoEYOLO 实践操作手册

## 适用人群
- 想快速上手的新手
- 想复现当前结果
- 想改进模型的开发者

---

## 第一部分：理论预热（15 分钟）

### 我应该先了解什么？

**最小必读内容：**
1. 这个项目是做什么的？→ [COMPREHENSIVE_GUIDE.md](COMPREHENSIVE_GUIDE.md) 第一节
2. 模型长什么样？→ [COMPREHENSIVE_GUIDE.md](COMPREHENSIVE_GUIDE.md) 第三节（模型架构）
3. 为什么这样设计？→ [COMPREHENSIVE_GUIDE.md](COMPREHENSIVE_GUIDE.md) 第四节（设计思想）

**大概需要 15 分钟理解这三个问题。**

然后你就可以开始操作了！

---

## 第二部分：快速上手（30 分钟）

### 场景 A：我只是想看看现有模型效果如何

**前置条件：**
- Python 环境已配置
- 三个训练好的 checkpoint 存在
- CUDA 环境正常

**操作步骤：**

```bash
# 步骤 1：进入项目目录
cd ./MoEYOLO

# 步骤 2：运行评估脚本
python evaluate_moe_v1.py \
  --base-ckpt runs/detect/runs/moeyolo/moe_v1_base/weights/best.pt \
  --ground-expert-ckpt runs/detect/runs/moeyolo/moe_v1_ground_expert/weights/best.pt \
  --tiny-expert-ckpt runs/detect/runs/moeyolo/moe_v1_tiny_obstacle_expert/weights/best.pt \
  --device 1 \
  --batch 32 \
  --sample-images 50 \
  --report artifacts/reports/my_evaluation.json

# 步骤 3：查看结果
cat artifacts/reports/my_evaluation.json | python -m json.tool | head -100
```

**输出解读：**

看 JSON 文件中的这些指标：
```json
{
  "metrics": {
    "base": {
      "map50_95": 0.46        # ← 看这个（越接近 1 越好）
    },
    "cascade_runtime": {
      "trigger_rate": 0.6     # ← 60% 的图激活了专家
    }
  }
}
```

**耗时：** ~5-10 分钟（取决于样本数）

---

### 场景 B：我想从零开始训练一个模型

**前置条件：**
- 有 3 块 24GB 的 GPU（或者调整参数）
- 原始数据在 `data/merged_yolo_detect/` 

**完整操作流程：**

```bash
# 进入项目目录
cd ./MoEYOLO

# 【第 1 步】先做一个快速验证（可选，1-2 分钟）
# 确保代码没问题
python train_moe_cascade.py \
  --mode base \
  --device 1,2,3 \
  --batch 96 \
  --imgsz 640 \
  --epochs-base 1 \
  --name-prefix quick_test

# 【第 2 步】运行完整训练（这是会花时间的部分）
python train_moe_cascade.py \
  --mode all \
  --device 1,2,3 \
  --batch 96 \
  --imgsz 640 \
  --epochs-base 30 \
  --epochs-expert 20 \
  --name-prefix moe_v2

# 【第 3 步】等待...（去喝杯咖啡，~1 小时）
# 你会看到进度条，每个 epoch 打印一次

# 【第 4 步】训练完后，评估新模型
python evaluate_moe_v1.py \
  --base-ckpt runs/detect/runs/moeyolo/moe_v2_base/weights/best.pt \
  --ground-expert-ckpt runs/detect/runs/moeyolo/moe_v2_ground_expert/weights/best.pt \
  --tiny-expert-ckpt runs/detect/runs/moeyolo/moe_v2_tiny_obstacle_expert/weights/best.pt \
  --device 1 \
  --batch 32 \
  --sample-images 100 \
  --report artifacts/reports/eval_v2.json

# 【第 5 步】比较新旧版本
echo "=== 版本 1 ==="
cat artifacts/reports/eval_moe_v1_smoke.json | grep map50_95
echo "=== 版本 2 ==="
cat artifacts/reports/eval_v2.json | grep map50_95
```

**耗时：** ~2 小时（含等待训练）

**如果遇到 OOM，改这里：**
```bash
# 把这行
--batch 96 --imgsz 640

# 改成
--batch 48 --imgsz 512
```

---

## 第三部分：解决常见问题

### 问题 1：训练卡住了/很慢

**症状：** 训练开始后，进度条不动或很慢

**诊断：**
```bash
# 打开另一个终端，检查 GPU 使用情况
nvidia-smi

# 如果看不到进程，说明可能卡在某个阶段
# 如果 GPU 占用率很低（< 10%），说明受 CPU 或 I/O 限制
```

**解决：**
- 通常是正常的，YOLO 初始化需要一些时间
- 等待几分钟看是否开始打印进度
- 如果真的卡住了，按 Ctrl+C 停止，检查错误

### 问题 2：看不懂输出日志

**输出示例：**
```
[2026-03-18 10:30:15] Training base model...
Epoch 1/30: 100%|██████████| 68/68 [00:45<00:00, 1.50it/s]
val: Scanning cached labels...
[val] AP50: 0.653, AP50-95: 0.431
Epoch 2/30: ...
```

**翻译：**
- `100%|██████████| 68/68` - 进度条，68 个 batch 全部完成
- `[00:45<00:00]` - 用了 45 秒，还需要 0 秒
- `1.50it/s` - 每秒处理 1.5 个 batch
- `AP50: 0.653` - 某个指标达到 0.653
- `Epoch 2/30` - 现在在第 2 个 epoch，总共 30 个

### 问题 3：评估脚本报错

**常见错误 1：FileNotFoundError**
```
FileNotFoundError: .../best.pt no such file or directory
```
解决：检查路径对不对
```bash
ls -la runs/detect/runs/moeyolo/moe_v1_base/weights/best.pt
# 如果看不到文件，说明训练没完成或路径写错了
```

**常见错误 2：CUDA out of memory**
```
RuntimeError: CUDA out of memory
```
解决：减小 batch
```bash
# 把 --batch 32 改成 --batch 16
python evaluate_moe_v1.py ... --batch 16 ...
```

### 问题 4：想改变训练参数

#### 改变 GPU 数量

```bash
# 原来：用 GPU 1,2,3（3 块）
--device 1,2,3

# 改成：用 GPU 0,1（2 块）
--device 0,1

# 改成：只用 GPU 0（1 块，但会慢很多）
--device 0
```

#### 改变模型大小

```bash
# 改成快速版本（小模型）
python train_moe_cascade.py \
  --mode all \
  --device 1,2,3 \
  --batch 96 \
  --imgsz 640 \
  --epochs-base 30 \
  --epochs-expert 20 \
  --pretrain "yolo11n.pt" \  # ← 改这里（nano 版本）
  --name-prefix moe_nano

# 改成高精度版本（大模型）
python train_moe_cascade.py \
  --mode all \
  --device 1,2,3 \
  --batch 64 \
  --imgsz 640 \
  --epochs-base 50 \
  --epochs-expert 30 \
  --pretrain "yolo11l.pt" \  # ← 改这里（large 版本）
  --name-prefix moe_large
```

---

## 第四部分：做出自己的改进

### 改进 1：添加更多数据

**做什么：** 为了提高 mAP，最好的办法是加数据

**怎么做：**

1. 获取新图像和标注，放在 `data/` 下
2. 转换为 YOLO 格式（`class_id x_c y_c w h` 每行一个）
3. 修改 `data/merged_yolo_detect/data.yaml`：

```yaml
# 改这部分（路径指向你的新数据）
path: ./data/merged_yolo_detect
train: images/train
val: images/val

# 这部分保持不变
nc: 8
names: ['blind_road', 'crosswalk', ...]
```

4. 重新训练：
```bash
python train_moe_cascade.py --mode all --device 1,2,3 --name-prefix moe_v3
```

### 改进 2：调整模型触发规则

**做什么：** 改变什么时候激活专家模型

**怎么做：**

编辑 `moe_yolo/router.py`：

```python
class RuleBasedRouter:
    def should_activate_ground_expert(self, detections):
        """
        原规则：检测框 < 3 个就激活
        """
        # 改成：检测框 < 2 个就激活（更严格）
        return len(detections) < 2
        
        # 或改成：检测框 < 5 个就激活（更宽松）
        # return len(detections) < 5
```

然后重新评估：
```bash
python evaluate_moe_v1.py ... --report artifacts/reports/eval_new_router.json
```

对比新旧版本的 `trigger_rate` 和 `mAP`。

### 改进 3：调整训练超参数

**做什么：** 优化学习率、权重衰减等参数

**怎么做：**

编辑 `train_moe_cascade.py` 中的 `train_one()` 函数：

```python
# 找到这一行
model = YOLO(model_path)

# 在它下面添加（训练开始前）
model.overrides.update({
    'lr0': 0.01,           # 初始学习率（默认 0.01）
    'lrf': 0.01,           # 最终学习率（默认 0.01）
    'weight_decay': 0.0005, # 权重衰减（默认 0.0005）
})
```

### 改进 4：添加第三个专家

**做什么：** 为某个特定问题添加特化模型

**例子：** 添加"夜间照片专家"或"雨天专家"

**怎么做：**

1. 编辑 `train_moe_cascade.py`，在 `build_expert_subset()` 中添加：

```python
elif expert_name == "night_expert":
    # 在夜间图上表现更好的模型
    subset_classes = ["blind_road", "cone", "tubular_marker"]
    subset_name = "night_expert"
    # 只选择夜间的图片
    # （假设你的标注中有时间信息）
    ...
```

2. 编辑 `moe_yolo/cascade.py`，在 `__init__()` 中添加：

```python
self.night_expert_model = YOLO(night_expert_path)
```

3. 编辑 `moe_yolo/router.py`，添加路由规则：

```python
def should_activate_night_expert(self, image, detections):
    # 检测图像亮度
    brightness = calculate_brightness(image)
    return brightness < 50  # 亮度很低 → 激活夜间专家
```

4. 在 `cascade.py` 的 predict() 中调用这个新规则

---

## 第五部分：理解输出指标

### 如何读 JSON 评估报告？

**典型结构：**
```json
{
  "timestamp": "2026-03-18T15:50:13",
  
  "metrics": {
    "base": {
      "precision": 0.739,           # ← 88.5% 预测对了
      "recall": 0.648,              # ← 64.8% 的真实目标被找到
      "map50": 0.675,               # ← 宽松定义下的准确率
      "map50_95": 0.460,            # ← 严格定义下的准确率（最重要）
      
      "per_class_map50_95": {
        "blind_road": 0.744,        # ← 这个类单独的准确率
        "crosswalk": 0.801,
        "cone": 0.496,              # ← 这个类比较差
        ...
      }
    },
    
    "ground_expert": {              # ← 地面专家只有 2 类
      "precision": 0.894,
      "recall": 0.825,
      "map50_95": 0.688,
      "per_class_map50_95": {
        "blind_road": 0.642,
        "crosswalk": 0.733
      }
    },
    
    "tiny_obstacle_expert": {       # ← 微小物体专家有 6 类
      "precision": 0.676,
      "recall": 0.588,
      "map50_95": 0.359,
      ...
    },
    
    "cascade_runtime": {
      "num_images": 5,              # ← 测试了 5 张图
      "trigger_rate": 0.6,          # ← 60% 的图激活了专家
      "avg_detections_per_image": 4.8,  # ← 平均每张 4.8 个检测框
      "latency_ms_avg": 241.88,     # ← 平均处理时间 241ms
      "latency_ms_p95": 1080.95,    # ← 95% 的图在 1.08s 内处理完
      
      "trigger_reason_counts": {
        "many_tiny_objects": 2,     # ← 2 张图因为小物体多而激活
        "too_few_boxes": 1          # ← 1 张图因为检测框太少而激活
      }
    }
  }
}
```

### 如何判断模型好不好？

**快速检查清单：**

```
✓ mAP50-95 > 0.5         ← 很好！
⚠ 0.4 < mAP50-95 <= 0.5  ← 还可以，有改进空间
✗ mAP50-95 <= 0.4        ← 需要改进（加数据，调参）

✓ trigger_rate 40-70%     ← 正常
⚠ trigger_rate > 80%      ← 基础模型效果不好
⚠ trigger_rate < 20%      ← 专家用处不大（但也许基础模型已够好）

✓ latency_ms_avg < 300    ← 可以实时处理
⚠ 300-500 ms              ← 有点慢，但可接受
✗ > 500 ms                ← 太慢，需要优化
```

### 如何调试性能不好的类别？

假设 `cone` 的 mAP50-95 只有 0.496（很低）

**诊断步骤：**

1. 查看是 precision 还是 recall 的问题：
```python
"per_class": {
  "cone": {
    "precision": 0.45,  # ← 低 = 误报多
    "recall": 0.92,     # ← 高 = 漏检少
    "map50_95": 0.496
  }
}
```
这说明：模型**用力过猛**，把太多东西识别成 cone

2. 查看图像中有多少 cone：
```bash
# 在数据集标注文件中搜索
grep "^2 " data/merged_yolo_detect/labels/val/*.txt | wc -l
# 假设结果是 300，说明验证集有 300 个 cone 对象
```

3. 根据原因改进：
```
如果是误报太多（precision 低）:
  ← 原因：样本太少，模型容易过拟合
  ← 解决：
     1. 添加更多 cone 的训练图（数据增强）
     2. 增加训练 epoch（但小心过拟合）
     3. 用微小物体专家（专门处理小物体）

如果是漏检太多（recall 低）:
  ← 原因：模型没有学到 cone 的特征
  ← 解决：
     1. 检查标注是否正确
     2. 增加 cone 的样本权重
     3. 换更大的模型（YOLO11l）
     4. 增加输入分辨率（--imgsz 832）
```

---

## 第六部分：从这里往前走

### 后续学习建议

**Level 1：理解（完成级别）**
- ✅ 能运行训练脚本
- ✅ 能理解 JSON 报告
- ✅ 能根据错误提示解决问题

**Level 2：优化（进阶级别）**
- 改变训练参数
- 添加新的数据
- 调整模型结构（如添加新专家）
- 分析模型失败的例子

**Level 3：创新（专家级别）**
- 设计新的路由规则
- 开发新的融合策略
- 集成到业务系统
- 部署和监控

### 推荐项目（自己可以尝试的）

1. **项目 1：特定场景优化**
   - 收集某个恶劣场景的图（如雨天、夜间、雾天）
   - 为每个场景训练一个专家
   - 对比效果

2. **项目 2：模型压缩**
   - 用蒸馏（distillation）把大模型压缩到小模型
   - 在手机或边缘设备上跑

3. **项目 3：实时推理系统**
   - 整合到 Web 或移动端
   - 做一个演示界面

### 进阶阅读

如果你想深入理解，建议阅读：

1. **目标检测基础**
   - Faster R-CNN 论文（框架的前身）
   - YOLO v8 论文

2. **多专家系统**
   - Mixture of Experts 论文
   - Conditional Computation

3. **优化和部署**
   - Knowledge Distillation
   - Neural Architecture Search

---

## 第七部分：快速参考卡

### "我想..." - 快速查找

| 我想... | 用这个命令 | 文件 |
|---------|---------|------|
| 训练一个模型 | `python train_moe_cascade.py --mode all ...` | train_moe_cascade.py |
| 评估模型效果 | `python evaluate_moe_v1.py ...` | evaluate_moe_v1.py |
| 查看训练进度 | `tail -f runs/.../log.txt` | 日志文件 |
| 在现有图上推理 | 见"代码示例"下方 | cascade.py |
| 改变触发条件 | 编辑 router.py | moe_yolo/router.py |
| 添加新专家 | 编辑 train_moe_cascade.py | - |
| 理解参数含义 | [COMPREHENSIVE_GUIDE.md](#参数含义详解) | - |
| 看架构图 | [ARCHITECTURE_DETAILED.md](#系统总体架构) | - |

### 常用命令片段

```bash
# 一键训练 + 评估
cd MoEYOLO && \
python train_moe_cascade.py --mode all --device 1,2,3 --name-prefix my_v1 && \
python evaluate_moe_v1.py \
  --base-ckpt runs/detect/runs/moeyolo/my_v1_base/weights/best.pt \
  --ground-expert-ckpt runs/detect/runs/moeyolo/my_v1_ground_expert/weights/best.pt \
  --tiny-expert-ckpt runs/detect/runs/moeyolo/my_v1_tiny_obstacle_expert/weights/best.pt \
  --report artifacts/reports/eval_my_v1.json

# 快速测试（1 epoch）
python train_moe_cascade.py \
  --mode base \
  --device 0 \
  --batch 32 \
  --imgsz 512 \
  --epochs-base 1 \
  --name-prefix quick_test

# 看评估结果的关键指标
python -c "
import json
with open('artifacts/reports/eval_moe_v1_smoke.json') as f:
    data = json.load(f)
    print('Base mAP50-95:', data['metrics']['base']['map50_95'])
    print('Trigger Rate:', data['metrics']['cascade_runtime']['trigger_rate'])
    print('Avg Latency:', data['metrics']['cascade_runtime']['latency_ms_avg'], 'ms')
"
```

---

## 总结

这份手册的目的是让你能够：
1. **5 分钟内**快速评估现有模型
2. **1 小时内**训练一个新模型
3. **30 分钟内**做出小的改进
4. **长期内**理解和优化整个系统

下一步？选择一个场景开始尝试吧！🚀

---

## 需要帮助？

遇到问题时按这个顺序查：
1. 本手册的"解决常见问题"部分
2. [COMPREHENSIVE_GUIDE.md](COMPREHENSIVE_GUIDE.md) 的"常见问题"
3. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) 的"常见错误和解决"  
4. 源代码注释
5. 官方文档（Ultralytics）
