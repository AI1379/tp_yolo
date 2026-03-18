# MoEYOLO

本目录给出一个更偏工程落地的方案：

- 不直接上“全模型动态 MoE”。
- 先采用“级联检测 + 轻量路由”的方案验证收益。
- 保留向共享 backbone + 稀疏专家 head 过渡的路径。

这个方案适合你当前目标：

- 端侧设备（普通 Android）实时运行。
- 类别多且数据分布不均衡。
- 小目标较多，单一模型容易漏检。

## 1. 核心思路

### Stage-1: Always-on 主模型

- 使用一个轻量 YOLO 模型覆盖全部类别。
- 目标是低延迟与高召回，作为默认路径。

### Stage-2: 触发式专家模型

- 当主模型结果显示“不确定/复杂场景”时触发专家模型。
- 专家模型按类别簇强化：
	- `ground_expert`: 盲道、横道。
	- `tiny_obstacle_expert`: cone、barrier 等小障碍。

### Route: 轻量路由

- 初版：规则路由（稳定、易部署）。
- 下一版：学习型 router（小 MLP 或逻辑回归）。

## 2. 目录结构

```text
MoEYOLO/
	README.md
	ARCHITECTURE.md
	run_demo.py
	train_moe_cascade.py
	evaluate_moe_v1.py
	moe_yolo/
		__init__.py
		config.py
		router.py
		cascade.py
		train_router_stub.py
```

## 3. 文件说明

- `ARCHITECTURE.md`
	- 详细设计文档：可行性、架构、训练与评估体系。

- `moe_yolo/config.py`
	- 级联推理和路由阈值配置。

- `moe_yolo/router.py`
	- 规则路由器实现。
	- 支持基于低置信度比例、小目标数量、框数量异常触发专家。

- `moe_yolo/cascade.py`
	- 主模型 + 专家模型级联管线。
	- 包含输出融合（按类别 NMS）。

- `moe_yolo/train_router_stub.py`
	- Router 训练脚手架（numpy 逻辑回归 baseline）。
	- 用于先验证“触发与否”的可学习性。

- `run_demo.py`
	- 单图最小可运行示例。

- `train_moe_cascade.py`
	- 级联训练入口（base + experts）。
	- 默认设备为 `1,2,3`，即四卡机器上的后三张 3090。

- `evaluate_moe_v1.py`
	- 自动评估脚本。
	- 输出 base、experts 的 val 指标，并统计 cascade 触发率与时延。

## 4. 快速开始

在仓库根目录安装依赖后执行：

```bash
cd MoEYOLO
python run_demo.py --image /path/to/image.jpg --base-model yolo11m.pt --ground-expert yolo11m.pt --tiny-expert yolo11m.pt
```

说明：

- 这是“初步代码”，用于验证流程，不代表最终最优结构。
- 你可以先用同一个权重占位，后续再替换为真正专家模型。

## 5. Router 训练脚手架

输入格式为 `jsonl`，每行示例：

```json
{"features": [12, 0.58, 5, 0.41], "label": 1}
```

运行：

```bash
cd MoEYOLO
python -m moe_yolo.train_router_stub --train /path/to/train.jsonl --val /path/to/val.jsonl --out router_weights.npz
```

## 6. 训练（使用后三张 3090）

### 快速试跑（先看流程和日志）

```bash
cd MoEYOLO
python train_moe_cascade.py \
	--mode all \
	--device 1,2,3 \
	--epochs-base 1 \
	--epochs-expert 1 \
	--fraction 0.2 \
	--batch 96 \
	--imgsz 640 \
	--name-prefix moe_try
```

### 初步正式训练

```bash
cd MoEYOLO
python train_moe_cascade.py \
	--mode all \
	--device 1,2,3 \
	--epochs-base 30 \
	--epochs-expert 20 \
	--fraction 1.0 \
	--batch 96 \
	--imgsz 640 \
	--name-prefix moe_v1
```

说明：

- `--mode base` 只训练主模型。
- `--mode experts` 只训练专家模型（默认从 `--base-model` 初始化）。
- `--mode all` 先训主模型，再用主模型权重初始化两个专家。
- 专家子集数据会写到 `MoEYOLO/artifacts/expert_subsets/`。
- 脚本默认关闭 AMP（更适合离线环境），需要时可手动加 `--amp`。

## 7. 建议的下一步迭代

1. 先固定规则路由阈值，跑出触发率与增益曲线。
2. 用误检/漏检样本构建专家训练集，训练真正专家权重。
3. 将规则路由替换为学习型 router，比较精度-延迟 tradeoff。
4. 做端侧实测，重点看 P95 延迟与温升。

## 8. 自动评估

在训练完成后，可以用以下命令自动生成评估报告：

```bash
cd MoEYOLO
python evaluate_moe_v1.py \
	--base-ckpt ./runs/detect/runs/moeyolo/moe_v1_base/weights/best.pt \
	--ground-expert-ckpt ./runs/detect/runs/moeyolo/moe_v1_ground_expert/weights/best.pt \
	--tiny-expert-ckpt ./runs/detect/runs/moeyolo/moe_v1_tiny_obstacle_expert/weights/best.pt \
	--device 1 \
	--split val \
	--sample-images 300 \
	--report artifacts/reports/eval_moe_v1.json
```

报告内容包括：

- base / experts 的 precision、recall、mAP50、mAP50-95。
- 每类 mAP50-95（per-class）。
- cascade 触发率、平均检测框数、平均时延、P95 时延与触发原因统计。

## 9. 注意事项

- 当前代码不改动仓库其他目录，方便在混乱主分支中独立演进。
- 该实现优先保证可读与可验证，后续可再做工程加速（批处理、异步、ROI 裁剪等）。