# 📚 MoEYOLO 文档导航中心

欢迎！👋 这里是所有文档的入口。根据你的需求选择合适的文档开始阅读。

---

## 🎯 按场景快速导航

### 场景 1：我只有 5 分钟

→ 快速了解这个项目是什么

**推荐阅读：**
1. [COMPREHENSIVE_GUIDE.md - 项目概述](COMPREHENSIVE_GUIDE.md#项目概述)（2 分钟）
2. [README.md](README.md)（3 分钟）

**就够了！** ✅

---

### 场景 2：我想快速运行一个评估

→ 看看当前模型的效果

**推荐阅读：**
1. [HANDS_ON_GUIDE.md - 快速上手/场景 A](HANDS_ON_GUIDE.md#场景-a我只是想看看现有模型效果如何)（5 分钟）
2. 照着步骤操作（5 分钟）

**预期时间：** 10 分钟 ⏱️

---

### 场景 3：我想理解整个系统如何工作

→ 深入学习架构和设计

**推荐阅读顺序：**
1. [COMPREHENSIVE_GUIDE.md - 核心概念讲解](COMPREHENSIVE_GUIDE.md#核心概念讲解)（15 分钟）
2. [COMPREHENSIVE_GUIDE.md - 模型架构详解](COMPREHENSIVE_GUIDE.md#模型架构详解)（15 分钟）
3. [ARCHITECTURE_DETAILED.md - 推理流程详解](ARCHITECTURE_DETAILED.md#推理流程详解)（20 分钟）
4. [ARCHITECTURE_DETAILED.md - 训练流程详解](ARCHITECTURE_DETAILED.md#训练流程详解)（15 分钟）

**预期时间：** 1-1.5 小时 ⏱️

**结果：** 你会理解：
- ✅ 为什么需要多个专家模型
- ✅ 每个部分是怎样工作的
- ✅ 如何评估模型性能

---

### 场景 4：我想实际动手训练/改进模型

→ 学实用技能，做出自己的改进

**推荐阅读顺序：**
1. [HANDS_ON_GUIDE.md - 场景 B](HANDS_ON_GUIDE.md#场景-b我想从零开始训练一个模型)（5 分钟阅读）
2. 照着步骤运行训练（1-2 小时等待）
3. [HANDS_ON_GUIDE.md - 第四部分：做出自己的改进](HANDS_ON_GUIDE.md#第四部分做出自己的改进)
4. 尝试一个改进（20-30 分钟）

**预期时间：** 2.5-3 小时（含训练等待）⏱️

**你会学到：**
- ✅ 如何训练完整的模型
- ✅ 如何解释评估结果
- ✅ 如何做出有意义的改进

---

### 场景 5：我遇到了问题，什么都不会

→ 快速找到解决方案

**按优先级查找：**

1. **首先** → [HANDS_ON_GUIDE.md - 第三部分：解决常见问题](HANDS_ON_GUIDE.md#第三部分解决常见问题)
2. **其次** → [QUICK_REFERENCE.md - 常见错误和解决](QUICK_REFERENCE.md#🐛-常见错误和解决)
3. **再次** → [COMPREHENSIVE_GUIDE.md - 常见问题](COMPREHENSIVE_GUIDE.md#常见问题)

---

## 📖 完整文档列表

### 📘 核心文档

#### 1. **COMPREHENSIVE_GUIDE.md** （最详细）
- **什么是：** 完整的理论教材
- **长度：** ~4000 字
- **适合人群：** 想彻底理解的人
- **包含内容：**
  - 项目概述
  - 核心概念（从零开始讲）
  - 模型架构（每个部分是什么）
  - 设计思想（为什么这样做）
  - 模型使用流程
  - 训练详细说明
  - 评估脚本原理
  - 参数含义详解
  - 常见问题 FAQ
- **阅读时间：** 2-3 小时
- **何时读：** 想建立完整的理论框架

#### 2. **QUICK_REFERENCE.md** （最实用）
- **什么是：** 速查表和参考卡
- **长度：** ~3000 字
- **适合人群：** 已有基础知识，需要快速查找
- **包含内容：**
  - 快速参考卡片
  - 常用命令速查
  - 指标速查表
  - 文件对应关系
  - 参数调优指南
  - 常见错误和解决
  - 进阶话题（添加专家、改路由等）
- **阅读时间：** 30-60 分钟（或按需查阅）
- **何时用：** 需要快速找某个命令或参数含义

#### 3. **ARCHITECTURE_DETAILED.md** （最深入）
- **什么是：** 架构和流程的详细可视化
- **长度：** ~3500 字
- **适合人群：** 想了解内部实现细节
- **包含内容：**
  - 系统总体架构图
  - 推理流程详解（带数字例子）
  - 训练流程详解（分三阶段）
  - 数据流与转换（每一步是什么）
  - 显存与计算分配（为什么会 OOM）
  - 优化决策树（遇到问题如何诊断）
- **阅读时间：** 1.5-2 小时
- **何时读：** 要改代码或深入理解计算原理

#### 4. **HANDS_ON_GUIDE.md** （最实用）
- **什么是：** 实践操作手册
- **长度：** ~2500 字
- **适合人群：** 想立即动手的人
- **包含内容：**
  - 理论预热（15 分钟速成）
  - 快速上手（两个完整场景）
  - 解决常见问题
  - 做出子身的改进（4 个例子）
  - 理解输出指标
  - 后续学习建议
  - 快速参考卡
- **阅读时间：** 30 分钟（理论）+ 2 小时（实践）
- **何时读：** 想从"了解"走向"会用"

### 🔗 其他文档

#### 5. **README.md**
- **什么是：** 项目概览
- **适合人群：** 刚接触项目的人
- **何时读：** 最开始

#### 6. **这份文件（文档导航中心）**
- **什么是：** 你现在在读的！
- **用途：** 帮你找到合适的文档

---

## 🗺️ 按知识深度级别

### 初级（入门级）⭐

**目标：** 能跟随教程运行代码

**必读（必须按顺序）：**
1. README.md（了解项目）
2. HANDS_ON_GUIDE.md - 理论预热（5 分钟）
3. HANDS_ON_GUIDE.md - 快速上手场景 A（5 分钟练习）
4. QUICK_REFERENCE.md - 快速参考卡片（查阅）

**学习时间：** ~30 分钟

**学完后你能：**
- ✅ 运行评估脚本
- ✅ 理解 JSON 输出
- ✅ 读懂常用命令

---

### 中级（应用级）⭐⭐

**目标：** 能训练自己的模型和做简单改进

**必读（按推荐顺序）：**
1. COMPREHENSIVE_GUIDE.md - 核心概念 + 模型架构
2. HANDS_ON_GUIDE.md - 完整场景（训练）
3. HANDS_ON_GUIDE.md - 改进实例
4. QUICK_REFERENCE.md - 参数调优指南

**学习时间：** ~2-3 小时

**学完后你能：**
- ✅ 从零训练一个完整模型
- ✅ 解读性能指标
- ✅ 改变训练参数看效果
- ✅ 添加新的训练数据

---

### 高级（深入级）⭐⭐⭐

**目标：** 能深入理解系统，做出创新改进

**必读（全部深入）：**
1. COMPREHENSIVE_GUIDE.md - 全部读完
2. ARCHITECTURE_DETAILED.md - 全部读完
3. 源代码 (train_moe_cascade.py, cascade.py, router.py)
4. 外部资源（YOLO 论文、MoE 论文）

**学习时间：** ~8-10 小时

**学完后你能：**
- ✅ 修改核心算法
- ✅ 设计新的路由规则
- ✅ 添加新的专家模型
- ✅ 指导他人使用系统

---

## 💡 学习建议

### 第一次接触？建议这样看：

```
Day 1（1 小时）:
  ├─ 15 分钟：读 README.md
  ├─ 15 分钟：读 COMPREHENSIVE_GUIDE.md 项目概述 + 核心概念
  ├─ 15 分钟：读 HANDS_ON_GUIDE.md 理论预热
  └─ 15 分钟：试着运行一个评估

Day 2（2 小时）:
  ├─ 60 分钟：读 COMPREHENSIVE_GUIDE.md 架构 + 设计思想
  ├─ 30 分钟：读 HANDS_ON_GUIDE.md 场景 B（训练）
  └─ 30 分钟：或等待训练，或读 ARCHITECTURE_DETAILED.md
  
Day 3（如需要）:
  ├─ 读 ARCHITECTURE_DETAILED.md 训练和显存分析
  ├─ 尝试 HANDS_ON_GUIDE.md 的改进项目
  └─ 参考 QUICK_REFERENCE.md 调试问题
```

### 如何查阅文档？

**如果你需要查找某个东西：**

1. 用 Ctrl+F（在你的编辑器/浏览器）搜索关键词
2. 查阅相应章节

**例如：**
- 想知道 "mAP" 是什么？ → Ctrl+F "mAP" → COMPREHENSIVE_GUIDE.md
- 想找到运行训练的命令？ → Ctrl+F "python train" → QUICK_REFERENCE.md 或 HANDS_ON_GUIDE.md
- 想理解 OOM 原因？ → Ctrl+F "OOM" → ARCHITECTURE_DETAILED.md 显存部分
- 遇到错误？ → HANDS_ON_GUIDE.md 常见问题 或 QUICK_REFERENCE.md

---

## 🎓 专题速查

### 想学的专题 → 对应文档

| 专题 | 查阅位置 |
|------|---------|
| **项目介绍** | README.md 或 COMPREHENSIVE_GUIDE.md - 项目概述 |
| **术语解释** | COMPREHENSIVE_GUIDE.md - 核心概念讲解 |
| **模型架构** | COMPREHENSIVE_GUIDE.md - 模型架构详解 或 ARCHITECTURE_DETAILED.md - 系统总体架构 |
| **为什么这样设计** | COMPREHENSIVE_GUIDE.md - 设计思想 |
| **怎么用模型** | COMPREHENSIVE_GUIDE.md - 模型使用流程 |
| **权重文件位置** | QUICK_REFERENCE.md - 文件和对应关系 |
| **常用命令** | QUICK_REFERENCE.md - 常用命令速查 |
| **怎么训练** | HANDS_ON_GUIDE.md - 场景 B |
| **参数是什么意思** | COMPREHENSIVE_GUIDE.md - 参数含义详解 或 QUICK_REFERENCE.md - 参数速查 |
| **评估指标** | COMPREHENSIVE_GUIDE.md - 评估脚本原理 或 HANDS_ON_GUIDE.md - 理解输出指标 |
| **显存不够怎么办** | ARCHITECTURE_DETAILED.md - 显存与计算分配 或 HANDS_ON_GUIDE.md - 问题 1 |
| **模型改进方法** | HANDS_ON_GUIDE.md - 做出自己的改进 |
| **添加新数据** | HANDS_ON_GUIDE.md - 改进 1：添加更多数据 |
| **调整触发规则** | HANDS_ON_GUIDE.md - 改进 2 或 QUICK_REFERENCE.md - 进阶话题 |
| **添加新专家** | HANDS_ON_GUIDE.md - 改进 4 或 QUICK_REFERENCE.md - 进阶话题 |
| **遇到 OOM 错误** | HANDS_ON_GUIDE.md - 问题 1 或 ARCHITECTURE_DETAILED.md - 显存部分 |
| **怎么读 JSON 报告** | HANDS_ON_GUIDE.md - 理解输出指标 |

---

## 📞 需要帮助？

### 我不知道从哪里开始？

1. 你有 5 分钟吗？ → 看"场景 1：我只有 5 分钟"
2. 你想快速测试？ → 看"场景 2：我想快速运行一个评估"
3. 你想从头理解？ → 看"场景 3：我想理解整个系统"
4. 你想实际操作？ → 看"场景 4：我想实际动手训练"

### 我遇到了问题？

1. **首先** → 这份文件的"按场景快速导航"最后有"场景 5"
2. **然后** → HANDS_ON_GUIDE.md 或 QUICK_REFERENCE.md 的错误部分
3. **最后** → 查相关文档的 FAQ 部分

### 文档之间的关系？

```
【高层理论】
COMPREHENSIVE_GUIDE.md（完整教材）
      |
      ├─ 让你理解 WHY（为什么）
      └─ 让你理解 WHAT（是什么）

      ↓

【中层架构】
ARCHITECTURE_DETAILED.md（流程和细节）
      |
      ├─ 让你理解 HOW（怎样工作）
      └─ 让你看到数字例子

      ↓

【底层操作】
HANDS_ON_GUIDE.md（实战手册）
QUICK_REFERENCE.md（速查表）
      |
      ├─ 让你学会 DO（怎样做）
      └─ 让你快速查找

      ↓

【最终结果】
你能成功使用、改进、部署这个系统 ✨
```

---

## 🌟 文档特色

### COMPREHENSIVE_GUIDE.md

✅ **最完整**：从零基础讲起
✅ **最易懂**：大量类比和例子
⚠️ **稍长**：需要 2-3 小时

### QUICK_REFERENCE.md

✅ **最快速**：按需查阅，不用全读
✅ **最实用**：直接告诉你怎么做
⚠️ **需要基础**：假设你已理解基本概念

### ARCHITECTURE_DETAILED.md

✅ **最深入**：讲到原理细节
✅ **最全面**：包含显存、计算等工程细节
⚠️ **最复杂**：需要较强理论基础

### HANDS_ON_GUIDE.md

✅ **最实战**：完整的操作步骤
✅ **最解决问题**：从诊断到方案
⚠️ **需要环境**：需要实际的 GPU 和数据

---

## ✨ 开始吧！

不要过度阅读，选择一个场景，立即开始吧！📚

**建议流程：**
```
了解 → 运行 → 理解 → 改进 → 学习 → 重复
 ▲                                    ▼
 └────────────────────────────────────┘
```

每一次循环，你就能向前进一步！🚀

---

**最后提示：** 这份导航文件（文档主导航中心）本身就放在 MoEYOLO 目录下，遇到问题时回来这里找索引。
