# 基于 YOLO 的一个盲道检测模型

## 依赖安装

项目使用 `uv` 管理依赖。

```bash
uv sync # Install PyTorch CPU
uv sync --extra cu128 # With CUDA 12.8
uv sync --extra cu130 # With CUDA 13.0

uv sync --extra cu128 --dev # With CUDA 12.8 and development dependencies (pip, ipython, pytest, etc.)
```
