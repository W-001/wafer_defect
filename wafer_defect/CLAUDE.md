# CLAUDE.md

晶圆缺陷分类算法 - 基于 DINOv3 的分层开放集分类框架。

## 环境配置

```shell
conda activate py310
```

## 架构概述

```
输入图像 → DINOv3 Backbone (冻结) → 共享特征塔
                                        ↓
                              ┌─────────┴─────────┐
                              ↓                   ↓
                         Gate Head          Fine Head
                        (二分类)           (多分类)
                              ↓                   ↓
                           Nuisance           Defect Type
```

**核心特性**：
- DINOv3 Backbone (冻结)
- 共享特征塔 + Gate/Fine 分类头
- Dinomaly2 异常检测 (arXiv 2510.17611v2)
- 可选三视角融合

## 常用命令

### 训练

```shell
cd C:/Code/Work/DefectClass_dinov3

# 合成数据 (快速测试)
PYTHONPATH=. python wafer_defect/train.py --synthetic --epochs 10 --device cpu

# 真实数据
PYTHONPATH=. python wafer_defect/train.py --data_dir /path/to/wafer_data --use_dinov3 --epochs 10 --device cuda
```

## 数据格式

真实数据文件夹结构:
```
data/
├── Nuisance/
│   └── *.jpg
├── Scratch/
│   └── *.jpg
└── Particle/
    └── *.jpg
```

**三视角模式** (需 `--three_views`):
- 文件名中 `I00K`/`I01K`/`I02K` 表示同一缺陷的三个视角

## 项目结构

```
wafer_defect/
├── models/              # 模型定义
│   ├── backbone.py      # DINOv3 封装
│   ├── classification.py # 分类分支 (Gate + Fine)
│   ├── dinomaly2.py     # Dinomaly2 异常检测
│   ├── defect_model.py  # 主模型
│   └── fusion.py        # 三视角融合
├── losses/              # 损失函数
│   ├── gate_loss.py     # Gate 损失
│   ├── fine_loss.py     # Fine 损失
│   ├── metric_loss.py   # Metric 损失
│   └── dinomaly_loss.py # Dinomaly2 损失
├── engine/              # 训练引擎
│   ├── trainer.py        # 训练器
│   ├── sampler.py        # 长尾采样
│   └── collate.py       # 数据整理
├── data/                # 数据处理
│   ├── dataset.py        # 数据集
│   └── preprocessor.py  # 预处理
├── inference.py          # 推理入口
└── train.py             # 训练入口
```

## 新增模块

- [x] `models/classification.py` - 分类分支
- [x] `models/dinomaly2.py` - Dinomaly2 分支
- [x] `models/defect_model.py` - 主模型
- [x] `losses/` - 损失函数模块
- [x] `engine/` - 训练引擎
- [x] `inference.py` - 推理入口

## 技术细节

### Gate 分类
- 二分类: Nuisance (无缺陷) vs Defect (有缺陷)
- 使用 weighted CE，defect_weight=3.0 惩罚漏检

### Fine 分类
- 多分类: 缺陷类型
- 仅在 defect 样本上计算损失

### Dinomaly2 异常检测
- 基于 Loose Reconstruction
- Noisy Bottleneck + Linear Attention
- Context-Aware Recentering

### 开放集检测
- 组合策略: Dinomaly2 + 距离类中心
- 识别未知缺陷类型

## 注意事项

1. **DINOv3 权重**: `dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`
2. **PYTHONPATH**: 必须包含项目根目录
3. **图像尺寸**: 默认 392×392 (DINOv3)
4. **底部裁剪**: 默认裁剪 40px (标尺区域)
