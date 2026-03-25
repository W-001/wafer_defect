# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

晶圆缺陷分类算法 - 基于 DINOv3 的分层开放集分类框架。

**核心特性**：
- DINOv3 / Swin backbone
- 三视角融合
- Gate (Nuisance vs Defect) + Fine (Defect细分类) 两级分类
- 异常检测 (未知defect发现)
- 噪声标签鲁棒训练

## 环境配置

```shell
conda activate py310
```

## 常用命令

### 训练

```shell
cd C:/Code/Work/DefectClass_dinov3

# 真实数据 (文件夹结构)
PYTHONPATH=. python wafer_defect/train.py \
    --data_dir /path/to/wafer_data \
    --use_dinov3 --epochs 10 --device cuda

# 合成数据 (快速测试)
PYTHONPATH=. python wafer_defect/train.py \
    --synthetic --epochs 10 --num_samples 200 --device cpu
```

### 数据格式

真实数据文件夹结构:
```
data/
├── Nuisance/
│   ├── D234569@123456W1234567890F12345678I00K12345678.jpg
│   ├── D234569@123456W1234567890F12345678I01K12345678.jpg
│   └── D234569@123456W1234567890F12345678I02K12345678.jpg
├── Scratch/
│   └── ...
└── Particle/
    └── ...
```

**三视角识别规则:**
- 文件名中 `I` 后面 `K` 前面的两位数字(00/01/02)表示同一defect的视角序号
- `I00K` = 第一视角，`I01K` = 第二视角，`I02K` = 第三视角
- 三张视角照片组合为一个样本

**路径说明:**
- 所有路径均使用相对于项目根目录的相对路径
- 服务器上运行时只需调整 `--data_dir` 参数

## 自动同步机制

### Git Hooks 自动化
- **pre-commit hook**: 自动检测变更的模块并更新时间戳到 TODO.md
- **post-commit hook**: 自动推送到 GitHub

### 同步流程
```
[修改代码] → git add . → git commit → [pre-commit: 更新TODO.md] → [post-commit: 推送到GitHub]
```

### 自动更新 TODO.md 的模块
| 模块 | 检测关键词 |
|------|-----------|
| DINOv3 backbone | `*backbone*` |
| 三视角融合 | `*fusion*` |
| Gate Head | `*gate_head*`, `*gate*` |
| Fine Head | `*fine_head*`, `*fine*` |
| 异常检测 | `*anomaly*` |
| 数据集 | `*dataset*` |
| 训练脚本 | `*train*` |
| 损失函数 | `*losses*`, `*loss*.py` |
| 训练引擎 | `*engine*`, `*trainer.py*` |
| 评估指标 | `*utils*`, `*metrics.py*` |

## 开发流程

1. **直接修改代码**
2. **提交**: `git add <files> && git commit -m "type(scope): description"`
3. **自动完成**:
   - pre-commit hook 检测变更模块并更新 TODO.md
   - post-commit hook 自动推送到 GitHub

## Skills 自动加载

开发过程中自动使用：

| Task | Skill |
|------|-------|
| 代码简化/重构 | `simplify` |
| 代码审查 | `code-review:code-review` |
| 调试问题 | `superpowers:systematic-debugging` |
| 性能优化 | `simplify` |
| 测试验证 | `superpowers:verification-before-completion` |
| 创意功能 | `superpowers:brainstorming` |
| 制定计划 | `planning-with-files:plan-zh` |

## 项目结构

```
wafer_defect/
├── configs/           # 配置文件
├── data/dataset.py   # 数据集 + 合成数据
├── models/
│   ├── backbone.py       # DINOv3 封装
│   ├── fusion.py         # 三视角融合
│   ├── gate_head.py      # Nuisance vs Defect
│   ├── fine_head.py      # Defect细分类
│   ├── anomaly_head.py   # 异常检测
│   └── full_model.py     # 完整模型
├── losses/__init__.py   # 损失函数
├── engine/trainer.py    # 训练引擎
├── utils/metrics.py    # 评估指标
└── train.py           # 主训练脚本
```

## 架构说明

### 数据流
```
[B, 3, C, H, W] → backbone → [B*3, D] → fusion → [B, D]
                                                      ↓
                              gate_head ← gate_logits (Nuisance/Defect)
                                   ↓
                              fine_head ← fine_logits (Defect类型)
                                   ↓
                            anomaly_head ← 距离类中心分数
```

### 损失函数
```
L_total = L_gate + λ1*L_fine + λ2*L_metric
L_gate = weighted_CE(defect_weight=3.0)
L_fine = CE(仅在is_defect=True样本)
L_metric = SupCon(拉近同类/推远异类)
```

## 新增模块检查

- [ ] 在 `models/__init__.py` 中导出
- [ ] 在 `losses/__init__.py` 中导出 (如适用)
- [ ] 添加单元测试 (如适用)

## 关键文件

| 文件 | 说明 |
|------|------|
| `train.py` | 主训练脚本 |
| `models/full_model.py` | WaferDefectModel 完整模型 |
| `data/dataset.py` | 数据集 + SyntheticWaferGenerator |
| `losses/__init__.py` | CombinedLoss 层级损失 |
| `TODO.md` | 项目进度追踪 |

## 注意事项

1. **DINOv3 权重**: `dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`
2. **PYTHONPATH**: 必须包含 `C:/Code/Work/DefectClass_dinov3`
3. **设备**: GPU训练用 `--device cuda`
