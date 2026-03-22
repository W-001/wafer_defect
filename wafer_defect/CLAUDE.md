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
# 使用 py310 conda 环境
# Python: 3.10+, PyTorch >= 2.3.0

# 激活环境并设置 PYTHONPATH
export PYTHONPATH=C:/Code/Work/DefectClass_dinov3
conda activate py310
```

## 常用命令

### 训练

```shell
# DINOv3 backbone (GPU)
cd C:/Code/Work/DefectClass_dinov3
PYTHONPATH=. /c/Users/Xiaofan/.conda/envs/py310/python.exe wafer_defect/train.py \
    --use_dinov3 \
    --epochs 10 \
    --num_samples 200 \
    --num_defect_classes 10 \
    --batch_size 4 \
    --device cuda

# 简单 CNN backbone (快速测试, CPU)
PYTHONPATH=. /c/Users/Xiaofan/.conda/envs/py310/python.exe wafer_defect/train.py \
    --epochs 10 \
    --num_samples 200 \
    --device cpu
```

### Git 操作

```shell
# 提交代码规范
git add <files>
git commit -m "type(scope): description

- feat: 新功能
- fix: Bug修复
- docs: 文档更新
- refactor: 代码重构
- test: 测试相关
- chore: 构建/工具相关

示例:
feat(gate): 添加高斯噪声增强gate分类鲁棒性
fix(fusion): 修复attention融合时的维度错误
docs(dataset): 更新数据集加载说明"

# 推送到 GitHub
git push origin <branch>
```

## 项目结构

```
wafer_defect/
├── configs/           # 配置文件
├── data/             # 数据集加载 + 合成数据
├── models/           # 模型组件
│   ├── backbone.py       # DINOv3 封装
│   ├── fusion.py         # 三视角融合
│   ├── gate_head.py      # Nuisance vs Defect
│   ├── fine_head.py      # Defect细分类
│   ├── anomaly_head.py   # 异常检测
│   └── full_model.py     # 完整模型
├── losses/           # 损失函数
├── engine/           # 训练引擎
├── utils/           # 工具函数
└── train.py         # 主训练脚本
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
L_gate = weighted_CE(defect_weight=3.0)  # 提高defect漏检惩罚
L_fine = CE(仅在is_defect=True样本)
L_metric = SupCon(拉近同类/推远异类)
```

## 开发规范

### 代码更新流程

1. **修改代码后自动同步**：
   - 每次代码变更后自动执行 `git add` 和 `git commit`
   - Commit 信息格式: `type(scope): description`
   - 自动推送到 GitHub

2. **TODO.md 同步更新**：
   - 完成新功能后更新 TODO.md 对应状态
   - 添加新的 TODO 项时说明优先级

3. **Commit 信息规范**：
   ```
   feat(gate): 实现基于能量分数的动态阈值
   - 添加能量阈值计算逻辑
   - 更新 gate 决策函数支持自适应阈值

   fix(fusion): 修复多视角权重为NaN的问题
   - 当视角特征全零时添加eps保护

   docs(train): 更新训练脚本参数说明
   - 添加 --freeze_backbone 参数
   - 更新 README.md
   ```

### 新增模块检查清单

- [ ] 在 `models/__init__.py` 中导出
- [ ] 在 `losses/__init__.py` 中导出 (如适用)
- [ ] 添加单元测试 (如适用)
- [ ] 更新 TODO.md 状态
- [ ] 更新本文件 (如需新增命令)

## 关键文件

| 文件 | 说明 |
|------|------|
| `train.py` | 主训练脚本 |
| `models/full_model.py` | WaferDefectModel 完整模型 |
| `data/dataset.py` | 数据集 + SyntheticWaferGenerator |
| `losses/__init__.py` | CombinedLoss 层级损失 |
| `TODO.md` | 项目进度追踪 |

## 注意事项

1. **DINOv3 权重路径**: `dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`
2. **PYTHONPATH**: 必须包含 `C:/Code/Work/DefectClass_dinov3`
3. **设备**: GPU 训练使用 `--device cuda`，否则自动选择
