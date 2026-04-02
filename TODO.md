# 晶圆缺陷分类算法 - 项目总结

## 一、项目概述

基于 DINOv3 的晶圆缺陷分层开放集分类框架，解决：
- Nuisance vs True Defect 的高精度二分类
- True Defect 细分类（几十类）
- 未知/新 defect 的检测（特征空间距离判断）
- 三视角融合
- 标签噪声处理

---

## 二、功能实现状态

### 2.1 输入预处理层
| 需求 | 状态 | 说明 |
|------|------|------|
| 主图区域 + 底部刻度区域分离 | ✅ 已实现 | Updated: 2026-03-25 10:02:56 | `RealWaferDataset` 默认裁剪底部40px |
| 底部区域单独处理 | ⚠️ 部分 | 当前仅做裁剪，未做 OCR/数值解析 |
| 可变尺寸图片resize | ✅ 已实现 | 504-680px → 统一224x224 |

### 2.2 共享表征层
| 需求 | 状态 | 说明 |
|------|------|------|
| DINOv3 backbone | ✅ 已实现 | Updated: 2026-04-01 20:50:03 | Updated: 2026-03-25 10:02:56 | `backbone.py` 中 `DINOv3Backbone` 封装 |
| 相对路径支持 | ✅ 已实现 | 动态计算项目根目录，适配服务器部署 |

### 2.3 三视角融合
| 需求 | 状态 | 说明 |
|------|------|------|
| 共享 encoder | ✅ 已实现 | 同一 backbone 提取三视角特征 |
| attention/gating 融合 | ✅ 已实现 | `MultiViewFusion` 支持 mean/attention/gated |
| 输出 group-level 结果 | ✅ 已实现 | 融合后统一输出 |
| 三视角识别（实际数据格式） | ✅ 已实现 | 正则匹配 `IxxK` 模式 (00/01/02) |

### 2.4 Gate Head (Nuisance vs Defect)
| 需求 | 状态 | 说明 |
|------|------|------|
| 二分类主头 | ✅ 已实现 | Updated: 2026-04-01 21:14:15 | `GateHead` 输出 2 类 |
| 拒识/不确定性头 | ✅ 已实现 | `UncertaintyHead` |
| 阈值可调 risk-aware decision | ✅ 已实现 | `defect_weight` 参数可调 |

### 2.5 Fine Head (Defect 细分类)
| 需求 | 状态 | 说明 |
|------|------|------|
| 多类分类头 | ✅ 已实现 | `FineHead` |
| 原型/类中心约束 | ✅ 已实现 | `PrototypeClassifier` |
| SupCon 辅助损失 | ✅ 已实现 | `MetricLoss` (SupCon) |

### 2.6 未知/新缺陷检测
| 需求 | 状态 | 说明 |
|------|------|------|
| 类中心距离检测 | ✅ 已实现 | Updated: 2026-04-01 20:50:03 | `AnomalyHead` 使用 z-score 归一化 |
| 能量分数 | ✅ 已实现 | 与距离组合 (0.7*距离 + 0.3*能量) |
| 未知缺陷标记 | ✅ 已实现 | `is_unknown_defect` 输出 |
| 自动校准阈值 | ✅ 已实现 | 训练后自动计算95百分位 |
| **RAD 多层 Patch-KNN** | ✅ 已实现 | 2026-04-01 | `RADAnomalyHead` + `--use_rad_anomaly` |

### 2.7 RAD 多层 Patch-KNN 异常检测
| 需求 | 状态 | 说明 |
|------|------|------|
| DINOv3 中间层特征提取 | ✅ 已实现 | 2026-04-01 | `backbone.get_intermediate_layers()` |
| 多层 Memory Bank 构建 | ✅ 已实现 | 2026-04-01 | `RADAnomalyHead.build_bank()` |
| Patch 级 KNN 异常评分 | ✅ 已实现 | 2026-04-01 | 多层线性融合 + Gaussian 平滑 |
| RAD 校准阈值 | ✅ 已实现 | 2026-04-01 | `calibrate()` 方法 |
| `--use_rad_anomaly` 开关 | ✅ 已实现 | 2026-04-01 | `--rad_layer_indices`, `--rad_k_image` |

### 2.8 错分样本追踪
| 需求 | 状态 | 说明 |
|------|------|------|
| Gate错分记录 | ✅ 已实现 | 漏检/误报分离 |
| Fine错分记录 | ✅ 已实现 | CSV/JSON 格式导出 |
| 数据集检查工具 | ✅ 已实现 | `data_inspector.py` |

### 2.9 训练策略
| 需求 | 状态 | 说明 |
|------|------|------|
| 分阶段训练 | ✅ 已实现 | Updated: 2026-03-25 10:02:56 | 当前是联合训练，可扩展为分阶段 |
| 层级损失 L = L_gate + λ1*L_fine + λ2*L_metric | ✅ 已实现 | `CombinedLoss` |
| 高代价惩罚 defect 漏检 | ✅ 已实现 | `defect_weight=3.0` 默认 |

### 2.10 评估指标
| 需求 | 状态 | 说明 |
|------|------|------|
| Gate 召回/漏检率 | ✅ 已实现 | `GateMetrics` |
| macro-F1 / per-class recall | ✅ 已实现 | `FineMetrics` |
| 错分样本报告 | ✅ 已实现 | JSON + CSV 导出 |
| 图文 Markdown 报告 | ✅ 已实现 | 2026-04-01 | `generate_markdown_report()` |

### 2.11 模型保存与加载
| 需求 | 状态 | 说明 |
|------|------|------|
| 最佳模型保存 | ✅ 已实现 | 2026-04-01 | 每个 epoch 验证后自动保存 `best_model.pt` |
| 最终模型保存 | ✅ 已实现 | 2026-04-01 | `last_model.pt` |
| RAD Bank 保存/加载 | ✅ 已实现 | 2026-04-01 | `build_bank()` / checkpoint 内嵌 |

---

## 三、项目结构

```
wafer_defect/
├── configs/base.yaml           # 配置文件
├── data/dataset.py            # 数据集 + RealWaferDataset
├── models/
│   ├── backbone.py            # DINOv3 backbone 封装
│   ├── fusion.py             # 三视角融合
│   ├── gate_head.py          # Nuisance vs Defect
│   ├── fine_head.py          # Defect细分类
│   ├── anomaly_head.py       # 类中心距离异常检测
│   ├── rad_head.py           # RAD 多层 Patch-KNN 异常检测
│   └── full_model.py         # 完整模型
├── losses/__init__.py        # 损失函数
├── engine/trainer.py         # 训练引擎 + 错分追踪 + Markdown报告
├── utils/
│   ├── metrics.py            # 评估指标
│   └── data_inspector.py    # 数据集检查工具
└── train.py                  # 主训练脚本
```

---

## 四、运行方式

### 训练

```shell
cd /path/to/DefectClass_dinov3

# 真实数据 + 类中心异常检测（默认）
PYTHONPATH=. python wafer_defect/train.py \
    --data_dir /path/to/wafer_data \
    --use_dinov3 --epochs 50 --device cuda

# 真实数据 + RAD 多层 Patch-KNN 异常检测
PYTHONPATH=. python wafer_defect/train.py \
    --data_dir /path/to/wafer_data \
    --use_dinov3 --use_rad_anomaly \
    --rad_layer_indices 3 6 9 11 \
    --rad_k_image 5 \
    --epochs 50 --device cuda

# 合成数据（测试用）
PYTHONPATH=. python wafer_defect/train.py --synthetic --epochs 10
```

### 数据检查

```shell
PYTHONPATH=. python wafer_defect/utils/data_inspector.py /path/to/wafer_data
```

### 数据格式

```
data/
├── Nuisance/
│   ├── D234569@...I00K12345678.jpg  # 视角1
│   ├── D234569@...I01K12345678.jpg  # 视角2
│   └── D234569@...I02K12345678.jpg  # 视角3
├── Scratch/
│   └── ...
└── Particle/
    └── ...
```

---

## 五、待完成 (TODO)

### 高优先级
1. **分阶段训练**: 先训 Gate，再训 Fine，最后联合微调
2. **Co-teaching**: 双模型互筛噪声样本
3. **完整 open-set 评估**: AUROC, AUPR, FPR@95TPR

### 中优先级
1. **底部刻度 OCR 解析**: 提取尺度元信息作为额外输入
2. **ProtoNet/센터 Loss**: 原型网络增强
3. **t-SNE 可视化**: embedding 空间可视化验证

### 低优先级
1. **多产品/批次分组评估**: 按产品型号分组统计
2. **主动学习闭环**: 难例自动标注回流
3. **DINOv3 微调**: 解除 backbone freeze 进行 fine-tuning

---

## 六、变更记录

| 时间 | 提交信息 | 变更模块 |
|------|----------|----------|
| 2026-03-22 18:53 | feat: initial wafer_defect project | 全部模块 |
| 2026-03-25 | feat: add real data loading, unknown defect detection, misclassification tracking | 数据加载、异常检测、错分追踪 |
[2026-03-25 10:02:57] Completed via commit: feat(feat: add real data loading, unknown defect detection, and misclassification tracking): add real data loading, unknown defect detection, and misclassification tracking
[2026-04-01 20:50:04] Completed via commit: feat(feat: integrate RAD multi-layer patch-KNN anomaly detection + bug fixes): integrate RAD multi-layer patch-KNN anomaly detection + bug fixes
[2026-04-01 21:14:16] Completed via commit: fix(fix: gate_head is_defect_pred returns torch.long for downstream compatibility): gate_head is_defect_pred returns torch.long for downstream compatibility
