# CampusFlowGuard 实验结果摘要

更新时间：2026-07-17

## 1. 摘要范围与证据

本文只汇总已经完成并保存到磁盘的真实实验结果，没有重新训练、重新选择模型或改写历史指标。主要证据文件如下：

- BCE Baseline：`artifacts/baseline_freeze/unsw_trusted_baseline_v1/metrics/baseline_summary.csv`
- 冻结 split：`artifacts/baseline_freeze/unsw_trusted_baseline_v1/splits/split_manifest.json`
- Focal 对照：`artifacts/experiments/focal_loss/unsw_focal_loss_v1/metrics/bce_vs_focal_aggregate.csv`
- Focal 逐 seed 指标：`artifacts/experiments/focal_loss/unsw_focal_loss_v1/metrics/bce_vs_focal_by_seed.csv`
- 阈值优化：`artifacts/threshold_optimization/unsw_trusted_baseline_v1/threshold_optimization.json`
- 阈值 test 对比：`artifacts/threshold_optimization/unsw_trusted_baseline_v1/test_threshold_comparison.csv`
- 风险映射：`configs/risk/risk_mapping.yaml`
- 风险示例：`artifacts/examples/risk_level_examples.json`

数据文件为 `NF-UNSW-NB15-v2.csv`，本地 SHA-256 为 `05019e3ac8d55b3f074f9f042623bbbb1af2527b8b78558c69353ccb2ccefa3f`。本地数据质量与实验已验证，但发布来源、许可证和上游发布方哈希仍为“待人工确认”。

## 2. 数据规模与实验协议

- 原始 CSV 共 2,390,275 行；可信 Baseline 使用固定分层抽取的 120,000 行资源限制实验池，不代表全量数据实验。
- `split_seed=20260715`，train/validation/test 分别为 96,000/12,000/12,000 行，即 80%/10%/10%。
- train 含良性 92,182 行、攻击 3,818 行；validation 和 test 各含良性 11,523 行、攻击 477 行。
- 先划分，再分别在各分区内部构造长度为 8 的窗口；窗口目标为最后一行，不跨分区。
- 预处理仅在 train 上拟合。仅 train 使用 `balanced_sampling`；validation 和 test 保持自然类别分布。
- validation 用于训练方式、最佳权重和阈值选择；test 不参与调参，只在规则锁定后用于最终评估。
- 数据没有已确认的时间字段，本实验不声称时间外推能力。
- BCE 与 Focal 均使用三个固定模型 seed：`20260715`、`20260716`、`20260717`。模型结构和参数量保持相同，参数量为 568,017。

## 3. BCE Baseline

可信 Baseline 使用 Binary Cross-Entropy、Record Projection、2 层 Transformer、2 个 attention heads、internal size 128、Last Token、窗口长度 8 和 `balanced_sampling`。训练使用 early stopping，并恢复 validation 最佳权重；每个 seed 的决策阈值仅由 validation F1 选择。

三 seed test 指标均值 ± 标准差：

| Precision | Recall | F1 | FAR | Balanced Accuracy | ROC-AUC | PR-AUC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8765 ± 0.0134 | 0.9811 ± 0.0076 | 0.9258 ± 0.0091 | 0.0057 ± 0.0007 | 0.9877 ± 0.0039 | 0.9985 ± 0.0004 | 0.9469 ± 0.0202 |

三个 validation 选定阈值分别为 0.9847、0.9876 和 0.6599。由于不同 seed 的概率校准不同，阈值不能脱离对应权重直接互换。

## 4. Focal Loss 对照

Focal 实验复用同一冻结 split、预处理、模型结构、`balanced_sampling`、early stopping 和三个 seed。唯一训练变化是把 BCE 替换为 Binary Focal Cross-Entropy；固定 `gamma=2.0`、`apply_class_balancing=false`、`alpha=0.25`、`from_logits=false`，未进行 Focal 参数搜索。阈值仍仅由 validation 选择，test 未参与 loss 参数或阈值选择。

| Loss | Precision | Recall | F1 | FAR | Balanced Accuracy | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BCE 均值 | 0.8765 | 0.9811 | 0.9258 | 0.0057 | 0.9877 | 0.9985 | 0.9469 |
| Focal 均值 | 0.9106 | 0.9518 | 0.9306 | 0.0039 | 0.9740 | 0.9990 | 0.9653 |
| Focal - BCE | +0.0340 | -0.0294 | +0.0048 | -0.0019 | -0.0137 | +0.0005 | +0.0184 |

主要变化是 Precision 提高约 3.40 个百分点、PR-AUC 提高约 1.84 个百分点、F1 提高约 0.48 个百分点，FAR 降低约 0.19 个百分点；同时 Recall 降低约 2.94 个百分点，Balanced Accuracy 降低约 1.37 个百分点。因此 Focal 在当前三 seed 资源限制实验中更偏向减少误报并提高预测精度，不能表述为所有指标全面优于 BCE。

## 5. Validation 阈值优化

阈值实验直接复用冻结 BCE 权重和预测概率，没有重新训练。每个 seed 在 validation 上扫描 0.000 至 1.000、步长 0.001，并锁定三种策略：固定 0.5、validation F1 最大、validation FAR 不超过 0.5% 时 Recall 最大。锁定后，test 每个 seed 只执行一次推理，其概率用于三个已锁定策略的并列评估。

以下为 `test_threshold_comparison.csv` 中三个 seed 的算术均值：

| 策略 | Precision | Recall | F1 | FAR | Balanced Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 固定阈值 0.5 | 0.8498 | 0.9965 | 0.9172 | 0.0073 | 0.9946 |
| validation F1 最大 | 0.8762 | 0.9832 | 0.9266 | 0.0058 | 0.9887 |
| validation FAR 约束 | 0.8811 | 0.9734 | 0.9249 | 0.0054 | 0.9840 |

相对固定 0.5，validation F1 策略的平均 F1 提高约 0.94 个百分点、Precision 提高约 2.64 个百分点、FAR 降低约 0.16 个百分点，但 Recall 和 Balanced Accuracy 有所下降。FAR 约束在 validation 上成立不意味着 test FAR 必然不超过 0.5%；实际三个 test FAR 为 0.4950%、0.5644% 和 0.5731%。

## 6. 风险等级输出

风险等级是攻击概率和已锁定决策阈值之上的确定性后处理，不是新模型，也不改变训练或 test 指标：

- Low：`probability < decision_threshold`
- Medium：`decision_threshold <= probability < high_cutoff`
- High：`probability >= high_cutoff`
- `high_cutoff = decision_threshold + 0.75 × (1 - decision_threshold)`

已有 BCE seed `20260717` 示例使用 validation F1 阈值 0.659，对应 `high_cutoff=0.91475`：概率 0.20、0.80、0.95 分别映射为 Low、Medium、High。当前 Focal 本地推理使用其自身 validation 阈值 0.775108，不能与 BCE 示例阈值混用。

## 7. 结论边界

当前结果支持以下有限结论：严格隔离协议下，BCE 已形成稳定的非零攻击检出 Baseline；在保持结构与 split 不变时，Focal 显示了降低 FAR、提高 Precision 和 PR-AUC，同时牺牲部分 Recall 的可复现实验权衡；validation 阈值优化能够调整 Precision、Recall 与 FAR 的取舍；风险等级能够把概率输出转成可展示的分层告警。

这些结果基于 120,000 行实验池，不是全量论文复现；数据无已确认时间字段，不支持时间外推声明；数据来源与许可证尚待人工确认；系统尚未达到生产部署条件。
