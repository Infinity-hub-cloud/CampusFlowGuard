# CampusFlowGuard strict 数据划分协议

## 适用范围

本协议用于 `NF-UNSW-NB15-v2.csv` 的资源受限可信 baseline。当前文件没有显式时间字段，因此本协议只进行固定随机种子的分层划分，不声称时间外推、未来流量检测或跨时间泛化能力。数据来源、上游许可证和发布方哈希仍为“待人工确认”。

## 固定实验池与划分

1. 读取全文件 `Attack` 标签并映射为二元标签：`Benign=0`，其他攻击类型为 `1`。
2. 使用固定 `split_seed=20260715`，从全文件分层抽取 120,000 行资源受限实验池。该池约占原文件 2,390,275 行的 5.0%，不是全量 baseline。
3. 在实验池内按类别分层划分 train/validation/test = 80%/10%/10%。三个索引集合必须互斥。
4. 原始 CSV 行索引保存为压缩 NPZ；JSON manifest 保存数据 SHA-256、schema SHA-256、seed、数量、类别分布和范围。后续运行必须复用索引，不能重新随机划分。

## 预处理边界

数值缺失值中位数与标准化参数只在 train 行上拟合；类别 OrdinalEncoder 的词表也只在 train 行上拟合。validation/test 只能调用 transform，未见类别映射为保留的 unknown 编号。禁止在全数据上拟合任何预处理步骤。

## 窗口边界

先完成行级划分，再分别在 train、validation、test 内按原始 CSV 行索引排序并构造窗口。窗口大小为 8，目标为窗口最后一行。随机分层会使同一集合内相邻行在原 CSV 中可能存在间隔，但任何窗口都不得包含其他集合的行。不得先在全表生成窗口后拆分。

## 训练与选择边界

- `none`、`class_weight`、`balanced_sampling` 只改变 train 的训练方式；validation/test 始终保持自然分布。
- 三种方式的小试只查看 validation，按固定阈值 0.5 的非零 TP 和 F1 选择标准基线方案；禁止读取 test 结果进行方案选择。
- 主 baseline 使用 validation PR-AUC 做 early stopping，并恢复最佳 validation 权重。
- 决策阈值只在 validation 上按最大 F1 选择；选定后原样应用到 test。
- test 对每个模型 seed 只进行一次最终评估。

## 可复现性与限制

固定 split seed 和三个模型 seed；保存配置副本、划分索引、数据指纹、最佳权重、逐 epoch 日志、最终指标与图。CPU 浮点实现仍可能因底层并行库产生微小差异。该资源受限结果不能写成 FlowTransformer 论文全量复现结果。
