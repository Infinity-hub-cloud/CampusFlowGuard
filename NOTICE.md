# CampusFlowGuard 来源与修改声明

## 上游项目

- 项目：FlowTransformer
- 原作者：liamdm / liam@riftcs.com
- 规范上游：<https://github.com/liamdm/FlowTransformer.git>
- 本地审计基线 commit：`52236c8c9feccdeb44acf578789db7b8b36bacea`
- 上游许可证：GNU Affero General Public License v3.0（AGPL-3.0）
- 完整许可证：根目录 `LICENSE`

## 改写关系

CampusFlowGuard 参考 FlowTransformer 上游源代码、论文实现结构和实验流程后进行自主改写。项目保留 FlowTransformer 的来源、原作者和 AGPL-3.0 许可证说明，不将 FlowTransformer 的模型架构、Record Projection、Transformer 编码、分类头或原始实现思想声明为 CampusFlowGuard 原创。

`src/campus_flow_guard/baseline.py` 与 `src/campus_flow_guard/train.py` 已明确标记为“参考上游源代码后的自主改写”。当前本地比对未发现整文件直接复制或可由连续文本证据确认的逐段复制，但该结果只用于来源追溯，不改变已确认的参考改写关系，也不构成法律意见。

## CampusFlowGuard 新增工作

相对于上游参考实现，CampusFlowGuard 当前新增和重构的工程内容主要包括：

- 固定随机种子、保存并复用 split 索引的数据协议；
- train-only 预处理、分区内独立构窗和无泄漏单元测试；
- `none`、`class_weight`、`balanced_sampling` 的 Baseline 对照；
- 配置、数据指纹、日志、权重、指标和图表的追溯归档；
- 在相同 split、结构和 seed 下进行的 Focal Loss 受控实验；
- 仅使用 validation 的阈值优化与 FAR 约束策略；
- Low / Medium / High 风险分级；
- 权重和预处理器哈希校验、本地 CSV 推理、FastAPI 和最小演示材料；
- 中文文档、环境诊断、许可证核查和提交合规材料。

Focal Loss、类别平衡、early stopping、FastAPI、TensorFlow/Keras 等方法或组件本身不是本项目原创；其第三方许可证见 `docs/third_party_licenses.md`。

## 提交范围

当前作品仅提交竞赛评委私有评审。私有评审场景不改变本项目对上游来源、版权和 AGPL-3.0 的保留。提交包应包含本项目对应源代码、根目录 `LICENSE`、本 `NOTICE.md` 和必要的第三方声明。

数据集 `NF-UNSW-NB15-v2.csv` 的发布许可和再分发条件仍待人工确认；原始数据及真实数据派生样例不应因本 NOTICE 被视为获得分发授权。模型权重由 CampusFlowGuard 实验流程自行训练，但是否纳入提交包仍需结合数据许可和竞赛要求确认。

## 修改声明

CampusFlowGuard modifications and independent engineering additions: CampusFlowGuard contributors, 2026.

This project is provided without warranty under the terms of the GNU Affero General Public License v3.0. See `LICENSE` for the complete terms.
