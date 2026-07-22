# CampusFlowGuard 竞赛技术报告素材

更新时间：2026-07-18

> 口径声明：本文只整理当前工程中已有、可追溯的代码、配置、日志、指标和演示产物。所有模型结果均基于固定分层抽取的 120,000 行资源限制实验池，不是 2,390,275 行全量实验，也不是 FlowTransformer 论文全量复现结果。

## 1. 项目背景与问题

校园网和实验室网络具有终端类型多、协议行为复杂、正常流量占比高等特点。入侵检测不仅要发现攻击，还要控制误报，否则大量无效告警会增加人工研判成本。CampusFlowGuard 面向合法、防御性的网络流量检测场景，目标是把真实流量数据检查、无泄漏模型实验、概率阈值决策、风险分级和本地展示组织成可运行、可验证、可追溯的闭环。

项目参考 FlowTransformer 上游源代码后进行自主改写，规范上游为 `https://github.com/liamdm/FlowTransformer.git`。FlowTransformer 原项目和 CampusFlowGuard 当前提交版本采用 AGPL-3.0；项目保留原作者、来源和修改说明，不将上游架构、源代码或实现思想声明为个人原创，也不开发漏洞利用、攻击自动化、规避检测或破坏功能。

当前工程聚焦三个实际问题：

1. 避免预处理、窗口构造、训练方式和阈值选择中的数据泄漏。
2. 在攻击样本较少的自然分布下兼顾 Recall、Precision 与 False Alarm Rate（FAR）。
3. 将实验模型封装为可在本机复现的 CSV 推理、风险告警和最小展示接口，同时保留数据、split、配置、权重和输出之间的追溯关系。

## 2. 系统总体流程

### 2.1 实验流程

```text
真实 NF-UNSW-NB15-v2.csv
  -> 全文件质量检查与 37 特征 schema 核对
  -> 固定 seed 分层抽取 120,000 行实验池
  -> 保存并复用 train / validation / test 原始行索引
  -> 仅在 train 拟合数值与类别预处理器
  -> 在三个分区内部独立构造长度为 8 的窗口
  -> Record Projection + 2-layer Transformer + Last Token
  -> 仅用 validation 选择训练方式、最佳权重与决策阈值
  -> test 只在规则锁定后进行最终评估
  -> 保存指标、日志、权重、图和数据指纹
```

### 2.2 本地检测流程

```text
流量 CSV
  -> 校验冻结预处理器要求的 37 个字段
  -> 复用 train-only 预处理器
  -> 在输入文件内部按行序构造 8 行窗口
  -> Focal Transformer 输出 attack probability
  -> validation 锁定阈值输出二元检测结果
  -> Low / Medium / High 风险映射
  -> JSON 检测汇总、逐窗口结果或 FastAPI 响应
```

输入 N 行产生 `N - 7` 个检测窗口，窗口最后一行是目标。输入文件的行序决定窗口上下文；数据没有已确认的时间字段，因此系统不把行序解释为严格时间顺序，也不声称具备时间外推能力。

## 3. 无泄漏 Baseline 方法

### 3.1 数据与划分

本地 `NF-UNSW-NB15-v2.csv` 共 2,390,275 行、45 列，文件 SHA-256 为 `05019e3ac8d55b3f074f9f042623bbbb1af2527b8b78558c69353ccb2ccefa3f`。全文件检查确认 37 个模型特征和 `Attack` 标签存在；良性 2,295,222 行，攻击 95,053 行，未发现缺失值、无穷值、完全重复行、常量列或时间字段。

可信 Baseline 使用 `split_seed=20260715` 分层抽取 120,000 行实验池，再分层划分为：

| 分区 | 行数 | 良性 | 攻击 | 用途 |
| --- | ---: | ---: | ---: | --- |
| train | 96,000 | 92,182 | 3,818 | 拟合预处理器和模型；允许训练集平衡 |
| validation | 12,000 | 11,523 | 477 | 训练方式、最佳权重和阈值选择 |
| test | 12,000 | 11,523 | 477 | 规则锁定后的最终评估 |

三个索引集合互斥并保存为 NPZ。先完成行级划分，再在各分区内部按原始行索引排序并独立构窗，禁止先在全表构窗后拆分。数值缺失值中位数、标准化参数和类别编码词表只在 train 上拟合，validation/test 只执行 transform。

### 3.2 模型与训练规则

主 Baseline 使用 Record Projection、2 层 Transformer、2 个 attention heads、internal size 128、Last Token 分类头和窗口大小 8，共 568,017 个参数。三个模型 seed 为 `20260715`、`20260716`、`20260717`。

`none`、`class_weight`、`balanced_sampling` 的 1 epoch 小试仅比较 validation；三种策略只作用于 train，validation/test 始终维持自然分布。最终选择 `balanced_sampling`，该处理属于 Baseline 类别不平衡处理，不作为创新点。正式训练采用 early stopping，监控 validation 指标并恢复最佳权重；每个 seed 的决策阈值仅由 validation F1 选择。

### 3.3 Baseline 可追溯性

冻结归档 `artifacts/baseline_freeze/unsw_trusted_baseline_v1/` 保存配置、split manifest 与索引、数据指纹、训练日志、模型摘要、最佳权重、三 seed 指标、混淆矩阵和 ROC/PR 曲线。冻结 manifest SHA-256 为 `71e6e8741907a41cc88423ee8792814a45203a3523c1d65039a70516ef7da09a`。

## 4. Focal Loss、Validation 阈值优化与风险分级

### 4.1 Focal Loss 对照

Focal 实验复用相同 split、预处理、模型结构、`balanced_sampling`、early stopping 和三个 seed，唯一训练变化是将 BCE 替换为 Binary Focal Cross-Entropy。固定参数为 `gamma=2.0`、`apply_class_balancing=false`、`alpha=0.25`、`from_logits=false`，没有进行 Focal 超参数搜索。test 未参与 loss 参数、权重或阈值选择。

这一实验的意义是检验损失函数在当前不平衡数据上的 Precision、Recall 和 FAR 权衡，而不是宣称 Focal Loss 为项目原创算法。

### 4.2 Validation 阈值优化

阈值实验复用冻结 BCE 权重和预测概率，不重新训练。每个 seed 仅在 validation 上扫描 0.000 至 1.000、步长 0.001，并比较：

- 固定阈值 0.5；
- validation F1 最大阈值；
- validation FAR 不超过 0.5% 时 Recall 最高阈值。

阈值锁定后，test 每个 seed 只推理一次，所得概率用于三个已锁定策略的并列评估。validation 上满足 FAR 约束不保证 test FAR 必然仍低于 0.5%，报告必须分别呈现两者。

### 4.3 风险分级

风险等级是概率与锁定阈值之上的确定性后处理，不是新模型，也不改变训练指标：

- Low：`probability < decision_threshold`
- Medium：`decision_threshold <= probability < high_cutoff`
- High：`probability >= high_cutoff`
- `high_cutoff = decision_threshold + 0.75 * (1 - decision_threshold)`

阈值与具体权重绑定，不能跨模型混用。当前本地 Focal 推理使用 seed `20260717`、validation F1 阈值 `0.7751080393791199`，对应 `high_cutoff=0.94377700984478`。

## 5. 实验设置与真实结果

### 5.1 结果口径

下表均为 120,000 行实验池上的三个固定 seed test 结果均值 ± 标准差。test 不参与训练方式、loss 参数、最佳权重或阈值选择。

| Loss | Precision | Recall | F1 | FAR | Balanced Accuracy | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BCE | 0.8765 ± 0.0134 | 0.9811 ± 0.0076 | 0.9258 ± 0.0091 | 0.0057 ± 0.0007 | 0.9877 ± 0.0039 | 0.9985 ± 0.0004 | 0.9469 ± 0.0202 |
| Focal | 0.9106 ± 0.0084 | 0.9518 ± 0.0128 | 0.9306 ± 0.0022 | 0.0039 ± 0.0004 | 0.9740 ± 0.0062 | 0.9990 ± 0.0001 | 0.9653 ± 0.0050 |

Focal 相比 BCE 的三 seed 平均变化为：Precision `+0.0340`、Recall `-0.0294`、F1 `+0.0048`、FAR `-0.0019`、Balanced Accuracy `-0.0137`、ROC-AUC `+0.0005`、PR-AUC `+0.0184`。因此当前证据支持“Focal 降低误报并提高 Precision、F1 和 PR-AUC，但牺牲部分 Recall 与 Balanced Accuracy”的权衡，不支持“所有指标全面提升”的结论。

### 5.2 BCE 阈值策略对比

以下为已有 test 阈值对比表中三个 seed 的算术均值：

| 策略 | Precision | Recall | F1 | FAR | Balanced Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 固定 0.5 | 0.8498 | 0.9965 | 0.9172 | 0.0073 | 0.9946 |
| validation F1 最大 | 0.8762 | 0.9832 | 0.9266 | 0.0058 | 0.9887 |
| validation FAR 约束 | 0.8811 | 0.9734 | 0.9249 | 0.0054 | 0.9840 |

validation F1 策略相对固定 0.5 提高平均 Precision 和 F1、降低 FAR，但 Recall 与 Balanced Accuracy 有所下降。该结果说明阈值是安全运营取舍的一部分，不能只报告单一准确率。

### 5.3 建议引用的现有图表

- BCE 三 seed 混淆矩阵、ROC 和 PR 图：`artifacts/baseline_freeze/unsw_trusted_baseline_v1/plots/`
- BCE 与 Focal 指标对比图：`artifacts/experiments/focal_loss/unsw_focal_loss_v1/plots/bce_vs_focal_metrics.png`
- validation 阈值扫描与 test 策略图：`artifacts/threshold_optimization/unsw_trusted_baseline_v1/plots/`

## 6. 本地推理、FastAPI 与演示流程

### 6.1 本地推理

本地入口复用冻结 preprocessing，校验其 SHA-256，并加载 validation PR-AUC 选出的 Focal seed `20260717` 最佳权重。输入 CSV 必须包含 37 个特征且至少 8 行；输出包括输入与权重元数据、逐窗口攻击概率、二元检测、风险等级和汇总统计。

```powershell
python -m campus_flow_guard.inference `
  --config configs\inference\local_focal.yaml `
  --input artifacts\examples\local_inference_input.csv `
  --output artifacts\examples\local_inference_output.json
```

当前 TensorFlow 2.17.1 为 Windows CPU-only 构建；CPU 推理已验证，GPU 方案尚未验证。

### 6.2 FastAPI 最小展示

FastAPI 直接封装已有推理模块，不改变模型和决策逻辑。服务默认只监听 `127.0.0.1:8000`，提供原生 HTML 上传页和 `POST /predict`。上传仅允许 `.csv`，上限 20 MiB，响应包含总流量数、评估窗口数、攻击数、风险等级统计和最多 20 条样本预测。

```powershell
python -m campus_flow_guard.api --config configs\api\local_demo.yaml
```

默认启动、首页 HTTP 200、真实 CSV 的 `POST /predict` HTTP 200 和 `Ctrl+C` 停止流程均已有实际验证记录。该接口是本地演示，不连接外部 API，也未达到生产部署条件。

### 6.3 固定演示

```powershell
python scripts\run_demo.py
```

`mixed_demo.csv` 来自冻结 validation 分区，仅含 16 行、37 个模型特征，不含源/目的 IP、`Attack` 或 `Label`。固定拼接为前 8 条良性来源记录和前 8 条攻击来源记录，该顺序只服务于可重复展示，不表示真实时间顺序。

实际演示中，16 行输入产生 9 个窗口，检测为攻击 8 个，风险统计为 Low 1、Medium 6、High 2。该结果是演示输出，不是新增 test 指标或正式实验结论。输入、split、权重和输出哈希记录在 `artifacts/demo/demo_manifest.json` 与 `artifacts/demo/mixed_demo_report.json`。

## 7. 创新点、局限性及后续工作

### 7.1 可核验的工程与实验贡献

1. **无泄漏、可冻结的数据协议**：保存原始行索引，先划分再独立构窗，train-only 拟合预处理器，并用测试验证索引互斥和窗口边界。
2. **训练、选择与最终评估职责分离**：仅在 train 做平衡，仅用 validation 选择训练方式、权重和阈值，test 只作最终评估。
3. **同结构、同 split、同 seed 的受控损失对照**：将 BCE 与 Focal 的差异限制在 loss，量化 Precision、Recall、FAR 和 PR-AUC 的真实权衡。
4. **面向告警运营的阈值与风险层**：支持 validation F1 和 FAR 约束策略，并把概率映射为可追溯的 Low/Medium/High 告警；阈值、模型与来源绑定。
5. **带完整性校验的本地交付链路**：推理时核验权重和预处理器 SHA-256，输出输入指纹、模型 seed、阈值来源和逐窗口结果；提供 CLI、FastAPI 和 validation-only 小型演示材料。

上述贡献主要是实验正确性、安全应用和工程可追溯性改进。Transformer、Record Projection、Focal Loss、类别平衡、early stopping、FastAPI 等基础方法或组件不是本项目原创。

### 7.2 当前局限性

- 结果只基于 120,000 行资源限制实验池，尚无全量数据实验。
- 数据没有显式时间字段，随机分层 split 不支持时间外推结论；分区内按索引排序的窗口还可能连接原 CSV 中不连续的记录。
- 当前任务是二元 Attack/Benign 检测，没有完成九类攻击的多分类识别。
- 数据仅覆盖当前本地 `NF-UNSW-NB15-v2.csv`，尚无跨数据集、跨校园或真实在线流量泛化验证。
- 当前 TensorFlow 只使用 CPU；WSL2/GPU、RTX 5060 与相关 CUDA/cuDNN 组合尚未验证。
- FastAPI 是本地最小展示，尚未完成生产级鉴权、并发、审计、部署和完整安全测试。
- 数据发布来源、许可证、下载证据和上游发布件哈希仍待人工确认。
- CampusFlowGuard 当前提交版本已落实 AGPL-3.0；具体版权主体、文件级声明和 Python 完整传递依赖许可证清单仍待确认。

### 7.3 后续工作

1. 人工补齐数据发布页面、许可证、下载记录和上游哈希证据。
2. 在保持冻结 Baseline 的前提下，以新实验标识开展全量数据、实体隔离或具有可靠时间字段的数据评估。
3. 评估攻击多分类、跨数据集泛化和概率校准，继续严格限定 validation/test 职责。
4. 补充竞赛报告模板适配、系统架构图、实验图表题注、演示截图和视频材料。
5. 对本地 API 完成交付前安全检查，并生成最终代码、配置、权重、报告和演示产物哈希清单。

## 8. 材料索引与待确认项

| 材料 | 路径 | 状态 |
| --- | --- | --- |
| 项目状态与冻结边界 | `docs/project_state.md` | 已有 |
| 实验结果摘要 | `docs/experiment_summary.md` | 已有 |
| Baseline 冻结报告 | `docs/baseline_report.md` | 已有 |
| 数据划分协议 | `docs/data_split_protocol_zh.md` | 已有 |
| 数据质量报告 | `docs/dataset_quality_report_zh.md` | 已有 |
| 原仓库审计 | `docs/original_repository_audit_zh.md` | 已有 |
| 演示数据说明 | `docs/demo_data_manifest.md` | 已有 |
| 数据正式来源、许可证与上游哈希 | 尚无确认证据 | 待人工确认 |
| 竞赛官方报告模板、页数和格式要求 | 当前项目未保存 | 待人工提供或确认 |
| 系统架构成图、运行截图、演示视频 | 当前项目未完成 | 待制作 |
| 独立代码许可证与依赖许可证清单 | 当前尚未冻结 | 待人工确认 |

本文可作为技术报告正文的事实底稿，但不能替代竞赛官方模板、许可证审查或最终提交前的逐项验收。
