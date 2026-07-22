# UNSW-NB15 相关 CSV 数据质量报告

## 检查范围与可追溯性

- 检查时间：2026-07-15（本机执行）
- 检查范围：全文件，不是抽样。
- 文件：`D:\Users\LENOVO\Desktop\AI_safety_competition_data\raw\NF-UNSW-NB15-v2.csv`
- 格式：CSV。
- 大小：441,867,785 bytes（421.398 MiB）。
- SHA-256：`05019e3ac8d55b3f074f9f042623bbbb1af2527b8b78558c69353ccb2ccefa3f`。
- 行数 / 列数：2,390,275 / 45。
- 机器可读原始检查结果：[dataset_profile.json](../artifacts/metrics/dataset_profile.json)。

已由代码直接确认：文件名与原 FlowTransformer 的 `demonstration.ipynb` 和 `FlowTransformer_demo.ipynb` 中使用的 `NF-UNSW-NB15-v2.csv` 一致。该本地文件的实际下载者、下载时间、发布许可证和其 SHA-256 是否等同于上游发布件，均**待确认**；本阶段没有联网访问或重新下载数据，不能将文件名一致写成来源已验证。

## 来源状态

原仓库 `demonstration.ipynb` 明确记录了 UQ NIDS 数据集页面及一个下载接口，且代码期望从压缩包中的 `fe6cb615d161452c_MOHANAD_A4706/data/NF-UNSW-NB15-v2.csv` 解出该文件。该事实可作为复现线索，具体链接、手工复核步骤和限制见 [dataset_download_guide_zh.md](dataset_download_guide_zh.md)。链接当前可访问性、上游文件哈希与许可证均未验证。

## 字段与标签核验

实际 CSV 的 45 个字段包括 37 个 schema 特征列、`Attack` 标签列，以及 7 个 schema 未纳入的字段：

```text
IPV4_SRC_ADDR, IPV4_DST_ADDR, DNS_QUERY_ID, DNS_QUERY_TYPE,
DNS_TTL_ANSWER, FTP_COMMAND_RET_CODE, Label
```

`configs/datasets/dataset_schema.yaml` 所列的 37 个特征和 `Attack` 标签均存在，故 `schema_missing_columns` 为空。额外字段不是错误数据的证据，也不应通过修改 schema 将其悄然纳入基线输入；后续特征选择必须单独设计、记录并经实验验证。

当前 schema 的 9 个类别字段均存在。实际 CSV 将它们读为数值编码（`int64` 或 `float64`），这与“以离散编号作为类别字段”的处理意图不冲突，但其编码语义、缺失码和未知类别策略仍待后续数据工程阶段确认。schema 未声明每列的预期 pandas dtype，因此本阶段只能报告实际 dtype，不能声称完成了严格的类型契约验证。

`Attack` 是字符串多分类标签，`Benign` 是良性值；`Label` 是额外的二元整型列。仅读取两列得到的交叉表表明：`Label=0` 的 2,295,222 行全部对应 `Attack=Benign`，`Label=1` 的 95,053 行全部对应九种非良性 `Attack` 值。该映射在此文件中成立，故 schema 的 `label_column: Attack` 与 `benign_label: Benign` 匹配实际数据。

## 质量检查结果

| 项目 | 实际结果 |
| --- | --- |
| 缺失值 / NaN | 0 / 0 |
| 数值 `inf` | 0 |
| 完全重复行 | 0 |
| 常量列 | 0 |
| 时间字段 | schema 为 `null`；字段名规则未发现时间候选列 |
| 良性 | 2,295,222（96.0233%） |
| 攻击（所有非 Benign `Attack`） | 95,053（3.9767%） |

攻击标签分布：Exploits 31,551；Fuzzers 22,310；Generic 16,560；Reconnaissance 12,779；DoS 5,794；Analysis 2,299；Backdoor 2,169；Shellcode 1,427；Worms 164。该类别分布高度不平衡，尤其 Worms 很少；本阶段没有执行重采样、划分或训练。

配置类别列的全文件不同取值数为：`CLIENT_TCP_FLAGS` 16、`L4_SRC_PORT` 64,601、`TCP_FLAGS` 17、`ICMP_IPV4_TYPE` 256、`ICMP_TYPE` 1,803、`PROTOCOL` 255、`SERVER_TCP_FLAGS` 15、`L4_DST_PORT` 64,625、`L7_PROTO` 86。端口列高基数是事实统计，不表示已经决定编码方式。

## Schema 结论

- 字段存在性：通过，所需 37 个特征与 `Attack` 均存在。
- 标签映射：通过，`Attack=Benign` 与 `Label=0` 一一对应；非良性攻击类型与 `Label=1` 一一对应。
- 类别字段：配置字段均存在；实际为数值编码，语义和编码方案待确认。
- 类型匹配：待确认。schema 缺少预期 dtype 定义，不能仅凭字段存在宣称类型契约成立。
- 时间顺序：待确认。文件没有显式时间字段，不能据此证明 `LastRows` 适合时间外推评估。

## 后续限制

不得在未定义防泄漏切分规则前进行窗口构造或模型训练。进入 Phase 1-C 前，应先确认数据来源与许可证、决定是否保留额外字段，并设计按时间或实体边界隔离的 train/validation/test 划分方案。
