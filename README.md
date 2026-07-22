# CampusFlowGuard

CampusFlowGuard 是面向校园网与实验室网络的防御性 AI 网络入侵检测项目。项目参考 FlowTransformer 上游源代码后进行自主改写，围绕真实流量数据建立可追溯的数据检查、无泄漏训练评估、阈值选择、风险分级、本地推理和最小展示接口。本项目不开发漏洞利用、攻击自动化、规避检测或破坏功能。

当前项目已完成可信 BCE Baseline、validation 阈值优化、Low/Medium/High 风险映射、Focal Loss 对照、本地 CSV 推理和 FastAPI 最小展示。项目状态与冻结边界见 `docs/project_state.md`。

## FlowTransformer 来源、许可证与原创性边界

- 规范上游：`https://github.com/liamdm/FlowTransformer.git`
- 原作者：`liamdm / liam@riftcs.com`
- 上游许可证：GNU Affero General Public License v3.0（AGPL-3.0）
- 本地审计基线 commit：`52236c8c9feccdeb44acf578789db7b8b36bacea`
- 改写关系：CampusFlowGuard 参考上游源代码、架构和实验流程后进行自主改写；`baseline.py`、`train.py` 已在文件头明确标记。
- 本项目提交版本保留并适用 AGPL-3.0，完整文本位于根目录 `LICENSE`，来源与修改说明位于 `NOTICE.md`。

FlowTransformer 的 Record Projection、Transformer 编码、分类头、模型构建思想和原始实现不属于 CampusFlowGuard 原创。CampusFlowGuard 的新增工作主要是严格 split 与窗口隔离、train-only 预处理、可追溯实验归档、validation-only 阈值优化、风险分级、Focal 受控对照、本地推理完整性校验、FastAPI 展示、测试和中文交付文档。Focal Loss、类别平衡、early stopping、TensorFlow/Keras 和 FastAPI 等通用方法或组件本身也不作为原创算法宣称。

作品当前仅提交竞赛评委私有评审；私评场景不改变 AGPL-3.0、上游版权和来源说明的保留。后续若公开发布或提供网络服务，仍需按 AGPL-3.0 和第三方许可证履行相应义务。

## 技术流程

当前可运行检测链路如下：

```text
流量 CSV
  → 校验 37 个模型特征
  → 复用冻结的 train-only 预处理器
  → 在输入 CSV 内按行序构造 8 行窗口
  → 窗口最后一行作为检测目标
  → Record Projection + 2-layer Transformer + Last Token
  → Focal Loss 最佳 validation 模型输出 attack probability
  → validation 选定阈值进行二元检测
  → Low / Medium / High 风险分级
  → 逐窗口结果与检测汇总
```

本地推理使用 Focal seed `20260717` 的最佳权重。该模型由 validation PR-AUC 选择，决策阈值 `0.7751080393791199` 由 validation F1 选择；test 未参与模型或阈值选择。

## 冻结边界

- Baseline split manifest、原始行索引和 train/validation/test 规则已冻结。
- 预处理只在 train 上拟合；validation 用于模型与阈值选择；test 只用于最终评估。
- 冻结归档位于 `artifacts/baseline_freeze/unsw_trusted_baseline_v1/`，不得覆盖。
- 当前结果基于 120,000 行资源限制实验池，不是全量 FlowTransformer 论文复现。
- 数据来源、上游许可证和发布方哈希仍为“待人工确认”。

## 环境安装

### 前提

- Windows PowerShell。
- Python `>=3.11,<3.12`；当前验证环境为 Python 3.11。
- 项目根目录为 `D:\Users\LENOVO\Desktop\AI_safety_competition_code`。
- 本地推理需要现有 Focal 权重、冻结 preprocessing 和配置文件。生成工件默认不提交 Git，新环境必须从项目交付包恢复这些文件。

进入项目目录并安装 ML 与本地 API 可选依赖：

```powershell
Set-Location D:\Users\LENOVO\Desktop\AI_safety_competition_code
python -m pip install --editable ".[ml,api]"
```

该命令已于 2026-07-17 在当前工程实际执行成功。主要运行依赖由 `pyproject.toml` 声明，包括 TensorFlow 2.17.1、Keras 3.15.0、FastAPI 0.115.6、Uvicorn 0.34.0 和 python-multipart 0.0.20。

运行单元测试：

```powershell
python -m unittest discover -s tests\unit -p "test_*.py" -v
```

## 快速演示

项目已提供三个来自冻结 validation 分区的 16 行小型 CSV。生成规则、来源索引、数据与 split 哈希见 `docs/demo_data_manifest.md` 和 `artifacts/demo/demo_manifest.json`。演示输入只含 37 个模型特征，不含 IP 地址或标签字段。

在项目根目录运行：

```powershell
python scripts\run_demo.py
```

该命令加载现有最佳 Focal Loss 权重，检测 `artifacts/demo/mixed_demo.csv`，在终端输出总流量数、攻击数和 Low/Medium/High 统计，并将完整逐窗口结果保存为 `artifacts/demo/mixed_demo_report.json`。该命令已于 2026-07-17 在当前环境实际验证。

需要从冻结 validation 索引重新生成演示 CSV 时运行：

```powershell
python scripts\prepare_demo_data.py
```

重新生成会读取本地真实数据文件，并在写入前校验数据 SHA-256、冻结 split SHA-256 和 validation/test 隔离关系；不会训练模型，也不会修改冻结归档。

## 本地 CSV 推理

使用项目自带的真实 validation 示例：

```powershell
python -m campus_flow_guard.inference `
  --config configs\inference\local_focal.yaml `
  --input artifacts\examples\local_inference_input.csv `
  --output artifacts\examples\local_inference_output.json
```

该命令已实际验证。示例输入为 16 行，生成 9 个窗口；输出保存在 `artifacts/examples/local_inference_output.json`。

替换 `--input` 和 `--output` 即可检测其他本地 CSV。已有输出路径会被写入新结果，正式使用时建议为每次运行提供独立文件名以便追溯。

## FastAPI 启动、访问与关闭

在项目根目录前台启动本地服务：

```powershell
python -m campus_flow_guard.api --config configs\api\local_demo.yaml
```

默认只监听本机地址：

```text
http://127.0.0.1:8000
```

浏览器打开该地址后，选择 CSV 并点击“开始检测”。页面使用原生 HTML、CSS 和 JavaScript，不连接外部 API。

也可以在另一个 PowerShell 窗口直接调用接口：

```powershell
curl.exe -F "file=@artifacts\examples\local_inference_input.csv" `
  http://127.0.0.1:8000/predict
```

默认启动、首页 HTTP 200、真实 CSV 的 `POST /predict` HTTP 200 和服务进程停止流程已于 2026-07-17 验证。

关闭服务：回到运行 Uvicorn 的 PowerShell 窗口，按 `Ctrl+C`。关闭后 `http://127.0.0.1:8000` 将不再可访问。

如果 8000 端口已被占用，可使用入口已支持的端口覆盖参数：

```powershell
python -m campus_flow_guard.api `
  --config configs\api\local_demo.yaml `
  --port 8001
```

此时浏览器地址相应改为 `http://127.0.0.1:8001`。

## CSV 输入要求

### 基本要求

- 文件扩展名必须为 `.csv`；FastAPI 当前拒绝其他扩展名。
- CSV 必须至少包含 8 行。窗口大小为 8，N 行输入产生 `N - 7` 个检测窗口。
- CSV 行顺序直接决定窗口上下文。调用方必须提供具有稳定、合理顺序的流记录；项目不会声称该顺序等同于时间顺序。
- CSV 不要求提供 `Attack` 或 `Label`；额外字段会被忽略。
- 数值字段必须能够被 pandas 和冻结预处理器解析；不应包含正负无穷值。
- FastAPI 单次上传大小上限为 20 MiB；本地 CLI 不使用该上传限制。

### 必需的 37 个特征

字段名必须与 `configs/datasets/dataset_schema.yaml` 一致：

```text
NUM_PKTS_UP_TO_128_BYTES
SRC_TO_DST_SECOND_BYTES
OUT_PKTS
OUT_BYTES
NUM_PKTS_128_TO_256_BYTES
DST_TO_SRC_AVG_THROUGHPUT
DURATION_IN
L4_SRC_PORT
ICMP_TYPE
PROTOCOL
SERVER_TCP_FLAGS
IN_PKTS
NUM_PKTS_512_TO_1024_BYTES
CLIENT_TCP_FLAGS
TCP_WIN_MAX_IN
NUM_PKTS_256_TO_512_BYTES
SHORTEST_FLOW_PKT
MIN_IP_PKT_LEN
LONGEST_FLOW_PKT
L4_DST_PORT
MIN_TTL
DST_TO_SRC_SECOND_BYTES
NUM_PKTS_1024_TO_1514_BYTES
DURATION_OUT
FLOW_DURATION_MILLISECONDS
TCP_FLAGS
MAX_TTL
SRC_TO_DST_AVG_THROUGHPUT
ICMP_IPV4_TYPE
MAX_IP_PKT_LEN
RETRANSMITTED_OUT_BYTES
IN_BYTES
RETRANSMITTED_IN_BYTES
TCP_WIN_MAX_OUT
L7_PROTO
RETRANSMITTED_OUT_PKTS
RETRANSMITTED_IN_PKTS
```

## 输出字段说明

### 本地推理 JSON

顶层字段：

- `metadata`：输入路径和 SHA-256、输入行数、窗口大小、模型 seed、validation 选择指标、权重 SHA-256 和阈值来源。
- `detection_summary`：整体检测汇总。
- `detections`：每个窗口的预测列表。

`detection_summary` 主要字段：

- `input_row_count`：输入 CSV 总行数。
- `window_count`：实际检测窗口数。
- `detected_attack_count`：攻击概率大于等于决策阈值的窗口数。
- `detected_attack_ratio`：攻击窗口占全部窗口的比例。
- `risk_level_counts`：Low、Medium、High 数量。
- `maximum_attack_probability`：最高攻击概率。
- `mean_attack_probability`：平均攻击概率。
- `decision_threshold`：二元检测阈值。
- `high_cutoff`：Medium 与 High 的分界值。

`detections` 每项主要字段：

- `window_index`：当前输出中的窗口序号。
- `target_row_number`：输入 CSV 中作为窗口目标的零基行号。
- `attack_probability`：模型输出的攻击概率。
- `detected_attack`：是否达到决策阈值。
- `risk_level`：Low、Medium 或 High。
- `decision_threshold`、`high_cutoff`：该预测使用的阈值边界。

### `POST /predict` 响应

- `total_flows`：上传 CSV 的总流量行数。
- `evaluated_windows`：评估窗口数。
- `attack_count`：检测为攻击的窗口数。
- `risk_level_counts`：Low、Medium、High 统计。
- `sample_predictions`：最多 20 条样本预测；数量由 `configs/api/local_demo.yaml` 控制。

风险等级是概率的后处理表示，不是新模型：低于决策阈值为 Low；达到决策阈值但低于高风险边界为 Medium；达到高风险边界为 High。

## 常见错误排查

### `No module named campus_flow_guard`

确认当前目录正确，并重新执行：

```powershell
python -m pip install --editable ".[ml,api]"
```

### 缺少 FastAPI、Uvicorn 或 multipart

典型表现为 `No module named fastapi`、`No module named uvicorn` 或文件上传依赖错误。执行同一安装命令补齐项目已声明的 API 依赖：

```powershell
python -m pip install --editable ".[ml,api]"
```

### 浏览器显示无法连接

确认启动命令所在窗口仍在运行，并检查终端是否显示启动失败。默认地址只在本机有效：`http://127.0.0.1:8000`。

### `[Errno 10048]` 或端口已占用

说明 8000 端口已有进程监听。关闭占用进程，或使用 `--port 8001` 启动并访问 `http://127.0.0.1:8001`。

### HTTP 415

上传文件扩展名不是配置允许的 `.csv`。请使用 CSV 文件，不要只修改非 CSV 文件的扩展名来绕过检查。

### HTTP 413

上传文件超过 `configs/api/local_demo.yaml` 中的 20 MiB 限制。使用本地 CLI 分析文件，或在明确评估内存风险后另建 API 配置；不要静默修改已记录配置。

### HTTP 422 或“缺少必需字段”

检查 CSV 是否包含全部 37 个字段，字段名的大小写和下划线必须完全一致。还需确认数值列可解析、CSV 至少有 8 行。

### 权重或 preprocessing SHA-256 不匹配

推理入口会拒绝加载被替换的权重或预处理器。应从项目交付包恢复配置所指向的正确工件，不要通过修改哈希配置绕过校验。

### TensorFlow 未发现 GPU

当前已验证环境使用 CPU 推理；未发现 GPU 不阻止本地运行，但首次加载模型和大 CSV 推理可能需要等待。GPU 兼容性仍需在独立环境验证。

## 数据、许可证与第三方声明

- 数据文件、抓包、模型权重和生成工件默认不提交 Git。
- `NF-UNSW-NB15-v2.csv` 的本地数据质量已经验证，但数据来源、许可证和上游发布方哈希仍待人工确认。
- CampusFlowGuard 参考并自主改写自 FlowTransformer：`https://github.com/liamdm/FlowTransformer.git`。
- 本提交版本适用 GNU Affero General Public License v3.0；完整文本位于根目录 `LICENSE`，备份位于 `LICENSES/FlowTransformer-AGPL-3.0.txt`。
- 上游版权、改写关系和本项目新增工作见 `NOTICE.md`；第三方依赖见 `docs/third_party_licenses.md`。
- 不得将 FlowTransformer 架构、源代码或论文实现整体声明为 CampusFlowGuard 原创。

## 相关文档

- `docs/project_state.md`：当前状态、冻结内容和后续任务。
- `docs/baseline_report.md`：可信 Baseline 冻结报告。
- `docs/data_split_protocol_zh.md`：无泄漏数据划分和窗口协议。
- `docs/dataset_quality_report_zh.md`：真实数据质量检查结果。
- `docs/environment_diagnosis_zh.md`：环境诊断与兼容性说明。
- `docs/original_repository_audit_zh.md`：原始 FlowTransformer 仓库审计。
