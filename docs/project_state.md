# CampusFlowGuard 项目状态快照

更新时间：2026-07-18

## 1. 项目目标

CampusFlowGuard 面向校园网和实验室网络的合法、防御性网络入侵检测场景，目标是在正确理解和复现 FlowTransformer 工作的基础上，建立可运行、可验证、可追溯的 AI 流量检测作品。

项目当前覆盖真实 UNSW-NB15 相关流量数据检查、无泄漏实验协议、Transformer 检测模型、阈值优化、风险等级映射、本地 CSV 推理和最小 FastAPI 展示。所有实验结果必须能够追溯到数据指纹、split、配置、随机种子、权重和指标文件；未经实际运行的结果不得写成已验证。

CampusFlowGuard 参考 FlowTransformer 上游源代码后进行自主改写，规范上游为 `https://github.com/liamdm/FlowTransformer.git`，上游及本提交版本采用 AGPL-3.0。FlowTransformer 架构、源代码和实现思想不属于 CampusFlowGuard 原创；CampusFlowGuard 新增工程工作单独说明。数据来源、许可和发布方哈希中尚未确认的部分继续标记为“待人工确认”。

## 2. 已完成阶段

### Phase 0：审计与工程准备

- 完成原始 FlowTransformer 仓库只读审计。
- 完成 CampusFlowGuard `src` layout、环境配置、许可证说明和基础测试初始化。
- 完成 Windows、Python、TensorFlow/Keras 及相关依赖环境诊断。

### Phase 1：数据与训练闭环

- 建立 UNSW-NB15 schema 和数据检查工具。
- 对真实 `NF-UNSW-NB15-v2.csv` 完成文件哈希、字段、标签、缺失值、重复行、常量列和类别分布检查。
- 完成真实数据到预处理、窗口、Transformer、训练、预测和指标输出的最小闭环。

### Phase A：可信 Baseline

- 使用固定分层 120,000 行资源限制实验池。
- 固定 `split_seed=20260715`，train/validation/test 为 96,000/12,000/12,000 行。
- 先划分再分别构窗，预处理仅在 train 上拟合。
- validation 选择 `balanced_sampling`；test 仅作最终评估。
- 固定三个模型 seed：`20260715`、`20260716`、`20260717`。
- 完成 BCE Baseline 三 seed 指标、权重、日志、混淆矩阵、ROC/PR 曲线和冻结归档。

### Phase B：阈值、风险与 Focal 实验

- 完成仅基于 validation 的阈值扫描与策略对比。
- 完成固定阈值、validation F1 最优阈值和 validation FAR 约束阈值评估。
- 完成 Low/Medium/High 风险等级映射。
- 在相同 split、模型结构、训练规则和 seed 下完成 BCE 与 Focal Loss 对照。
- Focal 实验唯一训练变化为 loss；未使用 test 选择 loss 参数或阈值。

### Phase C：本地推理与展示

- 完成本地 CSV 推理入口，输出逐窗口攻击概率、风险等级和检测汇总。
- 选择 validation PR-AUC 最优的 Focal seed `20260717` 权重用于本地推理。
- 完成 `POST /predict` FastAPI 接口和无前端框架的最小 HTML 上传页面。
- 使用真实 validation 示例 CSV 完成本地 CLI 和 FastAPI 端到端验证。
- 完成 README 运行说明更新；安装、本地推理、默认 FastAPI 启动、首页访问、真实 CSV 请求和服务停止流程已在当前工程验证。
- 完成 Phase C-3.2 最小可复现演示材料：三个 16 行、37 特征的 validation-only CSV、可追溯 manifest、固定演示脚本和 JSON 检测报告。
- 演示数据不含 IP 地址或标签字段；`mixed_demo.csv` 的拼接顺序不代表时间顺序，未使用 test 进行选择或调参。

### Phase D：比赛材料整理

- 完成已有真实实验结果摘要，统一汇总 BCE Baseline、Focal Loss、validation 阈值优化和风险等级映射。
- 摘要明确记录 120,000 行资源限制实验池、固定 split、test 不参与调参以及主要指标变化与结论边界。
- 汇总文档位于 `docs/experiment_summary.md`；本阶段未重新训练或修改历史实验结果。
- 完成竞赛技术报告事实素材，覆盖项目背景、系统流程、无泄漏 Baseline、Focal/阈值/风险方法、真实结果、本地展示、工程贡献、局限性和后续工作。
- 技术报告素材位于 `docs/competition_report_materials.md`，明确区分已有证据、非原创基础组件、缺失材料和待人工确认事项。
- 完成第三方依赖与许可证初步核查，覆盖直接 Python 依赖、环境候选与开发工具、FlowTransformer、真实数据、模型权重和网页/图片资源。
- 核查文档位于 `docs/third_party_licenses.md`；数据再分发、权重是否进入私评包、完整传递依赖和版权主体仍待人工确认。
- 完成 Phase D-3.1 源码权利来源复核准备：复用文件哈希、token 相似度、连续片段和核心模型人工对照结果，形成逐文件来源记录。
- 文本比对未发现整文件直接复制或可识别的逐段复制；项目负责人已确认整体改写关系，`baseline.py`、`train.py` 已标记为“参考上游源代码后的自主改写”。相似度不作为法律结论。
- 新增 `docs/code_provenance_audit.md` 与 `docs/submission_compliance_checklist.md`，记录潜在 AGPL 依赖闭包、版权声明建议、提交排除项和必须人工核实的页面字段。
- 完成许可证与来源说明落实：根目录保留完整 AGPL-3.0 `LICENSE`，新增 `NOTICE.md`，README、第三方声明、来源审计和提交清单统一使用 GitHub 规范上游与自主改写口径。
- 当前作品仅提交竞赛评委私有评审；私评不取消 AGPL-3.0、上游版权、来源和对应源码保留要求。

## 3. 当前冻结内容

以下内容不得被后续阶段覆盖或静默替换：

- 固定数据文件指纹：`NF-UNSW-NB15-v2.csv`，SHA-256 为 `05019e3ac8d55b3f074f9f042623bbbb1af2527b8b78558c69353ccb2ccefa3f`。
- Baseline split manifest 和原始行索引。
- train/validation/test 隔离规则、窗口目标为最后一行的规则。
- 预处理只在 train 上拟合的规则。
- validation 用于训练方式、最佳权重和阈值选择的规则。
- test 只用于最终评估、不得用于调参或方案选择的规则。
- `unsw_trusted_baseline_v1` 配置、三 seed 指标、日志、图和最佳权重。
- Baseline 冻结归档：`artifacts/baseline_freeze/unsw_trusted_baseline_v1/`。
- 冻结归档 manifest SHA-256：`71e6e8741907a41cc88423ee8792814a45203a3523c1d65039a70516ef7da09a`。
- 已完成的 Threshold、Risk、Focal Loss、本地推理和 API 示例结果。

任何后续模型、特征、loss、阈值策略或工程接口变化都必须使用新的配置和实验标识，并以冻结 Baseline 为对照，不得修改历史结果。

## 4. 当前运行方式

项目根目录：

```powershell
Set-Location D:\Users\LENOVO\Desktop\AI_safety_competition_code
```

安装当前项目、ML 和本地 API 可选依赖：

```powershell
python -m pip install --editable ".[ml,api]"
```

运行完整单元测试：

```powershell
python -m unittest discover -s tests\unit -p "test_*.py" -v
```

运行本地 CSV 推理：

```powershell
python -m campus_flow_guard.inference `
  --config configs\inference\local_focal.yaml `
  --input artifacts\examples\local_inference_input.csv `
  --output artifacts\examples\local_inference_output.json
```

运行最小可复现演示：

```powershell
python scripts\run_demo.py
```

重新从冻结 validation 索引生成演示数据：

```powershell
python scripts\prepare_demo_data.py
```

演示材料位于 `artifacts/demo/`，来源与生成规则见 `docs/demo_data_manifest.md`。

启动最小 FastAPI 展示：

```powershell
python -m campus_flow_guard.api --config configs\api\local_demo.yaml
```

启动后浏览器访问 `http://127.0.0.1:8000`，或调用：

```powershell
curl.exe -F "file=@artifacts\examples\local_inference_input.csv" `
  http://127.0.0.1:8000/predict
```

关闭服务：回到运行 Uvicorn 的 PowerShell 窗口，按 `Ctrl+C`。

当前 API 只监听本地地址，不连接外部 API。上传 CSV 必须包含训练预处理器要求的 37 个特征字段，并至少包含 8 行以构造一个窗口。

## 5. 后续任务

1. 人工确认 `NF-UNSW-NB15-v2.csv` 的发布页面、许可证、下载记录和上游哈希，并保存可追溯证据。
2. 基于已验证的固定演示脚本和演示数据说明，补充竞赛展示所需的运行截图。
3. 将 `docs/competition_report_materials.md` 适配到待确认的竞赛官方模板，并补充系统架构成图、图表题注、运行截图和演示视频。
4. 对本地 API 做交付前安全检查，包括上传大小、字段错误、临时文件清理、异常信息和仅本地监听验证。
5. 按 `docs/submission_compliance_checklist.md` 确认版权主体、数据许可、权重是否提交、竞赛官方规则和最终依赖 SBOM，再生成私评交付清单和哈希清单。

在以上事项完成前，不应宣称数据许可已确认、全量数据实验已完成、系统具备时间外推能力或已达到生产部署条件。
