# 第三方代码与许可证声明

## FlowTransformer

- 组件名称：FlowTransformer
- 原作者标识：`liamdm / liam@riftcs.com`
- 规范上游：`https://github.com/liamdm/FlowTransformer.git`
- 原始仓库本地基线：`C:\Users\LENOVO\FlowTransformer`
- 审计 commit：`52236c8c9feccdeb44acf578789db7b8b36bacea`
- 许可证：GNU Affero General Public License v3.0（AGPL-3.0）
- 许可证文本：根目录 `LICENSE`；备份为 `LICENSES/FlowTransformer-AGPL-3.0.txt`
- 当前关系：CampusFlowGuard 参考 FlowTransformer 上游源代码后进行自主改写；`baseline.py`、`train.py` 为明确标记的改写文件。
- 提交场景：仅提交竞赛评委私有评审。

FlowTransformer 源文件中的原作者版权信息不得删除。CampusFlowGuard 提交版本保留完整 AGPL-3.0、来源和修改说明；详细关系见根目录 `NOTICE.md` 与 `docs/code_provenance_audit.md`。

CampusFlowGuard 已完成真实 Baseline、Focal Loss、阈值、风险、本地推理和 FastAPI 演示。当前模型权重由本项目实验流程训练生成，不是上游预训练权重；权重是否随私评包提交仍需结合数据许可和竞赛要求确认。

## 数据集与其他依赖

数据集许可证、正式下载来源和允许用途仍待人工确认。Python 依赖许可证初步清单见 `docs/third_party_licenses.md`；若提交离线依赖，仍需生成最终 SBOM 和完整 LICENSE/NOTICE 目录。

本声明不构成法律意见。
