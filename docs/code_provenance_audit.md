# CampusFlowGuard 源码权利来源初步审计

审计日期：2026-07-18

## 1. 范围与方法

对比范围为当前 `src/campus_flow_guard/` 与只读本地基线 `C:\Users\LENOVO\FlowTransformer`。规范上游已确认为 `https://github.com/liamdm/FlowTransformer.git`；本地基线固定在 commit `52236c8c9feccdeb44acf578789db7b8b36bacea`，工作区干净，许可证为 AGPL-3.0，源码版权头为 `FlowTransformer 2023 by liamdm / liam@riftcs.com`。

当前 CampusFlowGuard 尚无首个 Git commit，本审计使用文件 SHA-256 作为版本快照。方法包括精确哈希、归一化 Python token 相似度、最长连续 token、符号/import 搜索，以及核心模型构建段人工逐行对照。相似度只用于发现候选，不构成法律结论，也不能证明不存在大幅改写或翻译后的派生关系。

## 2. 总体结果

### 2.1 直接复制文件

未发现。当前 10 个文件与上游 `framework/`、`implementations/` 共 21 个 Python 文件之间没有相同 SHA-256。

### 2.2 文本比对与已确认改写关系

文本比对未发现可识别的逐段复制或复制后小幅修改。最高全文件 token 相似度为 14.44%，对应 `api.py` 与上游预处理文件，人工检查属于通用 Python 语法巧合，没有功能对应关系。

项目负责人已确认 CampusFlowGuard 参考 FlowTransformer 上游源代码后进行自主改写。因此，低文本相似度只描述重写幅度，不用于否定项目级改写关系或缩小 AGPL-3.0 来源说明范围。

核心候选结果：

| 当前文件 | 上游候选 | token 相似度 | 最长连续匹配 | 解释 |
| --- | --- | ---: | ---: | --- |
| `baseline.py` | `framework/flow_transformer.py` | 8.62% | 10 tokens | 通用 sigmoid Dense 调用 |
| `baseline.py` | `implementations/transformers/basic/encoder_block.py` | 4.48% | 12 tokens | 通用 Keras MultiHeadAttention 调用 |
| `train.py` | `framework/flow_transformer.py` | 6.98% | 10 tokens | 通用 sigmoid Dense 调用 |
| `train.py` | `implementations/transformers/basic/encoder_block.py` | 6.42% | 12 tokens | 通用 Keras MultiHeadAttention 调用 |

### 2.3 参考上游源代码后的自主改写

`baseline.py` 和 `train.py` 已明确标记为“参考 FlowTransformer 上游源代码后的自主改写”。两者保留规范上游、原作者和 AGPL-3.0 说明；当前文件组织、数据协议、函数结构、层组合、命名、异常处理和训练评估逻辑与上游不同，且不直接导入上游 `framework` 或 `implementations`。

## 3. 逐文件记录

| 当前文件 | 对应上游文件 | 关系类型与证据 | 许可证可能影响 | 建议版权声明 |
| --- | --- | --- | --- | --- |
| `__init__.py` | 上游两个 `__init__.py` | 独立新增；无相同代码 | 未发现上游复制 | 待版权主体确定后使用 CampusFlowGuard 声明；统一 NOTICE 保留上游引用 |
| `train.py` | `flow_transformer.py`、`input_encodings.py`、`classification_heads.py`、`encoder_block.py` | 参考上游源代码后的自主改写；文件头已记录规范 URL、原作者和 AGPL | 纳入本提交版本 AGPL-3.0 来源范围 | 保留当前上游作者、URL、改写说明、根 LICENSE 和 NOTICE |
| `baseline.py` | 同上及上游训练/评估职责 | 参考上游源代码后的自主改写；功能组合对应，文本证据显示为较大幅度重写 | 纳入本提交版本 AGPL-3.0 来源范围，并向推理/API 链路传播来源说明 | 保留当前上游作者、URL、改写说明、根 LICENSE 和 NOTICE |
| `data_protocol.py` | `flow_transformer.py` 的 split/window 行为、`dataset_specification.py` | CampusFlowGuard 新增重构；保存互斥索引、train-only 预处理和分区构窗 | 作为当前 AGPL-3.0 提交作品的一部分随源码交付 | 保留新增工作说明，并在文档注明原行为审计来源 |
| `threshold_optimization.py` | 上游 evaluate 的宽泛概念 | CampusFlowGuard 新增；上游无 validation 阈值扫描和 FAR 约束模块 | 作为当前 AGPL-3.0 提交作品的一部分随源码交付 | 保留新增工作和 validation-only 说明 |
| `focal_experiment.py` | 无 Focal 对应模块 | CampusFlowGuard 新增；通过当前 `baseline.py` 复用模型 | 作为当前 AGPL-3.0 提交作品的一部分随源码交付 | 保留新增实验说明和内部复用链 |
| `inference.py` | 上游 build/evaluate 功能概念 | CampusFlowGuard 新增；上游无哈希校验、CSV CLI、风险 JSON 输出 | 作为当前 AGPL-3.0 提交作品的一部分随源码交付 | 保留模型架构、权重、阈值与上游来源说明 |
| `risk.py` | 无对应文件 | CampusFlowGuard 新增 | 作为当前 AGPL-3.0 提交作品的一部分随源码交付 | 保留风险映射版本和新增工作说明 |
| `api.py` | 无对应文件 | CampusFlowGuard 新增 FastAPI 包装 | 通过 `inference.py -> baseline.py` 形成依赖链；当前私评包随对应源码交付 | 保留来源 NOTICE；未来公开网络服务落实 AGPL 第 13 条 |
| `static/index.html` | 无对应文件 | CampusFlowGuard 新增，无外部媒体 | 纳入当前 AGPL-3.0 提交源码；未来公开服务时核对界面通知 | 保留页面新增工作说明，并预留许可证/源码入口 |

## 4. 核心实现差异

上游以可插拔 input encoding、sequential model 和 classification head 组件构建模型；当前 `baseline.py` 在一个函数内显式创建输入、embedding、projection、attention、feed-forward、Last Token、MLP 和输出层。上游数值特征逐列创建 Input，当前合并为数值张量；当前另有固定 split、哈希校验、balanced sampling、validation 阈值和冻结归档。

上游 Last Token 使用 `Lambda(lambda x: x[..., -1, :])`，当前使用 `Lambda(lambda tensor: tensor[:, -1, :])`。上游 encoder 是 Layer 类，当前在模型函数中循环调用标准 Keras 层。现有证据说明当前不是整文件复制后改名，而是参考上游源代码后的较大幅度自主改写；文本相似度不用于缩小 AGPL-3.0 范围。

## 5. AGPL-3.0 可能影响范围

本提交版本根目录保留并适用 AGPL-3.0。来源标记最直接的文件为 `baseline.py`、`train.py`；`data_protocol.py` 及其他模块属于 CampusFlowGuard 在同一提交作品中的新增或重构内容。

当前依赖链：

```text
baseline._build_record_projection_model
  -> threshold_optimization
  -> focal_experiment
  -> inference -> api -> static/index.html
```

为避免拆分许可证口径，当前提交版本按 AGPL-3.0 保留完整对应源码、根 `LICENSE` 和 `NOTICE.md`。依赖链中的阈值、Focal、推理、API 和 HTML 均应随同当前源码交付。作品目前仅供评委私有评审；若将来公开提供 FastAPI 网络服务，仍需人工落实 AGPL 第 13 条的源码获取方式和界面通知。模型权重是否提交不能由源码相似度决定。

当前落实措施：根目录保留完整 AGPL-3.0；新增 `NOTICE.md`；README 和文件头注明规范上游、作者与改写关系；提交当前对应源代码；不复制上游整仓；不把低相似度写成不受 AGPL 影响。文件级 SPDX 和具体版权主体仍待人工确定。

## 6. 当前源码哈希快照

| 文件 | SHA-256 |
| --- | --- |
| `__init__.py` | `46669cf6a3156e36c21529516770e6de626a8f51cb2761d8fb609c2988bd3bcd` |
| `api.py` | `3f393e198eedefe716f22de8d7465c55035423834758eea24c49fe2e36591239` |
| `baseline.py` | `e52e190b0a937958eaede9f00b648505e998fea241c0a997a5db887561a1dfd2` |
| `data_protocol.py` | `5bac23e729cb60fd6af3fed635b87ce3322da91eb91c4378db02a4208dc22867` |
| `focal_experiment.py` | `1f26a110f9b19c6ea96c222eda71bd88f3f0e7d7e5dd8bde32ca3c69f0b3a6a6` |
| `inference.py` | `2277dc042c87b443ea37335cc61c3c7ff5d7d67fc10a380cc5a9758abcd15de1` |
| `risk.py` | `af4b85de73b3da07aa0780ecfc767210d1bd9011b5fafe5f905e79bae2a4297f` |
| `static/index.html` | `b475467bd3013100f8a82e2dba4db7f801085c96ffb8dbcbb196773e2f5676d0` |
| `threshold_optimization.py` | `3953dbdeef92c343e58e80255325ba7afb889f1978f7796d2f5ef5821e253b79` |
| `train.py` | `138ed2cefdb5614b35db03f5d7fe17fc084b8f6b5e76c3d5641c58d779243bbc` |

## 7. 必须人工完成

1. 确认 CampusFlowGuard 具体版权主体和文件级版权年份/名称。
2. 保存 GitHub 规范上游、commit、许可证和作者信息的人工截图或归档。
3. 决定模型权重是否进入评委私评包，并核对数据许可。
4. 若未来公开部署网络服务，落实源码下载入口和适当法律通知。
5. 建立首个可审计 Git commit；提交前排除真实数据、环境、上游整仓和未批准权重。
