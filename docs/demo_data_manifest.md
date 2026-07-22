# 最小可复现演示数据说明

## 用途与范围

`artifacts/demo/` 中的三个 CSV 仅用于 CampusFlowGuard 本地推理演示，不用于训练、模型选择、阈值选择或正式指标评估。它们均来自冻结的 `unsw_trusted_baseline_v1` validation 分区，未使用 test 行。

数据文件为 `NF-UNSW-NB15-v2.csv`。本地文件 SHA-256 已由冻结 split manifest 记录并在生成时复核；该数据的发布来源、许可证和上游发布方哈希仍为“待人工确认”。

## 生成规则

运行：

```powershell
python scripts\prepare_demo_data.py
```

脚本按原始 CSV 行号升序，从冻结 validation 索引中选取：

- `benign_demo.csv`：前 16 条 `Attack == Benign` 记录；
- `attack_demo.csv`：前 16 条 `Attack != Benign` 记录；
- `mixed_demo.csv`：上述样例的前 8 条良性记录，再接前 8 条攻击记录。

这种混合顺序是为了提供稳定、易复现的展示输入，不代表原始时间顺序，也不能用于时间外推结论。精确原始行索引、文件大小和 SHA-256 保存在 `artifacts/demo/demo_manifest.json`。

## 字段与隐私处理

三个 CSV 只保留 `configs/datasets/dataset_schema.yaml` 声明的 37 个模型特征。以下字段不会写入演示 CSV：

- `IPV4_SRC_ADDR`
- `IPV4_DST_ADDR`
- `Attack`
- `Label`

因此样例不包含源/目的 IP，也不会把真实标签传入推理接口。每个文件仅 16 行，可构造 9 个长度为 8、目标为最后一行的推理窗口。

## 可重复性与边界

生成脚本会验证真实数据 SHA-256、冻结 split 文件 SHA-256、validation/test 互斥关系及字段白名单。相同冻结输入会得到相同的三个 CSV；单元测试同时复核文件哈希与来源索引。

模型输出是对窗口的预测，不是对样例真实标签的重新标注。`mixed_demo.csv` 的命名描述其来源构成，不保证每个窗口一定被当前模型判为攻击。

## 本次生成与演示结果

2026-07-17 实际生成文件：

| 文件 | 行数 | 列数 | SHA-256 |
| --- | ---: | ---: | --- |
| `benign_demo.csv` | 16 | 37 | `740c0d371550c36b36e138cb84b8be4561ebc389cd729a090b744e3ce1d41883` |
| `attack_demo.csv` | 16 | 37 | `07fa43fef16aab090d9900df32102c4f498cef6d623efafeb8a46dd77ed726a7` |
| `mixed_demo.csv` | 16 | 37 | `210c5a9eb2f35c9f2ce250e9f91a6016ca20023dfdf938b6f680b61180ddc6f9` |

实际执行 `python scripts\run_demo.py` 后，`mixed_demo.csv` 构造 9 个窗口，模型检测为攻击 8 个，风险等级为 Low 1、Medium 6、High 2。完整结果位于 `artifacts/demo/mixed_demo_report.json`；这些数值仅是演示输出，不是新增 test 指标或正式实验结论。
