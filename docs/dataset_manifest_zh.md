# 数据集清单与准备规范

## 1. 当前结论

本阶段只建立数据准备流程，没有下载、复制或检查真实数据文件。外部数据目录 `D:\Users\LENOVO\Desktop\AI_safety_competition_data` 当前未发现文件，因此本清单不包含任何行数、类别比例、缺失值或文件哈希结果。

## 2. 推荐数据集

推荐首先使用 **UNSW-NB15 的统一流格式 CSV** 作为基线候选，原因是原仓库的 `demonstration.ipynb` 使用该数据集，且 `NamedDatasetSpecifications.unified_flow_format` 与 `main.py` 的 `UNSW_NB15` 配置一致。

原仓库代码还列出以下数据集，但本阶段不为它们创建 schema，也不宣称文件已经存在：

| 名称 | 原代码文件名 | 原代码规范 | 划分方式 | 状态 |
|---|---|---|---|---|
| CSE_CIC_IDS | `NF-CSE-CIC-IDS2018-v2.csv` | `unified_flow_format` | `LastRows`, 1% | 待数据核验 |
| NSL-KDD | `NSL-KDD.csv` | `nsl_kdd` | `RandomRows`, 5% | 待数据核验；存在窗口泄漏风险 |
| UNSW_NB15 | `NF-UNSW-NB15-v2.csv` | `unified_flow_format` | `LastRows`, 2.5% | 推荐候选，文件待核验 |

## 3. 下载来源与真实性边界

原仓库 `demonstration.ipynb` 的 Markdown 指向 UQ NIDS 数据集页面：

`https://staff.itee.uq.edu.au/marius/NIDS_datasets/`

该 Notebook 的下载代码还使用一个 API 下载地址：

`https://api.rdm.uq.edu.au/production/files/8c6e2a00-ef9c-11ed-827d-e762de186848/download`

这些地址是原代码中直接出现的来源记录，不代表本阶段已访问、下载、验证可用性或确认许可。下载前必须记录实际 URL、访问时间、响应文件哈希、压缩包内部路径、数据集许可和版本说明。

## 4. 文件放置方式

推荐将原始文件放在外部数据根目录，避免进入 Git：

```text
D:\Users\LENOVO\Desktop\AI_safety_competition_data\
└── raw\
    └── NF-UNSW-NB15-v2.csv
```

项目内 `data/` 只保存目录说明和必要的小型元数据；项目数据布局为 `raw/` → `interim/` → `processed/`。不应把 CSV、Feather、Parquet、缓存、模型权重或抓包文件复制到 Git 工作区。

## 5. 原代码实际输入格式

原代码通过 `DatasetSpecification(include_fields, categorical_fields, class_column, benign_label, test_column)` 描述表格，`FlowTransformer.load_dataset()` 接收 CSV 路径或 DataFrame。UNSW-NB15 基线要求：

- 文件格式：CSV；
- 37 个特征字段：见 `configs/datasets/dataset_schema.yaml`；
- 标签字段：`Attack`；
- 良性标签：`Benign`；
- 类别字段：`CLIENT_TCP_FLAGS`、`L4_SRC_PORT`、`TCP_FLAGS`、`ICMP_IPV4_TYPE`、`ICMP_TYPE`、`PROTOCOL`、`SERVER_TCP_FLAGS`、`L4_DST_PORT`、`L7_PROTO`；
- 时间字段：原规范没有定义，当前为 `null`，不能据此声称数据已按时间排序；
- 独立 train/test 文件：原 `UNSW_NB15` 路径只有一个 CSV，使用 `LastRows` 在内存中生成评估掩码，故 schema 中 `train_file` 和 `test_file` 保持 `null`；
- 窗口：原示例 `window_size=8`，目标流位于窗口最后一行；
- 预处理：类别和数值预处理器应只用训练掩码行拟合，再变换全量行。

`Attack != Benign` 被原 `evaluate()` 当作恶意正类。真正加载数据前必须检查标签唯一值，不能只依赖配置文字。

## 6. 检查命令

数据文件存在后，先执行检查脚本，不要直接训练：

```powershell
Set-Location D:\Users\LENOVO\Desktop\AI_safety_competition_code
python scripts\inspect_dataset.py `
  D:\Users\LENOVO\Desktop\AI_safety_competition_data\raw\NF-UNSW-NB15-v2.csv `
  --schema configs\datasets\dataset_schema.yaml `
  --output logs\unsw_nb15_inspection.json
```

检查结果必须来自真实文件；文件不存在时脚本应失败并报告路径，不生成占位统计。

## 7. 后续数据工程与训练顺序

1. 固定来源、许可、下载时间、SHA-256、文件大小和列清单。
2. 执行 `inspect_dataset.py`，保存原始检查 JSON；审阅列、dtype、缺失、inf、重复和标签分布。
3. 明确时间排序或实体分组规则；原规范没有时间字段，必须待数据核验后决定 `LastRows` 是否合理。
4. 建立无泄漏的 train/validation/test 划分和窗口边界；禁止 RandomRows 评估行进入训练窗口上下文。
5. 只在训练部分拟合预处理器，保存版本化 schema 和预处理参数。
6. 运行窗口构造、输入编码和模型最小形状测试；此后才允许基线训练。
7. 训练与评估记录配置、种子、数据哈希、特征列表、阈值、TP/FP/TN/FN、Precision、Recall、F1、FAR、ROC-AUC、PR-AUC、参数量、训练时间和推理吞吐量。

当前阶段到第 2 步之前为止：没有真实数据报告，也没有训练结果。
