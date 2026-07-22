# FlowTransformer 原始仓库只读审计报告

## 1. 审计元数据与结论标记

- 审计对象：`C:\Users\LENOVO\FlowTransformer`
- 审计时间：2026-07-13 21:35:16（Asia/Shanghai）
- 审计方式：静态代码阅读、Git 只读检查、Notebook JSON 解析、无磁盘写入的最小逻辑复现；未执行完整数据预处理、模型构建或训练。
- 审计基线：分支 `master`，commit `52236c8c9feccdeb44acf578789db7b8b36bacea`
- 报告目标：`D:\Users\LENOVO\Desktop\AI_safety_competition_code\docs\original_repository_audit_zh.md`

本报告统一使用以下标记，避免把推测写成事实：

- **[代码确认]**：可由当前 commit 的代码、Git 元数据、文件内容或最小复现直接确认。
- **[高概率风险]**：代码结构明确显示存在风险，但影响程度依赖数据、依赖版本或运行方式。
- **[待实验验证]**：必须安装依赖、取得数据或运行训练后才能下结论。

## 2. 执行摘要

1. **[代码确认]** 原路径存在，是 Git 仓库；当前分支为 `master`，HEAD 为 `52236c8c9feccdeb44acf578789db7b8b36bacea`，审计开始时工作区干净、无未提交修改。
2. **[代码确认]** 代码实现的主链路是：CSV/Feather 读取 → 生成训练掩码 → 仅用训练行拟合预处理器 → 对全量行变换 → 按目标索引构造尾随窗口 → 特征拆分 → 输入编码 → Transformer → 分类头 → MLP → sigmoid → 50/50 类别平衡批训练 → 按原评估掩码评估。
3. **[代码确认]** `evaluate()` 将“非 benign_label”定义为正类，即正类代表恶意/攻击，负类代表良性；预测使用固定阈值 `> 0.5`，恰好等于 0.5 被判为负类。
4. **[代码确认]** TP、FP、TN、FN、Precision、Recall 和 F1 的布尔公式方向正确；False Alarm Rate 没有计算或返回。ROC-AUC、PR-AUC、最佳阈值也没有实现。
5. **[代码确认]** 训练批次采用近似 50/50 的恶意/良性平衡采样；评估集未做类别平衡，保留划分得到的实际类别分布。
6. **[代码确认]** `RandomRows`/散布式 `FilterColumn` 划分与相邻索引窗口组合时，训练窗口可能包含评估行特征，构成特征级数据泄漏。代码自身也发出相关警告，但“no class leakage”不能消除特征泄漏对实验估计的影响。
7. **[代码确认]** `LastRows` 的首批评估窗口会混入训练尾部行作为历史上下文，但按默认尾随窗口不会把评估行送入训练窗口；是否允许这种跨边界历史上下文应由实验协议明确规定。
8. **[代码确认]** early stopping 监控训练批次 loss，不监控验证指标；不保存或恢复最佳权重。
9. **[代码确认]** 当前环境为 Python 3.11.9，未安装 TensorFlow/Keras，因此模型构建、`jit_compile=True` 与完整训练均未验证。仓库没有依赖清单或版本锁定文件。
10. **[代码确认]** 存在可复现的导入错误、预处理缺陷和缓存键缺项；这些问题会影响可运行性、数据隔离或复现性，进入复现阶段前应先建立针对性回归用例。

## 3. 仓库与 Git 基线

### 3.1 仓库状态

| 项目 | 结果 | 证据级别 |
|---|---|---|
| 路径存在 | 是 | [代码确认] |
| `.git` 目录存在 | 是 | [代码确认] |
| Git 仓库 | 是 | [代码确认] |
| 当前分支 | `master` | [代码确认] |
| HEAD | `52236c8c9feccdeb44acf578789db7b8b36bacea` | [代码确认] |
| 远端跟踪状态 | `master...origin/master`，显示 up to date | [代码确认] |
| 工作区状态 | `nothing to commit, working tree clean` | [代码确认] |
| 未提交修改 | 无 | [代码确认] |

首次直接执行 Git 命令时，Git 因仓库所有者 `LENOVO` 与沙箱用户 `CodexSandboxOffline` 不同而报 `detected dubious ownership`。后续使用单次命令参数 `-c safe.directory=C:/Users/LENOVO/FlowTransformer` 完成只读查询，**没有修改全局 Git 配置**。

### 3.2 最近一次提交

```text
commit: 52236c8c9feccdeb44acf578789db7b8b36bacea
author: liamdm <liam@riftcs.com>
date: 2023-10-17T20:19:13+02:00
subject: Updated extended_concepts.ipynb
```

## 4. 文件清单

### 4.1 Python 文件（22 个）

```text
framework\__init__.py
framework\base_classification_head.py
framework\base_input_encoding.py
framework\base_preprocessing.py
framework\base_sequential.py
framework\dataset_specification.py
framework\enumerations.py
framework\flow_transformer.py
framework\flow_transformer_parameters.py
framework\framework_component.py
framework\model_input_specification.py
framework\sequential_input_encoding.py
framework\utilities.py
implementations\__init__.py
implementations\classification_heads.py
implementations\input_encodings.py
implementations\pre_processings.py
implementations\transformers\basic\decoder_block.py
implementations\transformers\basic\encoder_block.py
implementations\transformers\basic_transformers.py
implementations\transformers\named_transformers.py
main.py
```

### 4.2 Notebook（3 个）

```text
demonstration.ipynb
extended_concepts.ipynb
FlowTransformer_demo.ipynb
```

本阶段按要求详细阅读 `demonstration.ipynb` 与 `FlowTransformer_demo.ipynb`；`extended_concepts.ipynb` 已列入清单，但不在本阶段指定的详细阅读范围内。

### 4.3 README 与许可证

```text
readme.md
LICENCE.txt
```

仓库根目录未发现 `requirements*.txt`、`pyproject.toml`、`setup.py`、`setup.cfg`、`environment*.yml`、`Pipfile` 或 `poetry.lock`。

## 5. 文件级与函数级分析

### 5.1 `readme.md`

- **[代码确认]** 第 10–48 行描述了四个可替换组件：预处理、输入编码、序列模型、分类头，并展示 `load_dataset()`、`build_model()`、`evaluate()` 的基本用法。
- **[代码确认]** 第 54 行文档写作 `DataSpecification`，实际类名是 `DatasetSpecification`。
- **[代码确认]** README 没有安装步骤、Python/TensorFlow/Keras 版本、依赖锁定、随机种子策略、数据版本校验、评估阈值或完整指标定义。
- **[高概率风险]** 仅凭 README 无法搭建可复现环境，也无法判断论文实验与当前 commit 的精确对应关系。

### 5.2 `main.py`

- **[代码确认]** 第 18–41 行枚举输入编码、分类头和 Transformer；第 54–58 行组装默认 FlowTransformer。
- **[代码确认]** 第 43 行硬编码 `C:\Data\UQ\NIDS\Collected`；第 46–48 行在该路径下拼接三个数据集文件。
- **[代码确认]** 第 61–62 行默认选择列表第一个 CSE-CIC-IDS 数据集并加载；第 65–69 行构建并以 Adam、binary crossentropy、binary accuracy 和 `jit_compile=True` 编译；第 73 行训练评估。
- **[代码确认]** 第 36 行使用 `List[FunctionalComponent]`，但本文件没有显式导入 `typing.List`；当前写法依赖第 13 行的通配符导入碰巧把 `implementations.input_encodings` 中导入的 `List` 暴露出来。
- **[高概率风险]** 数据路径不可移植；通配符导入令 `List` 的可用性依赖被导入模块内部实现；脚本没有 `if __name__ == "__main__"`，导入 `main` 会立即尝试加载数据、创建缓存并训练。

### 5.3 `framework/flow_transformer.py`

#### `FlowTransformer.__init__()`（第 32–54 行）

- **[代码确认]** 可接收 `np.random.RandomState`，但未提供时第 39 行创建无显式种子的 `RandomState()`。
- **[代码确认]** 类级缓存 `inmem_cache` 可跨实例复用。

#### `build_model()`（第 56–96 行）

- **[代码确认]** 第 63–75 行为每个数值/分类特征分别创建形状为 `(window_size, feature_dimension)` 的 Keras Input。
- **[代码确认]** 第 77–87 行依次执行输入编码、分类头的 Transformer 前处理、序列模型、分类头。
- **[代码确认]** 第 89–93 行追加可配置 ReLU MLP、Dropout 和单节点 sigmoid 输出。
- **[代码确认]** 没有看到位置编码注入逻辑；在本次审阅的输入编码和 Transformer 实现中也未发现位置编码。
- **[待实验验证]** 缺少位置编码对论文指标、上下文顺序建模能力的实际影响需要消融实验确认。

#### `_load_preprocessed_dataset()`（第 98–274 行）

- **[代码确认]** 第 145–155 行支持字符串形式的 CSV/Feather 路径或 DataFrame；`pathlib.Path` 不满足 `isinstance(dataset, str)`，会被拒绝。
- **[代码确认]** 第 162–193 行先生成训练掩码；`LastRows` 取尾部，`RandomRows` 随机抽行，`FilterColumn` 自动把标记列中样本数最少的值当评估类。
- **[代码确认]** 第 169–170 行明确警告非 `LastRows` 模式可能泄漏评估特征。
- **[代码确认]** 第 195 行用 `set(...).difference(...)` 生成数值列集合；第 203、219 行迭代集合，特征顺序受 Python 哈希随机化影响。最小复现中 `PYTHONHASHSEED=1` 与 `2` 得到不同顺序。
- **[代码确认]** 第 221–225 行仅使用 `training_mask` 对应行拟合数值预处理器，然后变换全量数据；第 235–239 行对分类特征执行相同的“训练拟合、全量变换”模式。首次计算路径上没有直接用评估行拟合 min/max 或类别频次。
- **[代码确认]** 缓存键第 119–126 行没有包含 `evaluation_dataset_sampling`、随机种子、数据源路径/内容哈希、数据集规范、特征列表、`window_size`。命中缓存时第 139–143 行直接返回旧掩码和预处理结果。
- **[代码确认]** `evaluation_percent` 很小导致 `eval_n == 0` 时，第 173 行执行 `training_mask[-0:] = False`，等价于把**全部行**设为评估行。最小逻辑复现得到 `TRAIN_COUNT=0`。
- **[代码确认]** `FilterColumn` 第 183–192 行根据少数值自动判断评估标志，而不是按配置明确指定训练/测试语义。
- **[高概率风险]** 缓存键缺项可能使不同划分方法、不同数据版本或不同随机种子的运行复用同一个缓存，从而破坏实验隔离和追溯。
- **[高概率风险]** Feather 读取第 149 行只请求 `include_fields + class_column`；若使用 `FilterColumn`，测试标志列没有被包含，之后的末列检查可能失败。
- **[高概率风险]** 未校验 `evaluation_percent` 的合法区间，也未检查训练集/评估集是否为空。

#### `load_dataset()`（第 276–313 行）

- **[代码确认]** 默认缓存目录是相对当前工作目录的 `cache`；第 295–297 行用非递归 `os.mkdir()` 创建。
- **[代码确认]** 第 302–310 行从预处理 DataFrame 分离 `__training` 与 `__y`，其余列作为模型特征。
- **[高概率风险]** 从不同工作目录启动会得到不同缓存位置；多级父目录不存在时 `os.mkdir()` 会失败；传入 `Path` 对象作为数据路径不受支持。

#### `evaluate()`（第 315–530 行）

- **[代码确认]** 训练每批恶意样本数为 `int(0.5 * batch_size)`，其余为良性；偶数 batch 为严格 50/50，奇数 batch 近似平衡。
- **[代码确认]** 第 326 行定义 `y_mask = label != benign_label`，故 `True/1` 是恶意/攻击，`False/0` 是良性。
- **[代码确认]** 第 328–330 行仅从训练掩码中分别取恶意和良性目标索引；第 445–454 行分别采样后按“恶意在前、良性在后”拼接。
- **[代码确认]** 第 332、380–388 行对全部评估目标建立窗口，`eval_y` 没有重采样，因此评估集保持划分产生的真实类别分布。
- **[代码确认]** 第 342 行构造尾随窗口 `[i-window_size+1, i]`，窗口最后一行是被分类目标；标签取目标索引对应的 `y_mask`。
- **[代码确认]** 第 397–398 行使用固定阈值 `prediction > 0.5`；等于 0.5 判负。
- **[代码确认]** 第 406–409 行混淆矩阵方向正确：TP=预测恶意且真实恶意，FP=预测恶意且真实良性，TN=预测良性且真实良性，FN=预测良性且真实恶意。
- **[代码确认]** 第 415–418 行 Precision、Recall、F1 公式及分母为零保护正确。
- **[代码确认]** 第 411–413 行计算 sensitivity、specificity、balanced accuracy；第 421–433 行只保存 balanced accuracy 和 F1，没有保存 sensitivity、specificity、Precision、Recall 或 FAR。
- **[代码确认]** FAR 应为 `FP / (FP + TN)`，现有代码没有计算或返回。可由 `1 - specificity` 推导，但报告产物中没有该字段。
- **[代码确认]** 第 473–517 行 early stopping 依据训练批 loss：只要某个 batch loss 创全局新低，就把本 epoch 视为改善；不使用独立验证 loss、F1、PR-AUC 等指标。
- **[代码确认]** 停止条件是 `iters_since_loss_decrease > patience`，不是常见的 `>= patience`，因此会比参数字面含义多等待一个无改善 epoch。
- **[代码确认]** 没有调用 `save_weights`、ModelCheckpoint 或恢复最佳权重；返回的是停止时权重。
- **[代码确认]** 第 519 行只在 epoch 6、最后一个 epoch 或早停时评估；5 个 epoch 的示例只在最后评估。
- **[代码确认]** 当 `_train_ensure_flows_are_ordered_within_windows=False` 时，第 344–346 行先把二维随机索引 `.reshape(-1)` 成一维，再执行 `[:, -1]`，最小复现触发 `IndexError: too many indices for array`。默认参数为 True，所以默认路径不触发。
- **[高概率风险]** 每类可选训练样本少于对应 batch 配额时，`choice(..., replace=False)` 会失败；顺序采样模式下游标取模分母也可能为零或负数。
- **[高概率风险]** `eval_X` 以大量 DataFrame 窗口列表一次性物化并 `concat`，对百万级流数据存在显著内存压力。

#### `time()`（第 533–666 行）

- **[代码确认]** 基本复制训练批生成逻辑并计时 `predict_on_batch()`；同样包含未排序窗口的一维索引缺陷。
- **[代码确认]** 返回原始 batch 时间列表，不直接计算吞吐量，也不记录硬件、预热策略、数据版本或置信区间。

### 5.4 `framework/dataset_specification.py`

- **[代码确认]** `DatasetSpecification` 仅保存特征列、分类列、标签列、良性标签和可选测试列，没有时间列、分组列、流/主机实体列或排序规则。
- **[代码确认]** `NamedDatasetSpecifications` 内置 CSE-CIC-IDS、改进版 CSE-CIC-IDS、统一流格式、MQTT、NSL-KDD 规范。
- **[高概率风险]** `mqtt` 第 54–55 行把 `is_attack` 的 `benign_label` 配为字符串 `'1'`，命名语义可疑；必须对照真实数据字典确认，不能据此断言配置错误。
- **[待实验验证]** NSL-KDD 标签实际是 `normal` 还是 `normal.`、测试标志列是否位于末列，需对指定 CSV 版本核对。

### 5.5 `framework/flow_transformer_parameters.py`

- **[代码确认]** 公开参数只有窗口长度、MLP 层大小、MLP dropout。
- **[代码确认]** 私有训练开关默认为“窗口内保持原顺序”且“训练目标窗口随机抽取”。
- **[代码确认]** 没有公开的随机种子、阈值、验证集、采样比例、排序键、分组键或泄漏防护配置。

### 5.6 `implementations/pre_processings.py`

- **[代码确认]** 数值预处理用训练行的 min/range，执行减最小值、`log(x+1)`、按训练 range 对数归一化；默认不裁剪到 `[0,1]`。
- **[代码确认]** 若评估值低于训练最小值超过 1，第 47 行可产生 NaN。最小复现以训练 `[10,20]`、变换值 `0` 得到 NaN；变换后没有有限性复检。
- **[代码确认]** 分类编码第 67 行把结果初始化为 1，而注释第 71 行声明 0 代表未知值；第一个已知类别也映射为 1。最小复现确认已知首类与未知类都编码为 1。
- **[高概率风险]** 未知类别与最常见训练类别碰撞会掩盖分布漂移，并可能抬高或降低评估指标。
- **[高概率风险]** `fit_numerical()` 对空训练列、全 NaN 或异常 dtype 没有显式错误说明；数值清洗对原数组视图的原地赋值也需要数据副本语义测试。

### 5.7 `implementations/input_encodings.py`

- **[代码确认]** `NoInputEncoder` 需要 OneHot；整数格式时虽有转换分支，但其 `required_input_format` 固定为 OneHot。
- **[代码确认]** `RecordLevelEmbed` 先拼接所有特征，再做线性 Dense 投影。
- **[代码确认]** `CategoricalFeatureEmbed` 支持 Dense、Lookup、Projection；Lookup 使用 `Embedding(..., input_length=self.sequence_length)`。
- **[高概率风险]** `Embedding.input_length` 在较新 Keras 中已属于旧式/兼容性参数；确切行为取决于最终锁定版本。
- **[高概率风险]** 第 119、142 行在数值特征或分类特征列表为空时调用 `Concatenate`，单特征/空特征数据规范可能不兼容。

### 5.8 `implementations/classification_heads.py`

- **[代码确认]** 实现 Flatten、FeaturewiseEmbedding、GlobalAveragePooling、LastToken 和 CLS Token 分类头。
- **[代码确认]** LastToken 第 87 行取序列最后一个 token，与窗口目标位于最后一行的构造一致。
- **[代码确认]** CLS Token 第 122–133 行同时追加一个序列行和一个特征维指示列，之后继承 LastToken 取新增 token。
- **[高概率风险]** CLS 实现直接对 Functional KerasTensor 使用多个原生 `tf.*` 操作；Keras 3 对这类用法限制更严格，需在锁定版本上构建测试。

### 5.9 `implementations/transformers/basic_transformers.py`

- **[代码确认]** `BasicTransformer` 按层堆叠自定义 encoder 或 decoder block。
- **[代码确认]** 没有加入位置编码、padding mask 或显式 causal mask。
- **[高概率风险]** decoder 名称不等同于自回归模型；是否符合论文定义必须对照论文和原实验环境验证。

### 5.10 `implementations/transformers/named_transformers.py`

- **[代码确认]** GPTSmall 与 BERTSmall 都配置 12 层、768 internal size、12 heads、0.02 dropout。
- **[代码确认]** `head_size` 计算为浮点数但只进入参数字典，没有传给注意力层。
- **[代码确认]** BERT 层名前缀没有使用外部 `prefix`，多个模型组合时可能发生命名冲突。
- **[高概率风险]** GPT 路径调用的 decoder block 没有 causal mask，因此不是严格的 GPT 式因果注意力。

### 5.11 `implementations/transformers/basic/encoder_block.py` 与 `decoder_block.py`

- **[代码确认]** encoder 的 Keras MHA 使用 `key_dim=inner_dimension`，而不是常见的 `inner_dimension / num_heads`；具体参数量和论文设定需核对。
- **[代码确认]** encoder 第 111 行第二个残差连接使用 `attention_output + feed_forward_output`，而非标准结构常用的“注意力残差归一化结果 + FFN 结果”。这是明确的实现结构差异，但是否为作者有意设计尚不能断言。
- **[代码确认]** decoder 没有 causal mask；把 `inputs` 同时作为 target 和 encoder output，并复用同一个 MHA 实例执行两次注意力。
- **[代码确认]** decoder 复用 `dropout2` 与 `layernorm2` 两次。
- **[待实验验证]** 上述非标准结构对论文复现结果的影响必须通过原依赖环境和消融实验验证。

### 5.12 `demonstration.ipynb`

- **[代码确认]** Notebook 共 15 个单元；代码单元 execution count 为 35–41，部分单元保留输出。
- **[代码确认]** 单元 3 使用 `requests.get()` 下载 UNSW-NB15 ZIP，但没有 timeout、状态码检查、内容长度/哈希校验或流式下载。
- **[代码确认]** 单元 5 解压固定 ZIP 内路径并重命名；使用相对 `demonstration` 目录。
- **[代码确认]** 单元 11 用 `LastRows`、10% 评估比例；单元 13 以 `jit_compile=True` 编译；Notebook 没有训练调用。
- **[代码确认]** 单元 10 的 Markdown 声称“ensure only the testing data is used to fit the pre-processing”，与代码实际“只用 training rows 拟合”相反，是文档表述错误。
- **[高概率风险]** 下载不校验可能导致数据版本漂移或损坏文件被静默使用；ZIP 解压应在后续重构中加入路径穿越防护和哈希追溯。

### 5.13 `FlowTransformer_demo.ipynb`

- **[代码确认]** 所有代码单元 execution count 都为空，但单元 5–7 保留大量输出，因此嵌入输出不能视为本次审计或当前 commit/当前环境的已验证运行结果。
- **[代码确认]** 单元 0 使用 `from framework import NamedDatasetSpecifications, ...`；当前 `framework/__init__.py` 只有版权注释，最小导入复现得到 `ImportError`。
- **[代码确认]** 单元 2 使用 `! mkdir dataset`，单元 3 使用 POSIX `! mv /content/drive/...`，单元 4–5 硬编码 `/content/dataset` 与 `/content/cache_folder`，明显面向 Google Colab/Linux，不适用于原生 Windows。
- **[代码确认]** 单元 7 使用固定 `jit_compile=True` 并调用 `evaluate()`。
- **[代码确认]** Notebook 内嵌输出声称处理 2,390,275 行、模型 725,649 参数，并给出一次 epoch 4 的混淆矩阵与 F1；这些只是仓库文件中的历史输出，**本次没有复跑，不能写成已验证实验结果**。

### 5.14 其他直接相关文件

- **[代码确认]** `framework/sequential_input_encoding.py` 第 4–5 行使用缺少包前缀的 `from framework_component` 和 `from model_input_specification`。从仓库根目录导入时最小复现得到 `ModuleNotFoundError`。
- **[代码确认]** `framework/__init__.py` 没有导出 Notebook 所需符号。
- **[代码确认]** `framework/utilities.py` 用 pickle 保存/加载缓存元数据。
- **[高概率风险]** 若缓存文件来自不可信来源，pickle 反序列化可执行任意代码；缓存目录必须被视为可信本地工件，后续工程不应加载外部不可信 pickle。

## 6. 真实代码调用链

以下链路对应 `main.py` 的默认执行路径。

### 6.1 数据集读取与划分

```text
main.py:61-62
  └─ FlowTransformer.load_dataset(...)
       └─ FlowTransformer._load_preprocessed_dataset(...)
            ├─ pd.read_csv(...)/pd.read_feather(...)
            ├─ 创建 training_mask
            │    ├─ LastRows: 尾部 eval_n 行
            │    ├─ RandomRows: RandomState.choice 随机行
            │    └─ FilterColumn: 标记列少数值对应行
            └─ 生成 __training 与原始 __y
```

### 6.2 预处理

```text
_load_preprocessed_dataset()
  ├─ 数值列：清理非有限/超范围值
  ├─ StandardPreProcessing.fit_numerical(training rows)
  ├─ StandardPreProcessing.transform_numerical(all rows)
  ├─ StandardPreProcessing.fit_categorical(training rows)
  ├─ StandardPreProcessing.transform_categorical(all rows)
  └─ ModelInputSpecification(feature order/dimensions)
```

首次计算路径上，拟合操作只使用训练行；但缓存键缺少划分方法、种子和数据版本，错误缓存复用会绕过这一隔离保证。

### 6.3 窗口构造与模型前向

```text
FlowTransformer.evaluate()
  ├─ 选定训练目标索引/全部评估目标索引
  ├─ get_windows_for_indices(): X.iloc[i-window_size+1:i+1]
  ├─ samplewise_to_featurewise(): 每个特征形成
  │    (batch_or_eval_size, window_size, feature_dimension)
  └─ Keras Model
       ├─ per-feature Input
       ├─ input_encoding.apply()
       ├─ classification_head.apply_before_transformer()
       ├─ sequential_model.apply() / Transformer blocks
       ├─ classification_head.apply()
       ├─ Dense(ReLU) + Dropout MLP
       └─ Dense(1, sigmoid)
```

### 6.4 训练与评估

```text
训练：
  BatchYielder
    ├─ 每批分别抽取恶意和良性目标（约 50/50）
    ├─ 构造尾随窗口
    └─ m.train_on_batch(batch_X, batch_y)
  early stopping：训练 batch loss 是否创全局新低

评估：
  全部 ~training_mask 目标
    ├─ m.predict(eval_featurewise_X)
    ├─ score > 0.5
    ├─ TP/FP/TN/FN
    └─ sensitivity/specificity/balanced accuracy/precision/recall/F1
```

## 7. 数据划分与泄漏专项结论

### 7.1 训练预处理是否使用评估数据

- **[代码确认]** 在未命中缓存的正常路径中，数值 min/max 和分类频次只用 `training_mask=True` 的行拟合。
- **[代码确认]** 之后对训练和评估行一起执行 transform，这是常规做法，本身不等于拟合泄漏。
- **[高概率风险]** 缓存键缺少划分方法、随机种子和数据版本，可能加载另一实验的训练掩码/拟合参数；这会使上述隔离失效。

### 7.2 `RandomRows` 加相邻索引窗口

- **[代码确认]** 评估行散布在全表；训练目标窗口按原始相邻索引回看，并不检查上下文行的 `training_mask`。
- **[代码确认]** 最小逻辑复现：窗口大小 3、索引 5 为评估行、索引 6 为训练目标时，训练窗口 `[4,5,6]` 包含评估行 5。
- **结论：** 这是特征级数据泄漏。训练未读取评估行标签，但读取评估行特征仍会污染独立测试假设，特别是相邻流高度相关时。

### 7.3 `FilterColumn` 加相邻索引窗口

- **[代码确认]** 如果测试标志对应行在原数据中散布，窗口行为与 RandomRows 同类，也可能让训练窗口包含评估行特征。
- **[待实验验证]** 对具体 NSL-KDD 文件，需检查测试行是否成块排列、是否已排序、是否存在相邻重复实体。

### 7.4 `LastRows` 与边界

- **[代码确认]** 尾部评估区之前的训练目标只回看更早行，默认不会把尾部评估行纳入训练窗口。
- **[代码确认]** 第一个评估目标会回看训练尾部。例如 10 行、窗口 4、末 3 行评估时，第一个评估窗口为 `[4,5,6,7]`，其中 4–6 是训练行。
- **[高概率风险]** 这不是“测试特征进入训练”的反向泄漏，但它使评估样本上下文跨越划分边界。若部署设定允许使用历史已见流量，这可以合理；若要求测试窗口完全独立，则应留出 gap 或按完整窗口/会话划分。
- **[高概率风险]** `LastRows` 没有时间列排序或单调性校验，因此默认假定输入文件已经按预期时间/流顺序排列。
- **[待实验验证]** 必须对每个实际数据版本验证时间排序、相同主机/会话跨集合情况和重复流。

## 8. 标签、阈值与指标专项结论

| 项目 | 代码结论 | 状态 |
|---|---|---|
| 正类含义 | 标签字符串不等于 `benign_label`，即攻击/恶意 | [代码确认] |
| 负类含义 | 标签字符串等于 `benign_label`，即良性 | [代码确认] |
| 训练标签 | 恶意索引对应 1，良性索引对应 0 | [代码确认] |
| 预测阈值 | `score > 0.5` | [代码确认] |
| `score == 0.5` | 判为负类/良性 | [代码确认] |
| TP | 预测恶意且真实恶意 | [代码确认] |
| FP | 预测恶意且真实良性 | [代码确认] |
| TN | 预测良性且真实良性 | [代码确认] |
| FN | 预测良性且真实恶意 | [代码确认] |
| Precision | `TP/(TP+FP)` | [代码确认] |
| Recall/Detection Rate | `TP/(TP+FN)`，内部名 sensitivity/recall | [代码确认] |
| F1 | `2PR/(P+R)` | [代码确认] |
| False Alarm Rate | 未计算；应为 `FP/(FP+TN)` | [代码确认] |
| ROC-AUC | 未实现 | [代码确认] |
| PR-AUC | 未实现 | [代码确认] |
| 最佳阈值 | 未搜索、未保存 | [代码确认] |

标签比较通过 `self.y.astype('str') == str(benign_label)` 完成。**[高概率风险]** 若 CSV 将同一语义解析成不同字符串形式（例如 `1` 与 `1.0`、尾随空格、大小写差异），良性行可能被误标为攻击；需针对每个数据集做标签唯一值断言。

## 9. 平衡采样、评估分布与 early stopping

- **训练采样：[代码确认]** 每个 batch 分别无放回抽取恶意与良性，偶数 batch 严格 50/50；跨 batch 可重复抽到相同目标。
- **评估分布：[代码确认]** 对所有评估掩码行评估，没有 50/50 重采样，保留评估划分的类别比例。
- **训练/验证/测试结构：[代码确认]** 只有 training mask 与 evaluation mask，没有单独 validation mask。
- **early stopping：[代码确认]** 监控训练 batch loss 是否在 epoch 内创造全局最低值，不监控验证集指标。
- **最佳权重：[代码确认]** 不保存、不恢复。
- **[高概率风险]** 若后续反复根据 evaluation 结果选择模型、阈值或超参数，该集合将事实上承担验证集角色，不能再作为无偏最终测试集。

## 10. 硬编码路径与 Windows 兼容性

### 10.1 硬编码/运行时路径清单

| 文件与位置 | 路径/行为 | 结论 |
|---|---|---|
| `main.py:43` | `C:\Data\UQ\NIDS\Collected` | 特定 Windows 机器路径 |
| `framework/flow_transformer.py:293` | 默认相对目录 `cache` | 依赖当前工作目录 |
| `demonstration.ipynb` 单元 1 | 相对目录 `demonstration` | 可移植但依赖启动目录 |
| `demonstration.ipynb` 单元 3 | 固定远程下载 URL | 数据版本/完整性未锁定 |
| `demonstration.ipynb` 单元 5 | 固定 ZIP 内部路径 | 压缩包结构变化即失败 |
| `FlowTransformer_demo.ipynb` 单元 3 | `/content/drive/MyDrive/...` | Colab/Linux 路径 |
| `FlowTransformer_demo.ipynb` 单元 4 | `/content/dataset` | Colab/Linux 路径 |
| `FlowTransformer_demo.ipynb` 单元 5 | `/content/cache_folder` | Colab/Linux 路径 |

### 10.2 Windows 风险

- **[代码确认]** FlowTransformer_demo 的 `! mv` 是 POSIX 命令，原生 Windows Jupyter 默认 shell 通常不可用。
- **[代码确认]** 数据接口只认 `str`，不直接支持 `pathlib.Path`。
- **[高概率风险]** 相对缓存路径受 IDE/Notebook/命令行启动目录影响，容易读写到不同位置。
- **[高概率风险]** `os.mkdir()` 不递归，嵌套目录父级缺失会失败；并发创建存在竞态。
- **[高概率风险]** 大规模评估一次性构造全部窗口，Windows 内存压力和页面文件开销可能明显。

## 11. TensorFlow/Keras API 与当前环境兼容性

### 11.1 已确认的 API 用法

- `framework/flow_transformer.py:20-26`、输入编码、分类头及 Transformer block 多处先导入私有路径 `tensorflow._api.v2.v2.keras`，失败后回退 `tensorflow.keras`，随后又从独立顶层 `keras` 导入层。
- 多处同时使用 `tf.keras.*`、`tensorflow.keras` 和独立 `keras.*`。
- `input_encodings.py:138` 使用 `Embedding(input_length=...)`。
- `main.py:69`、两个 Notebook 使用 `jit_compile=True`。
- 训练循环使用 `train_on_batch()`，推理计时使用 `predict_on_batch()`。

### 11.2 风险判断

- **[高概率风险]** `tensorflow._api...` 是私有实现路径，不是稳定公共 API。
- **[高概率风险]** 混用 tf.keras 与独立 Keras 在 Keras 3/不同 TensorFlow 组合下可能产生 KerasTensor、Layer 或序列化不兼容。
- **[高概率风险]** `jit_compile=True` 要求 XLA 支持模型全部算子；不同 CPU/GPU、TensorFlow 版本及 Windows 环境可能编译失败或行为不同。
- **[代码确认]** 当前审计环境为 Python 3.11.9、NumPy 2.4.3、pandas 3.0.1，未安装 TensorFlow、tensorflow-intel、Keras、pyarrow；因此 `framework.flow_transformer` 导入即因缺少 TensorFlow 失败。
- **[待实验验证]** 当前机器上 `jit_compile=True` 是否可用、是否改善性能、是否改变数值结果，尚未验证。必须先锁定并安装兼容版本，再用最小模型和真实模型分别测试。

## 12. 导入错误与依赖问题

| 问题 | 位置 | 证据 |
|---|---|---|
| Notebook 包级导入失败 | `FlowTransformer_demo.ipynb` 单元 0；`framework/__init__.py` | [代码确认] 最小复现 `ImportError` |
| 缺包前缀导入失败 | `framework/sequential_input_encoding.py:4-5` | [代码确认] 最小复现 `ModuleNotFoundError` |
| TensorFlow 缺失 | 多个模型模块 | [代码确认] 当前环境导入失败 |
| `List` 隐式来自通配符 | `main.py:12-13,36` | [代码确认] 静态代码 |
| 无依赖清单/锁文件 | 仓库根目录 | [代码确认] 文件清点 |

所有 22 个 Python 文件均通过 Python AST 语法解析；这只能确认语法可解析，不能证明导入和运行成功。

## 13. 随机过程与不可复现来源

### 13.1 直接确认的随机源

1. `FlowTransformer.__init__()` 默认创建无种子的 `np.random.RandomState()`。
2. `RandomRows` 用 `self.rs.choice()` 选择评估行。
3. 训练批恶意/良性目标分别用 `self.rs.choice()`。
4. 未排序窗口分支使用全局 `np.random.choice()`，不受 `self.rs` 控制，且当前实现有索引维度错误。
5. Keras/TensorFlow 模型初始化与 Dropout 没有设置 TensorFlow 随机种子。
6. 数值特征通过 Python `set` 迭代，顺序受 `PYTHONHASHSEED` 影响。

### 13.2 复现信息缺口

- **[代码确认]** 示例不固定 Python、NumPy、TensorFlow/Keras 随机种子。
- **[代码确认]** 不启用确定性算子，也不记录硬件、驱动、线程数或 XLA 状态。
- **[代码确认]** 缓存键和训练结果不记录数据哈希、特征有序列表、划分索引哈希、随机种子、commit、依赖版本。
- **[代码确认]** 不保存模型最佳权重或最终权重。
- **[高概率风险]** 即使手动向 `FlowTransformer` 传入固定 `RandomState`，TensorFlow 初始化/Dropout、全局 NumPy 分支和 set 顺序仍未被统一控制。

## 14. 风险分级总表

### 14.1 已由代码直接确认

| 严重度 | 风险 | 位置 |
|---|---|---|
| 高 | RandomRows/散布 FilterColumn 的评估特征可进入训练窗口 | `flow_transformer.py:334-342` 与划分逻辑 `162-193` |
| 高 | 缓存键遗漏划分方法、种子、数据版本、规范、窗口大小 | `flow_transformer.py:112-143` |
| 高 | `eval_n == 0` 时 LastRows 把全部行设为评估 | `flow_transformer.py:163,172-173` |
| 高 | 未知分类值与第一已知类别都编码为 1 | `pre_processings.py:67-72` |
| 高 | 低于训练最小值的数值可经 log 产生 NaN，变换后不复检 | `pre_processings.py:37-55` |
| 中 | early stopping 只看训练 batch loss且不恢复最佳权重 | `flow_transformer.py:473-526` |
| 中 | 当前环境缺 TensorFlow/Keras，且仓库无依赖锁 | 仓库根目录/环境检查 |
| 中 | Notebook 包导出导入失败 | `FlowTransformer_demo.ipynb` 单元 0；`framework/__init__.py` |
| 中 | SequentialInputEncoding 导入失败 | `sequential_input_encoding.py:4-5` |
| 中 | 未排序窗口分支必然发生二维索引错误 | `flow_transformer.py:344-346,562-564` |
| 中 | 固定阈值且缺 FAR、ROC-AUC、PR-AUC、最佳阈值 | `flow_transformer.py:396-433` |
| 中 | 无独立验证集、无 checkpoint | `flow_transformer.py:evaluate()` |
| 中 | 数值特征 set 顺序不稳定 | `flow_transformer.py:195,203,219` |
| 低 | 文档与代码存在命名/训练集表述错误 | `readme.md:54`；`demonstration.ipynb` 单元 10 |

### 14.2 高概率风险

| 严重度 | 风险 | 需要的后续证据 |
|---|---|---|
| 高 | LastRows 未验证时间排序，可能不是时间外推评估 | 数据时间列、排序检查、数据版本说明 |
| 高 | 相同主机/会话/重复流可能跨训练评估集合 | 分组键、重复检测、实体交叉统计 |
| 高 | 私有 TensorFlow API与 Keras 混用在新版本不兼容 | 锁定依赖后的导入/构建测试 |
| 高 | `jit_compile=True` 在当前 Windows/硬件失败或数值不一致 | XLA 开关对照测试 |
| 中 | 标签字符串规范化差异导致正负类反转/错标 | 各数据集标签唯一值与计数断言 |
| 中 | MQTT 的 `benign_label='1'` 可能与 `is_attack` 语义冲突 | 官方数据字典与实际值核验 |
| 中 | 一次性构建全评估窗口导致内存不足 | 数据规模下内存峰值测量 |
| 中 | pickle 缓存被不可信文件替换时有代码执行风险 | 缓存来源/权限模型 |
| 中 | 无位置编码限制顺序表达能力 | 对照消融实验 |
| 中 | 非标准 encoder/decoder 残差与注意力结构影响论文复现 | 论文实现对照、模型摘要与消融 |

### 14.3 尚需运行实验验证

1. 论文使用的准确数据文件、下载地址、文件哈希和行数是否与 Notebook 历史输出一致。
2. 当前 commit 在锁定 TensorFlow/Keras 版本下能否完整构建所有编码、分类头和 Transformer 组合。
3. `jit_compile=True` 在当前 Windows CPU/GPU 环境的兼容性、性能和数值一致性。
4. 三个数据集的时间排序、重复行、实体交叉、类别分布和标签规范。
5. 论文指标能否在固定种子、多次重复实验下复现；本次没有运行训练，不能给出任何新指标。
6. 缓存命中/未命中、不同划分方法、不同种子之间是否发生交叉污染。
7. 最佳阈值应在独立验证集上选择，并在最终测试集一次性报告；原代码没有此流程。

## 15. 许可证与来源

- **[代码确认]** `LICENCE.txt` 是 “GNU AFFERO GENERAL PUBLIC LICENSE, Version 3, 19 November 2007” 完整文本。
- **[代码确认]** 主要 Python 源文件保留 `FlowTransformer 2023 by liamdm / liam@riftcs.com` 版权头；本阶段没有删除或修改。
- **[高概率风险]** 新项目若复制、修改或网络部署 AGPL-3.0 覆盖代码，需要保留许可证、原作者版权与修改说明，并履行 AGPL-3.0 对应源码提供义务。具体法律适用应由项目负责人确认，本报告不构成法律意见。
- **建议：** 新项目建立 `THIRD_PARTY_NOTICES.md` 或等价来源清单，记录原仓库 URL（待确认）、commit、复制文件、修改内容、许可证与数据集许可。

## 16. 本阶段实际执行命令与结果

以下均在原仓库根目录执行；未运行会修改原仓库的格式化、安装、训练或缓存生成命令。

1. `Test-Path`、`Get-ChildItem -Force`、递归查找 `AGENTS.md`：仓库和 `.git` 存在，未发现适用的 `AGENTS.md`。
2. 直接 `git branch/status/log`：失败，原因是 Git dubious ownership 安全检查；没有修改任何配置。
3. `git -c safe.directory=C:/Users/LENOVO/FlowTransformer branch --show-current`：成功，输出 `master`。
4. `git -c safe.directory=C:/Users/LENOVO/FlowTransformer rev-parse HEAD`：成功，输出完整 commit hash。
5. `git -c safe.directory=C:/Users/LENOVO/FlowTransformer log -1 ...`：成功，取得最近提交信息。
6. `git -c safe.directory=C:/Users/LENOVO/FlowTransformer status --porcelain=v1 --branch` 与 `git status`：成功，审计开始时工作区干净。
7. `rg --files -g '*.py'`、`rg --files -g '*.ipynb'`、README/许可证 glob：成功，生成文件清单。
8. `Get-Content` 带行号读取所有指定 Python/README 文件及相关 base/utilities/Transformer block 文件：成功。
9. `Get-Content -Raw | ConvertFrom-Json` 解析两个指定 Notebook 的代码单元、execution count 和输出类型：成功。
10. `rg -n` 检索路径、随机过程、TensorFlow/Keras API 和 import：成功。
11. `python --version` 与 `importlib.metadata`：成功；确认 Python 3.11.9，TensorFlow/Keras/pyarrow 未安装。
12. `PYTHONDONTWRITEBYTECODE=1 python -B` 导入 `framework.flow_transformer`：失败，确认缺少 TensorFlow。
13. `PYTHONDONTWRITEBYTECODE=1 python -B` 导入 `framework.sequential_input_encoding`：失败，确认缺包前缀导入错误。
14. `PYTHONDONTWRITEBYTECODE=1 python -B` 执行 FlowTransformer_demo 的 `from framework import ...`：失败，确认 `framework/__init__.py` 未导出符号。
15. `python -B` 对 22 个 Python 文件执行 `ast.parse`：成功，输出 `AST_OK=22`。
16. `python -B` 最小逻辑复现 LastRows 跨边界、RandomRows 训练窗口含评估行、`eval_n=0` 与未排序窗口索引：前三项输出符合代码分析；未排序分支如预期触发 `IndexError`。
17. `python -B` 最小复现分类未知值碰撞、数值低于训练 min 产生 NaN、不同 `PYTHONHASHSEED` 的 set 顺序：成功复现，并出现预期 RuntimeWarning。
18. `python -B` 用合成布尔标签复核混淆矩阵/F1/FAR 方向：成功，四格各 1 时 Precision、Recall、F1、FAR 均为 0.5。

19. `New-Item -ItemType Directory -Force` 创建新项目的 `docs`、`logs`、`artifacts`，随后 `Copy-Item` 将报告复制到指定路径：成功。
20. `Get-FileHash -Algorithm SHA256` 比较临时报告与目标报告：成功，两份文件哈希一致；最终哈希以交付校验输出为准。
21. `git -c safe.directory=... status --porcelain=v1 --branch` 与 `git diff --exit-code`：成功；最终仍为 `master...origin/master`，没有文件差异，`git diff` 退出码为 0。

## 17. 本阶段测试结果

- Git 只读基线检查：通过（采用单次 `safe.directory` 参数后）。
- 22 个 Python 文件 AST 解析：通过。
- 指定 Notebook JSON 解析：通过。
- 预处理模块导入：通过。
- FlowTransformer 模块导入：失败，当前环境缺 TensorFlow；属于已记录环境阻塞，不宣称模型可运行。
- FlowTransformer_demo 包级导入：失败，确认仓库导出错误。
- SequentialInputEncoding 导入：失败，确认相对/包导入错误。
- 窗口/切片/预处理边界最小复现：完成并复现上述缺陷。
- 完整训练、完整评估、`jit_compile=True`：未执行，状态为待确认。

## 18. 当前风险与下一阶段建议

### 当前风险

1. 当前环境缺少 TensorFlow/Keras/pyarrow，无法验证模型构建、缓存 Feather、XLA 和训练。
2. 数据文件不在本次审计范围内，时间顺序、标签、重复、实体交叉和数据版本均待确认。
3. Notebook 历史输出缺少环境、commit、种子和数据哈希，不能作为竞赛实验结果直接引用。
4. 缓存键、RandomRows 窗口和未知类别编码问题可能直接影响实验正确性，应优先建立回归用例。

### 下一阶段建议

下一阶段建议只做“可复现环境与缺陷复现测试设计”，暂不重构模型：

1. 在新项目中记录原 commit、Python/依赖版本候选和数据文件哈希方案。
2. 为缓存隔离、RandomRows 泄漏、LastRows gap、未知类别编码、数值越界、标签方向和指标公式建立小型合成数据测试。
3. 选择并锁定一个与原代码兼容的 TensorFlow/Keras 基线环境，分别测试 `jit_compile=False/True`。
4. 在任何真实训练前定义 train/validation/test 的时间或实体分组协议，禁止用最终测试集选择阈值。

本报告仅完成阶段任务 1 的只读审计，没有修改原仓库，也没有提前实施修复或工程重构。
