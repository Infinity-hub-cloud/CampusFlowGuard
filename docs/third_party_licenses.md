# CampusFlowGuard 第三方依赖与许可证初步核查

核查日期：2026-07-18

## 1. 核查范围与口径

本次依赖核查使用本地证据：`pyproject.toml`、`environment/`、Python import、当前解释器的 distribution metadata 与许可证文件清单、现有项目文档、FlowTransformer 本地 Git 仓库、实验配置和产物。项目负责人另已确认 FlowTransformer 规范上游、改写关系和评委私评场景；该确认已作为本阶段许可证落实依据。

本文是提交前的初步工程核查，不构成法律意见。表中的“需保留”含义是：如果提交包直接包含该依赖的源码、wheel、二进制文件、虚拟环境或打包后的可执行程序，应同时保留该 distribution 自带的版权、LICENSE、COPYING 和 NOTICE。若提交包只包含项目源码和依赖声明，建议不复制第三方包本体，而是保留本清单并让使用者通过包管理器安装。

当前环境不是最终冻结交付环境。`environment/requirements-cpu.txt` 自身标记为候选锁定、全新环境待验证；开发依赖尚未全部安装，传递依赖也未形成完整 SBOM。因此本文不能替代最终交付前对确切 wheel 的逐文件扫描。

## 2. 项目直接依赖

版本栏同时记录 `pyproject.toml` 声明和当前解释器实测版本。许可证来自当前安装包 metadata 或随包文件；没有本地证据的项不补猜。

| 依赖 | 声明 / 当前版本 | 本地确认的许可证 | 项目用途 | 随提交包保留许可证 | 初步分发建议 |
| --- | --- | --- | --- | --- | --- |
| NumPy | `>=1.26,<2` / 1.26.4 | 主体 BSD-3-Clause；当前 wheel 还记录 OpenBLAS、LAPACK、GCC runtime 等 bundled licenses | 数组、窗口、指标与模型输入 | 是；若捆绑 wheel，应原样保留完整 `numpy-*.dist-info/LICENSE.txt`，不能只放一份 BSD 文本 | 源码包仅声明依赖；不要复制当前 site-packages |
| Pandas | `>=2.2,<3` / 2.3.3 | BSD-3-Clause | CSV 读取、表格处理、日志与指标 | 是（随包分发依赖时），保留 distribution LICENSE | 仅声明依赖或提供环境文件 |
| scikit-learn | `>=1.5,<2` / 1.9.0 | BSD-3-Clause；安装包另含 vendored license 文件 | 分层划分、预处理、指标 | 是（随包分发依赖时），保留 COPYING 和 vendored notices | 仅声明依赖或提供环境文件 |
| PyArrow | `>=17,<26` / 25.0.0 | Apache-2.0 | 声明的数据格式依赖与 import 检查；当前核心训练/推理代码未直接 import | 是（随包分发依赖时），保留 LICENSE 与 NOTICE | 若最终提交不使用 Arrow 格式，可在后续依赖精简阶段评估；本阶段不修改声明 |
| PyYAML | `>=6,<7` / 6.0.3 | MIT | 配置和 schema 读取 | 是（随包分发依赖时），保留 LICENSE | 仅声明依赖或提供环境文件 |
| Matplotlib | `>=3.9,<4` / 3.10.8 | Matplotlib License；metadata classifier 为 PSF License；随包字体另有 DejaVu/STIX 许可证 | 生成混淆矩阵、ROC、PR 和对比图 | 是（随包分发依赖时），同时保留 Matplotlib 与字体许可证 | 仅声明依赖；项目生成的图另见资源核查 |
| TensorFlow / tensorflow-intel | `tensorflow==2.17.1` / 2.17.1 | Apache-2.0；当前安装包含 `tensorflow/THIRD_PARTY_NOTICES.txt` 和多项 bundled license | Transformer 构建、训练、评估和推理 | 是；若捆绑运行环境，必须保留 Apache LICENSE 和完整 THIRD_PARTY_NOTICES | 不打包当前 Python 环境；通过固定依赖安装 |
| Keras | `keras==3.15.0` / 3.15.0 | metadata 声明 Apache License 2.0；当前 distribution 未检出独立 LICENSE 文件 | 模型、层、loss、训练回调和权重加载 | 是（随包分发依赖时）；需从对应正式 distribution 补齐许可证文本 | 仅声明依赖；离线打包前人工补核官方 wheel |
| FastAPI | `fastapi==0.115.6` / 0.115.6 | MIT | 本地 `POST /predict` 和 HTML 页面服务 | 是（随包分发依赖时），保留 LICENSE | 仅声明 API 可选依赖 |
| Uvicorn | `uvicorn[standard]==0.34.0` / 0.34.0 | BSD-3-Clause | 本地 ASGI 服务启动 | 是（随包分发依赖时），保留 LICENSE | 仅声明 API 可选依赖；`standard` extras 的传递依赖仍需最终扫描 |
| python-multipart | `==0.0.20` / 0.0.20 | Apache-2.0 | FastAPI CSV 文件上传解析 | 是（随包分发依赖时），保留 LICENSE | 仅声明 API 可选依赖 |
| HTTPX | `>=0.27,<1` / 0.28.1 | BSD-3-Clause | FastAPI/Starlette TestClient 测试链路 | 是（随包分发依赖时），保留 LICENSE | 仅声明 API 可选依赖 |

## 3. 环境候选、构建与开发依赖

以下包出现在环境文件、构建配置或环境检查中，但不都属于应用代码的直接运行 import。

| 依赖 | 本地版本状态 | 本地确认的许可证 | 用途与注意事项 | 随提交包保留许可证 |
| --- | --- | --- | --- | --- |
| SciPy | 候选固定 1.17.1；当前 1.17.1 | BSD-3-Clause；distribution 另有 uarray、PocketFFT、Qhull 等许可证 | scikit-learn 等科学计算依赖；核心项目未直接 import | 是（若捆绑）；保留完整 distribution LICENSE/COPYING |
| Requests | 候选固定 2.32.5；当前 2.32.5 | Apache-2.0 | 环境诊断依赖；训练代码仅屏蔽其兼容性警告 | 是（若捆绑） |
| Pydantic | 候选固定 2.10.4；当前 2.13.4 | MIT | FastAPI 传递依赖；候选文件与当前环境版本不一致 | 是（若捆绑）；最终锁定前必须统一版本并复核 |
| pytest | 声明 8.3.4；当前未安装 | 待人工确认（本地无 distribution metadata） | 可选测试运行器；当前测试实际使用 `unittest` | 若随包提供则必须按其许可证保留；当前不要捆绑 |
| pytest-cov | 声明 6.0.0；当前未安装 | 待人工确认 | 测试覆盖率工具 | 同上 |
| Ruff | 声明 0.8.4；当前未安装 | 待人工确认 | 静态检查工具 | 同上 |
| setuptools | 构建要求 `>=68`；当前 65.5.0 | metadata classifier 为 MIT | PEP 517 构建后端；当前全局版本低于声明，构建隔离状态待复核 | 若捆绑构建工具则保留 LICENSE；通常仅声明构建依赖 |
| wheel | 构建要求未固定；当前 0.47.0 | MIT | wheel 构建 | 若捆绑则保留 LICENSE；通常仅声明构建依赖 |

`environment/environment-cpu.yml` 还固定 Python 3.11.9 和 pip 24.3.1。它们属于解释器与安装工具，不在上面的应用依赖表中；若提交完整离线运行环境，同样需要保留其许可证并纳入最终 SBOM。

## 4. 传递依赖与二进制包风险

本次没有把 TensorFlow、FastAPI、Uvicorn `standard` extras、Pandas、scikit-learn 等包的全部传递依赖逐项列出。当前 wheel 中已观察到 bundled 第三方组件和单独 NOTICE，说明不能用“所有 Python 包都是宽松许可证”作为提交结论。

提交包建议不包含 `.venv`、`site-packages` 或从当前全局 Python 复制出的 DLL。若竞赛要求离线运行，应先在全新、固定环境安装确切 wheel，再导出完整依赖版本、每个 distribution 的 LICENSE/NOTICE 和文件哈希；未完成该步骤前，完整 Python 环境属于“暂不应分发”。

## 5. FlowTransformer 原仓库

当前已确认：

- 审计 commit：`52236c8c9feccdeb44acf578789db7b8b36bacea`。
- 规范上游：`https://github.com/liamdm/FlowTransformer.git`。
- 本地审计仓库 `origin` 为 `git@gitee.com:ai-safety-creative-competition/FlowTransformer.git`，作为本地竞赛基线副本记录，不再作为规范上游地址。
- 原作者标识：`liamdm / liam@riftcs.com`，来自已有第三方声明。
- `LICENCE.txt` 是 GNU Affero General Public License v3.0（AGPL-3.0）。
- 根目录保留完整 `LICENSE`；`LICENSES/FlowTransformer-AGPL-3.0.txt` 与本地上游原文件 SHA-256 相同：`13fb48a6c31a72e0e2b69650b1cebe8954f7b73449848492dcfd22b25bd91afa`。
- CampusFlowGuard 参考 FlowTransformer 上游源代码后进行自主改写，提交版本适用 AGPL-3.0，详细说明见 `NOTICE.md`。
- 作品当前仅提交竞赛评委私有评审。

当前 CampusFlowGuard 没有复制整个原仓库，原仓库也未发现 `.h5`、`.keras` 或 `.ckpt` 权重文件。提交包应保留上游作者、规范 URL、修改关系、完整 AGPL-3.0、当前对应源代码和 `NOTICE.md`。源码相似度仅是本地证据，不作为法律结论。

- 私评包可包含：CampusFlowGuard 当前对应源代码、根 `LICENSE`、`NOTICE.md`、第三方声明和来源审计。
- 仅引用：FlowTransformer 整仓、论文和上游 commit；无需把上游整仓重复复制进提交包。
- 仍需人工确认：版权主体、最终私评包具体文件范围，以及将来公开发布/网络部署时的操作流程。

## 6. NF-UNSW-NB15-v2 数据

本地可确认的仅是来源线索和当前副本：

- 原 FlowTransformer Notebook 记录数据集页面 `https://staff.itee.uq.edu.au/marius/NIDS_datasets/`。
- Notebook 记录 UQ RDM 域名下载接口 `https://api.rdm.uq.edu.au/production/files/8c6e2a00-ef9c-11ed-827d-e762de186848/download`。
- 压缩包预期内部路径为 `fe6cb615d161452c_MOHANAD_A4706/data/NF-UNSW-NB15-v2.csv`。
- 当前本地 CSV SHA-256 为 `05019e3ac8d55b3f074f9f042623bbbb1af2527b8b78558c69353ccb2ccefa3f`，但该值不是上游确认哈希。

上述 URL 未联网复核。发布机构/权利人、数据版本说明、许可证、研究或竞赛使用条件、署名要求、再分发限制和本地文件下载链路均为“待人工确认”。域名属于 UQ 来源线索，不能据此自行推断完整法律发布主体。

在许可确认前：

- 暂不应分发：原始 CSV、压缩包、从真实记录提取的 `artifacts/demo/*.csv` 和 `artifacts/examples/local_inference_input.csv`。
- 可随报告保留：文件名、本地 SHA-256、行列统计、聚合指标和“来源待确认”声明；这些内容仍需避免被表述为上游授权证明。
- 仅可引用：现有未验证 URL 只能作为人工复核线索，不应标成已确认官方地址。

## 7. 模型权重来源与分发

本地实验配置、逐 epoch 日志、三个固定 seed 指标和权重文件形成了训练追溯链；原 FlowTransformer 仓库未发现模型权重文件。当前选用的 Focal 权重为：

- 文件：`artifacts/experiments/focal_loss/unsw_focal_loss_v1/models/seed20260717.weights.h5`
- 生成实验：`unsw_focal_loss_v1`
- seed：`20260717`
- SHA-256：`26e21e79fe5bb3d646a9749debfc178b600311e5931f8401ce993fbd1d8b768c`

因此可由本地工程证据确认：当前 Baseline 与 Focal 权重是 CampusFlowGuard 实验流程自行训练生成，不是从原仓库下载的预训练权重。但“自行训练”不自动等于“可提交”；权重仍受到训练数据许可、AGPL 来源说明和竞赛私评要求的共同影响。

当前是否把权重纳入私评包仍待确认。若提交，应同时保留权重 SHA-256、训练配置、数据指纹、split manifest、根 `LICENSE` 和 `NOTICE.md`；原始数据许可未确认前不得据此公开发布权重。

## 8. 网页、图标、图片与图表

本地扫描结果：

- `src/campus_flow_guard/static/index.html` 使用项目内嵌 HTML、CSS 和 JavaScript，不加载外部 CDN、Web 字体、图标库、图片或外部 API。
- 未发现项目网页使用的第三方 SVG、图标、照片、插画、音频或视频。
- 仓库中的 PNG 均为项目实验代码通过 Matplotlib 根据本地指标生成的混淆矩阵、ROC、PR 和对比图；未发现外部图片素材。
- 页面使用操作系统 `system-ui` 字体，不随提交包分发字体文件。

就当前本地材料而言，没有需要单独附带的第三方网页图片或图标许可证。若后续加入学校标识、竞赛标识、截图、照片、图标或视频，必须逐项记录作者、来源 URL、下载日期、许可证和允许的提交/公开范围。

## 9. 提交包初步建议

### 9.1 可直接分发

- 根 `LICENSE`、`NOTICE.md`、第三方声明和当前对应源代码，可用于已确认的评委私有评审包。
- 不含真实流记录的项目文档、配置、聚合指标、日志摘要和项目生成实验图。
- 本地网页源码当前未发现外部媒体资产，并纳入同一 AGPL-3.0 来源说明。

### 9.2 仅引用或通过包管理器获取

- Python 依赖：提交 `pyproject.toml` 与环境文件，不复制当前全局环境；由评审环境安装。
- FlowTransformer 原仓库源码：引用规范上游、commit、原作者和 AGPL-3.0，不复制整个仓库。
- 数据下载 URL：目前只能作为待复核线索；确认官方页面后再给出正式引用。

### 9.3 暂不应分发

- `NF-UNSW-NB15-v2.csv`、原压缩包以及真实数据派生的小型 CSV。
- 当前 `.venv`、`site-packages`、wheel/DLL 离线集合；尚无完整传递依赖许可证包。
- 模型权重的公开发布包；等待数据许可、代码来源和项目许可证确认。
- 来源或授权不明的后续图片、学校/竞赛标识、截图、字体和媒体文件。

## 10. 提交前待办

1. 人工访问并存档数据发布页面、发布机构/权利人、许可证、版本、署名和再分发条款，计算下载件与本地 CSV 哈希。
2. 保存规范上游 GitHub 页面、commit、许可证和作者信息截图，说明本地 Gitee 基线副本的用途。
3. 确认 CampusFlowGuard 版权主体；许可证已落实为 AGPL-3.0。
4. 在最终隔离环境生成精确依赖清单、传递依赖 SBOM、wheel 哈希及 LICENSE/NOTICE 目录。
5. 补核未安装的 pytest、pytest-cov、Ruff 以及 Keras wheel 的正式许可证文本。
6. 当前已确认评委私有评审；仍需决定真实样例、模型权重和离线依赖是否纳入私评包。

根目录 `NOTICE.md` 和更新后的 `LICENSES/THIRD_PARTY_NOTICES.md` 已记录当前改写、训练与私评状态；最终提交前仍应核对二者与实际包内容一致。
