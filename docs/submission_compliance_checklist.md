# CampusFlowGuard 提交包合规检查清单

更新日期：2026-07-18

> 本清单只依据本地材料准备，不是法律意见。人工确认应保存页面截图、许可证原文、日期、文件哈希或负责人签字记录。

## 1. 提交场景与权利主体

- [x] 当前提交场景确认为竞赛评委私有评审，不是公开发布。
- [ ] 确认 CampusFlowGuard 代码、文档、图表和权重的版权主体。
- [ ] 获取竞赛官方规则，确认源码、权重、数据样例、依赖和许可证是否必交。
- [x] 已确认 CampusFlowGuard 参考 FlowTransformer 上游源代码后自主改写；`baseline.py`、`train.py` 已标记。
- [x] 提交版本采用 AGPL-3.0，根目录保留完整 `LICENSE`。
- [ ] 版权主体确认后补充文件级版权年份/名称和 SPDX。
- [ ] 建立首个 Git commit，以 commit 和哈希冻结提交版本。

## 2. 必须随包保留的 LICENSE / NOTICE

### 2.1 当前应保留

- [x] 根目录 `LICENSE`：完整 GNU AGPL-3.0，不得删除或改写。
- [x] `NOTICE.md`：记录规范上游、原作者、改写关系和 CampusFlowGuard 新增工作。
- [x] `LICENSES/FlowTransformer-AGPL-3.0.txt`：保留上游许可证备份。
- [x] `LICENSES/THIRD_PARTY_NOTICES.md`：已更新当前训练、改写和私评状态。
- [x] `docs/third_party_licenses.md` 和 `docs/code_provenance_audit.md`。
- [x] README/NOTICE 已写明 FlowTransformer 作者、规范 URL、审计 commit、AGPL-3.0 和改写范围。

### 2.2 条件保留

- [x] `baseline.py`、`train.py` 保留 `FlowTransformer 2023 by liamdm / liam@riftcs.com`、规范 URL、自主改写说明和 AGPL-3.0 指引。
- [ ] 私评包确认包含当前完整对应源码，而不仅是模型权重或可执行文件。
- [ ] 若提交 wheel、DLL、离线环境或可执行包：保留每个 distribution 的 LICENSE、COPYING、NOTICE 和 bundled notices，尤其 TensorFlow、NumPy、SciPy、PyArrow、Matplotlib 字体和 Uvicorn extras。
- [ ] 若许可允许提交数据：附发布机构、版本、许可证、署名、再分发条款和上下游哈希。
- [ ] 若提交模型权重：附训练配置、seed、数据指纹、split manifest、权重哈希和允许分发的人工结论。

## 3. 默认不应提交的内容

### 3.1 数据

- [ ] 排除 `NF-UNSW-NB15-v2.csv` 和原压缩包。
- [ ] 排除 `artifacts/demo/benign_demo.csv`、`attack_demo.csv`、`mixed_demo.csv`。
- [ ] 排除 `artifacts/examples/local_inference_input.csv` 和其他真实流记录。
- [ ] 排除 PCAP/PCAPNG、IP 地址、抓包和任何未脱敏校园网/实验室流量。
- [ ] 许可确认前只保留聚合指标、哈希、schema 和“来源待确认”说明。

### 3.2 环境与依赖

- [ ] 排除 `.venv/`、Conda 环境、`site-packages/` 和解释器副本。
- [ ] 排除未完成许可证扫描的 wheel、DLL、CUDA/cuDNN、驱动和 Uvicorn extras 离线包。
- [ ] 排除 `__pycache__/`、测试/构建缓存、临时文件和本机绝对路径配置。
- [ ] 优先提交 `pyproject.toml`、环境声明和安装说明，由评审环境安装依赖。

### 3.3 第三方源码与媒体

- [ ] 不复制 `C:\Users\LENOVO\FlowTransformer` 整仓或 Notebook。
- [ ] 不加入来源未确认的学校/竞赛标识、图标、图片、字体、截图、音视频。
- [ ] 当前网页无外部资源；后续资源逐项记录作者、URL、日期、许可证和使用范围。
- [ ] 排除密钥、token、账号、`.env`、远程凭据和隐私信息。

## 4. 模型权重提交仍待确认

当前权重有项目训练日志、配置、seed 和哈希链，原仓库未发现预训练权重，因此本地证据支持“由本项目训练生成”。但公开分发权尚未确认。

- [x] 提交场景为评委私有评审。
- [ ] 确认竞赛是否强制提交权重、私评包是否会被主办方长期保存或二次公开。
- [ ] 确认数据许可是否允许生成和分发模型权重。
- [x] `baseline.py`/`train.py` 已按参考上游源码后的自主改写纳入 AGPL-3.0 来源说明。
- [ ] 确认 AGPL-3.0 与训练数据许可对权重提交的具体影响。
- [ ] 确认权重版权/权利主体。
- [ ] 任一项未确认时，权重不进入公开发布包。
- [ ] 批准提交时只提交必要最佳权重，避免重复 smoke/Baseline/Focal 文件。
- [ ] 当前 Focal 演示权重 SHA-256：`26e21e79fe5bb3d646a9749debfc178b600311e5931f8401ce993fbd1d8b768c`。

## 5. 必须人工打开核实的页面与字段

### 5.1 数据集页面

线索：`https://staff.itee.uq.edu.au/marius/NIDS_datasets/`

- [ ] 页面标题、维护机构和当前可访问性。
- [ ] `NF-UNSW-NB15-v2` 准确名称、版本、发布日期和文件清单。
- [ ] 发布机构、作者、权利人、DOI 和推荐引用。
- [ ] 许可证名称、完整文本或明确链接。
- [ ] 研究、教学、竞赛、商业和模型训练使用条件。
- [ ] 原文件、抽样记录、派生 CSV、特征和权重再分发限制。
- [ ] 署名、免责声明和引用格式。

### 5.2 UQ RDM 记录

线索：`https://api.rdm.uq.edu.au/production/files/8c6e2a00-ef9c-11ed-827d-e762de186848/download`

- [ ] 找到对应 landing page，而不只保存下载接口。
- [ ] 核实 title、作者、机构、日期、版本、DOI/record ID。
- [ ] 核实压缩包名称、大小、上游校验值和内部路径。
- [ ] 核实与数据集页一致的许可证及附加条款。
- [ ] 保存页面、许可文本、下载日期和原包哈希至外部 `metadata/`。

### 5.3 FlowTransformer 仓库

规范上游已确认为 `https://github.com/liamdm/FlowTransformer.git`。本地 `git@gitee.com:ai-safety-creative-competition/FlowTransformer.git` 仅作为审计基线副本记录。

- [x] 规范上游 URL 和作者 `liamdm` 已确认。
- [ ] 人工保存 GitHub 页面、许可证和作者信息截图/PDF。
- [ ] commit `52236c8c9feccdeb44acf578789db7b8b36bacea` 是否可定位。
- [ ] 该 commit 的 LICENSE、版权头、分支、tag 是否与本地一致。
- [ ] 记录本地 Gitee 仓库是镜像、fork 还是竞赛副本。
- [x] 当前私评提交保留根 LICENSE、NOTICE 和当前对应源码。
- [ ] 未来公开发布或网络运行时，再核实源码获取和界面通知流程。

### 5.4 Python 发行包

- [ ] 对最终每个直接和传递依赖打开实际安装来源页面，核实版本、许可证、文件哈希和 yanked 状态。
- [ ] 补核未安装的 pytest 8.3.4、pytest-cov 6.0.0、Ruff 0.8.4，以及 Keras 3.15.0 wheel LICENSE。
- [ ] TensorFlow Windows wheel 保存 Apache LICENSE 和完整 `THIRD_PARTY_NOTICES.txt`。
- [ ] Uvicorn `standard` 列出实际 extras 并逐项保存许可证。

### 5.5 竞赛官方规则

当前仅确认“评委私有评审”，项目仍没有官方规则 URL，需负责人提供并存档。

- [ ] 比赛全称、届次、主办/承办单位和作品权利条款。
- [ ] 是否要求公开源码、指定许可证或授权主办方使用/展示。
- [ ] 是否允许 AGPL 代码、公开数据、模型权重和外部下载链接。
- [ ] 是否要求数据来源证明、隐私/伦理声明和第三方授权书。
- [ ] 提交格式、大小、文件类型、外链规则和截止时间。
- [ ] 评审包是否公开、保存期限、访问主体和删除/撤回机制。

## 6. 最终技术验收

- [ ] 原 FlowTransformer 仓库保持干净。
- [ ] 冻结 Baseline manifest SHA-256 仍为 `71e6e8741907a41cc88423ee8792814a45203a3523c1d65039a70516ef7da09a`。
- [ ] 对最终目录生成文件清单、大小和 SHA-256，人工检查每个大文件。
- [ ] 在干净环境恢复并运行最小测试、CLI 推理和 FastAPI 演示。
- [ ] 搜索绝对路径、账号、IP、密钥、token、个人姓名和敏感日志。
- [ ] 核对 README、报告、NOTICE、许可证、代码、配置、权重和指标版本一致。
- [ ] 由负责人完成最终版权、数据和竞赛规则签字确认。
