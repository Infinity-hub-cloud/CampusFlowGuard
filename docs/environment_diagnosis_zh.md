# CampusFlowGuard 环境诊断报告

## 1. 诊断范围

- 诊断时间：2026-07-14（Asia/Shanghai）
- 当前 Python：`D:\python3.11\python.exe`
- 原仓库：`C:\Users\LENOVO\FlowTransformer`（只读导入检查）
- 新项目：`D:\Users\LENOVO\Desktop\AI_safety_competition_code`
- 本阶段没有安装或卸载软件，没有创建训练任务，没有加载数据集。

“当前已确认”只代表本机当前进程实际命令结果；环境文件中的新隔离环境尚未创建，均标记为“待验证”。

## 2. Windows 与工具链

### 2.1 Windows

| 项目 | 实际结果 |
|---|---|
| .NET OSDescription | `Microsoft Windows 10.0.26200` |
| 注册表 ProductName | `Windows 10 Home China` |
| DisplayVersion | `25H2` |
| Build | `26200.8655` |
| EditionID | `CoreCountrySpecific` |
| 架构 | x64 |
| PowerShell | Windows PowerShell 5.1.26100.8655 Desktop |

WMI/CIM 查询因沙箱权限返回“拒绝访问”，`cmd` 在沙箱 PATH 中不可用。注册表 ProductName、Conda user-agent（显示 Windows/11）和系统 build 的产品命名存在信息源差异，因此本报告保留原始字段，不自行把产品名改写成 Windows 11。

### 2.2 Python、pip、Conda、Git

| 工具 | 版本与位置 | 状态 |
|---|---|---|
| Python | 3.11.9，`D:\python3.11\python.exe` | 当前使用 |
| pip | 26.1.2，`D:\python3.11\Lib\site-packages\pip` | 可用 |
| Conda | 24.11.3，`D:\Anaconda` | 可用，当前未激活环境 |
| Conda base Python | 3.12.4 | 非当前 Python |
| Git | 2.45.1.windows.1 | 可用 |

`python -m pip check` 返回一个当前全局环境冲突：`opencv-python 5.0.0.93` 要求 NumPy >=2，但当前 NumPy 是 1.26.4。该冲突不是 CampusFlowGuard 引入的，但说明当前全局 Python 不适合作为正式复现环境。

导入 TensorFlow 等包时还出现 `RequestsDependencyWarning`：当前 urllib3/chardet/charset_normalizer 组合不符合 requests 的支持范围。正式环境应通过隔离环境重新解析并执行 `pip check`。

## 3. NVIDIA GPU、驱动与 CUDA

| 项目 | 实际结果 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| GPU 索引 | 0 |
| 总显存 | 8151 MiB（nvidia-smi 报告值） |
| 诊断时空闲显存 | 5668 MiB（瞬时值） |
| 驱动 | 592.01 |
| 驱动报告 CUDA 能力上限 | CUDA 13.1 |
| Compute Capability | 12.0 |
| 驱动模式 | WDDM |
| `nvidia-smi` | `C:\Windows\System32\nvidia-smi.exe` 可用，但不在当前 PATH |
| `nvcc` | 未找到 |
| 本地 CUDA Toolkit 目录 | 未找到 |
| CUDA/CUDNN 环境变量 | 未发现 |

`nvidia-smi` 中的 “CUDA Version 13.1” 表示驱动可支持的 CUDA 运行时上限，不证明本机已安装 CUDA Toolkit。当前未发现 `nvcc` 或 Toolkit 目录。

## 4. Python 依赖实测

以下版本来自当前 `D:\python3.11` 的包元数据与真实 import：

| 依赖 | 版本 | import 结果 |
|---|---:|---|
| TensorFlow | 2.17.1 (`tensorflow-intel` 2.17.1) | 成功 |
| Keras | 3.15.0 | 成功 |
| NumPy | 1.26.4 | 成功 |
| Pandas | 2.3.3 | 成功 |
| scikit-learn | 1.9.0 | 成功 |
| SciPy | 1.17.1 | 成功 |
| pyarrow | 25.0.0 | 成功 |
| matplotlib | 3.10.8 | 成功 |
| PyYAML | 6.0.3 | 成功 |
| requests | 2.32.5 | 成功，但出现依赖警告 |
| pytest | 未安装 | `ModuleNotFoundError` |
| FastAPI | 未安装 | `ModuleNotFoundError` |
| Uvicorn | 未安装 | `ModuleNotFoundError` |
| Pydantic | 未安装 | `ModuleNotFoundError` |

阶段 1 报告记录的是当时环境快照；本次实际检查发现 TensorFlow、Keras 和 pyarrow 已存在。阶段 1 报告不被覆盖，本报告作为 2026-07-14 的新快照记录环境变化。

## 5. TensorFlow 设备结果

TensorFlow 2.17.1 实际返回：

```text
built_with_cuda = false
built_with_gpu_support = false
physical_devices = [CPU:0]
gpu_devices = []
is_cuda_build = false
```

结论：当前 TensorFlow 是 Windows CPU-only 构建，只发现 CPU，没有使用 RTX 5060。NVIDIA GPU 和驱动存在不等于当前 TensorFlow 可用 GPU。

## 6. 原仓库基础 import 当前结果

未导入 `main.py`，因为它在模块顶层读取硬编码数据路径、创建缓存、构建并训练模型，导入会产生副作用。

| 模块 | 当前结果 |
|---|---|
| `framework` | 成功 |
| `framework.dataset_specification` | 成功 |
| `framework.flow_transformer_parameters` | 成功 |
| `implementations.pre_processings` | 成功 |
| `framework.flow_transformer` | 成功 |
| `implementations.input_encodings` | 成功 |
| `implementations.classification_heads` | 成功 |
| `implementations.transformers.basic_transformers` | 成功 |
| `implementations.transformers.named_transformers` | 成功 |
| `framework.sequential_input_encoding` | 失败：`ModuleNotFoundError: No module named 'framework_component'` |
| FlowTransformer_demo 的 `from framework import ...` | 失败：`ImportError: cannot import name 'NamedDatasetSpecifications' from 'framework'` |

这些结果只确认模块导入；没有构建 Keras 模型，Keras 3.15.0 下的 Functional API、CLS 分类头和 `jit_compile=True` 仍为“待验证”。

## 7. CPU 优先稳定复现环境

### 7.1 选择

候选环境采用 Python 3.11.9、TensorFlow 2.17.1、Keras 3.15.0 和 NumPy 1.26.4。选择依据是这些核心版本在当前机器已实际 import，且原仓库主要模块已能导入。其余数据与服务依赖使用固定版本，避免全局环境漂移。

重要限制：候选环境尚未在全新 Conda 环境执行完整安装、模型构建或训练，因此状态为**待验证**。如果 Keras 3 导致原模型构建失败，应先保留失败用例，再评估 TensorFlow/Keras 2.15 系列兼容基线，不能无证据地直接替换论文环境。

### 7.2 从零创建环境的 PowerShell 命令

```powershell
Set-Location D:\Users\LENOVO\Desktop\AI_safety_competition_code
conda env create -f environment\environment-cpu.yml
conda activate campus-flow-guard-cpu
python -m pip install -e . --no-deps
```

若同名环境已经存在，不要直接覆盖；先导出并核对已有环境，再决定是否更新或使用新环境名。

### 7.3 创建后的验证命令

```powershell
python --version
python -m pip --version
python -m pip check
python scripts\check_environment.py
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices())"
python -c "import keras, numpy, pandas, sklearn, pyarrow, matplotlib; print('imports ok')"
python -m pytest
```

原仓库导入验证必须显式提供只读路径，且不得导入 `main.py`：

```powershell
$env:FLOWTRANSFORMER_REPO = 'C:\Users\LENOVO\FlowTransformer'
python scripts\check_environment.py --original-repo $env:FLOWTRANSFORMER_REPO
```

## 8. 可选 GPU 方案

### 8.1 原生 Windows 现状

当前 TensorFlow 2.17.1 是 CPU-only；在原生 Windows 上不能仅通过现有 NVIDIA 驱动让该 wheel 自动获得 CUDA GPU。直接安装 CUDA Toolkit 也不会改变 CPU-only wheel 的构建属性。

### 8.2 建议路径

可选方案是 WSL2 + Ubuntu + NVIDIA 驱动透传，在 Linux 环境中选择明确支持 RTX 5060（Compute Capability 12.0）的 TensorFlow/CUDA/cuDNN 组合。由于本阶段没有查询并验证对应官方兼容矩阵，也没有在 WSL2 实际安装，以下均为**待验证**：

```powershell
wsl --status
wsl --list --verbose
# 仅在确认系统策略、磁盘空间和管理员权限后执行：
wsl --install -d Ubuntu-24.04
```

进入 WSL 后应先运行 `nvidia-smi`，再按 TensorFlow 官方 Linux GPU 安装说明创建独立环境。不得把 GPU 环境产生的指标与 CPU 基线混合报告；必须记录 TensorFlow、CUDA、cuDNN、驱动、GPU、XLA 和确定性设置。

RTX 5060 的 Compute Capability 12.0 较新，TensorFlow 2.17.1 是否包含适配代码尚未验证。若必须升级 TensorFlow/Keras 才支持该 GPU，应把它作为单独的 GPU 工程环境和兼容性实验，不替代 CPU 论文基线。

## 9. 本阶段实际诊断命令

```text
Get-ItemProperty HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion
[System.Runtime.InteropServices.RuntimeInformation]::OSDescription
python --version
python -m pip --version
conda --version
conda info --json
git --version
C:\Windows\System32\nvidia-smi.exe --query-gpu=...
C:\Windows\System32\nvidia-smi.exe
nvcc --version（未找到）
python -B 导入并记录依赖版本
python -B 导入原仓库基础模块
python -B -m pip check
```

本阶段未执行 `conda env create`、`pip install`、模型训练或数据下载。

## 10. 当前阻塞与待验证项

1. 当前全局 Python 存在 NumPy/OpenCV 依赖冲突和 requests 依赖警告。
2. pytest、FastAPI、Uvicorn、Pydantic 未安装；相关测试只能跳过或待新环境创建后执行。
3. 当前 TensorFlow 只能使用 CPU。
4. CPU 候选锁定环境尚未从零安装验证。
5. Keras 3 下模型构建、训练、最佳权重和 `jit_compile=True` 尚未验证。
6. WSL2 GPU 方案、RTX 5060 支持矩阵和 CUDA/cuDNN 组合尚未验证。

## 11. 新项目初始化与最小测试实录

### 11.1 初始化结果

- 已建立 `src` layout，包入口为 `src/campus_flow_guard/__init__.py`，当前版本 `0.1.0`。
- 已建立 README、`pyproject.toml`、`.gitignore`、环境文件、许可证目录、配置/数据/实验说明、脚本和单元测试目录。
- 已把基线与创新配置、实验目录分开；数据、日志、模型和抓包等生成文件已加入 `.gitignore`。
- 已原样复制 FlowTransformer 的 `LICENCE.txt` 到 `LICENSES/FlowTransformer-AGPL-3.0.txt`；两份文件 SHA-256 一致：`13FB48A6C31A72E0E2B69650B1CEBE8954F7B73449848492DCFD22B25BD91AFA`。
- 已初始化本地 Git 仓库，初始分支为 `main`；没有 commit，没有配置 remote，也没有推送。
- 阶段 1 报告 `docs/original_repository_audit_zh.md` 未覆盖，SHA-256 仍为 `37FBA2874C2D6D069B18C8AD067F9C28FE5C625C61A450EB2F4DCD2941FDDC0B`。

### 11.2 实际测试

1. 临时区静态校验：5 个 Python 文件通过 AST 解析；`pyproject.toml` 通过 `tomllib` 解析；`environment-cpu.yml` 通过 PyYAML 解析。
2. `python -B scripts/check_environment.py --original-repo C:\Users\LENOVO\FlowTransformer`：退出码 0；正确记录 TensorFlow CPU-only、缺失依赖和两个原仓库导入失败。
3. `python -B -m unittest discover -s tests -p 'test_*.py' -v`：共 5 项，3 项通过，2 项跳过。
4. 通过项：CampusFlowGuard 包导入、核心数据栈导入、TensorFlow/Keras 导入。
5. 跳过项：FastAPI/Uvicorn（未安装）、pytest import（未安装）。
6. 设置 `PYTHONPATH=src` 后导入 `campus_flow_guard`：成功，版本输出 `0.1.0`。

没有执行 `python -m pytest`，因为当前解释器未安装 pytest；这不是测试通过。CPU 环境文件会安装 pytest，必须在新环境创建后补跑。

### 11.3 中断恢复后的完整性检查

恢复任务时发现原仓库有 4 个导入检查生成的未跟踪 `__pycache__` 目录。确认其中仅含 `.pyc` 后已删除，原仓库最终恢复为 `master...origin/master` 且无工作区差异。原仓库内已有、被 Git 忽略的 `.venv` 未修改。
