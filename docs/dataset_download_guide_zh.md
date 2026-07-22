# NF-UNSW-NB15-v2 数据获取与复核指引

本指引不执行下载，也不保证外部链接当前可用。它仅记录原 FlowTransformer `demonstration.ipynb` 中已经出现的来源线索，供需要重新获取或核验本地文件时人工操作。

## 原仓库记录的来源

- 数据集页面：`https://staff.itee.uq.edu.au/marius/NIDS_datasets/`
- Notebook 中的下载接口：`https://api.rdm.uq.edu.au/production/files/8c6e2a00-ef9c-11ed-827d-e762de186848/download`
- Notebook 中预期的压缩包内部路径：`fe6cb615d161452c_MOHANAD_A4706/data/NF-UNSW-NB15-v2.csv`

以上链接来自原仓库代码，未在本阶段联网验证；不得将其当作已确认有效的官方 URL。数据集许可、使用限制和版本说明必须以实际访问到的发布页面或压缩包附带材料为准。

## 人工操作

1. 在浏览器访问上列数据集页面，确认发布方、许可证、版本、适用用途和下载文件清单。
2. 下载后保留原始压缩包、页面截图或许可文本及下载日期至外部数据根目录的 `metadata/`，不得提交 Git。
3. 仅解压目标 CSV 至 `D:\Users\LENOVO\Desktop\AI_safety_competition_data\raw\UNSW-NB15\`；不要覆盖已有 `raw\NF-UNSW-NB15-v2.csv`，除非先保留原版本和元数据。
4. 在 PowerShell 计算哈希并记录实际文件名和大小：

```powershell
Get-FileHash -Algorithm SHA256 `
  D:\Users\LENOVO\Desktop\AI_safety_competition_data\raw\UNSW-NB15\NF-UNSW-NB15-v2.csv
```

5. 使用项目脚本做全量质量检查；只有文件存在时才会生成真实结果：

```powershell
Set-Location D:\Users\LENOVO\Desktop\AI_safety_competition_code
python scripts\inspect_dataset.py `
  D:\Users\LENOVO\Desktop\AI_safety_competition_data\raw\UNSW-NB15\NF-UNSW-NB15-v2.csv `
  --schema configs\datasets\dataset_schema.yaml `
  --output artifacts\metrics\dataset_profile.json
```

当前根目录下已发现的文件位于 `raw\NF-UNSW-NB15-v2.csv`，其本机 SHA-256 见 [数据质量报告](dataset_quality_report_zh.md)。该哈希只标识当前本地副本，不是经上游发布方确认的期望哈希。
