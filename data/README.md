# 本地数据目录

数据文件不进入 Git。真实 UNSW-NB15 相关源文件保存在外部数据根目录 `D:\Users\LENOVO\Desktop\AI_safety_competition_data` 中；项目内此目录仅保留说明和可追溯元数据。

```text
AI_safety_competition_data/
  raw/          # 原始下载文件，不修改
    UNSW-NB15/  # 若获得原始 UNSW-NB15 发行包，放入此处
    NF-UNSW-NB15-v2.csv  # 当前已发现的实际 CSV，仍待确认官方来源
  interim/      # 仅由可追溯脚本生成的中间结果
  processed/    # 经审查的训练输入，不覆盖 raw
  metadata/     # 来源、版本、哈希与检查记录
```

每个数据版本应记录来源、许可证、下载时间、SHA-256、行数、特征列表和处理脚本版本。本阶段不训练、不窗口划分、不对原始文件进行预处理。
