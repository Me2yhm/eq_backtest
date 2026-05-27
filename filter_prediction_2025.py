import pandas as pd
from pathlib import Path

# 确保目标目录存在
output_dir = Path("/home/danny/backtest_verify/data")
output_dir.mkdir(parents=True, exist_ok=True)

# 读取 parquet 文件
df = pd.read_parquet("/ext/trq/predictions.parquet")

# 确保 trade_date 列是 datetime 类型
df["trade_date"] = pd.to_datetime(df["trade_date"])

# 筛选年份 >= 2026
filtered_df = df[df["trade_date"].dt.year >= 2025]

# 保存结果
filtered_df.to_parquet(output_dir / "predictions_2025.parquet")

print(f"筛选完成，共保留 {len(filtered_df)} 行，已保存至 {output_dir / 'predictions_2025.parquet'}")