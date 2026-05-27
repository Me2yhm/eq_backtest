import pandas as pd
from pathlib import Path

# 目标目录和文件
output_dir = Path("/home/danny/backtest_verify/data")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "predictions_2025_12.parquet"

# 读取原始数据
df = pd.read_parquet("/ext/trq/predictions.parquet")

# 确保 trade_date 列是 datetime 类型
df["trade_date"] = pd.to_datetime(df["trade_date"])

# 筛选：年份=2025 且 月份=12
mask = (df["trade_date"].dt.year == 2025) & (df["trade_date"].dt.month == 12)
filtered_df = df[mask]

# 保存
filtered_df.to_parquet(output_file)

print(f"筛选完成，共保留 {len(filtered_df)} 行，保存至 {output_file}")