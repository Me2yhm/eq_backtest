import pandas as pd

# 设置显示选项（避免列截断）
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# 读取文件
file_path = r"D:\develop\backtest_verify\eq-backtest\data_15min\predictions_2025_12.parquet"
df_full = pd.read_parquet(file_path)

# 确保 trade_date 列为 datetime 类型
df_full["trade_date"] = pd.to_datetime(df_full["trade_date"])

# 筛选出日期等于 2025-12-01 的行（忽略时间部分）
df = df_full[df_full["trade_date"].dt.date == pd.Timestamp("2025-12-01").date()]

# 打印结果
print(f"2025-12-01 共有 {len(df)} 条记录：\n")
print(df)

# 如果需要查看前几行，可以用 df.head()