import os
import pandas as pd
import numpy as np
from pathlib import Path
# import matplotlib.pyplot as plt

from utils import calculate_portfolio_metrics, calculate_and_plot_holding_periods, plot_position_heatmap, save_and_plot_final_results


START = '2020-01-01'

DATA = 'data/daily.pqt'
PREDS = 'data/preds_new/'
HORIZON = ['3d', '5d', '10d']
UNIVERSE = ['000300.XSHG', '000905.XSHG', '000852.XSHG']
BM_PATH = 'data/bm_open/ret_csi_1000.csv'
BM = 'csi_1000'

IS_SHORT = True
ST_OPEN = True

if IS_SHORT:
    POOL_NUM = 9999
    PORT_NUMS = [200, 300, 400]
else:
    POOL_NUM = 3800
    PORT_NUMS = [400, 450, 500, 550, 600, 650]

OUTPUT_PATH = 'output_short_susp_ST_' + str(POOL_NUM) + '/'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# SIMPLE = False
COST = 0.0004

# POOL_NUM = 1800
# PORT_NUMS = [400, 600, 800]

EXCLUDE_PERIOD = ('2024-1-1', '2024-12-31')


def gen_port(pool, thresh_in, thresh_out, size_cut = POOL_NUM, close_on_size_drop = True, plot_heatmap = True, output_path = 'output/'):
    """
    Generate daily portfolio based on prediction ranking.
    
    Logic:
    1. Portfolio size is fixed at thresh_in
    2. Each day, first close positions (if tradable), then open new ones (if tradable)
    3. Close: positions with pred_rank > thresh_out
       - Optionally also close if size_rank >= size_cut (controlled by close_on_size_drop)
       - But can only close if tradable == 1
    4. Open: fill remaining slots from top-ranked candidates that are tradable
    5. Non-tradable positions are forced to hold (cannot close even if we want to)
       - These positions are marked for forced close and will be closed at first opportunity
    
    Parameters:
    - close_on_size_drop: If False (default), don't close positions just because they dropped
                         out of size_cut, as long as pred_rank <= thresh_out
    - plot_heatmap: If True (default), plot position heatmap across size ranks and dates
    
    Note: Expects pool to have pre-calculated 'pred_rank' column
    """
    pos = None
    pos_df = []
    close_counts = []  # Track number of closes each day

    for date in pool.index.get_level_values(0).unique():
        daily_pool = pool.loc[date].copy()
        
        # Calculate pred_rank for stocks within size_cut AND current positions
        # This allows current holdings to be ranked even if they fell out of size pool
        if pos is None:
            # First day: rank all stocks within size pool
            rank_universe = (daily_pool['size_rank'] < size_cut) & (daily_pool['tradable'])
        else:
            # Subsequent days: rank stocks in size pool OR current positions
            rank_universe = (
                ((daily_pool['size_rank'] < size_cut) | (daily_pool.index.isin(pos.index))) &
                (daily_pool['tradable'])
            )
        
        daily_pool['pred_rank'] = np.nan
        daily_pool.loc[rank_universe, 'pred_rank'] = daily_pool.loc[rank_universe, 'pred'].rank(
            ascending = IS_SHORT, method='first'
        )

        if pos is None:
            # First day: open positions from top-ranked stocks
            # (already filtered for tradable and can_open)
            pos = daily_pool.loc[(daily_pool['pred_rank'] <= thresh_in) & (daily_pool['can_open'])].copy()
            to_close = None
            to_open = pos
            num_closes = 0
        else:
            existing_symbols = pos.index

            # Update existing positions with today's data
            available_symbols = existing_symbols.intersection(daily_pool.index)
            if len(available_symbols) > 0:
                pos.loc[available_symbols, :] = daily_pool.loc[available_symbols, :]

            # Handle missing symbols (delisted, suspended, etc.)
            # These cannot be traded (neither opened nor closed), so force hold
            missing_symbols = existing_symbols.difference(daily_pool.index)
            if len(missing_symbols) > 0:
                pos.loc[missing_symbols, 'ret'] = 0  # Suspended stocks earn 0 return
                pos.loc[missing_symbols, 'pred_rank'] = np.nan
                pos.loc[missing_symbols, 'size_rank'] = np.nan  # Ensure size_rank is set
                pos.loc[missing_symbols, 'tradable'] = False  # Cannot trade missing stocks

            # === STEP 1: CLOSE positions ===
            # Can ONLY close if tradable == 1
            # Conditions to close:
            # 1. pred_rank > thresh_out (poor ranking) - always close
            # 2. pred_rank is NaN (dropped out of eligible pool) - always close
            # 3. size_rank >= size_cut (dropped out of eligible pool) - only if close_on_size_drop=True
            # Note: NaN > thresh_out is False in pandas, so we need explicit isna() check
            if close_on_size_drop:
                want_to_close_mask = (
                    (pos['pred_rank'] > thresh_out) | 
                    (pos['pred_rank'].isna()) | 
                    (pos['size_rank'] >= size_cut)
                )
            else:
                want_to_close_mask = (
                    (pos['pred_rank'] > thresh_out) |
                    (pos['pred_rank'].isna())
                )
            
            can_close_mask = pos['tradable']
            
            # Actually close what we can
            to_close = pos.loc[want_to_close_mask & can_close_mask]
            num_closes = len(to_close)
            pos = pos.drop(index=to_close.index, errors='ignore')

            # === STEP 2: OPEN positions ===
            # Calculate available slots
            slots = max(thresh_in - len(pos), 0)
            
            if slots > 0:
                # Candidates: top-ranked, not already in position
                # (tradability already checked in eligible_mask when computing pred_rank)
                to_open_candidates = daily_pool.loc[
                    (daily_pool['pred_rank'].notna()) &  # Must be eligible and tradable
                    (~daily_pool.index.isin(pos.index)) & # Not already held
                    (daily_pool['can_open'])  # Within size cut
                ].sort_values('pred_rank')
                
                to_open = to_open_candidates.head(slots)
                pos = pd.concat([pos, to_open])
            else:
                to_open = daily_pool.iloc[0:0]  # Empty dataframe

        # Record position snapshot for this day
        pos_snapshot = pos.copy()
        pos_snapshot.insert(0, 'date', pd.Timestamp(date))
        pos_snapshot = pos_snapshot.reset_index().set_index(['date', 'symbol']).sort_index()
        pos_df.append(pos_snapshot)
        
        # Record close count
        close_counts.append({'date': pd.Timestamp(date), 'num_closes': num_closes})

        # Debug output (uncomment if needed)
        # print()
        # print(date, len(pos))
        # print('To Close:')
        # print(to_close)
        # print('To Open:')
        # print(to_open)
        # print('Pos:')
        # print(pos)
        # input()

    pos_df = pd.concat(pos_df).sort_index(level=[0, 1])
    close_counts_df = pd.DataFrame(close_counts).set_index('date')

    # Plot position heatmap
    if plot_heatmap:
        plot_position_heatmap(pos_df, port_num=thresh_in, is_short=IS_SHORT, output_path=output_path)

    return pos_df, close_counts_df

def prep_data(st_open = False):
    """
    Load and prepare data for backtesting.
    
    Returns:
    - pool: DataFrame with 2-level index (date, symbol) containing data and predictions
    """

    print('Preds:', PREDS)
    print('Benchmark:', BM)

    bm_ret = pd.read_csv(BM_PATH, index_col=0, parse_dates=True).squeeze()
    
    # print(bm_ret)
    # input()

    print('\nLoading Predictions ...')
    # Load parquet files from PREDS directory, filtered by horizon suffix after last '_'
    preds_dir = Path(PREDS)
    parquet_files = list(preds_dir.glob('*.parquet'))

    if not parquet_files:
        raise ValueError(f"No parquet files found in {PREDS}")

    def _get_horizon_suffix(path: Path) -> str:
        stem = path.stem
        if '_' not in stem:
            return ''
        return stem.rsplit('_', 1)[-1]

    horizon_set = set(HORIZON) if HORIZON else set()
    if horizon_set:
        parquet_files = [f for f in parquet_files if _get_horizon_suffix(f) in horizon_set]

    if not parquet_files:
        raise ValueError(f"No parquet files found in {PREDS} matching horizons: {HORIZON}")

    print(f'Found {len(parquet_files)} parquet file(s) after horizon filter:')
    for f in parquet_files:
        print(f'  - {f.name}')
    
    # Load and average all predictions
    all_preds = []
    for pf in parquet_files:
        pred_df = pd.read_parquet(pf).truncate(before=START)
        pred_df.index.name = 'date'
        pred_stacked = pred_df.stack()
        pred_stacked.index.set_names([pred_stacked.index.names[0], 'symbol'], inplace=True)
        all_preds.append(pred_stacked)
    
    # Average predictions across all files
    preds = pd.concat(all_preds, axis=1).mean(axis=1)
    preds.name = 'pred'

    print('\nPredictions:')
    print(preds)
    # input()

    # preds.head(10000).to_csv(OUTPUT_PATH + 'preds_first_10k.csv')
    # print(preds.tail())
    # input()

    print('\nLoading Data ...')
    data = pd.read_parquet(DATA)
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index(['date', 'symbol'], drop = True).sort_index().truncate(before=START)

    data['tradable'] = (data['turnover'] > 0) & (data['is_limit_up'] == False) & (data['is_limit_down'] == False)

    if st_open:
        data['can_open'] = data['tradable'] & (data['normal_days'] >= 10)
    else:
        data['can_open'] = data['tradable'] & (data['normal_days'] >= 10) & (data['is_ST'] == False)
    
    # Filter by UNIVERSE if specified
    if UNIVERSE is not None:
        data['can_open'] = data['can_open'] & data['index'].isin(UNIVERSE)

    print('\nData:')
    print(data)
    # print(data.columns)
    # input()

    print('\nGenerating Pool ...')
    pool = data.join(preds, how='left').sort_index()

    print('\nPool:')
    print(pool)
    input('\nPress Enter to Continue...')
    
    return pool, bm_ret


if __name__ == '__main__':

    pool, bm_ret = prep_data(st_open = ST_OPEN)

    df_cumret = []
    df_ret = []
    df_port_size = []
    df_close_counts = []
    metrics = {}

    for port_num in PORT_NUMS:
        print('\n  - Generating Portfolio with num:', port_num)

        # if SIMPLE:
        #     port = pool.loc[pool['pred_rank'] <= port_num]
        #     close_counts = pd.DataFrame()  # No close tracking in SIMPLE mode
        # else:
        port, close_counts = gen_port(pool, port_num, port_num + 200, size_cut = POOL_NUM, close_on_size_drop = True, output_path=OUTPUT_PATH)

        # print(port)
        port.to_csv(OUTPUT_PATH + 'port_' + str(port_num) + '.csv')
        # input()

        port_ret = port.groupby(['date'])['ret'].sum() / port_num

        wt = pd.Series(1, index = port.index, name = 'weight') / port_num
        wt = wt.reset_index().pivot(index = 'date', columns = 'symbol').fillna(0)
        
        # print(wt)
        # wt.to_csv(OUTPUT_PATH + 'port_weights.csv')
        # input()

        trades = wt.diff(axis = 0).abs()
        trades.iloc[0] = wt.iloc[0].abs()
        turnover = trades.sum(axis = 1)
        # port_ret.to_csv(OUTPUT_PATH + 'port_ret.csv')
        
        # print(port_ret)

        if IS_SHORT:
            excess_ret = bm_ret - port_ret - turnover * COST
        else:
            excess_ret = port_ret - bm_ret - turnover * COST
        
        # Exclude specified period
        if EXCLUDE_PERIOD:
            exclude_start = pd.to_datetime(EXCLUDE_PERIOD[0])
            exclude_end = pd.to_datetime(EXCLUDE_PERIOD[1])
            excess_ret.loc[(excess_ret.index >= exclude_start) & (excess_ret.index < exclude_end)] = 0
        
        excess_ret.name = f'Excess_Ret_{port_num}'
        df_ret.append(excess_ret)
        # excess_ret.to_csv(OUTPUT_PATH + 'excess_ret_' + str(port_num) + '.csv')

        # print(excess_ret.loc[excess_ret.isna()])
        # input()

        cumret = excess_ret.cumsum()
        cumret.name = f'Port_{port_num}'
        df_cumret.append(cumret)

        # Calculate daily portfolio size
        port_size = port.groupby('date').size()
        port_size.name = f'Port_{port_num}'
        df_port_size.append(port_size)
        
        # Store close counts
        if not close_counts.empty:
            close_counts.columns = [f'Port_{port_num}']
            df_close_counts.append(close_counts)

        metrics[port_num] = calculate_portfolio_metrics(excess_ret)
        metrics[port_num]['turnover'] = turnover.mean() * 242
        
        # Calculate and plot average holding days per date
        calculate_and_plot_holding_periods(port, port_num, is_short=IS_SHORT, output_path=OUTPUT_PATH)

    # Save and plot final results
    save_and_plot_final_results(metrics, df_ret, df_cumret, df_port_size, df_close_counts,
                               PORT_NUMS, POOL_NUM, BM, OUTPUT_PATH, IS_SHORT)
