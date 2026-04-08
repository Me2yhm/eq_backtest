import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def calculate_portfolio_metrics(daily_returns, ann_days = 242, risk_free_rate=0.0):
    """
    Calculate annualized return, volatility, Sharpe ratio, worst draw-down, and Calmar ratio
    from a daily return series.

    Parameters:
    - daily_returns: pandas Series of daily returns
    - risk_free_rate: float, annualized risk-free rate (default 0.0)

    Returns:
    - dict with the calculated metrics
    """
    # Annualized Return (geometric mean)
    cum_return = (1 + daily_returns).prod() - 1
    years = len(daily_returns) / ann_days
    annualized_return = (1 + cum_return) ** (1 / years) - 1

    # Annualized Volatility
    volatility = daily_returns.std() * np.sqrt(ann_days)

    # Sharpe Ratio
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility != 0 else np.nan

    # Worst Draw-Down
    cum_returns = (1 + daily_returns).cumprod()
    peak = cum_returns.expanding().max()
    drawdown = (cum_returns - peak) / peak
    worst_drawdown = drawdown.min()

    # Calmar Ratio
    calmar_ratio = annualized_return / abs(worst_drawdown) if worst_drawdown != 0 else np.nan

    return {
        'Annualized Return': annualized_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Worst Draw-Down': worst_drawdown,
        'Calmar Ratio': calmar_ratio
    }


def plot_holding_periods(holding_periods_avg, holding_periods_max, holding_periods_median, closed_holding_periods_avg, port_num, is_short=False, output_path='output/', show_max=False, show_median=True, show_closed=True):
    """
    Plot average, median, and optionally max holding days of all stocks in the portfolio across time.
    Also plots average holding days of closed positions.
    
    Parameters:
    - holding_periods_avg: pandas Series with date index and average holding days as values (open positions)
    - holding_periods_max: pandas Series with date index and max holding days as values
    - holding_periods_median: pandas Series with date index and median holding days as values
    - closed_holding_periods_avg: pandas Series with date index and average holding days of closed positions
    - port_num: portfolio number for title
    - is_short: if True, save to 'plots_short', else 'plots_long' (default: False)
    - show_max: if True, plot max holding days line (default: False)
    - show_median: if True, plot median holding days line (default: True)
    - show_closed: if True, plot closed positions holding days line (default: True)
    """
    if holding_periods_avg is None or len(holding_periods_avg) == 0:
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(holding_periods_avg.index, holding_periods_avg.values, linewidth=1.5, alpha=0.8, label='Avg (Open)', color='blue')
    if show_median:
        plt.plot(holding_periods_median.index, holding_periods_median.values, linewidth=1.5, alpha=0.8, label='Median (Open)', color='green')
    if show_max:
        plt.plot(holding_periods_max.index, holding_periods_max.values, linewidth=1.5, alpha=0.8, label='Max', color='orange')
    if show_closed and closed_holding_periods_avg is not None and len(closed_holding_periods_avg) > 0:
        plt.plot(closed_holding_periods_avg.index, closed_holding_periods_avg.values, linewidth=1, alpha=0.6, label='Avg (Closed)', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Holding Days')
    plt.title(f'Holding Days Over Time - Portfolio {port_num}')
    plt.grid(True, alpha=0.3)
    
    # Add overall mean line for average
    overall_mean = holding_periods_avg.mean()
    plt.axhline(overall_mean, color='blue', linestyle='--', 
               label=f'Avg Mean: {overall_mean:.1f} days', alpha=0.5)
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save figure
    plot_subdir = 'plots_short' if is_short else 'plots_long'
    plot_dir = os.path.join(output_path, plot_subdir)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'avg_holding_days_p{port_num}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def calculate_and_plot_holding_periods(port, port_num, is_short=False, output_path='output/'):
    """
    Calculate average and max holding days of current positions (days since last entry) over time.
    Also tracks average holding days of positions that closed each day.
    
    Parameters:
    - port: portfolio DataFrame with MultiIndex (date, symbol)
    - port_num: portfolio number for plotting
    - is_short: if True, save to 'plots_short', else 'plots_long' (default: False)
    
    Returns:
    - tuple of (avg_series, max_series, median_series, closed_avg_series) pandas Series of holding periods indexed by date
    """
    holding_periods_avg = {}
    holding_periods_max = {}
    holding_periods_median = {}
    closed_holding_periods_avg = {}
    symbol_entry_idx = {}  # Track entry date INDEX (not date itself)
    prev_symbols = set()
    
    sorted_dates = sorted(port.index.get_level_values('date').unique())
    
    for date_idx, date in enumerate(sorted_dates):
        daily_symbols = set(port.loc[date].index)
        
        # Identify new and exited symbols
        new_symbols = daily_symbols - prev_symbols
        exited_symbols = prev_symbols - daily_symbols
        
        # Calculate holding days for exited positions
        closed_holding_days = []
        for symbol in exited_symbols:
            if symbol in symbol_entry_idx:
                holding_days = date_idx - symbol_entry_idx[symbol]
                closed_holding_days.append(holding_days)
        
        # Store average holding days for closed positions
        if closed_holding_days:
            closed_holding_periods_avg[date] = np.mean(closed_holding_days)
        
        # Remove exited symbols from tracking
        for symbol in exited_symbols:
            symbol_entry_idx.pop(symbol, None)
        
        # Add new symbols with today's index as entry
        for symbol in new_symbols:
            symbol_entry_idx[symbol] = date_idx
        
        # Calculate holding days for all current positions
        daily_holding_days = []
        for symbol in daily_symbols:
            if symbol in symbol_entry_idx:
                holding_days = date_idx - symbol_entry_idx[symbol] + 1
                daily_holding_days.append(holding_days)
        
        # Calculate average, median, and max holding days for this date
        if daily_holding_days:
            holding_periods_avg[date] = np.mean(daily_holding_days)
            holding_periods_median[date] = np.median(daily_holding_days)
            holding_periods_max[date] = np.max(daily_holding_days)
        
        prev_symbols = daily_symbols
    
    # Convert to pandas Series
    holding_periods_avg_series = pd.Series(holding_periods_avg)
    holding_periods_median_series = pd.Series(holding_periods_median)
    holding_periods_max_series = pd.Series(holding_periods_max)
    closed_holding_periods_avg_series = pd.Series(closed_holding_periods_avg)
    
    # Print statistics
    if len(holding_periods_avg_series) > 0:
        print(f"\nHolding Period Statistics for Portfolio {port_num}:")
        print(f"  Total trading days: {len(sorted_dates)}")
        print(f"  Open positions - Average holding days: {holding_periods_avg_series.mean():.1f} days")
        print(f"  Open positions - Median holding days: {holding_periods_avg_series.median():.1f} days")
        print(f"  Open positions - Min daily avg: {holding_periods_avg_series.min():.1f} days")
        print(f"  Open positions - Max daily avg: {holding_periods_avg_series.max():.1f} days")
        print(f"  Open positions - Max holding (max): {holding_periods_max_series.max():.0f} days")
        
        if len(closed_holding_periods_avg_series) > 0:
            print(f"  Closed positions - Average holding days: {closed_holding_periods_avg_series.mean():.1f} days")
            print(f"  Closed positions - Median holding days: {closed_holding_periods_avg_series.median():.1f} days")
        
        plot_holding_periods(holding_periods_avg_series, holding_periods_max_series, holding_periods_median_series, closed_holding_periods_avg_series, port_num, is_short=is_short, output_path=output_path)
    
    return holding_periods_avg_series, holding_periods_max_series, holding_periods_median_series, closed_holding_periods_avg_series


def plot_portfolio_results(df_cumret, df_port_size, df_close_counts_combined, is_short=False, output_path='output/'):
    """
    Plot portfolio cumulative returns, drawdowns, portfolio size, and close counts.
    
    Parameters:
    - df_cumret: DataFrame with cumulative returns for each portfolio
    - df_port_size: DataFrame with portfolio size over time
    - df_close_counts_combined: DataFrame with daily close counts
    - is_short: if True, save to 'plots_short', else 'plots_long' (default: False)
    """
    # Create subplot: cumulative returns and portfolio size/closes
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Plot cumulative returns
    lines = df_cumret.plot(ax=ax1, grid=True, title='Portfolio Excess Return', linewidth = 2)
    ax1.set_ylabel('Cumulative Return')
    
    # Create secondary y-axis for drawdowns
    ax1_dd = ax1.twinx()
    
    # Calculate and plot drawdowns for each portfolio with matching colors
    for i, col in enumerate(df_cumret.columns):
        cum_returns = 1 + df_cumret[col]
        high_water_mark = cum_returns.expanding().max()
        drawdown = cum_returns - high_water_mark
        # Get the color from the corresponding line
        color = ax1.get_lines()[i].get_color()
        ax1_dd.plot(drawdown.index, drawdown, color=color, label=f'{col}_dd', linewidth=1, alpha = 0.6)
    
    ax1_dd.set_ylabel('Drawdown')
    ax1.legend(loc='upper left')
    ax1_dd.legend(loc='lower left')

    # Plot portfolio size and close counts on same axis with matching colors
    colors = plt.cm.tab10(range(len(df_port_size.columns)))
    
    for i, col in enumerate(df_port_size.columns):
        df_port_size[col].plot(ax=ax2, color=colors[i], label=col, alpha=1.0, linewidth=2)
        if not df_close_counts_combined.empty and col in df_close_counts_combined.columns:
            df_close_counts_combined[col].plot(ax=ax2, color=colors[i], label=f'{col}_closes', 
                                               alpha=0.5, linestyle='--')
    
    ax2.set_ylabel('Number of Stocks / Closes')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    ax2.set_title('Portfolio Size and Daily Closes')

    plt.tight_layout()
    
    # Save figure
    plot_subdir = 'plots_short' if is_short else 'plots_long'
    plot_dir = os.path.join(output_path, plot_subdir)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'portfolio_results.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_position_heatmap(pos_df, size_rank_bins = 20, port_num = None, is_short=False, output_path='output/'):
    """
    Plot heatmap showing distribution of positions across size_rank over time.
    
    Parameters:
    - pos_df: DataFrame with MultiIndex (date, symbol) and 'size_rank' column
    - size_rank_bins: number of bins to group size_ranks (default: 50)
    - port_num: portfolio number for filename suffix (default: None)
    - is_short: if True, save to 'plots_short', else 'plots_long' (default: False)
    """
    if pos_df.empty:
        return
    
    # Create a copy and reset index to work with date and size_rank
    df = pos_df.reset_index()
    
    # Check if the sum of positions is the same for all dates
    position_counts = df.groupby('date').size()
    if position_counts.nunique() > 1:
        print("WARNING: Portfolio size varies across dates:")
        print(position_counts.describe())
        print(f"Min: {position_counts.min()} | Max: {position_counts.max()} | Mean: {position_counts.mean():.1f}")
    
    # Create size_rank bins
    max_rank = df['size_rank'].max()
    bins = np.linspace(0, max_rank, size_rank_bins + 1)
    df['size_rank_bin'] = pd.cut(df['size_rank'], bins=bins, labels=False, include_lowest=True)
    
    # Count positions in each bin for each date
    heatmap_data = df.groupby(['date', 'size_rank_bin']).size().unstack(fill_value=0)
    
    # Create heatmap
    plt.figure(figsize=(14, 8))
    im = plt.imshow(heatmap_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
    
    # Set axis labels
    plt.ylabel('Size Rank Bin')
    plt.xlabel('Date')
    plt.title('Portfolio Position Distribution Across Size Ranks Over Time')
    
    # Set x-axis ticks to show dates at regular intervals
    date_labels = heatmap_data.index
    num_ticks = min(20, len(date_labels))  # Show max 20 date labels
    tick_indices = np.linspace(0, len(date_labels) - 1, num_ticks, dtype=int)
    plt.xticks(tick_indices, [date_labels[i].strftime('%Y-%m-%d') for i in tick_indices], rotation=45, ha='right')
    
    # Add vertical grid lines at year boundaries
    year_boundaries = []
    current_year = date_labels[0].year
    for i, date in enumerate(date_labels):
        if date.year != current_year:
            year_boundaries.append(i - 0.5)  # Place line between dates
            current_year = date.year
    
    for boundary in year_boundaries:
        plt.axvline(x=boundary, color='white', linewidth=1.5, linestyle='--', alpha=0.7)
    
    # Set y-axis ticks to show size rank ranges
    bin_labels = [f'{int(bins[i])}-{int(bins[i+1])}' for i in range(size_rank_bins)]
    plt.yticks(range(size_rank_bins), bin_labels)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Number of Positions')
    
    plt.tight_layout()
    
    # Save figure
    plot_subdir = 'plots_short' if is_short else 'plots_long'
    plot_dir = os.path.join(output_path, plot_subdir)
    os.makedirs(plot_dir, exist_ok=True)
    if port_num is not None:
        filename = os.path.join(plot_dir, f'position_heatmap_p{port_num}.png')
    else:
        filename = os.path.join(plot_dir, 'position_heatmap.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def save_and_plot_final_results(metrics, df_ret, df_cumret, df_port_size, df_close_counts, 
                               port_nums, pool_num, bm, output_path, is_short):
    """
    Save and plot final portfolio results.
    
    Parameters:
    - metrics: dict of portfolio metrics
    - df_ret: list of excess return series
    - df_cumret: list of cumulative return series
    - df_port_size: list of portfolio size series
    - df_close_counts: list of close count dataframes
    - port_nums: list of portfolio numbers
    - pool_num: pool size
    - bm: benchmark name
    - output_path: output directory path
    - is_short: whether short strategy
    """
    # Create metrics dataframe and save to CSV
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(output_path + 'portfolio_metrics_p' + str(pool_num) + '_' + str(bm) + '.csv')

    # Save metrics as a figure
    fig, ax = plt.subplots(figsize=(12, len(port_nums) * 0.8 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    # Format metrics for display
    metrics_display = metrics_df.copy()
    for col in metrics_display.columns:
        if col == 'turnover':
            metrics_display[col] = metrics_display[col].apply(lambda x: f'{x:.2f}')
        else:
            metrics_display[col] = metrics_display[col].apply(lambda x: f'{x:.4f}')
    
    table = ax.table(cellText=metrics_display.values,
                    rowLabels=metrics_display.index,
                    colLabels=metrics_display.columns,
                    cellLoc='center',
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style the header
    for i in range(len(metrics_display.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style the row labels
    for i in range(len(metrics_display)):
        table[(i+1, -1)].set_facecolor('#40466e')
        table[(i+1, -1)].set_text_props(weight='bold', color='white')
    
    strategy_type = 'Short' if is_short else 'Long'
    plt.title(f'Portfolio Metrics ({strategy_type} - Pool: {pool_num}, Benchmark: {bm})', 
             fontsize=12, weight='bold', pad=20)
    
    plot_subdir = 'plots_short' if is_short else 'plots_long'
    plot_dir = os.path.join(output_path, plot_subdir)
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'portfolio_metrics_p{pool_num}_{bm}.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

    print('\nPortfolio Metrics:')
    print(metrics_df)

    # Save returns and cumulative returns
    df_ret_combined = pd.concat(df_ret, axis=1)
    df_ret_combined.to_csv(output_path + 'port_ret_p' + str(pool_num) + '_' + str(bm) + '.csv')

    df_cumret_combined = pd.concat(df_cumret, axis=1)
    df_cumret_combined.to_csv(output_path + 'port_cumret_p' + str(pool_num) + '_' + str(bm) + '.csv')
    
    # Concatenate portfolio sizes and close counts
    df_port_size_combined = pd.concat(df_port_size, axis=1)
    df_close_counts_combined = pd.concat(df_close_counts, axis=1) if df_close_counts else pd.DataFrame()
    
    # Plot portfolio results
    plot_portfolio_results(df_cumret_combined, df_port_size_combined, df_close_counts_combined, is_short=is_short, output_path=output_path)
