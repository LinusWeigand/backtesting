import pandas as pd
import numpy as np
import random

def read_csv(filename):
    column_names = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

    df = pd.read_csv(
        filename,
        skiprows=3,
        header=None,
        names=column_names,
        index_col='Date',
        parse_dates=True,
    )

    return df

def clean_data(df):
    # Add missing Dates
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
    df = df.reindex(date_range)

    # Remove NaNs
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    return df

if __name__ == "__main__":
    tesla = read_csv('tesla_practice.csv')
    tesla = clean_data(tesla)

    df = pd.DataFrame({'close': tesla['Close'], 'open': tesla['Open']})

    df['short_ma'] = df['close'].ewm(span=50, adjust=False).mean()
    df['long_ma'] = df['close'].rolling(200).mean()

    conditions = [
        df['short_ma'] > df['long_ma'],
        df['short_ma'] < df['long_ma'],
        ]
    signal_choices = [1, -1]
    df['signal'] = np.select(conditions, signal_choices, default=0)

    # We dont care about the first entry being NaN because of warmup period
    df['lagged_signal'] = df['signal'].shift(1)

    df['open_to_next_open_return'] = df['open'].pct_change().shift(-1)

    strategy_conditions = [
        df['lagged_signal'] == 1,
        df['lagged_signal'] == -1
    ]

    strategy_return_choices = [
        df['open_to_next_open_return'],
        (1 / (1 + df['open_to_next_open_return'])) - 1
    ]
    df['strategy_return'] = np.select(strategy_conditions, strategy_return_choices, default=0)

    # Remove Warmup Period
    df = df.iloc[200:].copy()

    # last row of open_to_next_open_return and strategy_return are NaN
    if pd.isna(df['open_to_next_open_return'].iloc[-1]):
        df.loc[df.index[-1], 'open_to_next_open_return'] = 0

    if pd.isna(df['strategy_return'].iloc[-1]):
        df.loc[df.index[-1], 'strategy_return'] = 0

    trade_days = df[df['lagged_signal'].diff().ne(0)].copy()
    first_day = df.iloc[:1]
    if first_day.index[0] not in trade_days.index:
        trade_days = pd.concat([first_day, trade_days]).sort_index()

    trades_df = pd.DataFrame({
        'position': trade_days['lagged_signal'],
        'entry_date': trade_days.index,
        'entry_price': trade_days['open']
    })

    trades_df['exit_date'] = trades_df['entry_date'].shift(-1)
    trades_df['exit_price'] = trades_df['entry_price'].shift(-1)

    if not trades_df.empty:
        trades_df.loc[trades_df.index[-1], 'exit_date'] = df.index[-1]
        trades_df.loc[trades_df.index[-1], 'exit_price'] = df['open'].iloc[-1]

    trades_df = trades_df[trades_df['position'] != 0].copy()

    if not trades_df.empty:
        trades_df['pnl'] = np.where(
            trades_df['position'] == 1,
            trades_df['exit_price'] / trades_df['entry_price'],
            trades_df['entry_price'] / trades_df['exit_price']
        )
        trades_df['duration'] = trades_df['exit_date'] - trades_df['entry_date']

        wins = trades_df[trades_df['pnl'] > 1].shape[0]
        losses = trades_df[trades_df['pnl'] < 1].shape[0]
        win_loss_ratio = wins / losses if losses != 0 else float('inf')
        avg_duration = trades_df['duration'].mean()
        min_duration = trades_df['duration'].min()
        max_duration = trades_df['duration'].max()
    else:
        no_time = pd.Timedelta(0)
        wins, losses, win_loss_ratio, avg_duration, min_duration, max_duration = 0, 0, 0, no_time, no_time, no_time

    trades_pnl = trades_df['pnl'].prod()

    best_trade = trades_df['pnl'].max()
    worst_trade = trades_df['pnl'].min()
    avg_trade = trades_df['pnl'].mean()

    pnl_from_daily_returns = (1 + df['strategy_return']).prod()

    # Sharpe Ratio
    risk_free_rate = 0.045
    daily_risk_free_rate = risk_free_rate / 365
    excess_returns = df['strategy_return'] - daily_risk_free_rate
    mean_return = excess_returns.mean()
    std_return = excess_returns.std()
    sharpe_ratio = 0.
    if std_return > 0:
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252)

    # Sortino Ratio
    neg_excess_returns = excess_returns[excess_returns < 1]
    downside_std = neg_excess_returns.std()
    sortino_ratio = np.inf
    if downside_std > 0:
        sortino_ratio = (mean_return / downside_std) * np.sqrt(252)

    # Plot strategy performance / equity curve
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()

    # Max Drawdown
    running_max = df['cumulative_return'].expanding().max()
    drawdown = (df['cumulative_return'] / running_max) - 1
    max_drawdown = drawdown.min()

    print(f"Total return from Trades: {trades_pnl:.4f}")
    print(f"Total return from daily returns: {pnl_from_daily_returns:.4f}")

    print(f"Win/Loss Ratio: {win_loss_ratio:.2f} ({wins} wins / {losses} losses)")
    print(f"Min duration: {min_duration}")
    print(f"Avg duration: {avg_duration}")
    print(f"Max duration: {max_duration}")

    print(f"Best Trade: {best_trade}")
    print(f"Worst Trade: {worst_trade}")
    print(f"Avg Trade: {avg_trade}")

    print(f"Sharpe Ratio: {sharpe_ratio}")
    print(f"Sharpe Ratio: {sortino_ratio}")

    print(f"Max Drawdown: {max_drawdown}")

