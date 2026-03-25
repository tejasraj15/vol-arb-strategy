from collections import defaultdict
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from volForecaster import Model

MODEL = Model.EGARCH
RESULTS_DIR = f"{MODEL.name}_results"

MAX_POSITIONS = 5 # number of positions you can hold at once
MAX_PER_SECTOR = float('inf')
MAX_POSITION_SIZE = 0.10
MAX_DEPLOYED = 0.40
ONE_PER_TICKER = True
DRAWDOWN_HALT = 0.10

TICKER_SECTORS = {
    'AAPL': 'Tech',
    'AMD': 'Tech',
    'AMZN': 'Tech',
    'GOOG': 'Tech',
    'MSFT': 'Tech',
    'NVDA': 'Tech',
    'MU': 'Semis',
    'INTC': 'Semis',
    'NFLX': 'Consumer',
    'NKE': 'Consumer',
    'SBUX': 'Consumer',
    'DIS': 'Consumer',
    'TSLA': 'Auto',
    'WMT': 'Retail',
    'XOM': 'Energy',
    'PFE': 'Healthcare',
    'UNH': 'Healthcare',
    # 'BAC': 'Financials',
    # 'GS': 'Financials',
    'BA': 'Industrial',
    'CAT': 'Industrial',
    'GE': 'Industrial',
}

ALL_TICKERS = list(TICKER_SECTORS.keys())
INITIAL_CASH = 100_000

cash = INITIAL_CASH
available_tickers = set(ALL_TICKERS)
remaining_sector_counts = {sector: MAX_PER_SECTOR for sector in set(TICKER_SECTORS.values())}
open_positions = defaultdict(list) # exit_date -> list[(trade_id, pnl, ticker, principal)]
num_open_positions = 0

results = []

trade_records = {}
next_trade_id = 1

def can_place_trade(cost, ticker):
    deployed_capital = INITIAL_CASH - cash
    potential_deployment = (deployed_capital + cost) / INITIAL_CASH
    
    return num_open_positions < MAX_POSITIONS and ticker in available_tickers and remaining_sector_counts[TICKER_SECTORS[ticker]] > 0 and potential_deployment < MAX_DEPLOYED


# open all trade logs and sort
df = pd.DataFrame()
for ticker in ALL_TICKERS:
    data = pd.read_csv(f'{RESULTS_DIR}/trade_log/trade_log_{ticker.lower()}_SHORT_VOL.csv')
    data = data[['entry_date','exit_date','entry_credit','net_pnl','garch_forecast']]
    data['ticker'] = ticker
    df = pd.concat([df, data], ignore_index=True)

df = df.sort_values(by=['entry_date', 'garch_forecast'], ascending=[True, False])

df['entry_date'] = pd.to_datetime(df['entry_date'])
df['exit_date'] = pd.to_datetime(df['exit_date'])
all_dates = pd.date_range(start=df['entry_date'].min(), end=max(df['entry_date'].max(), df['exit_date'].max()))

for current_date in all_dates:
    # close expired positions
    for exit_date in list(open_positions.keys()):
        if exit_date <= current_date:
            # Close trades scheduled for this (or earlier) date
            positions_to_close = open_positions[exit_date]
            while positions_to_close:
                locked_principal_before = sum(item[3] for trades in open_positions.values() for item in trades)
                equity_before = cash + locked_principal_before
                deployed_before = locked_principal_before
                cash_before = cash

                trade_id, pnl, ticker, principal = positions_to_close.pop(0)
                cash += (principal + pnl)
                num_open_positions -= 1
                available_tickers.add(ticker)
                remaining_sector_counts[TICKER_SECTORS[ticker]] += 1

                locked_principal_after = locked_principal_before - principal
                equity_after = cash + locked_principal_after
                deployed_after = locked_principal_after

                if trade_id in trade_records:
                    trade_records[trade_id].update({
                        'exit_date_actual': current_date,
                        'exit_cash_before': cash_before,
                        'exit_cash_after': cash,
                        'exit_equity_before': equity_before,
                        'exit_equity_after': equity_after,
                        'exit_deployed_before': deployed_before,
                        'exit_deployed_after': deployed_after,
                    })
            del open_positions[exit_date]
    
    day_entries = df[df['entry_date'] == current_date]
    
    # place trades
    for _, trade in day_entries.iterrows():
        if can_place_trade(trade['entry_credit'], trade['ticker']):
            locked_principal_before = sum(item[3] for trades in open_positions.values() for item in trades)
            equity_before = cash + locked_principal_before
            deployed_before = locked_principal_before
            cash_before = cash

            trade_id = next_trade_id
            next_trade_id += 1

            cash -= trade['entry_credit']
            open_positions[trade['exit_date']].append((
                trade_id,
                trade['net_pnl'],
                trade['ticker'],
                trade['entry_credit'],
            ))
            num_open_positions += 1
            if ONE_PER_TICKER:
                available_tickers.remove(trade['ticker'])

            remaining_sector_counts[TICKER_SECTORS[ticker]] -= 1

            locked_principal_after = sum(item[3] for trades in open_positions.values() for item in trades)
            equity_after = cash + locked_principal_after
            deployed_after = locked_principal_after

            trade_records[trade_id] = {
                'trade_id': trade_id,
                'ticker': trade['ticker'],
                'entry_date': current_date,
                'exit_date_planned': trade['exit_date'],
                'entry_credit': float(trade['entry_credit']),
                'net_pnl': float(trade['net_pnl']),
                'garch_forecast': float(trade['garch_forecast']),
                'entry_cash_before': cash_before,
                'entry_cash_after': cash,
                'entry_equity_before': equity_before,
                'entry_equity_after': equity_after,
                'entry_deployed_before': deployed_before,
                'entry_deployed_after': deployed_after,
            }
            
            print(f"Trade: {current_date.date()}, {trade['ticker']} | Cash: {cash:.2f}")
    
    locked_principal = sum(item[3] for trades in open_positions.values() for item in trades)
    current_total_equity = cash + locked_principal
    deployed_capital = locked_principal
    
    results.append((current_date, current_total_equity, deployed_capital))
    # print(f"Date: {current_date} | Equity: {current_total_equity:.2f} | Free Cash: {cash:.2f}")

# plot results
dates = [r[0] for r in results]
cash_values = [r[1] for r in results]
deployed_values = [r[2] for r in results]


res_df = pd.DataFrame(results, columns=['date', 'equity', 'deployed_capital'])
res_df['date'] = pd.to_datetime(res_df['date'])
res_df.set_index('date', inplace=True)
res_df['daily_ret'] = res_df['equity'].pct_change().fillna(0)
res_df['margin_usage'] = res_df['deployed_capital'] / INITIAL_CASH

total_return = (res_df['equity'].iloc[-1] / INITIAL_CASH) - 1
days_total = (res_df.index[-1] - res_df.index[0]).days
annualized_return = (1 + total_return) ** (365 / days_total) - 1
annual_vol = res_df['daily_ret'].std() * np.sqrt(252)
sharpe_ratio = (res_df['daily_ret'].mean() / res_df['daily_ret'].std()) * np.sqrt(252) if res_df['daily_ret'].std() != 0 else 0
downside_rets = res_df['daily_ret'][res_df['daily_ret'] < 0]
sortino_ratio = (res_df['daily_ret'].mean() / downside_rets.std()) * np.sqrt(252) if len(downside_rets) > 0 else 0

res_df['peak'] = res_df['equity'].cummax()
res_df['drawdown'] = (res_df['equity'] - res_df['peak']) / res_df['peak']
max_drawdown = res_df['drawdown'].min()
worst_one_day_drawdown = res_df['daily_ret'].min()
average_margin_usage = res_df['margin_usage'].mean()
peak_margin_usage = res_df['margin_usage'].max()

print("\n" + "="*30)
print("STRATEGY PERFORMANCE")
print("="*30)
print(f"Total Return:         {total_return:.2%}")
print(f"Annualized Return:    {annualized_return:.2%}")
print(f"Max Drawdown:         {max_drawdown:.2%}")
print(f"Worst 1-Day DD:       {worst_one_day_drawdown:.2%}")
print(f"Sharpe Ratio:         {sharpe_ratio:.2f}")
print(f"Sortino Ratio:        {sortino_ratio:.2f}")
print(f"Average Margin Usage: {average_margin_usage:.2%}")
print(f"Peak Margin Usage:    {peak_margin_usage:.2%}")
print(f"Total Trades:         {len(df)}")

# plot graphs
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

equity_color = '#2E86AB'
deployed_color = '#A23B72'
drawdown_color = '#F18F01'

# Equity Curve
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(res_df.index, res_df['equity'], linewidth=2.5, color=equity_color, label='Portfolio Equity')
ax1.axhline(y=INITIAL_CASH, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Initial Capital')
ax1.fill_between(res_df.index, INITIAL_CASH, res_df['equity'], where=(res_df['equity'] >= INITIAL_CASH), 
                 alpha=0.2, color='green', interpolate=True)
ax1.fill_between(res_df.index, INITIAL_CASH, res_df['equity'], where=(res_df['equity'] < INITIAL_CASH), 
                 alpha=0.2, color='red', interpolate=True)
ax1.set_ylabel('Portfolio Value ($)', fontsize=11, fontweight='bold')
ax1.set_title('Volatility Arbitrage Strategy - Portfolio Performance', fontsize=14, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='upper left', framealpha=0.9)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Drawdown Chart
ax2 = fig.add_subplot(gs[1, 0])
ax2.fill_between(res_df.index, 0, res_df['drawdown'] * 100, color=drawdown_color, alpha=0.6)
ax2.plot(res_df.index, res_df['drawdown'] * 100, linewidth=1.5, color=drawdown_color)
ax2.set_ylabel('Drawdown (%)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Date', fontsize=10)
ax2.set_title('Portfolio Drawdown', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Deployed Capital
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(res_df.index, res_df['deployed_capital'], linewidth=2, color=deployed_color, label='Deployed Capital')
ax3.fill_between(res_df.index, 0, res_df['deployed_capital'], alpha=0.3, color=deployed_color)
ax3.axhline(y=MAX_DEPLOYED * INITIAL_CASH, color='red', linestyle='--', linewidth=1.5, 
            alpha=0.7, label=f'Max Deployment ({MAX_DEPLOYED:.0%})')
ax3.set_ylabel('Deployed Capital ($)', fontsize=11, fontweight='bold')
ax3.set_xlabel('Date', fontsize=10)
ax3.set_title('Capital Deployment', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(loc='upper left', framealpha=0.9)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Statistics Box
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('off')

# statistics text
stats_text = f"""
PERFORMANCE METRICS                                    RISK METRICS                                         EXECUTION METRICS

Total Return:              {total_return:>8.2%}              Max Drawdown:           {max_drawdown:>8.2%}              Total Trades:           {len(df):>8,}
Annualized Return:         {annualized_return:>8.2%}              Worst 1-Day DD:         {worst_one_day_drawdown:>8.2%}              Max Positions:          {MAX_POSITIONS:>8}
Annual Volatility:         {annual_vol:>8.2%}              Sharpe Ratio:           {sharpe_ratio:>8.2f}              Avg Margin Usage:       {average_margin_usage:>8.2%}
Sortino Ratio:             {sortino_ratio:>8.2f}              Downside Volatility:    {downside_rets.std() * np.sqrt(252):>8.2%}              Peak Margin Usage:      {peak_margin_usage:>8.2%}
"""

ax4.text(0.5, 0.5, stats_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='center', horizontalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, pad=1),
         fontfamily='monospace')

for ax in [ax1, ax2, ax3]:
    ax.tick_params(axis='x', rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha('right')

plt.suptitle(f'Strategy: {" + ".join(ALL_TICKERS)} | Initial Capital: ${INITIAL_CASH:,}', 
             fontsize=11, y=0.995, alpha=0.7)

os.makedirs(f"{RESULTS_DIR}/portfolio", exist_ok=True)

fig.savefig(f"{RESULTS_DIR}/portfolio/strategy_performance.png")


# Save trades
full_trade_log_df = pd.DataFrame(trade_records.values())
if not full_trade_log_df.empty:
    full_trade_log_df = full_trade_log_df.sort_values(['entry_date', 'garch_forecast'], ascending=[True, False])
full_trade_log_df.to_csv(f"{RESULTS_DIR}/portfolio/full_trade_log.csv", index=False)
print(f"\nSaved executed trade log: {RESULTS_DIR}/portfolio/full_trade_log.csv ({len(full_trade_log_df):,} trades)")