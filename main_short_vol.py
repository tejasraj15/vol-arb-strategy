import numpy as np
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
from preprocess_data import parse_data
from implied_vol_surface import ImpliedVolSurface
from dividend_yield import get_dividend_yield
from transactionCosts import TransactionCost
from regime_identifier import RegimeBlockerXGB
from earnings_blocker import EarningsBlocker
from volForecaster import VolForecaster
from position import Position

warnings.filterwarnings('ignore', category=RuntimeWarning)

IV_WINDOW = 60

def load_options_data(filepath, ticker=None):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df['exdate'] = pd.to_datetime(df['exdate'])
    
    if ticker:
        df = df[df['ticker'] == ticker].copy()
        if len(df) == 0:
            print(f"No data found for ticker '{ticker}'")
            print(f"Available tickers: {sorted(df['ticker'].unique())[:10]}... ({len(df['ticker'].unique())} total)")
    
    # Use mid_price if available, otherwise calculate from bid/offer
    if 'mid_price' in df.columns:
        df['market_price'] = df['mid_price']
    else:
        df['market_price'] = (df['best_bid'] + df['best_offer']) / 2
    
    # Convert strike price from cents/thousands to dollars
    # Check if strikes are very large (>10000 suggests they're in cents)
    if df['strike_price'].median() > 10000:
        df['strike_price'] = df['strike_price'] / 1000
    elif df['strike_price'].median() > 1000:
        df['strike_price'] = df['strike_price'] / 100
    
    # Calculate time to maturity in years
    df['maturity'] = df['days_to_expiry'] / 365
    
    return df

def get_iv_surface_for_date(options_df, date, spot_price, risk_free_rate=0.02, dividend_yield=0.0, options_by_date=None):
    if options_by_date is not None:
        date_options = options_by_date.get(date)
        if date_options is None:
            return None, None, None
        date_options = date_options.copy()
    else:
        date_options = options_df[options_df['date'] == date].copy()
    
    if len(date_options) == 0:
        return None, None, None
    
    calls = date_options[date_options['cp_flag'] == 'C'].copy()
    
    if len(calls) < 10:  # Need minimum data
        return None, None, None
    
    # Filter out problematic options that cause numerical issues
    # Remove expired or very short-dated options (< 2 days)
    calls = calls[calls['days_to_expiry'] >= 2].copy()
    
    # Remove options with zero or negative prices (invalid)
    calls = calls[calls['market_price'] > 0.01].copy()
    
    # Remove extreme OTM options (strike > 3x spot)
    calls = calls[calls['strike_price'] < spot_price * 3.0].copy()
    
    if len(calls) < 10:  # Recheck after filtering
        return None, None, None
    
    # Build price matrix using pivot
    pivot = calls.pivot_table(index='strike_price', columns='maturity', values='market_price', aggfunc='first')
    strikes = pivot.index.to_numpy()
    maturities = pivot.columns.to_numpy()
    market_prices = pivot.to_numpy(dtype=float)
    
    # Filter out strikes/maturities with too much missing data
    valid_strikes_mask = ~np.isnan(market_prices).all(axis=1)
    valid_maturities_mask = ~np.isnan(market_prices).all(axis=0)
    
    strikes = np.array(strikes)[valid_strikes_mask]
    maturities = np.array(maturities)[valid_maturities_mask]
    market_prices = market_prices[valid_strikes_mask][:, valid_maturities_mask]
    
    if len(strikes) < 5 or len(maturities) < 2:
        return None, None, None
    
    return strikes, maturities, market_prices

def get_atm_option_for_dte(options_df, date, spot_price, target_dte=45, dte_range=(30, 45), options_by_date=None):
    if options_by_date is not None:
        date_options = options_by_date.get(date)
        if date_options is None:
            return None
        date_options = date_options.copy()
    else:
        date_options = options_df[options_df['date'] == date].copy()
    
    if len(date_options) == 0:
        return None
    
    dte = date_options['days_to_expiry'].to_numpy()
    lo = np.searchsorted(dte, dte_range[0], side='left')
    hi = np.searchsorted(dte, dte_range[1], side='right')
    date_options = date_options.iloc[lo:hi].copy()
    
    if len(date_options) == 0:
        return None
    
    date_options['dte_diff'] = abs(date_options['days_to_expiry'] - target_dte)
    target_dte_actual = date_options.loc[date_options['dte_diff'].idxmin(), 'days_to_expiry']
    target_options = date_options[date_options['days_to_expiry'] == target_dte_actual].copy()
    
    strikes = target_options['strike_price'].unique()
    atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
    
    calls = target_options[(target_options['strike_price'] == atm_strike) & (target_options['cp_flag'] == 'C')]
    puts = target_options[(target_options['strike_price'] == atm_strike) & (target_options['cp_flag'] == 'P')]
    
    if len(calls) == 0 or len(puts) == 0:
        return None
    
    call_price = calls['market_price'].iloc[0]
    put_price = puts['market_price'].iloc[0]
    
    return {
        'strike': atm_strike,
        'dte': target_dte_actual,
        'maturity': target_dte_actual / 365,
        'call_price': call_price,
        'put_price': put_price,
        'straddle_price': call_price + put_price,
        'exdate': calls['exdate'].iloc[0]
    }


def get_iv_for_option(iv_calc, strike, maturity, call_price, put_price):
    try:
        call_iv = iv_calc.implied_volatility(call_price, strike, maturity, 'call')
        put_iv = iv_calc.implied_volatility(put_price, strike, maturity, 'put')
        # Average of call and put IV
        if not np.isnan(call_iv) and not np.isnan(put_iv):
            return (call_iv + put_iv) / 2
        elif not np.isnan(call_iv):
            return call_iv
        elif not np.isnan(put_iv):
            return put_iv
        return np.nan
    except:
        return np.nan


def should_enter(market_iv, garch_forecast, iv_history, regime_blocker, earnings_blocker, current_date):
    """
    Evaluate whether market conditions justify entering a short vol trade.
    Returns (signal, iv_diff, iv_percentile, is_blocked, entry_notes).
    """
    iv_diff = market_iv - garch_forecast
    garch_signal = iv_diff > 0.03

    if len(iv_history) >= 20:
        iv_array = np.array(iv_history)
        recent_iv = iv_array[-IV_WINDOW:]
        iv_mean = np.mean(recent_iv)
        iv_std = np.std(recent_iv)

        iv_zscore = (market_iv - iv_mean) / iv_std
        iv_expensive = iv_zscore > 1.0
        iv_percentile = (np.array(iv_history) < market_iv).sum() / len(iv_history) * 100
    else:
        iv_expensive = True
        iv_percentile = 50

    is_blocked = False
    if regime_blocker is not None:
        try:
            is_blocked = regime_blocker.isBlocked(current_date.strftime('%Y-%m-%d'))
        except Exception:
            is_blocked = False

    entry_notes = []
    if not garch_signal:
        entry_notes.append(f"Market IV not expensive (diff {iv_diff:.1%} < 3%)")
    if not iv_expensive:
        entry_notes.append(f"IV percentile {iv_percentile:.0f} < 80")
    if is_blocked:
        entry_notes.append("Stress regime")
    if earnings_blocker and earnings_blocker.should_block_entry(current_date):
        entry_notes.append(earnings_blocker.get_block_reason())
        return 'HOLD', iv_diff, iv_percentile, is_blocked, entry_notes

    signal = 'SELL' if (garch_signal and iv_expensive and not is_blocked) else 'HOLD'
    return signal, iv_diff, iv_percentile, is_blocked, entry_notes


def build_result_row(current_date, spot_price, garch_forecast, portfolio_value,
                      market_iv=np.nan, iv_spread=np.nan, iv_percentile=np.nan,
                      signal='HOLD', blocked=False, entry_notes='',
                      atm_strike=np.nan, atm_dte=np.nan, straddle_price=np.nan,
                      position_status='NONE', days_held=0, iv_change_pct=np.nan,
                      trade_pnl_dollars=0.0, traded=False):
    """Single schema for all result rows — prevents key mismatches between branches."""
    return {
        'date': current_date,
        'spot_price': spot_price,
        'forecast_iv': garch_forecast,
        'market_iv': market_iv,
        'iv_spread': iv_spread,
        'iv_percentile': iv_percentile,
        'signal': signal,
        'blocked': blocked,
        'entry_notes': entry_notes,
        'atm_strike': atm_strike,
        'atm_dte': atm_dte,
        'straddle_price': straddle_price,
        'position_status': position_status,
        'days_held': days_held,
        'iv_change_pct': iv_change_pct,
        'portfolio_value': portfolio_value,
        'trade_pnl_dollars': trade_pnl_dollars,
        'traded': traded,
    }

def output_diagnostics(volForecaster: VolForecaster):
    diagnostics = volForecaster.get_forecast_diagnostics()
    if diagnostics:
        print(f"\n  Forecast Quality Analysis:")
        print(f"    Mean Forecast:          {diagnostics['mean_forecast']:.2%}")
        print(f"    Mean Market IV:         {diagnostics['mean_market_iv']:.2%}")
        print(f"    Mean Forecast Error:    {diagnostics['mean_error']:.2%}")
        print(f"    RMSE:                   {diagnostics['rmse']:.2%}")
        print(f"    Correlation:            {diagnostics['correlation']:.3f}")
        print(f"    Vol Risk Premium:       {diagnostics['vol_risk_premium']:.2%}")

def output_performance_analysis(df_trades):
    if len(df_trades) > 0:
        print(f"\n  Trade Statistics:")
        print(f"    Total Trades:     {len(df_trades):>6}")
        print(f"    Winning Trades:   {(df_trades['net_pnl'] > 0).sum():>6}")
        print(f"    Losing Trades:    {(df_trades['net_pnl'] < 0).sum():>6}")
        print(f"    Win Rate:         {(df_trades['net_pnl'] > 0).mean()*100:>6.1f}%")
        
        print(f"\n  Holding Period:")
        print(f"    Mean:             {df_trades['days_held'].mean():>6.1f} days")
        print(f"    Median:           {df_trades['days_held'].median():>6.1f} days")
        print(f"    Min:              {df_trades['days_held'].min():>6} days")
        print(f"    Max:              {df_trades['days_held'].max():>6} days")
        
        print(f"\n  IV Change (Entry to Exit):")
        print(f"    Mean:             {df_trades['iv_change_pct'].mean():>7.2f}%")
        print(f"    Median:           {df_trades['iv_change_pct'].median():>7.2f}%")
        print(f"    Std:              {df_trades['iv_change_pct'].std():>7.2f}%")
        
        print(f"\n  P&L Statistics:")
        print(f"    Total Gross P&L:  ${df_trades['gross_pnl'].sum():>10,.2f}")
        print(f"    Total Net P&L:    ${df_trades['net_pnl'].sum():>10,.2f}")
        print(f"    Mean Trade P&L:   ${df_trades['net_pnl'].mean():>10,.2f}")
        print(f"    Median Trade P&L: ${df_trades['net_pnl'].median():>10,.2f}")
        print(f"    Best Trade:       ${df_trades['net_pnl'].max():>10,.2f}")
        print(f"    Worst Trade:      ${df_trades['net_pnl'].min():>10,.2f}")
        
        print(f"\n  Exit Reasons:")
        for reason, count in df_trades['exit_reason'].value_counts().items():
            print(f"    {reason}: {count}")
        
        print(f"\n  P&L Breakdown (SHORT VOL):")
        print(f"    Option P&L:         ${df_trades['option_pnl'].sum():>10,.2f}  (profit when IV drops)")
        print(f"    Gamma P&L:          ${df_trades['gamma_pnl'].sum():>10,.2f}  (cost from spot moves)")
        print(f"    Theta P&L:          ${df_trades['theta_pnl'].sum():>10,.2f}  (GAIN from decay)")
        print(f"    Delta Hedge P&L:    ${df_trades['delta_hedge_pnl'].sum():>10,.2f}")
        print(f"    Total Hedge P&L:    ${df_trades['hedge_pnl'].sum():>10,.2f}")
        
        print(f"\n  Transaction Costs:")
        print(f"    Entry Costs:        ${df_trades['entry_tc'].sum():>10,.2f}")
        print(f"    Exit Costs:         ${df_trades['exit_tc'].sum():>10,.2f}")
        print(f"    Hedge Rebal Costs:  ${df_trades['hedge_rebalance_costs'].sum():>10,.2f}")

def get_trading_dates(options_data, train_window, start_date=None, end_date=None):
    trading_dates = sorted(options_data['date'].unique())
    
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        trading_dates = [d for d in trading_dates if d >= start_date]
        print(f"Start date filter: {start_date.date()}")
    
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        trading_dates = [d for d in trading_dates if d <= end_date]
        print(f"End date filter: {end_date.date()}")
    
    print(f"{len(trading_dates)} trading dates with options data")
    print(f"Training window: {train_window} days, will start backtesting from day {train_window + 1}")
    
    if len(trading_dates) <= train_window:
        print(f"Error: Not enough data. Need >{train_window} days, have {len(trading_dates)}")
        return None
    
    return trading_dates

def get_regime_blocker(trading_dates, train_window, stock_data):
    backtest_start_date = trading_dates[train_window]
    historical_stock_data = stock_data.loc[:backtest_start_date].iloc[:-1]
    log_returns_historical = parse_data(historical_stock_data.reset_index(), price_col='prc')
    
    print(f"Training regime blocker on data up to {backtest_start_date.date()} ({len(log_returns_historical)} days)")
    
    try:
        regime_blocker = RegimeBlockerXGB(
            returns=log_returns_historical,
            stress_vol_percentile=90.0,
            stress_drawdown_threshold=-0.05,
            calm_vol_percentile=25.0,
            block_threshold=0.5,
            random_state=42,
            verbose=False
        )
    except Exception as e:
        print(f"Could not initialize regime blocker: {e}")
        print("Continuing without regime blocking")
        regime_blocker = None
    
    return regime_blocker

def rolling_window_backtest(ticker, train_window=126, refit_frequency=21, 
                            starting_capital=100000, position_size=8000,
                            start_date=None, end_date=None,
                            use_regime_blocker=True,
                            use_earnings_blocker=True,
                            earnings_csv='earnings_data.csv',
                            earnings_before=7, 
                            earnings_after=2, 
                            earnings_exit=3,
                            verbose=False):
    
    ticker_upper = ticker.upper()
    ticker_lower = ticker.lower()
    dividend_yield = get_dividend_yield(ticker_upper)
    sp_data_path = f"data/{ticker_lower}_stock_prices_2020_2024.csv"
    options_data_path = f"data/{ticker_lower}_options_2020_2024.csv"
    
    print(f"\nLoading data")
    
    stock_data = pd.read_csv(sp_data_path, parse_dates=['date'])
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data = stock_data.set_index('date').sort_index()
    
    options_data = load_options_data(options_data_path, ticker=ticker_upper)
    print(f"Loaded {len(stock_data)} days of {ticker_upper} stock data")
    print(f"Loaded {len(options_data)} option quotes for {ticker_upper}")

    options_by_date = dict(tuple(options_data.groupby('date', sort=False)))
    
    trading_dates = get_trading_dates(options_data, train_window, start_date, end_date)
    if not trading_dates:
        return None
    
    earnings_blocker = None
    if use_earnings_blocker:
        earnings_blocker = EarningsBlocker(
            ticker=ticker,
            earnings_csv=earnings_csv,
            before_days=earnings_before,
            after_days=earnings_after,
            exit_days=earnings_exit,
            verbose=verbose
        )
    
    regime_blocker = None
    if use_regime_blocker:
        regime_blocker = get_regime_blocker(trading_dates, train_window, stock_data)
    
    results = []
    trade_log = []
    portfolio_value = starting_capital
    active_position = None
    tc_calc = TransactionCost()
    iv_history = deque(maxlen=252)
    
    print("\nStarting backtest")
    print(f"  Starting Capital: ${starting_capital:,.2f}")
    print(f"  Position Size: ${position_size:,.2f}")
    volForecaster = VolForecaster(stock_data, train_window, refit_frequency, verbose)
    
    processed_count = 0
    skipped_count = 0
    blocked_count = 0
    trades_entered = 0
    trades_exited = 0
    total_dates = len(trading_dates)

    for i, current_date in enumerate(trading_dates):
        if i % 20 == 0:
            progress_pct = (i / total_dates) * 100
            print(f"\r  Progress: {i}/{total_dates} dates ({progress_pct:.1f}%) - Trades: {trades_entered} entered, {trades_exited} exited, Blocked: {blocked_count}", end="", flush=True)
        
        if i < train_window:
            continue  # Need minimum training data

        if current_date not in stock_data.index:
            skipped_count += 1
            if verbose:
                print(f"[{current_date.date()}] No stock data for this date")
            continue
        spot_price = stock_data.loc[current_date, 'prc']

        garch_forecast = volForecaster.get_forecast(current_date)

        if active_position is not None:
            updated = active_position.update(current_date, spot_price, options_by_date)
            if not updated:
                continue

            should_exit, exit_reason = active_position.check_exit(current_date, earnings_blocker)

            trade_record = None
            if should_exit:
                trade_record = active_position.close(current_date, spot_price, tc_calc, ticker_upper)
                trade_record['exit_reason'] = exit_reason
                portfolio_value += trade_record['net_pnl']
                trade_log.append(trade_record)
                if verbose:
                    print(f"\n[{current_date.date()}] EXIT: {exit_reason} | P&L: ${trade_record['net_pnl']:.2f}")
                trades_exited += 1

            results.append(build_result_row(
                current_date, spot_price, garch_forecast, portfolio_value,
                market_iv=trade_record['exit_iv'] if trade_record else active_position.current_iv,
                position_status='EXITED' if trade_record else 'HOLDING',
                days_held=trade_record['days_held'] if trade_record else active_position.days_held,
                iv_change_pct=trade_record['iv_change_pct'] if trade_record else active_position.iv_change_pct,
                trade_pnl_dollars=trade_record['net_pnl'] if trade_record else 0,
                traded=should_exit,
            ))
            processed_count += 1
            if should_exit:
                active_position = None
            continue
        
        try:
            iv_calc = ImpliedVolSurface(
                spot_price=spot_price,
                risk_free_rate=0.02,
                dividend_yield=dividend_yield,
                verbose=False
            )
            atm_option = get_atm_option_for_dte(
                options_data, current_date, spot_price,
                target_dte=45, dte_range=(30, 45),
                options_by_date=options_by_date
            )
            if atm_option is None:
                skipped_count += 1
                if verbose:
                    print(f"[{current_date.date()}] No 30-45 DTE options available")
                continue

            market_iv = get_iv_for_option(
                iv_calc, atm_option['strike'], atm_option['maturity'],
                atm_option['call_price'], atm_option['put_price']
            )
            if np.isnan(market_iv):
                skipped_count += 1
                continue

            iv_history.append(market_iv)
            volForecaster.record_market_iv(market_iv)

            signal, iv_diff, iv_percentile, is_blocked, entry_notes = should_enter(
                market_iv, garch_forecast, iv_history, regime_blocker, earnings_blocker, current_date
            )
            if is_blocked:
                blocked_count += 1

            if signal == 'SELL':
                active_position = Position.open(
                    current_date, spot_price, atm_option, market_iv, garch_forecast,
                    dividend_yield, ticker_upper, position_size, starting_capital,
                     iv_percentile, tc_calc, verbose
                )
                if active_position is None:
                    signal = 'HOLD'
                else:
                    trades_entered += 1
                    if verbose:
                        print(f"\n[{current_date.date()}] ENTRY (SELL): Strike ${atm_option['strike']:.0f}, "
                              f"DTE {atm_option['dte']}, IV {market_iv:.1%}, GARCH {garch_forecast:.1%}, "
                              f"Credit ${active_position.entry_credit:.0f}")

            results.append(build_result_row(
                current_date, spot_price, garch_forecast, portfolio_value,
                market_iv=market_iv,
                iv_spread=iv_diff,
                iv_percentile=iv_percentile if len(iv_history) >= 20 else np.nan,
                signal=signal,
                blocked=is_blocked,
                entry_notes='; '.join(entry_notes) if entry_notes else '',
                atm_strike=atm_option['strike'],
                atm_dte=atm_option['dte'],
                straddle_price=atm_option['straddle_price'],
                traded=signal == 'SELL',
            ))
            processed_count += 1

        except Exception as e:
            skipped_count += 1
            if verbose:
                print(f"[{current_date.date()}] Error: {str(e)}")
            continue
    
    print(f"\n\nProcessing results")
    print(f"  Processed: {processed_count}, Skipped: {skipped_count}, Blocked: {blocked_count}")
    print(f"  Trades entered: {trades_entered}, Trades exited: {trades_exited}")
    
    output_diagnostics(volForecaster)
    
    df_results = pd.DataFrame(results)
    df_trades = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    
    if len(df_results) == 0:
        print("No valid backtest results")
        return None
    
    output_performance_analysis(df_trades)
    
    # Entry criteria distribution
    if 'signal' in df_results.columns:
        print(f"\nSignal Distribution:")
        for sig in ['SELL', 'HOLD']:
            count = (df_results['signal'] == sig).sum()
            pct = count / len(df_results) * 100
            print(f"    {sig}: {count:>4} ({pct:>5.1f}%)")
    
    if 'iv_spread' in df_results.columns:
        iv_spreads = df_results['iv_spread'].dropna()
        if len(iv_spreads) > 0:
            print(f"\n  Market IV - GARCH Spread:")
            print(f"    Mean:             {iv_spreads.mean():>7.2%}")
            print(f"    Median:           {iv_spreads.median():>7.2%}")
            print(f"    % Above 2%:       {(iv_spreads > 0.02).mean()*100:>6.1f}%")
    
    print(f"\nFinal Portfolio Value: ${portfolio_value:,.2f}")
    print(f"Total Return:          {(portfolio_value - starting_capital) / starting_capital * 100:.2f}%")
    
    if earnings_blocker:
        earnings_blocker.print_stats()
    
    df_results.attrs['trade_log'] = df_trades
    
    return df_results


def plot_short_vol_results(results_df, save_path='results/graphs/backtest_short_vol_analysis.png'):
    if results_df is None or len(results_df) == 0:
        print("No results to plot")
        return

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.25)

    dates = pd.to_datetime(results_df['date'])
    portfolio_values = results_df['portfolio_value'].astype(float).values
    starting_value = float(portfolio_values[0])
    running_max = pd.Series(portfolio_values).cummax()
    drawdown = (portfolio_values - running_max) / running_max * 100

    trade_log = results_df.attrs.get('trade_log', pd.DataFrame())

    # Portfolio value
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, portfolio_values, linewidth=2, color='#2E86AB', label='Portfolio Value')
    ax1.fill_between(dates, starting_value, portfolio_values, alpha=0.25, color='#2E86AB')
    ax1.axhline(y=starting_value, color='red', linestyle='--', alpha=0.5, label='Starting Capital')

    if len(trade_log) > 0:
        entry_dates = pd.to_datetime(trade_log['entry_date'])
        exit_dates = pd.to_datetime(trade_log['exit_date'])

        # Mark entries/exits
        ax1.scatter(entry_dates, np.interp(mdates.date2num(entry_dates), mdates.date2num(dates), portfolio_values),
                    s=18, color='#E63946', alpha=0.75, label='Trade Entry')
        ax1.scatter(exit_dates, np.interp(mdates.date2num(exit_dates), mdates.date2num(dates), portfolio_values),
                    s=18, color='#6A994E', alpha=0.75, label='Trade Exit')

    ax1.set_title('Portfolio Value Over Time (Short Vol Strategy)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=11)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.grid(True, alpha=0.3)
    ax1.legend(ncols=3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Drawdown
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(dates, 0, drawdown, color='#A23B72', alpha=0.7)
    ax2.plot(dates, drawdown, linewidth=1.5, color='#A23B72')
    ax2.set_title('Drawdown (%)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Trade P&L distribution
    ax3 = fig.add_subplot(gs[2, 0])
    if len(trade_log) > 0 and 'net_pnl' in trade_log.columns:
        pnls = trade_log['net_pnl'].astype(float).values
        ax3.hist(pnls, bins=24, alpha=0.75, edgecolor='black', color='#E63946')
        ax3.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Break-even')
        ax3.set_title('Trade Net P&L Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Net P&L ($)', fontsize=10)
        ax3.set_ylabel('Frequency', fontsize=10)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.axis('off')
        ax3.text(0.5, 0.5, 'No completed trades to plot', ha='center', va='center')

    # Exit reasons
    ax4 = fig.add_subplot(gs[2, 1])
    if len(trade_log) > 0 and 'exit_reason' in trade_log.columns:
        reason_counts = trade_log['exit_reason'].value_counts().head(8)
        ax4.barh(reason_counts.index[::-1], reason_counts.values[::-1], color='#2E86AB', alpha=0.8, edgecolor='black')
        ax4.set_title('Top Exit Reasons', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Count', fontsize=10)
        ax4.grid(True, alpha=0.3, axis='x')
    else:
        ax4.axis('off')

    # Forecast vs Market IV + SELL signals
    ax5 = fig.add_subplot(gs[3, 0])
    if 'forecast_iv' in results_df.columns and 'market_iv' in results_df.columns:
        forecast_iv = results_df['forecast_iv'].astype(float).values * 100
        market_iv = results_df['market_iv'].astype(float).ffill().values * 100

        ax5.plot(dates, forecast_iv, label='Forecast (EGARCH + RP)', linewidth=2, color='#2E86AB', alpha=0.85)
        ax5.plot(dates, market_iv, label='Market IV', linewidth=2, color='#F18F01', alpha=0.85)

        if 'signal' in results_df.columns:
            sell_mask = (results_df['signal'] == 'SELL').values
            if sell_mask.any():
                ax5.fill_between(dates, 0, np.maximum(forecast_iv.max(), market_iv.max()) * 1.2,
                                 where=sell_mask, color='#E63946', alpha=0.10, label='SELL signal')

        ax5.set_title('Forecast vs Market IV (Signals Highlighted)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Implied Volatility (%)', fontsize=10)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    else:
        ax5.axis('off')

    # Metrics summary
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.axis('off')

    daily_returns = pd.Series(portfolio_values).pct_change().dropna()
    sharpe = 0.0
    max_dd = float(drawdown.min() / 100) if len(drawdown) > 0 else 0.0
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    total_return = (portfolio_values[-1] - starting_value) / starting_value
    years = max(1e-9, (dates.iloc[-1] - dates.iloc[0]).days / 365.0)
    annual_return = (portfolio_values[-1] / starting_value) ** (1 / years) - 1

    total_trades = int(len(trade_log)) if len(trade_log) > 0 else 0
    win_rate = float((trade_log['net_pnl'] > 0).mean() * 100) if total_trades > 0 and 'net_pnl' in trade_log.columns else 0.0
    total_net_pnl = float(trade_log['net_pnl'].sum()) if total_trades > 0 and 'net_pnl' in trade_log.columns else 0.0

    metrics_text = f"""
        SHORT VOL PERFORMANCE SUMMARY

        Total Return:        {total_return*100:>7.2f}%
        Annualized Return:   {annual_return*100:>7.2f}%
        Sharpe (daily):      {sharpe:>7.3f}
        Max Drawdown:        {max_dd*100:>7.2f}%

        Total Trades:        {total_trades:>7}
        Win Rate:            {win_rate:>7.1f}%
        Total Net P&L:       ${total_net_pnl:>10,.2f}
        """

    ax6.text(0.06, 0.5, metrics_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Short Vol Strategy - Performance Dashboard', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nCharts saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    else:
        ticker = 'AAPL'
        print(f"No ticker specified, using default: {ticker}")
        print(f"Usage: python main_short_vol.py <TICKER>")
        print(f"Example: python main_short_vol.py AAPL\n")
    
    results = rolling_window_backtest(
        ticker=ticker,
        train_window=126,
        refit_frequency=21,
        starting_capital=100000,
        position_size=8000,
        use_regime_blocker=True,
        start_date=None,
        end_date=None,
        verbose=True,
        earnings_csv=f'data/{ticker.lower()}_earnings_dates.csv'
    )
    
    if results is not None:
        results.to_csv(f"results/res/backtest_results_{ticker.lower()}_SHORT_VOL.csv", index=False)
        print(f"\nDaily results saved to results/res/backtest_results_{ticker.lower()}_SHORT_VOL.csv")
        
        trade_log = results.attrs.get('trade_log', pd.DataFrame())
        if len(trade_log) > 0:
            trade_log.to_csv(f"results/trade_log/trade_log_{ticker.lower()}_SHORT_VOL.csv", index=False)
            print(f"Trade log saved to results/trade_log/trade_log_{ticker.lower()}_SHORT_VOL.csv")

        plot_short_vol_results(results, save_path=f"results/graphs/backtest_short_vol_{ticker.lower()}_analysis.png")