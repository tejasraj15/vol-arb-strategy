import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
from preprocess_data import get_log_returns, parse_data
from garch import garch_modelling
from implied_vol_surface import ImpliedVolSurface
from Delta_Hedging import DeltaHedger
from transactionCosts import TransactionCost
from regime_identifier import RegimeBlockerXGB
from scipy.stats import norm

warnings.filterwarnings('ignore', category=RuntimeWarning)

DE_MEAN = "AR"
MODEL = "EGARCH"
DISTRIBUTION = {"GARCH": "normal", "EGARCH": "t"}[MODEL]
validity_checks = False

def load_options_data(filepath, ticker=None):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df['exdate'] = pd.to_datetime(df['exdate'])
    
    if ticker:
        df = df[df['ticker'] == ticker].copy()
        if len(df) == 0:
            print(f"  [!] Warning: No data found for ticker '{ticker}'")
            print(f"  Available tickers: {sorted(df['ticker'].unique())[:10]}... ({len(df['ticker'].unique())} total)")
    
    if 'mid_price' in df.columns:
        df['market_price'] = df['mid_price']
    else:
        df['market_price'] = (df['best_bid'] + df['best_offer']) / 2
    
    if df['strike_price'].median() > 10000:
        df['strike_price'] = df['strike_price'] / 1000
    elif df['strike_price'].median() > 1000:
        df['strike_price'] = df['strike_price'] / 100
    
    df['maturity'] = df['days_to_expiry'] / 365
    
    return df

def get_iv_surface_for_date(options_df, date, spot_price, risk_free_rate=0.02, dividend_yield=0.02):
    date_options = options_df[options_df['date'] == date].copy()
    
    if len(date_options) == 0:
        return None, None, None
    
    calls = date_options[date_options['cp_flag'] == 'C'].copy()
    
    if len(calls) < 10:
        return None, None, None
    
    calls = calls[calls['days_to_expiry'] >= 2].copy()
    calls = calls[calls['market_price'] > 0.01].copy()
    calls = calls[calls['strike_price'] < spot_price * 3.0].copy()
    
    if len(calls) < 10:
        return None, None, None
    
    strikes = sorted(calls['strike_price'].unique())
    maturities = sorted(calls['maturity'].unique())
    
    market_prices = np.full((len(strikes), len(maturities)), np.nan)
    
    for i, K in enumerate(strikes):
        for j, T in enumerate(maturities):
            price_data = calls[(calls['strike_price'] == K) & (calls['maturity'] == T)]
            if len(price_data) > 0:
                market_prices[i, j] = price_data['market_price'].iloc[0]
    
    valid_strikes_mask = ~np.isnan(market_prices).all(axis=1)
    valid_maturities_mask = ~np.isnan(market_prices).all(axis=0)
    
    strikes = np.array(strikes)[valid_strikes_mask]
    maturities = np.array(maturities)[valid_maturities_mask]
    market_prices = market_prices[valid_strikes_mask][:, valid_maturities_mask]
    
    if len(strikes) < 5 or len(maturities) < 2:
        return None, None, None
    
    return strikes, maturities, market_prices

def get_atm_option_for_dte(options_df, date, spot_price, target_dte=45, dte_range=(30, 45)):
    date_options = options_df[options_df['date'] == date].copy()
    
    if len(date_options) == 0:
        return None
    
    date_options = date_options[
        (date_options['days_to_expiry'] >= dte_range[0]) & 
        (date_options['days_to_expiry'] <= dte_range[1])
    ].copy()
    
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
        if not np.isnan(call_iv) and not np.isnan(put_iv):
            return (call_iv + put_iv) / 2
        elif not np.isnan(call_iv):
            return call_iv
        elif not np.isnan(put_iv):
            return put_iv
        return np.nan
    except:
        return np.nan

def combined_vol_arbitrage_backtest(ticker, train_window=126, refit_frequency=21,
                                    starting_capital=100000, position_size=8000,
                                    start_date=None, end_date=None,
                                    use_regime_blocker=True, 
                                    allocation_mode='split',  # 'split', 'separate', or 'priority'
                                    verbose=False):
    print("="*80)
    print("COMBINED LONG/SHORT VOL ARBITRAGE STRATEGY")
    print("="*80)
    print("\n  Strategy Rules:")
    print("    LONG:  GARCH > Market IV + 2%")
    print("    SHORT: Market IV > GARCH + 2%")
    print(f"    Allocation: {allocation_mode}")
    print("    Position: 45 DTE ATM straddles, delta hedged daily")
    print("    Exit: Day 7/10/14 dynamic rules based on IV changes")
    
    # Construct filenames
    ticker_upper = ticker.upper()
    ticker_lower = ticker.lower()
    sp_data_path = f"{ticker_lower}_stock_prices_2020_2024.csv"
    options_data_path = f"{ticker_lower}_options_2020_2024.csv"
    
    # Load data
    print(f"\n[1/6] Loading data for {ticker_upper}...")
    stock_data = pd.read_csv(sp_data_path, parse_dates=['date'])
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data = stock_data.set_index('date').sort_index()
    
    options_data = load_options_data(options_data_path, ticker=ticker_upper)
    print(f"  * Loaded {len(stock_data)} days of {ticker_upper} stock data")
    print(f"  * Loaded {len(options_data)} option quotes for {ticker_upper}")
    
    trading_dates = sorted(options_data['date'].unique())
    
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        trading_dates = [d for d in trading_dates if d >= start_date]
        print(f"  [i] Start date filter: {start_date.date()}")
    
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        trading_dates = [d for d in trading_dates if d <= end_date]
        print(f"  [i] End date filter: {end_date.date()}")
    
    print(f"  * {len(trading_dates)} trading dates with options data")
    
    # Initialize regime blocker
    regime_blocker = None
    if use_regime_blocker:
        print(f"\n[2/6] Initializing regime blocker...")
        backtest_start_date = trading_dates[train_window]
        historical_stock_data = stock_data.loc[:backtest_start_date].iloc[:-1]
        log_returns_historical = parse_data(historical_stock_data.reset_index(), price_col='prc')
        
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
            print("  * Regime blocker initialized successfully")
        except Exception as e:
            print(f"  [!] Warning: Could not initialize regime blocker: {e}")
            regime_blocker = None
    
    # Results tracking
    results = []
    long_trades = []
    short_trades = []
    
    # Portfolio tracking
    portfolio_value = starting_capital
    cash = starting_capital
    
    # Position tracking - can have BOTH long and short simultaneously
    long_position = None
    short_position = None
    
    tc_calc = TransactionCost()
    iv_history = []
    
    print("\n[3/6] Starting combined strategy backtest...")
    print(f"  Starting Capital: ${starting_capital:,.2f}")
    print(f"  Max Position Size: ${position_size:,.2f}")
    
    last_garch_fit_date = None
    garch_forecast = None
    
    processed_count = 0
    long_entries = 0
    short_entries = 0
    long_exits = 0
    short_exits = 0
    both_signals_count = 0
    
    # Vol forecasting setup
    forecast_history = []
    market_iv_history = []
    
    def calculate_vol_risk_premium(forecast_hist, market_hist, min_samples=60):
        if len(forecast_hist) < min_samples:
            return 0.03
        lookback = min(126, len(forecast_hist))
        recent_forecasts = np.array(forecast_hist[-lookback:])
        recent_markets = np.array(market_hist[-lookback:])
        risk_premium = np.median(recent_markets - recent_forecasts)
        return np.clip(risk_premium, 0.0, 0.10)
    
    def ensemble_vol_forecast(returns, base_forecast):
        forecasts = [base_forecast]
        if len(returns) >= 21:
            recent_rv = returns.tail(21).std() * np.sqrt(252)
            forecasts.append(recent_rv)
        if len(returns) >= 30:
            ewma_vol = returns.ewm(span=30).std().iloc[-1] * np.sqrt(252)
            forecasts.append(ewma_vol)
        return np.median(forecasts)
    
    def calculate_dynamic_position_size(base_size, current_iv, iv_history, position_type):
        if len(iv_history) < 20:
            return base_size
        
        # Calculate IV percentile
        iv_percentile = (np.array(iv_history) < current_iv).sum() / len(iv_history)
        
        if position_type == 'SHORT':
            if iv_percentile < 0.25:  # LOW IV = dangerous for short
                scale_factor = 0.5  # Small position
            else:
                scale_factor = 1.0  # Normal/high IV = safe for short
        else:  # LONG
            # Long vol: reduce size when IV is very low (less potential)
            if iv_percentile < 0.20:  # IV in bottom 20%
                scale_factor = 0.75
            else:
                scale_factor = 1.0
        
        return base_size * scale_factor
    
    def manage_position(position, position_type, current_date, spot_price, options_data, 
                       hedger, tc_calc, ticker_upper):
        days_held = (current_date - position['entry_date']).days
        
        # Get current option prices
        date_options = options_data[options_data['date'] == current_date]
        strike_options = date_options[date_options['strike_price'] == position['strike']]
        
        current_iv = None
        current_straddle_value = None
        
        if len(strike_options) > 0:
            calls = strike_options[strike_options['cp_flag'] == 'C']
            puts = strike_options[strike_options['cp_flag'] == 'P']
            
            if len(calls) > 0 and len(puts) > 0:
                calls = calls.iloc[(calls['exdate'] - position['exdate']).abs().argsort()[:1]]
                puts = puts.iloc[(puts['exdate'] - position['exdate']).abs().argsort()[:1]]
                
                if len(calls) > 0 and len(puts) > 0:
                    current_call_price = calls['market_price'].iloc[0]
                    current_put_price = puts['market_price'].iloc[0]
                    current_straddle_value = (current_call_price + current_put_price) * 100 * position['num_straddles']
                    
                    current_dte = max(1, position['entry_dte'] - days_held)
                    current_maturity = current_dte / 365
                    
                    iv_calc = ImpliedVolSurface(
                        spot_price=spot_price,
                        risk_free_rate=0.02,
                        dividend_yield=0.02,
                        verbose=False
                    )
                    current_iv = get_iv_for_option(iv_calc, position['strike'], current_maturity,
                                                   current_call_price, current_put_price)
        
        if current_straddle_value is None or current_iv is None:
            position['dte_remaining'] = max(1, position['entry_dte'] - days_held)
            return position, None
        
        # Track IV change
        iv_change = current_iv - position['entry_iv']
        iv_change_pct = iv_change * 100
        
        # Daily delta hedging
        current_maturity = max(0.01, (position['entry_dte'] - days_held) / 365)
        hedge_result = hedger.calculate_hedge_position(
            spot_price, position['strike'], current_maturity, current_iv
        )
        
        # Calculate greeks
        d1 = (np.log(spot_price / position['strike']) + 
              (0.02 - 0.02 + 0.5 * current_iv**2) * current_maturity) / \
             (current_iv * np.sqrt(current_maturity))
        
        call_delta = np.exp(-0.02 * current_maturity) * norm.cdf(d1)
        put_delta = -np.exp(-0.02 * current_maturity) * norm.cdf(-d1)
        straddle_delta = call_delta + put_delta
        
        single_gamma = np.exp(-0.02 * current_maturity) * norm.pdf(d1) / \
                       (spot_price * current_iv * np.sqrt(current_maturity))
        straddle_gamma = 2 * single_gamma
        
        prev_spot = position.get('prev_spot', position['entry_spot'])
        spot_change = spot_price - prev_spot
        
        prev_delta = position.get('prev_delta', 0)
        prev_gamma = position.get('prev_gamma', straddle_gamma)
        
        # P&L calculations
        delta_hedge_pnl = -prev_delta * spot_change * position['num_straddles']
        gamma_pnl = 0.5 * prev_gamma * (spot_change ** 2) * position['num_straddles']
        theta_pnl = hedger.calculate_theta_pnl(position['strike'], current_maturity,
                                               current_iv, days=1) * position['num_straddles']
        
        if 'prev_hedge_units' in position:
            hedge_adjustment = abs(hedge_result['hedge_units'] - position['prev_hedge_units'])
            hedge_rebalance_cost = hedge_adjustment * spot_price * 0.001
        else:
            hedge_rebalance_cost = 0
        
        daily_hedge_pnl = delta_hedge_pnl + gamma_pnl + theta_pnl - hedge_rebalance_cost
        
        # Accumulate
        position['cumulative_hedge_pnl'] = position.get('cumulative_hedge_pnl', 0) + daily_hedge_pnl
        position['cumulative_gamma_pnl'] = position.get('cumulative_gamma_pnl', 0) + gamma_pnl
        position['cumulative_theta_pnl'] = position.get('cumulative_theta_pnl', 0) + theta_pnl
        position['cumulative_delta_pnl'] = position.get('cumulative_delta_pnl', 0) + delta_hedge_pnl
        position['total_hedge_cost'] = position.get('total_hedge_cost', 0) + hedge_rebalance_cost
        
        # Update tracking
        position['prev_hedge_units'] = hedge_result['hedge_units']
        position['prev_spot'] = spot_price
        position['prev_delta'] = straddle_delta
        position['prev_gamma'] = straddle_gamma
        
        # Exit logic (different for long vs short)
        should_exit = False
        exit_reason = None
        
        if position_type == 'LONG':
            # Long vol exit rules
            if days_held >= 14:
                should_exit = True
                exit_reason = "Day 14 mandatory exit"
            elif days_held >= 10:
                if iv_change_pct >= 2.0:
                    should_exit = True
                    exit_reason = f"Day {days_held} take profit (IV +{iv_change_pct:.1f}%)"
                elif iv_change_pct <= -1.0:
                    should_exit = True
                    exit_reason = f"Day {days_held} stop loss (IV {iv_change_pct:.1f}%)"
            elif days_held >= 7:
                if iv_change_pct >= 3.0:
                    should_exit = True
                    exit_reason = f"Day {days_held} take profit (IV +{iv_change_pct:.1f}%)"
                elif iv_change_pct <= -2.0:
                    should_exit = True
                    exit_reason = f"Day {days_held} stop loss (IV {iv_change_pct:.1f}%)"
        
        else:  # SHORT
            # Short vol exit rules (OPPOSITE signs!) - TIGHTER STOPS for tail risk
            if days_held >= 14:
                should_exit = True
                exit_reason = "Day 14 mandatory exit"
            elif days_held >= 10:
                if iv_change_pct <= -2.0:  # IV dropped = good for short
                    should_exit = True
                    exit_reason = f"Day {days_held} take profit (IV {iv_change_pct:.1f}%)"
                elif iv_change_pct >= 1.0:  # IV rose = bad for short
                    should_exit = True
                    exit_reason = f"Day {days_held} stop loss (IV +{iv_change_pct:.1f}%)"
            elif days_held >= 7:
                if iv_change_pct <= -3.0:
                    should_exit = True
                    exit_reason = f"Day {days_held} take profit (IV {iv_change_pct:.1f}%)"
                elif iv_change_pct >= 1.5:  # TIGHTER: was 2.0, now 1.5%
                    should_exit = True
                    exit_reason = f"Day {days_held} stop loss (IV +{iv_change_pct:.1f}%)"
        
        if should_exit:
            # Calculate final P&L
            if position_type == 'LONG':
                option_pnl = current_straddle_value - position['entry_cost']
            else:  # SHORT
                option_pnl = position['entry_cost'] - current_straddle_value
            
            cumulative_hedge_pnl = position.get('cumulative_hedge_pnl', 0)
            
            exit_cost = tc_calc.calculate(
                price=current_straddle_value / (100 * position['num_straddles']),
                contracts=position['num_straddles'] * 2,
                ticker=ticker_upper
            )
            
            entry_cost_tc = position.get('entry_tc', 0)
            total_hedge_rebalance_costs = position.get('total_hedge_cost', 0)
            
            gross_pnl = option_pnl + cumulative_hedge_pnl
            net_pnl = gross_pnl - entry_cost_tc - exit_cost
            
            trade_info = {
                'position_type': position_type,
                'entry_date': position['entry_date'],
                'exit_date': current_date,
                'days_held': days_held,
                'strike': position['strike'],
                'entry_iv': position['entry_iv'],
                'exit_iv': current_iv,
                'iv_change': iv_change,
                'iv_change_pct': iv_change_pct,
                'entry_cost': position['entry_cost'],
                'exit_value': current_straddle_value,
                'option_pnl': option_pnl,
                'gamma_pnl': position.get('cumulative_gamma_pnl', 0),
                'theta_pnl': position.get('cumulative_theta_pnl', 0),
                'delta_hedge_pnl': position.get('cumulative_delta_pnl', 0),
                'hedge_pnl': cumulative_hedge_pnl,
                'gross_pnl': gross_pnl,
                'entry_tc': entry_cost_tc,
                'exit_cost': exit_cost,
                'hedge_rebalance_costs': total_hedge_rebalance_costs,
                'net_pnl': net_pnl,
                'exit_reason': exit_reason,
                'garch_forecast': position['garch_forecast'],
                'num_straddles': position['num_straddles']
            }
            
            return None, trade_info
        else:
            position['dte_remaining'] = max(1, position['entry_dte'] - days_held)
            position['current_iv'] = current_iv
            position['current_value'] = current_straddle_value
            return position, None
    
    # Main backtest loop
    for i, current_date in enumerate(trading_dates):
        
        if i % 20 == 0:
            progress_pct = (i / len(trading_dates)) * 100
            print(f"\r  Progress: {i}/{len(trading_dates)} ({progress_pct:.1f}%) - Long: {long_entries}/{long_exits}, Short: {short_entries}/{short_exits}, Both signals: {both_signals_count}", end="", flush=True)
        
        if i < train_window:
            continue
        
        if current_date not in stock_data.index:
            continue
        
        spot_price = stock_data.loc[current_date, 'prc']
        
        # Refit GARCH periodically
        days_since_fit = (current_date - last_garch_fit_date).days if last_garch_fit_date else refit_frequency + 1
        
        if days_since_fit >= refit_frequency or garch_forecast is None:
            historical_data = stock_data.loc[:current_date].iloc[-train_window-1:-1]
            log_returns = parse_data(historical_data, price_col='prc')
            
            try:
                _, sigma_forecast_base = garch_modelling(log_returns, DE_MEAN, MODEL, DISTRIBUTION, validity_checks=False)
                
                # Sanity check for EGARCH explosions
                if sigma_forecast_base > 2.0 or sigma_forecast_base < 0.05:
                    sigma_forecast_base = log_returns.std() * np.sqrt(252)
                    if verbose:
                        print(f"\n  [!] Warning: EGARCH forecast {sigma_forecast_base:.2%} out of bounds, using fallback")
                
                sigma_forecast_ensemble = ensemble_vol_forecast(log_returns, sigma_forecast_base)
                vol_risk_premium = calculate_vol_risk_premium(forecast_history, market_iv_history)
                garch_forecast = sigma_forecast_ensemble + vol_risk_premium
                
                last_garch_fit_date = current_date
                
            except Exception as e:
                if verbose:
                    print(f"\n  [{current_date.date()}] GARCH fit failed: {e}")
                continue
        
        # Manage existing positions
        hedger = DeltaHedger(spot_price, risk_free_rate=0.02, dividend_yield=0.02)
        
        if long_position is not None:
            long_position, long_exit_info = manage_position(
                long_position, 'LONG', current_date, spot_price, options_data,
                hedger, tc_calc, ticker_upper
            )
            if long_exit_info:
                long_trades.append(long_exit_info)
                portfolio_value += long_exit_info['net_pnl']
                long_exits += 1
                if verbose:
                    print(f"\n  [{current_date.date()}] LONG EXIT: {long_exit_info['exit_reason']} | P&L: ${long_exit_info['net_pnl']:.2f}")
        
        if short_position is not None:
            short_position, short_exit_info = manage_position(
                short_position, 'SHORT', current_date, spot_price, options_data,
                hedger, tc_calc, ticker_upper
            )
            if short_exit_info:
                short_trades.append(short_exit_info)
                portfolio_value += short_exit_info['net_pnl']
                short_exits += 1
                if verbose:
                    print(f"\n  [{current_date.date()}] SHORT EXIT: {short_exit_info['exit_reason']} | P&L: ${short_exit_info['net_pnl']:.2f}")
        
        # Check for new entry signals (only if no position in that direction)
        strikes, maturities, market_prices = get_iv_surface_for_date(options_data, current_date, spot_price)
        
        if strikes is None:
            continue
        
        try:
            iv_calc = ImpliedVolSurface(
                spot_price=spot_price,
                risk_free_rate=0.02,
                dividend_yield=0.02,
                strikes=strikes,
                maturities=maturities,
                market_prices=market_prices,
                verbose=False
            )
            
            if iv_calc.iv_surface is None:
                continue
            
            atm_option = get_atm_option_for_dte(options_data, current_date, spot_price,
                                                target_dte=45, dte_range=(30, 45))
            
            if atm_option is None:
                continue
            
            market_iv = get_iv_for_option(iv_calc, atm_option['strike'], atm_option['maturity'],
                                         atm_option['call_price'], atm_option['put_price'])
            
            if np.isnan(market_iv):
                continue
            
            iv_history.append(market_iv)
            if len(iv_history) > 252:
                iv_history = iv_history[-252:]
            
            if garch_forecast is not None:
                forecast_history.append(garch_forecast)
                market_iv_history.append(market_iv)
                if len(forecast_history) > 252:
                    forecast_history = forecast_history[-252:]
                    market_iv_history = market_iv_history[-252:]
            
            # Generate signals
            long_signal = garch_forecast > market_iv + 0.02  # GARCH > Market
            short_signal = market_iv > garch_forecast + 0.02  # Market > GARCH
            
            # Check IV percentile
            if len(iv_history) >= 20:
                iv_percentile = (np.array(iv_history) < market_iv).sum() / len(iv_history) * 100
                iv_cheap = iv_percentile < 25
            else:
                iv_cheap = True
                iv_percentile = 50
            
            # Check regime blocker
            is_blocked = False
            if regime_blocker is not None:
                try:
                    is_blocked = regime_blocker.isBlocked(current_date.strftime('%Y-%m-%d'))
                except:
                    is_blocked = False
            
            # Determine capital allocation
            if long_signal and short_signal:
                both_signals_count += 1
            
            # LONG ENTRY
            if long_signal and long_position is None and not is_blocked and iv_cheap:
                if allocation_mode == 'split' and short_signal:
                    long_size = position_size / 2
                elif allocation_mode == 'priority' and short_signal:
                    long_size = 0  # Skip long, prioritize short
                else:
                    long_size = position_size
                
                # Apply dynamic position sizing based on IV regime
                long_size = calculate_dynamic_position_size(long_size, market_iv, iv_history, 'LONG')
                
                if long_size > 0:
                    straddle_price_per_unit = atm_option['straddle_price'] * 100
                    num_straddles = max(1, int(long_size / straddle_price_per_unit))
                    entry_cost = straddle_price_per_unit * num_straddles
                    
                    entry_tc = tc_calc.calculate(
                        price=atm_option['straddle_price'],
                        contracts=num_straddles * 2,
                        ticker=ticker_upper
                    )
                    
                    hedge_result = hedger.calculate_hedge_position(
                        spot_price, atm_option['strike'], atm_option['maturity'], market_iv
                    )
                    
                    straddle_delta = hedger.calculate_straddle_delta(
                        spot_price, atm_option['strike'], atm_option['maturity'],
                        0.02, market_iv, 0.02
                    )
                    
                    d1 = (np.log(spot_price / atm_option['strike']) +
                          (0.02 - 0.02 + 0.5 * market_iv**2) * atm_option['maturity']) / \
                         (market_iv * np.sqrt(atm_option['maturity']))
                    
                    straddle_gamma = np.exp(-0.02 * atm_option['maturity']) * norm.pdf(d1) / \
                                     (spot_price * market_iv * np.sqrt(atm_option['maturity'])) * 2
                    
                    long_position = {
                        'entry_date': current_date,
                        'strike': atm_option['strike'],
                        'entry_dte': atm_option['dte'],
                        'dte_remaining': atm_option['dte'],
                        'exdate': atm_option['exdate'],
                        'entry_iv': market_iv,
                        'garch_forecast': garch_forecast,
                        'entry_cost': entry_cost,
                        'entry_tc': entry_tc,
                        'num_straddles': num_straddles,
                        'entry_spot': spot_price,
                        'prev_spot': spot_price,
                        'prev_hedge_units': hedge_result['hedge_units'],
                        'prev_delta': straddle_delta,
                        'prev_gamma': straddle_gamma,
                        'cumulative_hedge_pnl': 0,
                        'cumulative_gamma_pnl': 0,
                        'cumulative_theta_pnl': 0,
                        'cumulative_delta_pnl': 0,
                        'total_hedge_cost': 0
                    }
                    
                    long_entries += 1
                    if verbose:
                        print(f"\n  [{current_date.date()}] LONG ENTRY: Strike ${atm_option['strike']:.0f}, DTE {atm_option['dte']}, IV {market_iv:.1%}, GARCH {garch_forecast:.1%}")
            
            # SHORT ENTRY
            if short_signal and short_position is None and not is_blocked:
                if allocation_mode == 'split' and long_signal:
                    short_size = position_size / 2
                else:
                    short_size = position_size
                
                # Apply dynamic position sizing based on IV regime (CRITICAL for tail risk)
                short_size = calculate_dynamic_position_size(short_size, market_iv, iv_history, 'SHORT')
                
                if short_size > 0:
                    straddle_price_per_unit = atm_option['straddle_price'] * 100
                    num_straddles = max(1, int(short_size / straddle_price_per_unit))
                    entry_cost = straddle_price_per_unit * num_straddles
                    
                    entry_tc = tc_calc.calculate(
                        price=atm_option['straddle_price'],
                        contracts=num_straddles * 2,
                        ticker=ticker_upper
                    )
                    
                    hedge_result = hedger.calculate_hedge_position(
                        spot_price, atm_option['strike'], atm_option['maturity'], market_iv
                    )
                    
                    straddle_delta = hedger.calculate_straddle_delta(
                        spot_price, atm_option['strike'], atm_option['maturity'],
                        0.02, market_iv, 0.02
                    )
                    
                    d1 = (np.log(spot_price / atm_option['strike']) +
                          (0.02 - 0.02 + 0.5 * market_iv**2) * atm_option['maturity']) / \
                         (market_iv * np.sqrt(atm_option['maturity']))
                    
                    straddle_gamma = np.exp(-0.02 * atm_option['maturity']) * norm.pdf(d1) / \
                                     (spot_price * market_iv * np.sqrt(atm_option['maturity'])) * 2
                    
                    short_position = {
                        'entry_date': current_date,
                        'strike': atm_option['strike'],
                        'entry_dte': atm_option['dte'],
                        'dte_remaining': atm_option['dte'],
                        'exdate': atm_option['exdate'],
                        'entry_iv': market_iv,
                        'garch_forecast': garch_forecast,
                        'entry_cost': entry_cost,
                        'entry_tc': entry_tc,
                        'num_straddles': num_straddles,
                        'entry_spot': spot_price,
                        'prev_spot': spot_price,
                        'prev_hedge_units': hedge_result['hedge_units'],
                        'prev_delta': straddle_delta,
                        'prev_gamma': straddle_gamma,
                        'cumulative_hedge_pnl': 0,
                        'cumulative_gamma_pnl': 0,
                        'cumulative_theta_pnl': 0,
                        'cumulative_delta_pnl': 0,
                        'total_hedge_cost': 0
                    }
                    
                    short_entries += 1
                    if verbose:
                        print(f"\n  [{current_date.date()}] SHORT ENTRY: Strike ${atm_option['strike']:.0f}, DTE {atm_option['dte']}, IV {market_iv:.1%}, GARCH {garch_forecast:.1%}")
            
            # Record daily observation
            results.append({
                'date': current_date,
                'spot_price': spot_price,
                'forecast_iv': garch_forecast,
                'market_iv': market_iv,
                'long_signal': long_signal,
                'short_signal': short_signal,
                'both_signals': long_signal and short_signal,
                'long_position_active': long_position is not None,
                'short_position_active': short_position is not None,
                'portfolio_value': portfolio_value
            })
            processed_count += 1
            
        except Exception as e:
            if verbose:
                print(f"\n  [{current_date.date()}] Error: {str(e)}")
            continue
    
    print(f"\n\n[4/6] Processing results...")
    print(f"  Long trades: {long_entries} entered, {long_exits} exited")
    print(f"  Short trades: {short_entries} entered, {short_exits} exited")
    print(f"  Both signals appeared: {both_signals_count} times")
    
    df_results = pd.DataFrame(results)
    df_long_trades = pd.DataFrame(long_trades) if long_trades else pd.DataFrame()
    df_short_trades = pd.DataFrame(short_trades) if short_trades else pd.DataFrame()
    
    # Combine all trades for overall stats
    all_trades = long_trades + short_trades
    df_all_trades = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
    
    # Store in attrs for later access
    df_results.attrs['long_trades'] = df_long_trades
    df_results.attrs['short_trades'] = df_short_trades
    df_results.attrs['all_trades'] = df_all_trades
    
    print(f"\n[5/6] Performance Analysis...")
    
    if len(df_all_trades) > 0:
        print(f"\n  === COMBINED STRATEGY ===")
        print(f"    Total Trades:         {len(df_all_trades)}")
        print(f"    Long Trades:          {len(df_long_trades)}")
        print(f"    Short Trades:         {len(df_short_trades)}")
        print(f"    Win Rate:             {(df_all_trades['net_pnl'] > 0).mean()*100:.1f}%")
        print(f"    Total Net P&L:        ${df_all_trades['net_pnl'].sum():,.2f}")
        print(f"    Mean Trade P&L:       ${df_all_trades['net_pnl'].mean():,.2f}")
        print(f"    Best Trade:           ${df_all_trades['net_pnl'].max():,.2f}")
        print(f"    Worst Trade:          ${df_all_trades['net_pnl'].min():,.2f}")
        
        if len(df_long_trades) > 0:
            print(f"\n  === LONG VOL COMPONENT ===")
            print(f"    Trades:               {len(df_long_trades)}")
            print(f"    Win Rate:             {(df_long_trades['net_pnl'] > 0).mean()*100:.1f}%")
            print(f"    Total P&L:            ${df_long_trades['net_pnl'].sum():,.2f}")
            print(f"    Avg Trade:            ${df_long_trades['net_pnl'].mean():,.2f}")
            print(f"    Option P&L:           ${df_long_trades['option_pnl'].sum():,.2f}")
            print(f"    Theta P&L:            ${df_long_trades['theta_pnl'].sum():,.2f}")
        
        if len(df_short_trades) > 0:
            print(f"\n  === SHORT VOL COMPONENT ===")
            print(f"    Trades:               {len(df_short_trades)}")
            print(f"    Win Rate:             {(df_short_trades['net_pnl'] > 0).mean()*100:.1f}%")
            print(f"    Total P&L:            ${df_short_trades['net_pnl'].sum():,.2f}")
            print(f"    Avg Trade:            ${df_short_trades['net_pnl'].mean():,.2f}")
            print(f"    Option P&L:           ${df_short_trades['option_pnl'].sum():,.2f}")
            print(f"    Theta P&L:            ${df_short_trades['theta_pnl'].sum():,.2f}")
        
        # Calculate correlation if both strategies traded
        if len(df_long_trades) > 5 and len(df_short_trades) > 5:
            print(f"\n  === CORRELATION ANALYSIS ===")
            
            # Daily P&L streams
            long_daily_pnl = df_results['date'].map(
                df_long_trades.set_index('exit_date')['net_pnl']
            ).fillna(0)
            short_daily_pnl = df_results['date'].map(
                df_short_trades.set_index('exit_date')['net_pnl']
            ).fillna(0)
            
            # Only calculate correlation for days with actual P&L
            mask = (long_daily_pnl != 0) | (short_daily_pnl != 0)
            if mask.sum() > 5:
                correlation = np.corrcoef(long_daily_pnl[mask], short_daily_pnl[mask])[0, 1]
                print(f"    Long/Short Correlation: {correlation:.3f}")
                
                if correlation < -0.2:
                    print(f"    ✓ NEGATIVE correlation - strategies hedge each other!")
                elif correlation < 0.2:
                    print(f"    ✓ LOW correlation - good diversification")
                else:
                    print(f"    ⚠ POSITIVE correlation - limited diversification benefit")
    
    print(f"\n  Final Portfolio Value: ${portfolio_value:,.2f}")
    print(f"  Total Return:          {(portfolio_value - starting_capital) / starting_capital * 100:.2f}%")
    print(f"  Annualized Return:     {((portfolio_value / starting_capital) ** (1/4.5) - 1) * 100:.2f}%")
    
    # Calculate Sharpe Ratio and Max Drawdown
    if len(df_results) > 0:
        portfolio_series = pd.Series(df_results['portfolio_value'].values)
        returns = portfolio_series.pct_change().dropna()
        
        if len(returns) > 0:
            # Sharpe Ratio
            annual_return = returns.mean() * 252
            annual_vol = returns.std() * np.sqrt(252)
            risk_free_rate = 0.03  # 3% annual
            sharpe_ratio = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
            
            # Max Drawdown
            running_max = portfolio_series.cummax()
            drawdown = (portfolio_series - running_max) / running_max
            max_drawdown = drawdown.min()
            max_drawdown_dollars = (running_max - portfolio_series).max()
            
            # Sortino Ratio (downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_vol = downside_returns.std() * np.sqrt(252)
                sortino_ratio = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
            else:
                sortino_ratio = np.inf if annual_return > risk_free_rate else 0
            
            print(f"\n  === RISK-ADJUSTED METRICS ===")
            print(f"    Sharpe Ratio:          {sharpe_ratio:.3f}")
            print(f"    Sortino Ratio:         {sortino_ratio:.3f}")
            print(f"    Annual Volatility:     {annual_vol*100:.2f}%")
            print(f"    Max Drawdown:          {max_drawdown*100:.2f}%")
            print(f"    Max Drawdown ($):      ${max_drawdown_dollars:,.2f}")
            
            # Store metrics in results attrs
            df_results.attrs['sharpe_ratio'] = sharpe_ratio
            df_results.attrs['sortino_ratio'] = sortino_ratio
            df_results.attrs['max_drawdown'] = max_drawdown
            df_results.attrs['max_drawdown_dollars'] = max_drawdown_dollars
            df_results.attrs['annual_volatility'] = annual_vol
    
    print("\n  Backtest complete!")
    print("="*80)
    
    return df_results

def plot_combined_results(results_df, save_path='backtest_combined_analysis.png'):
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(6, 2, hspace=0.35, wspace=0.3)
    
    # Prepare data
    portfolio_values = results_df['portfolio_value'].values
    starting_value = portfolio_values[0]
    running_max = pd.Series(portfolio_values).cummax()
    drawdown = (portfolio_values - running_max) / running_max * 100
    dates = pd.to_datetime(results_df['date'])
    
    # Get trade logs
    long_trades = results_df.attrs.get('long_trades', pd.DataFrame())
    short_trades = results_df.attrs.get('short_trades', pd.DataFrame())
    all_trades = results_df.attrs.get('all_trades', pd.DataFrame())
    
    # 1. Portfolio Value
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, portfolio_values, linewidth=2, color='#2E86AB', label='Portfolio Value')
    ax1.fill_between(dates, starting_value, portfolio_values, alpha=0.3, color='#2E86AB')
    ax1.axhline(y=starting_value, color='red', linestyle='--', alpha=0.5, label='Starting Capital')
    
    # Mark long and short entries
    if len(long_trades) > 0:
        long_entries = long_trades['entry_date'].values
        for entry_date in long_entries:
            if entry_date in dates.values:
                idx = dates[dates == entry_date].index[0]
                ax1.scatter(entry_date, portfolio_values[idx], color='green', s=50, alpha=0.6, marker='^')
    
    if len(short_trades) > 0:
        short_entries = short_trades['entry_date'].values
        for entry_date in short_entries:
            if entry_date in dates.values:
                idx = dates[dates == entry_date].index[0]
                ax1.scatter(entry_date, portfolio_values[idx], color='red', s=50, alpha=0.6, marker='v')
    
    ax1.set_title('Portfolio Value Over Time (Combined Long/Short Strategy)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=11)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(dates, 0, drawdown, color='#A23B72', alpha=0.7)
    ax2.plot(dates, drawdown, linewidth=1.5, color='#A23B72')
    ax2.set_title('Drawdown (%)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # 3. Trade P&L Distribution by Type
    ax3 = fig.add_subplot(gs[2, 0])
    if len(all_trades) > 0:
        long_pnls = long_trades['net_pnl'].values if len(long_trades) > 0 else []
        short_pnls = short_trades['net_pnl'].values if len(short_trades) > 0 else []
        
        ax3.hist([long_pnls, short_pnls], bins=20, alpha=0.7, edgecolor='black',
                label=['Long Vol', 'Short Vol'], color=['#6A994E', '#E63946'])
        ax3.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Break-even')
        ax3.set_title('Trade P&L Distribution by Strategy', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Trade P&L ($)', fontsize=10)
        ax3.set_ylabel('Frequency', fontsize=10)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Win Rate Comparison
    ax4 = fig.add_subplot(gs[2, 1])
    if len(long_trades) > 0 and len(short_trades) > 0:
        long_win_rate = (long_trades['net_pnl'] > 0).mean() * 100
        short_win_rate = (short_trades['net_pnl'] > 0).mean() * 100
        combined_win_rate = (all_trades['net_pnl'] > 0).mean() * 100
        
        strategies = ['Long Vol', 'Short Vol', 'Combined']
        win_rates = [long_win_rate, short_win_rate, combined_win_rate]
        colors = ['#6A994E', '#E63946', '#2E86AB']
        
        bars = ax4.bar(strategies, win_rates, color=colors, alpha=0.7, edgecolor='black')
        ax4.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
        ax4.set_title('Win Rate by Strategy', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Win Rate (%)', fontsize=10)
        ax4.set_ylim(0, 100)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, rate in zip(bars, win_rates):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 5. Cumulative P&L by Strategy
    ax5 = fig.add_subplot(gs[3, 0])
    if len(long_trades) > 0 and len(short_trades) > 0:
        long_cum_pnl = long_trades['net_pnl'].cumsum()
        short_cum_pnl = short_trades['net_pnl'].cumsum()
        
        ax5.plot(range(1, len(long_cum_pnl)+1), long_cum_pnl, linewidth=2, 
                color='#6A994E', marker='o', markersize=3, label='Long Vol', alpha=0.8)
        ax5.plot(range(1, len(short_cum_pnl)+1), short_cum_pnl, linewidth=2,
                color='#E63946', marker='s', markersize=3, label='Short Vol', alpha=0.8)
        ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax5.set_title('Cumulative P&L by Strategy', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Trade Number', fontsize=10)
        ax5.set_ylabel('Cumulative P&L ($)', fontsize=10)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. Trade Count Over Time
    ax6 = fig.add_subplot(gs[3, 1])
    if len(all_trades) > 0:
        all_trades_df = pd.DataFrame(all_trades)
        all_trades_df['entry_date'] = pd.to_datetime(all_trades_df['entry_date'])
        all_trades_df['year_month'] = all_trades_df['entry_date'].dt.to_period('M')
        
        trade_counts = all_trades_df.groupby(['year_month', 'position_type']).size().unstack(fill_value=0)
        
        trade_counts.plot(kind='bar', stacked=True, ax=ax6, color=['#6A994E', '#E63946'],
                         alpha=0.7, edgecolor='black')
        ax6.set_title('Trade Count by Month and Type', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Month', fontsize=10)
        ax6.set_ylabel('Number of Trades', fontsize=10)
        ax6.legend(title='Strategy', labels=['Long Vol', 'Short Vol'])
        ax6.grid(True, alpha=0.3, axis='y')
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 7. IV Change Distribution by Strategy
    ax7 = fig.add_subplot(gs[4, 0])
    if len(long_trades) > 0 and len(short_trades) > 0:
        long_iv_changes = long_trades['iv_change_pct'].values
        short_iv_changes = short_trades['iv_change_pct'].values
        
        ax7.hist([long_iv_changes, short_iv_changes], bins=20, alpha=0.7, edgecolor='black',
                label=['Long Vol', 'Short Vol'], color=['#6A994E', '#E63946'])
        ax7.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax7.set_title('IV Change Distribution by Strategy', fontsize=12, fontweight='bold')
        ax7.set_xlabel('IV Change (%)', fontsize=10)
        ax7.set_ylabel('Frequency', fontsize=10)
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    # 8. Holding Period Distribution
    ax8 = fig.add_subplot(gs[4, 1])
    if len(all_trades) > 0:
        long_days = long_trades['days_held'].values if len(long_trades) > 0 else []
        short_days = short_trades['days_held'].values if len(short_trades) > 0 else []
        
        ax8.hist([long_days, short_days], bins=range(1, 16), alpha=0.7, edgecolor='black',
                label=['Long Vol', 'Short Vol'], color=['#6A994E', '#E63946'], align='left')
        ax8.axvline(x=7, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Day 7 check')
        ax8.axvline(x=10, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='Day 10 check')
        ax8.axvline(x=14, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Day 14 exit')
        ax8.set_title('Holding Period Distribution', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Days Held', fontsize=10)
        ax8.set_ylabel('Frequency', fontsize=10)
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    
    # 9. Forecast vs Market IV
    ax9 = fig.add_subplot(gs[5, 0])
    if 'forecast_iv' in results_df.columns and 'market_iv' in results_df.columns:
        ax9.plot(dates, results_df['forecast_iv'] * 100, label='GARCH Forecast',
                linewidth=2, color='#2E86AB', alpha=0.8)
        market_iv = results_df['market_iv'].ffill() * 100
        ax9.plot(dates, market_iv, label='Market IV',
                linewidth=2, color='#F18F01', alpha=0.8)
        
        # Shade regions where long/short signals active
        if 'long_signal' in results_df.columns:
            long_signal_mask = results_df['long_signal'].fillna(False)
            ax9.fill_between(dates, 0, 100, where=long_signal_mask, alpha=0.1, 
                           color='green', label='Long Signal Active')
        if 'short_signal' in results_df.columns:
            short_signal_mask = results_df['short_signal'].fillna(False)
            ax9.fill_between(dates, 0, 100, where=short_signal_mask, alpha=0.1,
                           color='red', label='Short Signal Active')
        
        ax9.set_title('Forecast vs Market IV with Signals', fontsize=12, fontweight='bold')
        ax9.set_ylabel('Implied Volatility (%)', fontsize=10)
        ax9.set_ylim(0, max(market_iv.max(), results_df['forecast_iv'].max() * 100) * 1.1)
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        ax9.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # 10. Performance Metrics Summary
    ax10 = fig.add_subplot(gs[5, 1])
    ax10.axis('off')
    
    # Get metrics
    sharpe = results_df.attrs.get('sharpe_ratio', 0)
    sortino = results_df.attrs.get('sortino_ratio', 0)
    max_dd = results_df.attrs.get('max_drawdown', 0)
    
    total_return = (portfolio_values[-1] - starting_value) / starting_value
    annual_return = (portfolio_values[-1] / starting_value) ** (1/4.5) - 1
    
    # Create metrics text
    metrics_text = f"""
    PERFORMANCE SUMMARY
    
    Total Return:        {total_return*100:>6.2f}%
    Annualized Return:   {annual_return*100:>6.2f}%
    
    Sharpe Ratio:        {sharpe:>6.3f}
    Sortino Ratio:       {sortino:>6.3f}
    Max Drawdown:        {max_dd*100:>6.2f}%
    
    Total Trades:        {len(all_trades):>6}
    Long Trades:         {len(long_trades):>6}
    Short Trades:        {len(short_trades):>6}
    
    Overall Win Rate:    {(all_trades['net_pnl'] > 0).mean()*100:>6.1f}%
    Long Win Rate:       {(long_trades['net_pnl'] > 0).mean()*100 if len(long_trades) > 0 else 0:>6.1f}%
    Short Win Rate:      {(short_trades['net_pnl'] > 0).mean()*100 if len(short_trades) > 0 else 0:>6.1f}%
    
    Correlation:         {results_df.attrs.get('correlation', 0):>6.3f}
    """
    
    ax10.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Combined Long/Short Vol Arbitrage Strategy - Comprehensive Analysis',
                fontsize=16, fontweight='bold', y=0.998)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n  * Charts saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    else:
        ticker = 'AAPL'
        print(f"No ticker specified, using default: {ticker}")
    
    # Run combined strategy
    results = combined_vol_arbitrage_backtest(
        ticker=ticker,
        train_window=126,
        refit_frequency=21,
        starting_capital=100000,
        position_size=8000,
        allocation_mode='split',  # Options: 'split', 'separate', 'priority'
        use_regime_blocker=True,
        start_date=None,
        end_date=None,
        verbose=False
    )
    
    if results is not None:
        # Save results
        results.to_csv(f"backtest_combined_{ticker.lower()}.csv", index=False)
        print(f"\nResults saved to backtest_combined_{ticker.lower()}.csv")
        
        # Save trade logs
        long_trades = results.attrs.get('long_trades', pd.DataFrame())
        short_trades = results.attrs.get('short_trades', pd.DataFrame())
        all_trades = results.attrs.get('all_trades', pd.DataFrame())
        
        if len(long_trades) > 0:
            long_trades.to_csv(f"trades_long_{ticker.lower()}.csv", index=False)
            print(f"Long trades saved to trades_long_{ticker.lower()}.csv")
        
        if len(short_trades) > 0:
            short_trades.to_csv(f"trades_short_{ticker.lower()}.csv", index=False)
            print(f"Short trades saved to trades_short_{ticker.lower()}.csv")
        
        if len(all_trades) > 0:
            all_trades.to_csv(f"trades_combined_{ticker.lower()}.csv", index=False)
            print(f"Combined trades saved to trades_combined_{ticker.lower()}.csv")
        
        # Generate comprehensive plots
        print("\nGenerating performance charts...")
        plot_combined_results(results, save_path=f'backtest_combined_analysis_{ticker.lower()}.png')