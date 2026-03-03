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
    
    # Use mid_price if available, otherwise calculate from bid/offer
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
    
    # Filter for calls only (for IV extraction)
    calls = date_options[date_options['cp_flag'] == 'C'].copy()
    
    if len(calls) < 10:  # Need minimum data
        return None, None, None
    
    # Filter out only the most problematic options that cause numerical issues
    # 1. Remove expired or very short-dated options (< 2 days) - these cause div/0
    calls = calls[calls['days_to_expiry'] >= 2].copy()
    
    # 2. Remove options with zero or negative prices (invalid data)
    calls = calls[calls['market_price'] > 0.01].copy()
    
    # 3. Remove extreme OTM options to avoid numerical instability (strike > 3x spot)
    calls = calls[calls['strike_price'] < spot_price * 3.0].copy()
    
    if len(calls) < 10:  # Recheck after filtering
        return None, None, None
    
    strikes = sorted(calls['strike_price'].unique())
    maturities = sorted(calls['maturity'].unique())
    
    # Build price matrix
    market_prices = np.full((len(strikes), len(maturities)), np.nan)
    
    for i, K in enumerate(strikes):
        for j, T in enumerate(maturities):
            price_data = calls[(calls['strike_price'] == K) & (calls['maturity'] == T)]
            if len(price_data) > 0:
                market_prices[i, j] = price_data['market_price'].iloc[0]
    
    # Filter out strikes/maturities with too much missing data
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
    
    # Find ATM strike (closest to spot)
    strikes = target_options['strike_price'].unique()
    atm_strike = min(strikes, key=lambda x: abs(x - spot_price))
    
    # Get call and put at ATM strike
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


def rolling_window_backtest(ticker, train_window=252, refit_frequency=21, 
                            starting_capital=100000, position_size=8000,
                            start_date=None, end_date=None,
                            use_regime_blocker=True, verbose=False):
    
    print("="*80)
    print("\n  Strategy Rules:")
    print("    Entry: GARCH > Market IV + 4%, VIX < 25th pctl, 30-45 DTE, no stress")
    print("    Position: Buy ATM straddle, $8k size, delta hedge daily")
    print("    Exit Day 7: IV +3% take profit, IV -2% stop loss")
    print("    Exit Day 10: IV +2% take profit, IV -1% stop loss")
    print("    Exit Day 14: Mandatory exit (theta acceleration)")
    
    ticker_upper = ticker.upper()
    ticker_lower = ticker.lower()
    sp_data_path = f"{ticker_lower}_stock_prices_2020_2024.csv"
    options_data_path = f"{ticker_lower}_options_2020_2024.csv"
    
    print(f"\n[1/6] Loading data for {ticker_upper}...")
    
    stock_data = pd.read_csv(sp_data_path, parse_dates=['date'])
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data = stock_data.set_index('date').sort_index()
    
    print(f"  [i] Using {ticker_upper} stock and options data")
    
    options_data = load_options_data(options_data_path, ticker=ticker_upper)
    print(f"  * Loaded {len(stock_data)} days of {ticker_upper} stock data")
    print(f"  * Loaded {len(options_data)} option quotes for {ticker_upper}")
    
    # Get available trading dates
    trading_dates = sorted(options_data['date'].unique())
    trading_dates_set = set(trading_dates)
    
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        trading_dates = [d for d in trading_dates if d >= start_date]
        print(f"  [i] Start date filter: {start_date.date()}")
    
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        trading_dates = [d for d in trading_dates if d <= end_date]
        print(f"  [i] End date filter: {end_date.date()}")
    
    print(f"  * {len(trading_dates)} trading dates with options data")
    print(f"  [i] Training window: {train_window} days, will start backtesting from day {train_window + 1}")
    
    if len(trading_dates) <= train_window:
        print(f"  [!] Error: Not enough data! Need >{train_window} days, have {len(trading_dates)}")
        return None
    
    print(f"\n[2/6] Computing historical IV percentiles...")

    regime_blocker = None
    if use_regime_blocker:
        print(f"\n[3/6] Initializing regime blocker...")
        backtest_start_date = trading_dates[train_window]
        
        historical_stock_data = stock_data.loc[:backtest_start_date].iloc[:-1]  # Exclude backtest start date
        log_returns_historical = parse_data(historical_stock_data.reset_index(), price_col='prc')
        
        print(f"  [i] Training regime blocker on data up to {backtest_start_date.date()} ({len(log_returns_historical)} days)")
        print(f"  [i] This ensures no look-ahead bias - blocker never sees backtest period data")
        
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
            print("  * Regime blocker initialized successfully (trained on pre-backtest data only)")
        except Exception as e:
            print(f"  [!] Warning: Could not initialize regime blocker: {e}")
            print("  * Continuing without regime blocking")
            regime_blocker = None
    else:
        print(f"\n[3/6] Regime blocker disabled (use_regime_blocker=False)")
    
    results = []
    trade_log = []
    
    portfolio_value = starting_capital
    cash = starting_capital
    
    active_position = None  # Will hold position details when in a trade
    
    tc_calc = TransactionCost()
    
    iv_history = []
    
    print("\n[4/6] Starting rolling window backtest...")
    print(f"  Starting Capital: ${starting_capital:,.2f}")
    print(f"  Position Size: ${position_size:,.2f}")
    last_garch_fit_date = None
    garch_forecast = None
    
    processed_count = 0
    skipped_count = 0
    blocked_count = 0
    trades_entered = 0
    trades_exited = 0
    total_dates = len(trading_dates)
    
    # Delta hedging tracking
    total_hedge_costs = 0
    total_hedge_pnl = 0
    
    forecast_history = []
    market_iv_history = []
    
    def calculate_vol_risk_premium(forecast_hist, market_hist, min_samples=60):
        if len(forecast_hist) < min_samples:
            # Not enough data: use default equity risk premium
            return 0.03  # 3% is typical for equities
        
        # Use last 60-126 observations for rolling calibration
        lookback = min(126, len(forecast_hist))
        recent_forecasts = np.array(forecast_hist[-lookback:])
        recent_markets = np.array(market_hist[-lookback:])
        
        risk_premium = np.median(recent_markets - recent_forecasts)
        risk_premium = np.clip(risk_premium, 0.0, 0.10)
        
        return risk_premium
    
    def ensemble_vol_forecast(returns, base_forecast):
        forecasts = [base_forecast]
        
        if len(returns) >= 21:
            recent_rv = returns.tail(21).std() * np.sqrt(252)
            forecasts.append(recent_rv)
        
        if len(returns) >= 30:
            ewma_vol = returns.ewm(span=30).std().iloc[-1] * np.sqrt(252)
            forecasts.append(ewma_vol)
        
        return np.median(forecasts)
    
    for i, current_date in enumerate(trading_dates):
        
        if i % 20 == 0:
            progress_pct = (i / total_dates) * 100
            print(f"\r  Progress: {i}/{total_dates} dates ({progress_pct:.1f}%) - Trades: {trades_entered} entered, {trades_exited} exited, Blocked: {blocked_count}", end="", flush=True)
        
        if i < train_window:
            continue  # Need minimum training data
        
        # Get spot price from stock data
        if current_date not in stock_data.index:
            skipped_count += 1
            if verbose:
                print(f"  [{current_date.date()}] No stock data for this date")
            continue
        spot_price = stock_data.loc[current_date, 'prc']
        
        days_since_fit = (current_date - last_garch_fit_date).days if last_garch_fit_date else refit_frequency + 1
        
        if days_since_fit >= refit_frequency or garch_forecast is None:
            if verbose:
                print(f"\n  [{current_date.date()}] Refitting GARCH model (window: {train_window} days)...")
            
            # Get historical returns for training
            historical_data = stock_data.loc[:current_date].iloc[-train_window-1:-1]
            log_returns = parse_data(historical_data, price_col='prc')
            
            try:
                _, sigma_forecast_base = garch_modelling(log_returns, DE_MEAN, MODEL, DISTRIBUTION, validity_checks=False)
                
                sigma_forecast_ensemble = ensemble_vol_forecast(log_returns, sigma_forecast_base)
                
                vol_risk_premium = calculate_vol_risk_premium(forecast_history, market_iv_history)
                
                garch_forecast = sigma_forecast_ensemble + vol_risk_premium
                
                last_garch_fit_date = current_date
                
                if verbose:
                    print(f"\n  [{current_date.date()}] Refit Model:")
                    print(f"    Base EGARCH:     {sigma_forecast_base:.2%}")
                    print(f"    Ensemble:        {sigma_forecast_ensemble:.2%}")
                    print(f"    Risk Premium:    {vol_risk_premium:.2%}")
                    print(f"    Final Forecast:  {garch_forecast:.2%}")
                
            except Exception as e:
                if verbose:
                    print(f"\n  [{current_date.date()}] GARCH fit failed: {e}, skipping...")
                continue
        
        
        if active_position is not None:
            days_held = (current_date - active_position['entry_date']).days
            
            # Get current option prices for the position
            current_option = get_atm_option_for_dte(
                options_data, current_date, spot_price,
                target_dte=active_position['dte_remaining'],
                dte_range=(max(1, active_position['dte_remaining'] - 5), active_position['dte_remaining'] + 5)
            )
            
            # Try to find the same strike if possible
            date_options = options_data[options_data['date'] == current_date]
            strike_options = date_options[date_options['strike_price'] == active_position['strike']]
            
            current_iv = None
            current_straddle_value = None
            
            if len(strike_options) > 0:
                # Find options with similar expiry
                calls = strike_options[strike_options['cp_flag'] == 'C']
                puts = strike_options[strike_options['cp_flag'] == 'P']
                
                if len(calls) > 0 and len(puts) > 0:
                    # Get option closest to original expiry
                    calls = calls.iloc[(calls['exdate'] - active_position['exdate']).abs().argsort()[:1]]
                    puts = puts.iloc[(puts['exdate'] - active_position['exdate']).abs().argsort()[:1]]
                    
                    if len(calls) > 0 and len(puts) > 0:
                        current_call_price = calls['market_price'].iloc[0]
                        current_put_price = puts['market_price'].iloc[0]
                        current_straddle_value = (current_call_price + current_put_price) * 100 * active_position['num_straddles']
                        
                        # Calculate current IV
                        current_dte = max(1, active_position['entry_dte'] - days_held)
                        current_maturity = current_dte / 365
                        
                        iv_calc = ImpliedVolSurface(
                            spot_price=spot_price,
                            risk_free_rate=0.02,
                            dividend_yield=0.02,
                            verbose=False
                        )
                        current_iv = get_iv_for_option(iv_calc, active_position['strike'], current_maturity, 
                                                       current_call_price, current_put_price)
            
            # If we couldn't get current values, estimate them
            if current_straddle_value is None or current_iv is None:
                # Skip this day for position management
                active_position['dte_remaining'] = max(1, active_position['entry_dte'] - days_held)
                continue

            iv_change = current_iv - active_position['entry_iv']
            iv_change_pct = iv_change * 100  # Convert to percentage points

            hedger = DeltaHedger(spot_price, risk_free_rate=0.02, dividend_yield=0.02)
            current_maturity = max(0.01, (active_position['entry_dte'] - days_held) / 365)
            
            hedge_result = hedger.calculate_hedge_position(
                spot_price, active_position['strike'], current_maturity, current_iv
            )
            
            from scipy.stats import norm

            d1 = (np.log(spot_price / active_position['strike']) + 
                  (0.02 - 0.02 + 0.5 * current_iv**2) * current_maturity) / \
                 (current_iv * np.sqrt(current_maturity))

            call_delta = np.exp(-0.02 * current_maturity) * norm.cdf(d1)
            put_delta = -np.exp(-0.02 * current_maturity) * norm.cdf(-d1)
            straddle_delta = call_delta + put_delta

            single_gamma = np.exp(-0.02 * current_maturity) * norm.pdf(d1) / \
                           (spot_price * current_iv * np.sqrt(current_maturity))
            straddle_gamma = 2 * single_gamma

            prev_spot = active_position.get('prev_spot', active_position['entry_spot'])
            spot_change = spot_price - prev_spot

            prev_delta = active_position.get('prev_delta', 0)
            prev_gamma = active_position.get('prev_gamma', straddle_gamma)

            delta_hedge_pnl = -prev_delta * spot_change * active_position['num_straddles']

            gamma_pnl = 0.5 * prev_gamma * (spot_change ** 2) * active_position['num_straddles']

            theta_pnl = hedger.calculate_theta_pnl(active_position['strike'], current_maturity, 
                                                   current_iv, days=1) * active_position['num_straddles'] * 100

            if 'prev_hedge_units' in active_position:
                hedge_adjustment = abs(hedge_result['hedge_units'] - active_position['prev_hedge_units'])
                hedge_rebalance_cost = hedge_adjustment * spot_price * 0.001  # 0.1% transaction cost
            else:
                hedge_rebalance_cost = 0
            
            # Total daily hedge P&L
            daily_hedge_pnl = delta_hedge_pnl + gamma_pnl + theta_pnl - hedge_rebalance_cost
            
            # Accumulate hedge P&L
            active_position['cumulative_hedge_pnl'] = active_position.get('cumulative_hedge_pnl', 0) + daily_hedge_pnl
            active_position['cumulative_gamma_pnl'] = active_position.get('cumulative_gamma_pnl', 0) + gamma_pnl
            active_position['cumulative_theta_pnl'] = active_position.get('cumulative_theta_pnl', 0) + theta_pnl
            active_position['cumulative_delta_pnl'] = active_position.get('cumulative_delta_pnl', 0) + delta_hedge_pnl
            active_position['total_hedge_cost'] = active_position.get('total_hedge_cost', 0) + hedge_rebalance_cost
            total_hedge_costs += hedge_rebalance_cost
            
            # Update position tracking for next day
            active_position['prev_hedge_units'] = hedge_result['hedge_units']
            active_position['prev_spot'] = spot_price
            active_position['prev_delta'] = straddle_delta  # Use calculated straddle delta
            active_position['prev_gamma'] = straddle_gamma  # Use calculated straddle gamma
            
            # Exit logic based on days held
            should_exit = False
            exit_reason = None
            
            if days_held >= 14:
                # Day 14: EXIT regardless
                should_exit = True
                exit_reason = "Day 14 mandatory exit"
            elif days_held >= 10:
                # Day 10-13: Check IV thresholds
                if iv_change_pct >= 2.0:
                    should_exit = True
                    exit_reason = f"Day {days_held} take profit (IV +{iv_change_pct:.1f}%)"
                elif iv_change_pct <= -1.0:
                    should_exit = True
                    exit_reason = f"Day {days_held} stop loss (IV {iv_change_pct:.1f}%)"
            elif days_held >= 7:
                # Day 7-9: Check IV thresholds
                if iv_change_pct >= 3.0:
                    should_exit = True
                    exit_reason = f"Day {days_held} take profit (IV +{iv_change_pct:.1f}%)"
                elif iv_change_pct <= -2.0:
                    should_exit = True
                    exit_reason = f"Day {days_held} stop loss (IV {iv_change_pct:.1f}%)"
            
            if should_exit:
                # Calculate P&L components
                # 1. Option value change (vega P&L from IV change)
                option_pnl = current_straddle_value - active_position['entry_cost']
                
                # 2. Cumulative hedge P&L (gamma + theta + delta hedge)
                cumulative_hedge_pnl = active_position.get('cumulative_hedge_pnl', 0)
                
                # 3. Exit transaction costs
                exit_cost = tc_calc.calculate(
                    price=current_straddle_value / (100 * active_position['num_straddles']),
                    contracts=active_position['num_straddles'] * 2,
                    ticker=ticker_upper
                )
                
                # Entry transaction costs (should have been tracked)
                entry_cost_tc = active_position.get('entry_tc', 0)
                
                # Total hedge rebalancing costs
                total_hedge_rebalance_costs = active_position.get('total_hedge_cost', 0)
                
                # Net P&L = Option P&L + Hedge P&L - Entry TC - Exit TC
                gross_pnl = option_pnl + cumulative_hedge_pnl
                net_pnl = gross_pnl - entry_cost_tc - exit_cost
                
                portfolio_value += net_pnl
                trades_exited += 1
                
                # Log the trade with detailed P&L breakdown
                trade_log.append({
                    'entry_date': active_position['entry_date'],
                    'exit_date': current_date,
                    'days_held': days_held,
                    'strike': active_position['strike'],
                    'entry_iv': active_position['entry_iv'],
                    'exit_iv': current_iv,
                    'iv_change': iv_change,
                    'iv_change_pct': iv_change_pct,
                    'entry_cost': active_position['entry_cost'],
                    'exit_value': current_straddle_value,
                    'option_pnl': option_pnl,  # P&L from option value change
                    'gamma_pnl': active_position.get('cumulative_gamma_pnl', 0),  # Gamma scalping profits
                    'theta_pnl': active_position.get('cumulative_theta_pnl', 0),  # Theta decay costs
                    'delta_hedge_pnl': active_position.get('cumulative_delta_pnl', 0),  # Delta hedge P&L
                    'hedge_pnl': cumulative_hedge_pnl,  # Total hedge P&L
                    'gross_pnl': gross_pnl,
                    'entry_tc': entry_cost_tc,
                    'exit_cost': exit_cost,
                    'hedge_rebalance_costs': total_hedge_rebalance_costs,
                    'net_pnl': net_pnl,
                    'exit_reason': exit_reason,
                    'garch_forecast': active_position['garch_forecast'],
                    'num_straddles': active_position['num_straddles']
                })
                
                if verbose:
                    print(f"\n  [{current_date.date()}] EXIT: {exit_reason} | P&L: ${net_pnl:.2f}")
                
                active_position = None
            else:
                active_position['dte_remaining'] = max(1, active_position['entry_dte'] - days_held)
                active_position['current_iv'] = current_iv
                active_position['current_value'] = current_straddle_value

            results.append({
                'date': current_date,
                'spot_price': spot_price,
                'forecast_iv': garch_forecast,
                'market_iv': current_iv if current_iv else np.nan,
                'position_status': 'HOLDING' if active_position else 'EXITED',
                'days_held': days_held,
                'iv_change_pct': iv_change_pct,
                'portfolio_value': portfolio_value,
                'trade_pnl_dollars': net_pnl if should_exit else 0,
                'traded': should_exit
            })
            processed_count += 1
            continue
        
        strikes, maturities, market_prices = get_iv_surface_for_date(
            options_data, current_date, spot_price
        )
        
        if strikes is None:
            skipped_count += 1
            if verbose:
                print(f"  [{current_date.date()}] Insufficient options data, skipping...")
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
                skipped_count += 1
                continue
            
            atm_option = get_atm_option_for_dte(
                options_data, current_date, spot_price,
                target_dte=45, dte_range=(30, 45)
            )
            
            if atm_option is None:
                skipped_count += 1
                if verbose:
                    print(f"  [{current_date.date()}] No 30-45 DTE options available")
                continue
            
            market_iv = get_iv_for_option(
                iv_calc, atm_option['strike'], atm_option['maturity'],
                atm_option['call_price'], atm_option['put_price']
            )
            
            if np.isnan(market_iv):
                skipped_count += 1
                continue
            
            iv_history.append(market_iv)
            if len(iv_history) > 252:
                iv_history = iv_history[-252:]  # Keep last 252 days

            if garch_forecast is not None:
                forecast_history.append(garch_forecast)
                market_iv_history.append(market_iv)
                if len(forecast_history) > 252:
                    forecast_history = forecast_history[-252:]
                    market_iv_history = market_iv_history[-252:]

            iv_diff = garch_forecast - market_iv
            garch_signal = iv_diff > 0.02  # 2 percentage points (was 4%)

            if len(iv_history) >= 20:  # Need some history
                iv_percentile = (np.array(iv_history) < market_iv).sum() / len(iv_history) * 100
                iv_cheap = iv_percentile < 25
            else:
                iv_cheap = True  # Not enough history, assume cheap
                iv_percentile = 50

            dte_available = True

            is_blocked = False
            if regime_blocker is not None:
                try:
                    is_blocked = regime_blocker.isBlocked(current_date.strftime('%Y-%m-%d'))
                except:
                    is_blocked = False

            if garch_signal and iv_cheap and dte_available and not is_blocked:
                signal = 'BUY'
            else:
                signal = 'HOLD'
                if is_blocked:
                    blocked_count += 1

            entry_notes = []
            if not garch_signal:
                entry_notes.append(f"GARCH diff {iv_diff:.1%} < 4%")
            if not iv_cheap:
                entry_notes.append(f"IV percentile {iv_percentile:.0f} >= 25")
            if is_blocked:
                entry_notes.append("Stress regime")

            if signal == 'BUY':
                straddle_price_per_unit = atm_option['straddle_price'] * 100  # Per straddle (100 shares)
                num_straddles = max(1, int(position_size / straddle_price_per_unit))
                entry_cost = straddle_price_per_unit * num_straddles

                entry_tc = tc_calc.calculate(
                    price=atm_option['straddle_price'],
                    contracts=num_straddles * 2,  # 2 contracts per straddle
                    ticker=ticker_upper
                )

                hedger = DeltaHedger(spot_price, risk_free_rate=0.02, dividend_yield=0.02)
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
                from scipy.stats import norm
                straddle_gamma = np.exp(-0.02 * atm_option['maturity']) * norm.pdf(d1) / \
                                 (spot_price * market_iv * np.sqrt(atm_option['maturity'])) * 2  # x2 for straddle

                active_position = {
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
                
                trades_entered += 1
                
                if verbose:
                    print(f"\n  [{current_date.date()}] ENTRY: Strike ${atm_option['strike']:.0f}, "
                          f"DTE {atm_option['dte']}, IV {market_iv:.1%}, GARCH {garch_forecast:.1%}, "
                          f"Cost ${entry_cost:.0f}")
            
            results.append({
                'date': current_date,
                'spot_price': spot_price,
                'forecast_iv': garch_forecast,
                'market_iv': market_iv,
                'iv_spread': iv_diff,
                'iv_percentile': iv_percentile if len(iv_history) >= 20 else np.nan,
                'signal': signal,
                'blocked': is_blocked,
                'entry_notes': '; '.join(entry_notes) if entry_notes else '',
                'atm_strike': atm_option['strike'],
                'atm_dte': atm_option['dte'],
                'straddle_price': atm_option['straddle_price'],
                'portfolio_value': portfolio_value,
                'trade_pnl_dollars': 0,
                'traded': signal == 'BUY'
            })
            processed_count += 1
            
        except Exception as e:
            skipped_count += 1
            if verbose:
                print(f"  [{current_date.date()}] Error: {str(e)}")
            continue
    
    print(f"\n\n[5/6] Processing results...")
    print(f"  Processed: {processed_count}, Skipped: {skipped_count}, Blocked: {blocked_count}")
    print(f"  Trades entered: {trades_entered}, Trades exited: {trades_exited}")

    if len(forecast_history) > 0 and len(market_iv_history) > 0:
        print(f"\n  Forecast Quality Analysis:")
        forecast_arr = np.array(forecast_history)
        market_arr = np.array(market_iv_history)
        
        forecast_error = forecast_arr - market_arr
        print(f"    Mean Forecast:          {forecast_arr.mean():.2%}")
        print(f"    Mean Market IV:         {market_arr.mean():.2%}")
        print(f"    Mean Forecast Error:    {forecast_error.mean():.2%}")
        print(f"    RMSE:                   {np.sqrt((forecast_error**2).mean()):.2%}")
        print(f"    Correlation:            {np.corrcoef(forecast_arr, market_arr)[0,1]:.3f}")
        print(f"    Calibrated Risk Prem:   {calculate_vol_risk_premium(forecast_history, market_iv_history):.2%}")
    
    df_results = pd.DataFrame(results)
    df_trades = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    
    if len(df_results) == 0:
        print("  [!] No valid backtest results")
        return None
    
    print(f"  * Generated {len(df_results)} daily observations")
    print(f"  * Completed {len(df_trades)} round-trip trades")
    print(f"  * Regime blocker prevented {blocked_count} entries during stress periods")
    
    print("\n[6/6] Performance Analysis...")
    
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
        
        print(f"\n  P&L Breakdown:")
        print(f"    Option P&L:         ${df_trades['option_pnl'].sum():>10,.2f}")
        print(f"    Gamma P&L:          ${df_trades['gamma_pnl'].sum():>10,.2f}")
        print(f"    Theta P&L:          ${df_trades['theta_pnl'].sum():>10,.2f}")
        print(f"    Delta Hedge P&L:    ${df_trades['delta_hedge_pnl'].sum():>10,.2f}")
        print(f"    Total Hedge P&L:    ${df_trades['hedge_pnl'].sum():>10,.2f}")
        
        print(f"\n  Transaction Costs:")
        print(f"    Entry Costs:        ${df_trades['entry_tc'].sum():>10,.2f}")
        print(f"    Exit Costs:         ${df_trades['exit_cost'].sum():>10,.2f}")
        print(f"    Hedge Rebal Costs:  ${df_trades['hedge_rebalance_costs'].sum():>10,.2f}")
    
    # Entry criteria distribution
    if 'signal' in df_results.columns:
        print(f"\n  Signal Distribution:")
        for sig in ['BUY', 'HOLD']:
            count = (df_results['signal'] == sig).sum()
            pct = count / len(df_results) * 100
            print(f"    {sig}: {count:>4} ({pct:>5.1f}%)")
    
    if 'iv_spread' in df_results.columns:
        iv_spreads = df_results['iv_spread'].dropna()
        if len(iv_spreads) > 0:
            print(f"\n  GARCH - Market IV Spread:")
            print(f"    Mean:             {iv_spreads.mean():>7.2%}")
            print(f"    Median:           {iv_spreads.median():>7.2%}")
            print(f"    % Above 4%:       {(iv_spreads > 0.04).mean()*100:>6.1f}%")
    
    print(f"\n  Final Portfolio Value: ${portfolio_value:,.2f}")
    print(f"  Total Return:          {(portfolio_value - starting_capital) / starting_capital * 100:.2f}%")
    
    print("\n  Backtest complete!")
    print("="*80)
    
    df_results.attrs['trade_log'] = df_trades
    
    return df_results

def calculate_performance_metrics(results_df, starting_capital=100000, annual_risk_free_rate=0.02):
    
    metrics = {}
    
    trade_log = results_df.attrs.get('trade_log', pd.DataFrame())
    
    if len(trade_log) == 0:
        print("\n  [!] No trades executed during backtest period")
        # Return minimal metrics
        metrics['total_trades'] = 0
        metrics['starting_capital'] = starting_capital
        metrics['ending_capital'] = starting_capital
        metrics['total_pnl'] = 0
        metrics['total_return'] = 0
        return metrics
    
    # Calculate returns based on portfolio value changes
    portfolio_values = results_df['portfolio_value'].values
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    portfolio_returns = pd.Series(returns)
    
    trade_pnl = trade_log['net_pnl']
    
    metrics['starting_capital'] = starting_capital
    metrics['starting_capital'] = starting_capital
    metrics['ending_capital'] = portfolio_values[-1]
    metrics['total_pnl'] = portfolio_values[-1] - starting_capital
    metrics['total_return'] = (portfolio_values[-1] - starting_capital) / starting_capital
    
    metrics['total_trades'] = len(trade_log)
    metrics['total_observations'] = len(results_df)
    metrics['mean_return'] = portfolio_returns.mean() if len(portfolio_returns) > 0 else 0
    metrics['median_return'] = portfolio_returns.median() if len(portfolio_returns) > 0 else 0
    metrics['std_return'] = portfolio_returns.std() if len(portfolio_returns) > 0 else 0
    metrics['min_return'] = portfolio_returns.min() if len(portfolio_returns) > 0 else 0
    metrics['max_return'] = portfolio_returns.max() if len(portfolio_returns) > 0 else 0
    
    metrics['mean_trade_pnl'] = trade_pnl.mean()
    metrics['total_trade_pnl'] = trade_pnl.sum()
    metrics['mean_holding_days'] = trade_log['days_held'].mean()
    
    metrics['mean_iv_change'] = trade_log['iv_change_pct'].mean()
    metrics['trades_with_iv_increase'] = (trade_log['iv_change'] > 0).sum()
    metrics['trades_with_iv_decrease'] = (trade_log['iv_change'] < 0).sum()
    
    metrics['total_option_pnl'] = trade_log['option_pnl'].sum()
    metrics['total_gamma_pnl'] = trade_log['gamma_pnl'].sum()
    metrics['total_theta_pnl'] = trade_log['theta_pnl'].sum()
    metrics['total_delta_hedge_pnl'] = trade_log['delta_hedge_pnl'].sum()
    metrics['total_hedge_pnl'] = trade_log['hedge_pnl'].sum()
    
    metrics['total_entry_costs'] = trade_log['entry_tc'].sum()
    metrics['total_exit_costs'] = trade_log['exit_cost'].sum()
    metrics['total_hedge_rebalance_costs'] = trade_log['hedge_rebalance_costs'].sum()
    metrics['total_transaction_costs'] = metrics['total_entry_costs'] + metrics['total_exit_costs'] + metrics['total_hedge_rebalance_costs']
    metrics['transaction_cost_pct'] = (metrics['total_transaction_costs'] / starting_capital) * 100
    
    metrics['win_rate'] = (trade_pnl > 0).mean()
    metrics['num_wins'] = (trade_pnl > 0).sum()
    metrics['num_losses'] = (trade_pnl < 0).sum()
    
    metrics['exits_take_profit'] = trade_log['exit_reason'].str.contains('profit', case=False).sum()
    metrics['exits_stop_loss'] = trade_log['exit_reason'].str.contains('stop', case=False).sum()
    metrics['exits_mandatory'] = trade_log['exit_reason'].str.contains('mandatory|Day 14', case=False).sum()
    
    portfolio_series = pd.Series(portfolio_values)
    
    running_max = portfolio_series.cummax()
    drawdown = (portfolio_series - running_max) / running_max
    metrics['max_drawdown'] = drawdown.min()
    metrics['max_drawdown_pct'] = metrics['max_drawdown'] * 100
    metrics['max_drawdown_dollars'] = (running_max - portfolio_series).max()
    
    if len(portfolio_returns) > 0 and metrics['std_return'] > 0:
        daily_rf = annual_risk_free_rate / 252
        excess_returns = portfolio_returns - daily_rf
        metrics['sharpe_ratio'] = excess_returns.mean() / portfolio_returns.std() * np.sqrt(252)
    else:
        metrics['sharpe_ratio'] = 0
    
    # Sortino Ratio (downside deviation)
    if len(portfolio_returns) > 0:
        downside_returns = portfolio_returns[portfolio_returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            if downside_std > 0:
                daily_rf = annual_risk_free_rate / 252
                metrics['sortino_ratio'] = (portfolio_returns.mean() - daily_rf) / downside_std * np.sqrt(252)
            else:
                metrics['sortino_ratio'] = 0
        else:
            metrics['sortino_ratio'] = np.inf if portfolio_returns.mean() > 0 else 0
    else:
        metrics['sortino_ratio'] = 0
    
    # Calmar Ratio (return / max drawdown)
    if metrics['max_drawdown'] < 0:
        metrics['calmar_ratio'] = metrics['total_return'] / abs(metrics['max_drawdown'])
    else:
        metrics['calmar_ratio'] = np.inf if metrics['total_return'] > 0 else 0
    
    # Average win/loss in dollars
    if metrics['num_wins'] > 0:
        metrics['avg_win'] = trade_pnl[trade_pnl > 0].mean()
    else:
        metrics['avg_win'] = 0
    
    if metrics['num_losses'] > 0:
        metrics['avg_loss'] = trade_pnl[trade_pnl < 0].mean()
    else:
        metrics['avg_loss'] = 0
    
    # Profit factor
    total_wins = trade_pnl[trade_pnl > 0].sum()
    total_losses = abs(trade_pnl[trade_pnl < 0].sum())
    if total_losses > 0:
        metrics['profit_factor'] = total_wins / total_losses
    else:
        metrics['profit_factor'] = np.inf if total_wins > 0 else 0
    
    return metrics

def plot_backtest_results(results_df, save_path='backtest_analysis.png'):
    
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.3)
    
    portfolio_values = results_df['portfolio_value'].values
    starting_value = portfolio_values[0]
    running_max = pd.Series(portfolio_values).cummax()
    drawdown = (portfolio_values - running_max) / running_max * 100
    dates = pd.to_datetime(results_df['date'])
    
    trade_log = results_df.attrs.get('trade_log', pd.DataFrame())
    
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, portfolio_values, linewidth=2, color='#2E86AB', label='Portfolio Value')
    ax1.fill_between(dates, starting_value, portfolio_values, alpha=0.3, color='#2E86AB')
    ax1.axhline(y=starting_value, color='red', linestyle='--', alpha=0.5, label='Starting Capital')
    ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=11)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(dates, 0, drawdown, color='#A23B72', alpha=0.7)
    ax2.plot(dates, drawdown, linewidth=1.5, color='#A23B72')
    ax2.set_title('Drawdown (%)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    ax3 = fig.add_subplot(gs[2, 0])
    if 'iv_spread' in results_df.columns:
        iv_spreads = results_df['iv_spread'].dropna() * 100
        ax3.hist(iv_spreads, bins=30, color='#F18F01', alpha=0.7, edgecolor='black')
        ax3.axvline(x=4, color='green', linestyle='--', linewidth=2, label='Entry threshold (4%)')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero spread')
    ax3.set_title('GARCH - Market IV Spread Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('IV Spread (%)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[2, 1])
    if len(trade_log) > 0:
        ax4.hist(trade_log['net_pnl'], bins=20, color='#6A994E', alpha=0.7, edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax4.axvline(x=trade_log['net_pnl'].mean(), color='blue', linestyle='-', linewidth=2, 
                   label=f"Mean: ${trade_log['net_pnl'].mean():.0f}")
    ax4.set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Trade P&L ($)', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[3, 0])
    if len(trade_log) > 0:
        ax5.hist(trade_log['iv_change_pct'], bins=20, color='#BC4B51', alpha=0.7, edgecolor='black')
        ax5.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax5.axvline(x=3, color='green', linestyle='--', alpha=0.7, label='Day 7 profit (+3%)')
        ax5.axvline(x=-2, color='red', linestyle='--', alpha=0.7, label='Day 7 stop (-2%)')
    ax5.set_title('IV Change During Trade (Entry to Exit)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('IV Change (%)', fontsize=10)
    ax5.set_ylabel('Frequency', fontsize=10)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[3, 1])
    if len(trade_log) > 0:
        holding_days = trade_log['days_held']
        ax6.hist(holding_days, bins=range(1, 16), color='#5C8A97', alpha=0.7, edgecolor='black', align='left')
        ax6.axvline(x=7, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Day 7 check')
        ax6.axvline(x=10, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='Day 10 check')
        ax6.axvline(x=14, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Day 14 exit')
    ax6.set_title('Holding Period Distribution', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Days Held', fontsize=10)
    ax6.set_ylabel('Frequency', fontsize=10)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    ax7 = fig.add_subplot(gs[4, 0])
    if 'forecast_iv' in results_df.columns and 'market_iv' in results_df.columns:
        ax7.plot(dates, results_df['forecast_iv'] * 100, label='GARCH Forecast', 
                 linewidth=2, color='#2E86AB', alpha=0.8)
        market_iv = results_df['market_iv'].fillna(method='ffill') * 100
        ax7.plot(dates, market_iv, label='Market IV', 
                 linewidth=2, color='#F18F01', alpha=0.8)
    ax7.set_title('Forecast vs Market IV', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Implied Volatility (%)', fontsize=10)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    ax8 = fig.add_subplot(gs[4, 1])
    if len(trade_log) > 0:
        cumulative_pnl = trade_log['net_pnl'].cumsum()
        trade_numbers = range(1, len(trade_log) + 1)
        ax8.plot(trade_numbers, cumulative_pnl, linewidth=2, color='#2E86AB', marker='o', markersize=4)
        ax8.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax8.fill_between(trade_numbers, 0, cumulative_pnl, alpha=0.3, color='#2E86AB')
    ax8.set_title('Cumulative Trade P&L', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Trade Number', fontsize=10)
    ax8.set_ylabel('Cumulative P&L ($)', fontsize=10)
    ax8.grid(True, alpha=0.3)
    
    plt.suptitle('Volatility Arbitrage Strategy - 45 DTE Straddle Backtest', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n  * Charts saved to {save_path}")
    plt.close()

def print_performance_summary(metrics):
    
    print("\n" + "="*80)
    print("PERFORMANCE METRICS SUMMARY")
    print("="*80)
    
    print("\n[Portfolio Performance]")
    print(f"  Starting Capital:    ${metrics['starting_capital']:>12,.2f}")
    print(f"  Ending Capital:      ${metrics['ending_capital']:>12,.2f}")
    print(f"  Total P&L:           ${metrics['total_pnl']:>12,.2f}")
    print(f"  Total Return:        {metrics['total_return']*100:>11.2f}%")
    
    if metrics['total_trades'] == 0:
        print("\n  [i] No trades executed - cannot calculate risk metrics")
        return
    
    print("\n[Risk-Adjusted Returns]")
    print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>10.3f}")
    print(f"  Sortino Ratio:       {metrics['sortino_ratio']:>10.3f}")
    print(f"  Calmar Ratio:        {metrics['calmar_ratio']:>10.3f}")
    
    print("\n[Return Statistics]")
    print(f"  Mean Trade Return:   {metrics['mean_return']*100:>9.2f}%")
    print(f"  Median Trade Return: {metrics['median_return']*100:>9.2f}%")
    print(f"  Std Dev:             {metrics['std_return']*100:>9.2f}%")
    print(f"  Min Return:          {metrics['min_return']*100:>9.2f}%")
    print(f"  Max Return:          {metrics['max_return']*100:>9.2f}%")
    
    print("\n[Risk Metrics]")
    print(f"  Max Drawdown:        {metrics['max_drawdown_pct']:>9.2f}%")
    print(f"  Max Drawdown ($):    ${metrics['max_drawdown_dollars']:>11,.2f}")
    
    print("\n[Trading Statistics]")
    print(f"  Total Observations:  {metrics.get('total_observations', metrics['total_trades']):>10}")
    print(f"  Actual Trades:       {metrics['total_trades']:>10}")
    print(f"  Mean Holding Days:   {metrics.get('mean_holding_days', 0):>10.1f}")
    print(f"  Win Rate:            {metrics['win_rate']*100:>9.2f}%")
    print(f"  Winning Trades:      {metrics['num_wins']:>10}")
    print(f"  Losing Trades:       {metrics['num_losses']:>10}")
    print(f"  Avg Win:             ${metrics['avg_win']:>10,.2f}")
    print(f"  Avg Loss:            ${metrics['avg_loss']:>10,.2f}")
    print(f"  Mean Trade P&L:      ${metrics['mean_trade_pnl']:>10,.2f}")
    print(f"  Profit Factor:       {metrics['profit_factor']:>10.3f}")
    
    print("\n[Exit Analysis]")
    print(f"  Take Profit Exits:   {metrics.get('exits_take_profit', 0):>10}")
    print(f"  Stop Loss Exits:     {metrics.get('exits_stop_loss', 0):>10}")
    print(f"  Day 14 Exits:        {metrics.get('exits_mandatory', 0):>10}")
    
    print("\n[IV Change Statistics]")
    print(f"  Mean IV Change:      {metrics.get('mean_iv_change', 0):>9.2f}%")
    print(f"  Trades IV Up:        {metrics.get('trades_with_iv_increase', 0):>10}")
    print(f"  Trades IV Down:      {metrics.get('trades_with_iv_decrease', 0):>10}")
    
    print("\n[P&L Breakdown]")
    print(f"  Option P&L:          ${metrics.get('total_option_pnl', 0):>10,.2f}")
    print(f"  Gamma P&L:           ${metrics.get('total_gamma_pnl', 0):>10,.2f}")
    print(f"  Theta P&L:           ${metrics.get('total_theta_pnl', 0):>10,.2f}")
    print(f"  Delta Hedge P&L:     ${metrics.get('total_delta_hedge_pnl', 0):>10,.2f}")
    print(f"  Total Hedge P&L:     ${metrics.get('total_hedge_pnl', 0):>10,.2f}")
    
    print("\n[Transaction Costs]")
    print(f"  Entry Costs:         ${metrics.get('total_entry_costs', 0):>10,.2f}")
    print(f"  Exit Costs:          ${metrics.get('total_exit_costs', 0):>10,.2f}")
    print(f"  Hedge Rebal Costs:   ${metrics.get('total_hedge_rebalance_costs', 0):>10,.2f}")
    print(f"  Total Costs:         ${metrics['total_transaction_costs']:>10,.2f}")
    print(f"  Costs as % Capital:  {metrics['transaction_cost_pct']:>9.2f}%")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    import sys
    
    # Get ticker from command line argument or use default
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    else:
        ticker = 'AAPL'  # Default ticker
        print(f"No ticker specified, using default: {ticker}")
        print(f"Usage: python main.py <TICKER>")
        print(f"Example: python main.py AAPL\n")
    
    # Run rolling window backtest with new 45 DTE strategy
    results = rolling_window_backtest(
        ticker=ticker,
        train_window=126,  # 6 months for GARCH training (better regime capture)
        refit_frequency=21,  # Refit GARCH every 21 days (monthly)
        starting_capital=100000,  # $100k starting capital
        position_size=8000,  # $8k per trade
        use_regime_blocker=True,  # Enable regime blocking
        start_date=None,  # Start date
        end_date=None,  # End date
        verbose=True
    )
    
    if results is not None:
        # Save daily results
        results.to_csv(f"backtest_results_{ticker.lower()}.csv", index=False)
        print(f"\nDaily results saved to backtest_results_{ticker.lower()}.csv")
        
        # Save trade log if available
        trade_log = results.attrs.get('trade_log', pd.DataFrame())
        if len(trade_log) > 0:
            trade_log.to_csv(f"trade_log_{ticker.lower()}.csv", index=False)
            print(f"Trade log saved to trade_log_{ticker.lower()}.csv")
        
        # Calculate performance metrics
        print("\nCalculating performance metrics...")
        metrics = calculate_performance_metrics(results, starting_capital=100000)
        
        # Print performance summary
        print_performance_summary(metrics)
        
        # Create visualizations
        print("\nGenerating performance charts...")
        plot_backtest_results(results, save_path=f'backtest_analysis_{ticker.lower()}.png')
        
        # Display sample results
        print("\nSample Daily Results (first 10 observations):")
        display_cols = ['date', 'spot_price', 'forecast_iv', 'market_iv', 'iv_spread', 'signal']
        available_cols = [c for c in display_cols if c in results.columns]
        print(results[available_cols].head(10).to_string(index=False))
        
        # Display sample trades
        if len(trade_log) > 0:
            print("\nSample Trades (first 5 trades):")
            trade_cols = ['entry_date', 'exit_date', 'days_held', 'entry_iv', 'exit_iv', 
                         'iv_change_pct', 'net_pnl', 'exit_reason']
            available_trade_cols = [c for c in trade_cols if c in trade_log.columns]
            print(trade_log[available_trade_cols].head(5).to_string(index=False))