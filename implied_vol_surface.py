import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


class ImpliedVolSurface:
    
    def __init__(self, spot_price, risk_free_rate, dividend_yield=0.0, strikes=None, maturities=None, market_prices=None, verbose=False):
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.strikes = np.asarray(strikes) if strikes is not None else None
        self.maturities = np.asarray(maturities) if maturities is not None else None
        self.market_prices = market_prices
        self.verbose = verbose
        self.iv_surface = None
        
        if self.strikes is not None and self.maturities is not None and self.market_prices is not None:
            self.iv_surface = self._compute_iv_surface()
    
    def black_scholes_call(self, S, K, T, r, sigma, q=0.0):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price
    
    def black_scholes_put(self, S, K, T, r, sigma, q=0.0):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        return put_price
    
    def vega(self, S, K, T, r, sigma, q=0.0):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega_value = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        return vega_value
    
    def implied_volatility(self, market_price, K, T, option_type='call', initial_guess=0.2):
        if option_type == 'call':
            objective = lambda sigma: self.black_scholes_call(
                self.spot_price, K, T, self.risk_free_rate, sigma, self.dividend_yield
            ) - market_price
        else:
            objective = lambda sigma: self.black_scholes_put(
                self.spot_price, K, T, self.risk_free_rate, sigma, self.dividend_yield
            ) - market_price
        
        try:
            iv = brentq(objective, 0.001, 5.0)
            return iv
        except ValueError:
            return np.nan
    
    def _compute_iv_surface(self):
        iv_surface = np.zeros((len(self.strikes), len(self.maturities)))
        
        for i, K in enumerate(self.strikes):
            for j, T in enumerate(self.maturities):
                price = self.market_prices[i, j]
                
                # Skip if price is invalid or too close to zero
                if np.isnan(price) or price <= 0.01:
                    iv_surface[i, j] = np.nan
                    continue
                
                try:
                    iv = self.implied_volatility(price, K, T, 'call')
                    iv_surface[i, j] = iv
                except (ValueError, RuntimeError):
                    iv_surface[i, j] = np.nan
        
        return iv_surface
    
    def generate_surface_data(self, strikes, maturities, market_prices, option_type='call'):
        strikes = np.asarray(strikes)
        maturities = np.asarray(maturities)
        
        iv_surface = np.zeros((len(strikes), len(maturities)))
        
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                price = market_prices[i, j]
                iv = self.implied_volatility(price, K, T, option_type)
                iv_surface[i, j] = iv
        
        return strikes, maturities, iv_surface
    
    def calculate_straddle_cost(self, K, T, sigma):
        call_price = self.black_scholes_call(self.spot_price, K, T, self.risk_free_rate, sigma, self.dividend_yield)
        put_price = self.black_scholes_put(self.spot_price, K, T, self.risk_free_rate, sigma, self.dividend_yield)
        straddle_cost = call_price + put_price
        return straddle_cost
    
    def calculate_straddle_greeks(self, K, T, sigma):
        d1 = (np.log(self.spot_price / K) + (self.risk_free_rate - self.dividend_yield + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_delta = np.exp(-self.dividend_yield * T) * norm.cdf(d1)
        put_delta = -np.exp(-self.dividend_yield * T) * norm.cdf(-d1)
        straddle_delta = call_delta + put_delta
        
        gamma = np.exp(-self.dividend_yield * T) * norm.pdf(d1) / (self.spot_price * sigma * np.sqrt(T))
        
        vega = self.spot_price * np.exp(-self.dividend_yield * T) * norm.pdf(d1) * np.sqrt(T) / 100
        
        call_theta = (-self.spot_price * np.exp(-self.dividend_yield * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                      + self.dividend_yield * self.spot_price * np.exp(-self.dividend_yield * T) * norm.cdf(d1)
                      - self.risk_free_rate * K * np.exp(-self.risk_free_rate * T) * norm.cdf(d2)) / 365
        
        put_theta = (-self.spot_price * np.exp(-self.dividend_yield * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                     - self.dividend_yield * self.spot_price * np.exp(-self.dividend_yield * T) * norm.cdf(-d1)
                     + self.risk_free_rate * K * np.exp(-self.risk_free_rate * T) * norm.cdf(-d2)) / 365
        
        straddle_theta = call_theta + put_theta
        
        return {
            'delta': straddle_delta,
            'gamma': gamma,
            'vega': vega,
            'theta': straddle_theta,
            'cost': self.calculate_straddle_cost(K, T, sigma)
        }
    
    def print_optimal_straddle_greeks(self, optimal, metric):
        print(f"\n{metric.upper()} STRADDLE")
        print("-" * 50)
        print(f"Strike:        ${optimal['strike']:.2f}")
        print(f"Maturity:      {optimal['maturity']:.3f} years ({optimal['maturity']*12:.1f} months)")
        print(f"IV:            {optimal['iv']:.2%}")
        print(f"Cost:          ${optimal['cost']:.4f}")
        print(f"Delta:         {optimal['delta']:.4f}")
        print(f"Gamma:         {optimal['gamma']:.6f}")
        print(f"Vega:          {optimal['vega']:.4f}")
        print(f"Theta:         {optimal['theta']:.4f}")
        if 'vega_theta_ratio' in optimal:
            print(f"Vega/Theta:    {optimal['vega_theta_ratio']:.2f}")
    
    def find_optimal_straddle(self, strikes, maturities, iv_surface, metric='cheapest'):
        results = []
        
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                sigma = iv_surface[i, j]
                if np.isnan(sigma):
                    continue
                
                greeks = self.calculate_straddle_greeks(K, T, sigma)
                
                atm_distance = abs(K - self.spot_price) / self.spot_price
                vega_theta_ratio = abs(greeks['vega'] / greeks['theta']) if greeks['theta'] != 0 else 0
                
                results.append({
                    'strike': K,
                    'maturity': T,
                    'iv': sigma,
                    'atm_distance': atm_distance,
                    'cost': greeks['cost'],
                    'delta': greeks['delta'],
                    'gamma': greeks['gamma'],
                    'vega': greeks['vega'],
                    'theta': greeks['theta'],
                    'vega_theta_ratio': vega_theta_ratio,
                    'strike_idx': i,
                    'maturity_idx': j
                })
        
        df_results = pd.DataFrame(results)
        
        if metric == 'cheapest':
            optimal = df_results.loc[df_results['cost'].idxmin()]
        elif metric == 'highest_gamma':
            optimal = df_results.loc[df_results['gamma'].idxmax()]
        elif metric == 'best_vega_carry':
            optimal = df_results.loc[df_results['vega_theta_ratio'].idxmax()]
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if self.verbose:
            self.print_optimal_straddle_greeks(optimal, metric)
        
        return optimal, df_results
    
    def calculate_straddle_pnl_at_forecast(self, K, T, current_iv, forecast_iv, spot_move=0):
        current_cost = self.calculate_straddle_cost(K, T, current_iv)
        
        forecast_value = self.calculate_straddle_cost(K, T, forecast_iv)
        iv_pnl = forecast_value - current_cost
        
        spot_move_pnl = 0
        if spot_move != 0:
            new_spot = self.spot_price * (1 + spot_move)
            greeks = self.calculate_straddle_greeks(K, T, forecast_iv)
            spot_change = new_spot - self.spot_price
            spot_move_pnl = greeks['delta'] * spot_change + 0.5 * greeks['gamma'] * (spot_change ** 2)
        
        total_pnl = iv_pnl + spot_move_pnl
        
        return {
            'current_cost': current_cost,
            'forecast_value': forecast_value,
            'iv_pnl': iv_pnl,
            'iv_pnl_pct': (iv_pnl / current_cost * 100) if current_cost != 0 else 0,
            'spot_move_pnl': spot_move_pnl,
            'total_pnl': total_pnl,
            'total_pnl_pct': (total_pnl / current_cost * 100) if current_cost != 0 else 0,
            'current_iv': current_iv,
            'forecast_iv': forecast_iv,
            'iv_difference': forecast_iv - current_iv,
            'iv_difference_pct': ((forecast_iv - current_iv) / current_iv * 100) if current_iv != 0 else 0
        }
