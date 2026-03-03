import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


class DeltaHedger:
    def __init__(self, spot_price, risk_free_rate, dividend_yield=0.0):
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
    
    def black_scholes_call(self, S, K, T, r, sigma, q=0.0):
        if T <= 0:
            return max(S - K, 0)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price
    
    def black_scholes_put(self, S, K, T, r, sigma, q=0.0):
        if T <= 0:
            return max(K - S, 0)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        return put_price
    
    def calculate_call_delta(self, S, K, T, r, sigma, q=0.0):
        if T <= 0:
            return 1.0 if S > K else 0.0
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return np.exp(-q * T) * norm.cdf(d1)
    
    def calculate_put_delta(self, S, K, T, r, sigma, q=0.0):
        if T <= 0:
            return -1.0 if S < K else 0.0
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return -np.exp(-q * T) * norm.cdf(-d1)
    
    def calculate_straddle_delta(self, S, K, T, r, sigma, q=0.0):
        call_delta = self.calculate_call_delta(S, K, T, r, sigma, q)
        put_delta = self.calculate_put_delta(S, K, T, r, sigma, q)
        straddle_delta = call_delta + put_delta
        return straddle_delta
    
    def calculate_hedge_position(
        self,
        S,
        K,
        T,
        sigma,
        position_sign: int = 1,
        num_straddles: int = 1,
        contract_multiplier: int = 100,
        round_shares: bool = False,
    ):
        if num_straddles <= 0 or contract_multiplier <= 0:
            raise ValueError("num_straddles and contract_multiplier must be positive")
        if position_sign not in (-1, 1):
            raise ValueError("position_sign must be +1 (long) or -1 (short)")

        straddle_delta = self.calculate_straddle_delta(
            S, K, T, self.risk_free_rate, sigma, self.dividend_yield
        )

        option_delta_shares = position_sign * straddle_delta * contract_multiplier * num_straddles

        hedge_shares = -option_delta_shares
        if round_shares:
            hedge_shares = float(int(round(hedge_shares)))

        if hedge_shares > 0:
            hedge_direction = 'LONG'
        elif hedge_shares < 0:
            hedge_direction = 'SHORT'
        else:
            hedge_direction = 'FLAT'

        hedge_notional = hedge_shares * S

        residual_delta_shares = option_delta_shares + hedge_shares
        is_delta_neutral = abs(residual_delta_shares) < 1e-6

        return {
            'straddle_delta': straddle_delta,
            'position_sign': position_sign,
            'num_straddles': num_straddles,
            'contract_multiplier': contract_multiplier,
            'option_delta_shares': option_delta_shares,
            # Backward-compatible key; interpret as shares
            'hedge_units': hedge_shares,
            'hedge_shares': hedge_shares,
            'hedge_direction': hedge_direction,
            'hedge_cost': hedge_notional,
            'residual_delta': residual_delta_shares,
            'is_delta_neutral': is_delta_neutral,
        }
    
    def needs_rehedge(self, old_delta, new_delta, rehedge_threshold=0.10):
        delta_change = abs(new_delta - old_delta)
        return delta_change >= rehedge_threshold
    
    def analyze_rehedge_points(self, K, T, sigma, spot_range=0.10):
        spot_moves = np.linspace(-spot_range, spot_range, 21)
        spots = self.spot_price * (1 + spot_moves)
        
        rehedge_data = []
        initial_delta = self.calculate_straddle_delta(self.spot_price, K, T, self.risk_free_rate, sigma, self.dividend_yield)
        
        for spot in spots:
            current_delta = self.calculate_straddle_delta(spot, K, T, self.risk_free_rate, sigma, self.dividend_yield)
            delta_change = abs(current_delta - initial_delta)
            needs_rehedge = delta_change >= 0.10

            new_hedge = self.calculate_hedge_position(spot, K, T, sigma)
            
            rehedge_data.append({
                'spot_price': spot,
                'spot_move_pct': spot_moves[len(rehedge_data)]*100,
                'delta': current_delta,
                'delta_change': delta_change,
                'needs_rehedge': needs_rehedge,
                'new_hedge_units': new_hedge['hedge_units'],
                'new_hedge_direction': new_hedge['hedge_direction']
            })
        
        return pd.DataFrame(rehedge_data)
    
    def calculate_gamma_pnl(self, K, T, sigma, spot_move):
        d1 = (np.log(self.spot_price / K) + (self.risk_free_rate - self.dividend_yield + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        gamma = np.exp(-self.dividend_yield * T) * norm.pdf(d1) / (self.spot_price * sigma * np.sqrt(T))
        gamma_pnl = 0.5 * gamma * (spot_move ** 2)
        return gamma_pnl
    
    def calculate_vega_pnl(self, K, T, current_iv, forecast_iv):
        call_curr = self.black_scholes_call(self.spot_price, K, T, self.risk_free_rate, current_iv, self.dividend_yield)
        put_curr = self.black_scholes_put(self.spot_price, K, T, self.risk_free_rate, current_iv, self.dividend_yield)
        straddle_curr = call_curr + put_curr

        call_forecast = self.black_scholes_call(self.spot_price, K, T, self.risk_free_rate, forecast_iv, self.dividend_yield)
        put_forecast = self.black_scholes_put(self.spot_price, K, T, self.risk_free_rate, forecast_iv, self.dividend_yield)
        straddle_forecast = call_forecast + put_forecast

        vega_pnl = straddle_forecast - straddle_curr
        vega_pnl_pct = (vega_pnl / straddle_curr * 100) if straddle_curr != 0 else 0

        d1 = (np.log(self.spot_price / K) + (self.risk_free_rate - self.dividend_yield + 0.5 * current_iv**2) * T) / (current_iv * np.sqrt(T))
        vega = self.spot_price * np.exp(-self.dividend_yield * T) * norm.pdf(d1) * np.sqrt(T) / 100
        
        return {
            'current_value': straddle_curr,
            'forecast_value': straddle_forecast,
            'vega_pnl': vega_pnl,
            'vega_pnl_pct': vega_pnl_pct,
            'vega_per_1pct': vega,
            'iv_move': (forecast_iv - current_iv) * 100
        }
    
    def calculate_theta_pnl(self, K, T, sigma, days=1):
        d1 = (np.log(self.spot_price / K) + (self.risk_free_rate - self.dividend_yield + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        call_theta = (-self.spot_price * np.exp(-self.dividend_yield * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                      + self.dividend_yield * self.spot_price * np.exp(-self.dividend_yield * T) * norm.cdf(d1)
                      - self.risk_free_rate * K * np.exp(-self.risk_free_rate * T) * norm.cdf(d2))

        put_theta = (-self.spot_price * np.exp(-self.dividend_yield * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                     - self.dividend_yield * self.spot_price * np.exp(-self.dividend_yield * T) * norm.cdf(-d1)
                     + self.risk_free_rate * K * np.exp(-self.risk_free_rate * T) * norm.cdf(-d2))
        
        straddle_theta = (call_theta + put_theta) / 365 * days
        return straddle_theta
    
    def simulate_hedge_pnl(self, K, T, current_iv, forecast_iv, spot_moves=None):
        if spot_moves is None:
            spot_moves = np.linspace(-0.10, 0.10, 21)
        
        vega_analysis = self.calculate_vega_pnl(K, T, current_iv, forecast_iv)
        vega_pnl = vega_analysis['vega_pnl']
        
        results = []
        for move in spot_moves:
            spot_change = self.spot_price * move
            gamma_pnl = self.calculate_gamma_pnl(K, T, current_iv, spot_change)
            theta_pnl = self.calculate_theta_pnl(K, T, current_iv, days=1)

            total_pnl = gamma_pnl + vega_pnl + theta_pnl
            
            results.append({
                'spot_move_pct': move * 100,
                'new_spot': self.spot_price * (1 + move),
                'gamma_pnl': gamma_pnl,
                'vega_pnl': vega_pnl,
                'theta_pnl': theta_pnl,
                'total_pnl': total_pnl
            })
        
        return pd.DataFrame(results)
    
    def plot_rehedge_requirements(self, K, T, sigma, spot_range=0.10):
        df = self.analyze_rehedge_points(K, T, sigma, spot_range)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.plot(df['spot_price'], df['delta'], 'b-', linewidth=2, label='Straddle Delta')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax1.fill_between(df['spot_price'], -0.10, 0.10, alpha=0.2, color='green', label='No Rehedge Zone (±0.10)')
        ax1.set_ylabel('Delta', fontsize=10)
        ax1.set_title('Straddle Delta vs Spot Price', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        rehedge_mask = df['needs_rehedge']
        ax2.scatter(df[rehedge_mask]['spot_price'], df[rehedge_mask]['delta_change'], 
                   color='red', s=100, label='Rehedge Required', zorder=5)
        ax2.scatter(df[~rehedge_mask]['spot_price'], df[~rehedge_mask]['delta_change'], 
                   color='green', s=50, label='No Rehedge', alpha=0.5)
        ax2.axhline(y=0.10, color='red', linestyle='--', alpha=0.5, label='Rehedge Threshold')
        ax2.set_xlabel('Spot Price', fontsize=10)
        ax2.set_ylabel('Delta Change from Initial', fontsize=10)
        ax2.set_title('Rehedge Triggers', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_hedge_pnl_breakdown(self, K, T, current_iv, forecast_iv):
        df = self.simulate_hedge_pnl(K, T, current_iv, forecast_iv)
        
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.fill_between(df['spot_move_pct'], 0, df['gamma_pnl'], alpha=0.6, label='Gamma P&L')
        ax.fill_between(df['spot_move_pct'], df['gamma_pnl'], 
                       df['gamma_pnl'] + df['vega_pnl'], alpha=0.6, label='Vega P&L')
        ax.fill_between(df['spot_move_pct'], df['gamma_pnl'] + df['vega_pnl'], 
                       df['total_pnl'], alpha=0.6, label='Theta P&L')
        
        ax.plot(df['spot_move_pct'], df['total_pnl'], 'k-', linewidth=2, label='Total P&L')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        ax.set_xlabel('Spot Move (%)', fontsize=10)
        ax.set_ylabel('P&L', fontsize=10)
        ax.set_title('Delta-Hedged Straddle P&L Breakdown (Gamma + Vega + Theta)', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def print_hedge_summary(self, K, T, sigma, forecast_iv=None):
        hedge = self.calculate_hedge_position(self.spot_price, K, T, sigma)
        
        print("\n" + "="*80)
        print("DELTA HEDGE SUMMARY")
        print("="*80)
        
        print(f"\nPosition Details:")
        print(f"  Spot Price:              ${self.spot_price:.2f}")
        print(f"  Strike:                  ${K:.2f}")
        print(f"  Time to Expiration:      {T:.3f} years ({T*12:.0f} months)")
        print(f"  Implied Volatility:      {sigma:.2%}")
        
        print(f"\nDelta Analysis:")
        print(f"  Straddle Delta:          {hedge['straddle_delta']:+.6f}")
        print(f"  Hedge Direction:         {hedge['hedge_direction']}")
        print(f"  Hedge Shares:            {abs(hedge['hedge_shares']):.6f}")
        print(f"  Hedge Notional:          ${abs(hedge['hedge_cost']):.2f}")
        print(f"  Residual Delta (shares): {hedge['residual_delta']:+.6f}")
        print(f"  Delta Neutral:           {'✓ YES' if hedge['is_delta_neutral'] else '✗ NO'}")
        
        if forecast_iv:
            vega_analysis = self.calculate_vega_pnl(K, T, sigma, forecast_iv)
            print(f"\nVega Exposure (if IV → {forecast_iv:.2%}):")
            print(f"  Current Value:           ${vega_analysis['current_value']:.4f}")
            print(f"  Forecast Value:          ${vega_analysis['forecast_value']:.4f}")
            print(f"  Vega P&L:                ${vega_analysis['vega_pnl']:+.4f} ({vega_analysis['vega_pnl_pct']:+.1f}%)")
        
        print("\n" + "="*80)