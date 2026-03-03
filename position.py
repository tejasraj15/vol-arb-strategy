import numpy as np
import pandas as pd
from scipy.stats import norm
from delta_Hedging import DeltaHedger
from hedging_transaction_costs import TransactionCostCalculator
from transactionCosts import TransactionCost
from implied_vol_surface import ImpliedVolSurface

RISK_FREE_RATE = 0.02
CONTRACT_MULTIPLIER = 100
STOP_LOSS_MULTIPLIER = 2.0
IV_SPIKE_THRESHOLD = 0.04
IV_DROP_THRESHOLD = -0.03
IV_CHECK_AFTER_DAYS = 3
MANDATORY_EXIT_DTE = 14


class Position:
    """Manages the full lifecycle of a single short straddle position."""

    def __init__(self, entry_date, strike, entry_dte, exdate, entry_iv,
                 garch_forecast, entry_credit, entry_tc, num_straddles,
                 entry_spot, initial_hedge_shares, initial_hedge_cost,
                 straddle_gamma, dividend_yield):

        self.entry_date = entry_date
        self.strike = strike
        self.entry_dte = entry_dte
        self.exdate = exdate
        self.entry_iv = entry_iv
        self.garch_forecast = garch_forecast
        self.entry_credit = entry_credit
        self.entry_tc = entry_tc
        self.num_straddles = num_straddles
        self.entry_spot = entry_spot
        self.dividend_yield = dividend_yield

        # Hedge state — updated daily by _rebalance_hedge()
        self._prev_spot = entry_spot
        self._hedge_shares = initial_hedge_shares
        self._prev_gamma = straddle_gamma

        self._cumulative_hedge_pnl = 0.0
        self._cumulative_gamma_pnl = 0.0
        self._cumulative_theta_pnl = 0.0
        self._cumulative_delta_pnl = 0.0
        self._total_hedge_cost = initial_hedge_cost

        self.current_iv: float | None = None
        self.current_straddle_value: float | None = None
        self.days_held: int = 0

    @property
    def iv_change(self) -> float | None:
        if self.current_iv is None:
            return None
        return self.current_iv - self.entry_iv

    @property
    def iv_change_pct(self) -> float | None:
        iv_chg = self.iv_change
        return None if iv_chg is None else iv_chg * 100

    @property
    def dte_remaining(self) -> int:
        return max(0, self.entry_dte - self.days_held)

    @classmethod
    def open(cls, current_date, spot_price, atm_option, market_iv, garch_forecast,
             dividend_yield, ticker, position_size, starting_capital,
             tc_calc: TransactionCost, verbose=False) -> "Position | None":
        """Returns None if the trade should be skipped (size / risk guard)."""
        straddle_price_per_unit = atm_option['straddle_price'] * CONTRACT_MULTIPLIER
        num_straddles = max(1, int(position_size / straddle_price_per_unit))
        entry_credit = straddle_price_per_unit * num_straddles

        # Risk guard: cap at 15% of starting capital
        max_notional_risk = starting_capital * 0.15
        if entry_credit > max_notional_risk:
            if verbose:
                print(f"\n  [{current_date.date()}] SKIPPED: Position too large "
                      f"(${entry_credit:.0f} > ${max_notional_risk:.0f})")
            return None

        entry_tc = tc_calc.calculate(
            price=atm_option['straddle_price'],
            contracts=num_straddles * 2,
            ticker=ticker.upper()
        )

        hedger = DeltaHedger(spot_price, risk_free_rate=RISK_FREE_RATE, dividend_yield=dividend_yield)
        hedge_result = hedger.calculate_hedge_position(
            spot_price, atm_option['strike'], atm_option['maturity'],
            market_iv, position_sign=-1,
            num_straddles=num_straddles,
            contract_multiplier=CONTRACT_MULTIPLIER,
        )
        initial_hedge_shares = hedge_result['hedge_shares']
        if abs(initial_hedge_shares) > 0:
            initial_hedge_cost_info = TransactionCostCalculator.calculate(
                shares=abs(initial_hedge_shares),
                price=spot_price,
                is_buy=initial_hedge_shares > 0
            )
            initial_hedge_cost = initial_hedge_cost_info.total_cost
        else:
            initial_hedge_cost = 0

        d1 = (np.log(spot_price / atm_option['strike']) +
              (RISK_FREE_RATE - dividend_yield + 0.5 * market_iv ** 2) * atm_option['maturity']) / \
             (market_iv * np.sqrt(atm_option['maturity']))
        straddle_gamma = np.exp(-dividend_yield * atm_option['maturity']) * norm.pdf(d1) / \
                         (spot_price * market_iv * np.sqrt(atm_option['maturity'])) * 2

        return cls(
            entry_date=current_date,
            strike=atm_option['strike'],
            entry_dte=atm_option['dte'],
            exdate=atm_option['exdate'],
            entry_iv=market_iv,
            garch_forecast=garch_forecast,
            entry_credit=entry_credit,
            entry_tc=entry_tc,
            num_straddles=num_straddles,
            entry_spot=spot_price,
            initial_hedge_shares=initial_hedge_shares,
            initial_hedge_cost=initial_hedge_cost,
            straddle_gamma=straddle_gamma,
            dividend_yield=dividend_yield,
        )

    def update(self, current_date, spot_price, options_by_date: dict) -> bool:
        """Reprice, recalculate IV, and rebalance the delta hedge. Returns False if unable to price."""
        self.days_held = (current_date - self.entry_date).days

        current_call, current_put = self._find_current_option_prices(current_date, options_by_date)
        if current_call is None:
            return False

        self.current_straddle_value = (current_call + current_put) * CONTRACT_MULTIPLIER * self.num_straddles

        current_maturity = max(0.01, self.dte_remaining / 365)

        iv_calc = ImpliedVolSurface(
            spot_price=spot_price,
            risk_free_rate=RISK_FREE_RATE,
            dividend_yield=self.dividend_yield,
            verbose=False
        )
        self.current_iv = self._get_current_iv(iv_calc, current_maturity, current_call, current_put)
        if self.current_iv is None:
            return False

        self._rebalance_hedge(spot_price, current_maturity)
        return True

    def check_exit(self, current_date, earnings_blocker=None) -> tuple[bool, str | None]:
        """Returns (should_exit, reason). Priority: earnings exit → stop loss → DTE → IV rules."""
        if self.current_iv is None or self.current_straddle_value is None:
            return False, None

        option_pnl = self.entry_credit - self.current_straddle_value
        iv_chg = self.iv_change

        if earnings_blocker and earnings_blocker.should_force_exit(current_date):
            return True, earnings_blocker.get_block_reason()

        if option_pnl < -STOP_LOSS_MULTIPLIER * self.entry_credit:
            return True, f"Position stop loss (2x premium): P&L ${option_pnl:.0f}"

        if self.dte_remaining <= MANDATORY_EXIT_DTE:
            return True, f"Mandatory exit (DTE <= {MANDATORY_EXIT_DTE}): DTE {self.dte_remaining}"

        if self.days_held >= IV_CHECK_AFTER_DAYS:
            if iv_chg >= IV_SPIKE_THRESHOLD:
                return True, f"Day {self.days_held} stop loss (IV {self.iv_change_pct:.1f}%)"
            if iv_chg <= IV_DROP_THRESHOLD:
                return True, f"Day {self.days_held} take profit (IV {self.iv_change_pct:.1f}%)"

        return False, None

    def close(self, current_date, spot_price, tc_calc: TransactionCost, ticker: str) -> dict:
        """Unwind hedge, calculate net P&L, and return a trade record dict."""
        option_pnl = self.entry_credit - self.current_straddle_value
        iv_chg = self.iv_change

        exit_tc = tc_calc.calculate(
            price=self.current_straddle_value / (CONTRACT_MULTIPLIER * self.num_straddles),
            contracts=self.num_straddles * 2,
            ticker=ticker.upper()
        )

        hedge_unwind_cost = 0.0
        if abs(self._hedge_shares) > 0:
            hedge_unwind_cost_info = TransactionCostCalculator.calculate(
                shares=abs(self._hedge_shares),
                price=spot_price,
                is_buy=self._hedge_shares < 0,
            )
            hedge_unwind_cost = hedge_unwind_cost_info.total_cost

        gross_pnl = option_pnl + self._cumulative_hedge_pnl
        net_pnl = gross_pnl - self.entry_tc - exit_tc - hedge_unwind_cost

        return {
            'entry_date': self.entry_date,
            'exit_date': current_date,
            'days_held': self.days_held,
            'strike': self.strike,
            'entry_iv': self.entry_iv,
            'exit_iv': self.current_iv,
            'iv_change': iv_chg,
            'iv_change_pct': self.iv_change_pct,
            'entry_credit': self.entry_credit,
            'exit_cost': self.current_straddle_value,
            'option_pnl': option_pnl,
            'gamma_pnl': self._cumulative_gamma_pnl,
            'theta_pnl': self._cumulative_theta_pnl,
            'delta_hedge_pnl': self._cumulative_delta_pnl,
            'hedge_pnl': self._cumulative_hedge_pnl,
            'gross_pnl': gross_pnl,
            'entry_tc': self.entry_tc,
            'exit_tc': exit_tc,
            'hedge_rebalance_costs': self._total_hedge_cost,
            'hedge_unwind_cost': hedge_unwind_cost,
            'net_pnl': net_pnl,
            'garch_forecast': self.garch_forecast,
            'num_straddles': self.num_straddles,
        }

    def _find_current_option_prices(self, current_date, options_by_date) -> tuple[float | None, float | None]:
        date_options = options_by_date.get(current_date)
        if date_options is None:
            return None, None

        strike_options = date_options[date_options['strike_price'] == self.strike]
        calls = strike_options[strike_options['cp_flag'] == 'C']
        puts = strike_options[strike_options['cp_flag'] == 'P']

        if calls.empty or puts.empty:
            return None, None

        calls = calls.iloc[(calls['exdate'] - self.exdate).abs().argsort()[:1]]
        puts = puts.iloc[(puts['exdate'] - self.exdate).abs().argsort()[:1]]

        return calls['market_price'].iloc[0], puts['market_price'].iloc[0]

    def _get_current_iv(self, iv_calc, current_maturity, call_price, put_price) -> float | None:
        try:
            call_iv = iv_calc.implied_volatility(call_price, self.strike, current_maturity, 'call')
            put_iv = iv_calc.implied_volatility(put_price, self.strike, current_maturity, 'put')
            ivs = [v for v in [call_iv, put_iv] if not np.isnan(v)]
            return float(np.mean(ivs)) if ivs else None
        except Exception:
            return None

    def _rebalance_hedge(self, spot_price, current_maturity):
        hedger = DeltaHedger(spot_price, risk_free_rate=RISK_FREE_RATE, dividend_yield=self.dividend_yield)
        hedge_result = hedger.calculate_hedge_position(
            spot_price, self.strike, current_maturity, self.current_iv,
            position_sign=-1, num_straddles=self.num_straddles,
            contract_multiplier=CONTRACT_MULTIPLIER,
        )

        spot_change = spot_price - self._prev_spot
        daily_delta_pnl = self._hedge_shares * spot_change

        hedge_rebalance_cost = 0.0
        target_shares = hedge_result['hedge_shares']
        if self.days_held % 2 == 0:  # rebalance every other day to reduce gamma bleed
            shares_to_trade = target_shares - self._hedge_shares
            if abs(shares_to_trade) > 0:
                cost_info = TransactionCostCalculator.calculate(
                    shares=abs(shares_to_trade),
                    price=spot_price,
                    is_buy=shares_to_trade > 0
                )
                hedge_rebalance_cost = cost_info.total_cost
            self._hedge_shares = target_shares

        d1 = (np.log(spot_price / self.strike) +
              (RISK_FREE_RATE - self.dividend_yield + 0.5 * self.current_iv ** 2) * current_maturity) / \
             (self.current_iv * np.sqrt(current_maturity))
        gamma = np.exp(-self.dividend_yield * current_maturity) * norm.pdf(d1) / \
                (spot_price * self.current_iv * np.sqrt(current_maturity)) * 2

        gamma_pnl = -0.5 * self._prev_gamma * (spot_change ** 2) * self.num_straddles * CONTRACT_MULTIPLIER
        theta_pnl = -hedger.calculate_theta_pnl(
            self.strike, current_maturity, self.current_iv, days=1
        ) * self.num_straddles * CONTRACT_MULTIPLIER

        self._cumulative_hedge_pnl += daily_delta_pnl - hedge_rebalance_cost
        self._cumulative_delta_pnl += daily_delta_pnl
        self._cumulative_gamma_pnl += gamma_pnl
        self._cumulative_theta_pnl += theta_pnl
        self._total_hedge_cost += hedge_rebalance_cost

        self._prev_spot = spot_price
        self._prev_gamma = gamma
