import numpy as np
import pandas as pd


class EarningsBlocker:
    def __init__(self, ticker, earnings_csv='earnings_data.csv',
                 before_days=7, after_days=2, exit_days=3, verbose=False):
        self.ticker = ticker.upper()
        self.before_days = before_days
        self.after_days = after_days
        self.exit_days = exit_days
        self.verbose = verbose
        
        self.earnings_dates = self._load_earnings(earnings_csv)
        self.enabled = len(self.earnings_dates) > 0
        
        self.entries_blocked = 0
        self.forced_exits = 0
        self.last_block_reason = None
    
    def _load_earnings(self, filepath):
        """Load and sort earnings dates for this ticker."""
        try:
            df = pd.read_csv(filepath)
            df['rdq'] = pd.to_datetime(df['rdq'])
            
            ticker_data = df[df['tic'] == self.ticker]
            
            if len(ticker_data) == 0:
                if self.verbose:
                    print(f"  [EarningsBlocker] No earnings data for {self.ticker}")
                return np.array([])
            
            earnings_dates = ticker_data['rdq'].sort_values().values
            
            if self.verbose:
                print(f"  [EarningsBlocker] Loaded {len(earnings_dates)} earnings dates for {self.ticker}")
                print(f"    First: {pd.Timestamp(earnings_dates[0]).date()}")
                print(f"    Last:  {pd.Timestamp(earnings_dates[-1]).date()}")
            
            return earnings_dates
            
        except Exception as e:
            if self.verbose:
                print(f"  [EarningsBlocker] Error loading earnings: {e}")
            return np.array([])
    
    def _find_nearest_earnings(self, current_date):
        if not self.enabled:
            return None, None
        
        current_dt = pd.Timestamp(current_date)
        current_np = current_dt.to_datetime64()
        idx = np.searchsorted(self.earnings_dates, current_np)
        
        prev_earnings = None
        next_earnings = None
        
        if idx > 0:
            prev_earnings = pd.Timestamp(self.earnings_dates[idx - 1])
        
        if idx < len(self.earnings_dates):
            next_earnings = pd.Timestamp(self.earnings_dates[idx])
        
        return prev_earnings, next_earnings
    
    def should_block_entry(self, current_date):
        if not self.enabled:
            return False
        
        prev_earn, next_earn = self._find_nearest_earnings(current_date)
        current_dt = pd.Timestamp(current_date)
        
        # Check previous earnings (within after_days)
        if prev_earn is not None:
            days_since = (current_dt - prev_earn).days
            if 0 <= days_since <= self.after_days:
                self.last_block_reason = f"{days_since}d after earnings ({prev_earn.date()})"
                self.entries_blocked += 1
                if self.verbose:
                    print(f"  [EarningsBlocker] Entry blocked: {self.last_block_reason}")
                return True
        
        # Check next earnings (within before_days)
        if next_earn is not None:
            days_until = (next_earn - current_dt).days
            if 0 <= days_until <= self.before_days:
                self.last_block_reason = f"{days_until}d before earnings ({next_earn.date()})"
                self.entries_blocked += 1
                if self.verbose:
                    print(f"  [EarningsBlocker] Entry blocked: {self.last_block_reason}")
                return True
        
        return False
    
    def should_force_exit(self, current_date):
        if not self.enabled:
            return False
        
        _, next_earn = self._find_nearest_earnings(current_date)
        
        if next_earn is not None:
            current_dt = pd.Timestamp(current_date)
            days_until = (next_earn - current_dt).days
            
            if 0 < days_until <= self.exit_days:
                self.last_block_reason = f"Earnings in {days_until}d - forced exit ({next_earn.date()})"
                self.forced_exits += 1
                if self.verbose:
                    print(f"  [EarningsBlocker] Forced exit: {self.last_block_reason}")
                return True
        
        return False
    
    def get_block_reason(self):
        return self.last_block_reason
    
    def get_stats(self):
        return {
            'ticker': self.ticker,
            'enabled': self.enabled,
            'earnings_count': len(self.earnings_dates),
            'entries_blocked': self.entries_blocked,
            'forced_exits': self.forced_exits,
            'total_interventions': self.entries_blocked + self.forced_exits
        }
    
    def print_stats(self):
        stats = self.get_stats()
        print(f"\n  === EARNINGS BLOCKER STATS ({stats['ticker']}) ===")
        print(f"    Enabled:          {stats['enabled']}")
        print(f"    Earnings loaded:  {stats['earnings_count']}")
        print(f"    Entries blocked:  {stats['entries_blocked']}")
        print(f"    Forced exits:     {stats['forced_exits']}")
        print(f"    Total blocks:     {stats['total_interventions']}")
    
    def __repr__(self):
        return f"EarningsBlocker({self.ticker}, before={self.before_days}, after={self.after_days}, exit={self.exit_days})"