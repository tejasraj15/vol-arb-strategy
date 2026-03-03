from dataclasses import dataclass


@dataclass
class TransactionCostBreakdown:
    """Detailed breakdown of transaction costs"""
    total_cost: float
    commission: float
    sec_fee: float
    finra_taf: float
    clearing_fee: float
    pass_through_fees: float
    shares_traded: float
    trade_value: float  # Dollar value of trade
    cost_bps: float  # Cost in basis points of trade value
    is_buy: bool
    
    def __str__(self):
        trade_type = "BUY" if self.is_buy else "SELL"
        return f"""
Transaction Cost Breakdown ({trade_type}):
═══════════════════════════════════════
Commission:          ${self.commission:,.2f}
SEC Fee (sells):     ${self.sec_fee:,.2f}
FINRA TAF (sells):   ${self.finra_taf:,.2f}
Clearing Fee:        ${self.clearing_fee:,.2f}
Pass-Through Fees:   ${self.pass_through_fees:,.2f}
───────────────────────────────────────
TOTAL COST:          ${self.total_cost:,.2f}
═══════════════════════════════════════
Shares Traded:       {self.shares_traded:,.0f}
Trade Value:         ${self.trade_value:,.2f}
Cost (bps):          {self.cost_bps:.2f} bps
"""


class TransactionCostCalculator:
    """Calculate transaction costs for a single stock trade based on IBKR fee structure."""
    
    # IBKR Tiered Commission Structure (per share)
    COMMISSION_TIERS = [
        (300_000, 0.0035),      # First 300K shares/month
        (3_000_000, 0.0020),    # 300K - 3M shares
        (20_000_000, 0.0015),   # 3M - 20M shares
        (100_000_000, 0.0010),  # 20M - 100M shares
        (float('inf'), 0.0005)  # >100M shares
    ]
    
    # Per-order constraints
    MIN_COMMISSION_PER_ORDER = 0.35
    MAX_COMMISSION_RATE = 0.01  # 1% of trade value
    
    # Regulatory fees (per share or per dollar)
    SEC_FEE_PER_DOLLAR = 0.0000278  # Sells only: $27.80 per million
    FINRA_TAF_PER_SHARE = 0.000166  # Sells only, capped at $8.30
    FINRA_TAF_CAP = 8.30
    
    # Clearing fees
    CLEARING_FEE_PER_SHARE = 0.00020  # NSCC/DTC, buys and sells
    
    # Pass-through fees (multipliers on commission)
    NYSE_FEE_MULTIPLIER = 0.000175
    FINRA_FEE_MULTIPLIER = 0.000563
    
    @staticmethod
    def calculate(
        shares: float,
        price: float,
        is_buy: bool = True
    ) -> TransactionCostBreakdown:
        calculator = TransactionCostCalculator()
        return calculator.calculate_costs(shares, price, is_buy)
    
    def calculate_costs(
        self,
        shares: float,
        price: float,
        is_buy: bool = True
    ) -> TransactionCostBreakdown:
        shares = abs(shares)  # Ensure positive
        
        if shares <= 0 or price <= 0:
            return TransactionCostBreakdown(
                total_cost=0.0,
                commission=0.0,
                sec_fee=0.0,
                finra_taf=0.0,
                clearing_fee=0.0,
                pass_through_fees=0.0,
                shares_traded=0.0,
                trade_value=0.0,
                cost_bps=0.0,
                is_buy=is_buy
            )
        
        trade_value = shares * price
        
        # Commission
        commission = self._calculate_commission(shares, trade_value)
        
        # Regulatory fees (sells only)
        sec_fee = 0.0
        finra_taf = 0.0
        if not is_buy:
            sec_fee = trade_value * self.SEC_FEE_PER_DOLLAR
            finra_taf = min(shares * self.FINRA_TAF_PER_SHARE, self.FINRA_TAF_CAP)
        
        # Clearing fees (buys and sells)
        clearing_fee = shares * self.CLEARING_FEE_PER_SHARE
        
        # Pass-through fees (based on commission)
        pass_through = commission * (self.NYSE_FEE_MULTIPLIER + self.FINRA_FEE_MULTIPLIER)
        
        # Calculate total cost
        total_cost = commission + sec_fee + finra_taf + clearing_fee + pass_through
        
        # Calculate basis points
        cost_bps = (total_cost / trade_value) * 10000 if trade_value > 0 else 0
        
        return TransactionCostBreakdown(
            total_cost=total_cost,
            commission=commission,
            sec_fee=sec_fee,
            finra_taf=finra_taf,
            clearing_fee=clearing_fee,
            pass_through_fees=pass_through,
            shares_traded=shares,
            trade_value=trade_value,
            cost_bps=cost_bps,
            is_buy=is_buy
        )
    
    def _calculate_commission(self, shares: float, trade_value: float) -> float:
        # Use first tier rate for conservative estimate (no monthly aggregation)
        rate_per_share = self.COMMISSION_TIERS[0][1]
        commission = shares * rate_per_share
        
        # Apply minimum
        commission = max(commission, self.MIN_COMMISSION_PER_ORDER)
        
        # Apply maximum (1% of trade value)
        max_commission = trade_value * self.MAX_COMMISSION_RATE
        commission = min(commission, max_commission)
        
        return commission