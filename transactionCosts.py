class TransactionCost:
    def __init__(
        self,
        us_per_contract=0.65,          # USD
        us_min_per_order=1.00,         # USD
        us_regulatory_per_contract=0.05,  # ORF + OCC approx
        us_fx_bps=3,                   # 0.03%
        uk_per_contract=1.00           # GBP
    ):
        self.us_per_contract = us_per_contract
        self.us_min_per_order = us_min_per_order
        self.us_regulatory_per_contract = us_regulatory_per_contract
        self.us_fx_bps = us_fx_bps
        self.uk_per_contract = uk_per_contract

    def is_us_option(self, ticker: str) -> bool:
        # needs to be made more complex in future
        return ticker.isupper() and ticker.isalpha()

    def calculate(
        self,
        price: float,
        contracts: int,
        ticker: str,
        fx_rate: float = 1.0
    ) -> float:
        """
        Returns total transaction cost in BASE CURRENCY
        (GBP if fx_rate converts USD → GBP)
        """

        if self.is_us_option(ticker):
            commission = max(
                self.us_min_per_order,
                contracts * self.us_per_contract
            )

            regulatory = contracts * self.us_regulatory_per_contract
            fx_cost = price * contracts * (self.us_fx_bps / 10_000)

            total_usd = commission + regulatory + fx_cost
            return total_usd * fx_rate

        else:
            return contracts * self.uk_per_contract
