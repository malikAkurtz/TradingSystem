class Portfolio:
    def __init__(self, intial_capital=10000):
        self.initial_capital = intial_capital
        self.cash = intial_capital
        self.position = None
        self.trade_type = None # "long" meaning we bought SPXL and "short" meaning we bought SPXS
        self.units = 0
        self.holdings_value = 0.0
        self.equity_curve = []

    def update_on_fill(self, trade_type, side, quantity, price):
        trade_cost = price * quantity

        if side == "BUY":
            self.cash -= trade_cost
            self.units += quantity
        elif side == "SELL":
            self.cash += trade_cost
            self.units -= quantity

        self.holding_symbol = trade_type

    def update_equity(self, current_price):
        self.holdings_value = self.units * current_price
        total_equity = self.holdings_value + self.cash
        self.equity_curve.append(total_equity)
        return total_equity
    
    def get_current_position(self):
        return self.trade_type
    
    def flatten_position(self, current_price):
        if self.units > 0:
            self.update_on_fill("SELL", self.units, current_price)

        self.units = 0

    def get_equity_curve(self):
        return self.equity_curve