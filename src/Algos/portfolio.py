
class Portfolio:
    def __init__(self, intial_capital=10000):
        self.initial_capital = intial_capital
        self.cash = intial_capital
        self.holdings = {"SPY" : 0}
        self.holdings_value = 0.0
        self.equity_curve = []

    def update_on_fill(self, symbol, side, quantity, price):
        trade_cost = price * quantity

        if side == "BUY":
            self.cash -= trade_cost
            self.holdings[symbol] += quantity
        elif side == "SELL":
            self.cash += trade_cost
            self.holdings[symbol] -= quantity

    def update_equity(self, row):
        self.holdings_value = 0
        for symbol, units in self.holdings.items():
            self.holdings_value += units * row[symbol + "_close"]
        total_equity = self.holdings_value + self.cash
        self.equity_curve.append(total_equity)
        return total_equity
    
    def get_current_positions(self):
        return self.holdings
    
    def flatten_positions(self, row):
        for symbol, units in self.holdings.items():
            if units > 0:
                self.update_on_fill(symbol, "SELL", units, row[symbol + "_close"])
                self.holdings[symbol] = 0
            elif units < 0:
                self.update_on_fill(symbol, "BUY", units, row[symbol + "_close"])
                self.holdings[symbol] = 0


    def get_equity_curve(self):
        return self.equity_curve