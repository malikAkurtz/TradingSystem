import strategy
import pandas as pd

class BollingerBandStrat(strategy):
    def __init__(self, BB_period=20, BB_std=2):
        self.bb_period = BB_period
        self.bb_std = BB_std


    def calculate_indicators(self, df: pd.DataFrame):
        df["SMA"] = df["close"].rolling(window=self.bb_period).mean()
        df["STD"] = df["close"].rolling(window=self.bb_period).std();
        df["Upper_BB"] = df["SMA"] + (self.bb_std * df["STD"])
        df["Lower_BB"] = df["SMA"] - (self.bb_std * df["STD"])

        return df
    
    def generate_signals(self, row, holdings):
        close = row['close']
        upper_BB = row['Upper_BB']
        lower_BB = row['Lower_BB']

        if holdings['SPXL'] == 0 and holdings['SPXS'] == 0:
            # If the latest bar closes above the upper bollinger band
            if close > upper_BB:
                # Buy leveraged SPY
                return "BUY_LONG"
            # If the latest bar closes below the lower bollinger band
            elif close < lower_BB:
                # Buy leveraged inverse SPY
                return "BUY_SHORT"
        # If we are in a position
        elif holdings['SPXL'] != 0:
            # If the latest bar closes above the upper bollinger band
            if close > upper_BB:
                # Sell 1/4 of position
                return "PARTIAL_PROFIT_LONG"
            # If the latest bar closes below the lower bollinger band
            elif close < lower_BB:
                # Flatten position
                return "CLOSE_LONG"
        elif holdings['SPXS'] != 0:
            # If the latest bar closes below the lower bollinger band
            if close < lower_BB:
                # Sell 1/4 of position
                return "PARTIAL_PROFIT_SHORT"
            # If the latest bar closes above the upper bollinger band
            elif close > upper_BB:
                # Flatten position
                return "CLOSE_SHORT"
        
        return None
