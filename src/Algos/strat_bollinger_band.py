from strategy import Strategy
import pandas as pd
import time
import datetime
from datetime import time




class BollingerBandStrat(Strategy):
    def __init__(self, BB_period=20, BB_std=2):
        self.bb_period = BB_period
        self.bb_std = BB_std
        self.start_trade_window = time(17,30)
        self.end_trade_window = time(19,20)
        self.eod_time = time(21,00)
        self.position_open = False
        self.roundtrip_complete = False
        self.today = None



    def calculate_indicators(self, df: pd.DataFrame):
        df["SMA"] = df["SPY_close"].rolling(window=self.bb_period).mean()
        df["STD"] = df["SPY_close"].rolling(window=self.bb_period).std();
        df["Upper_BB"] = df["SMA"] + (self.bb_std * df["STD"])
        df["Lower_BB"] = df["SMA"] - (self.bb_std * df["STD"])

        return df
    
    def generate_signals(self, timestamp, row, holdings):
        # if its a new day, we have yet to make a roundtrip trade today
        print(self.position_open, self.roundtrip_complete)
        if (self.today != timestamp.day):
            self.today = timestamp.day
            self.roundtrip_complete = False

        # if weve already made a roundtrip trade today, return none
        if self.roundtrip_complete:
            return None
        
        # if we dont have enough data for indicators, return none
        if (pd.isna(row['SMA'])):
            return None

        
        current_dt = datetime.datetime.combine(datetime.date.today(), timestamp.time())
        eod_dt = datetime.datetime.combine(datetime.date.today(), self.eod_time)
        diff = eod_dt - current_dt

        # if we are less than 5 minutes from eod and in positions, close positions
        if (abs(diff) <= datetime.timedelta(minutes=5) and (current_dt < eod_dt)):
            if holdings["SPY"] > 0:
                self.position_open = False
                return "CLOSE_LONG"
            elif holdings["SPY"] < 0:
                self.position_open = False
                return "CLOSE_SHORT"    
        
        # if we are not in a position, and not during lunch time, return none
        if (not self.position_open) and not (self.start_trade_window <= timestamp.time() <= self.end_trade_window):
            return None
                
        close = row['SPY_close']
        upper_BB = row['Upper_BB']
        lower_BB = row['Lower_BB']

        if not self.position_open and self.start_trade_window <= timestamp.time() <= self.end_trade_window:
            # If the latest bar closes above the upper bollinger band
            if close > upper_BB:
                # Buy leveraged SPY
                self.position_open = True
                return "BUY_LONG"
            # If the latest bar closes below the lower bollinger band
            elif close < lower_BB:
                # Buy leveraged inverse SPY
                self.position_open = True
                return "SELL_SHORT"
        # If we are in a position
        elif holdings["SPY"] > 0:
            # If the latest bar closes above the upper bollinger band
            if close > upper_BB:
                # Sell 1/4 of position
                if holdings["SPY"] == 25:
                    self.position_open = False
                    self.roundtrip_complete = True
                return "PARTIAL_PROFIT_LONG"
            # If the latest bar closes below the lower bollinger band
            elif close < lower_BB:
                # Flatten position
                self.position_open = False
                self.roundtrip_complete = True
                return "CLOSE_LONG"
        elif holdings["SPY"] < 0:
            # If the latest bar closes below the lower bollinger band
            if close < lower_BB:
                # Sell 1/4 of position
                if holdings["SPY"] == -25:
                    self.position_open = False
                    self.roundtrip_complete = True
                return "PARTIAL_PROFIT_SHORT"
            # If the latest bar closes above the upper bollinger band
            elif close > upper_BB:
                # Flatten position
                self.position_open = False
                self.roundtrip_complete = True
                return "CLOSE_SHORT"
        
        
        return None
