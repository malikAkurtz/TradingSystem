import pandas as pd
from datetime import time
from .data_handler import DataHandler
from .strat_bollinger_band import BollingerBandStrat
from .portfolio import Portfolio
from .execution import SimulatedExecutionHandler

class BacktestEngine:
    def __init__(self, data_handler, strategy, portfolio, execution_handler, start_time=time(12,30), end_time=time(14,30), eod_time=time(16,00)):
        self.data_hander = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        self.start_time = start_time
        self.end_time = end_time
        self.eod_time = eod_time
        self.trades = []

    def run_backtest(self):
        # load the data
        df = self.data_hander.load_data()
        # add indicators to data
        df = self.strategy.calculate_indicators(df)

        # iterate over every row in the dataframe
        for timestamp, row in df.iterrows():
            # get the current state of the portfolio
            position = self.portfolio.get_current_position()
            
            # need at least an initial amount of rows to calculate indicators
            if (pd.isna(row['SMA'])):
                continue

            current_time = timestamp.time()

            if (position is not None and (self.eod_time.hour == current_time.hour and self.end_time.minute - current_time.minte <= 5)):
                self.portfolio.flatten_position(row['SPY_close'])
                continue

            if not (self.start_time <= current_time <= self.end_time):
                continue

            signal = self.strategy.generate_signals(row, position)

            if signal is not None:
                if signal == "BUY_LONG":
                    fill_price, fill_quantity = self.execution_handler("BUY", 4, row['SPXL'])
                    self.portfolio.update_on_fill("SPXL", "BUY", fill_quantity, fill_price)
                elif signal == "BUY_SHORT":
                    fill_price, fill_quantity = self.execution_handler("BUY", 4, row['SPXS'])
                    self.portfolio.update_on_fill("SPXS", "BUY", fill_quantity, fill_price)
                elif signal == "PARTIAL_PROFIT_LONG":
                    fill_price, fill_quantity = self.execution_handler("SELL", 1, row['SPXL'])
                    self.portfolio.update_on_fill("SPXL", "SELL", fill_quantity, fill_price)
                elif signal == "PARTIAL_PROFIT_SHORT":
                    fill_price, fill_quantity = self.execution_handler("SELL", 1, row['SPXS'])
                    self.portfolio.update_on_fill("SPXS", "SELL", fill_quantity, fill_price)
                elif signal == "CLOSE_LONG":
                    self.portfolio.flatten_position(row['SPXL'])
                elif signal == "CLOSE_SHORT":
                    self.portfolio.flatten_position(row['SPXS'])

                self.trades.append((timestamp, signal, row['SPY']))

            self.portfolio.update_equity(row)
        
        return self.portfolio.get_equity_curve(), self.trades
            