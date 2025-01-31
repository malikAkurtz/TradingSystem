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

            if (self.start_time <= current_time <= self.end_time):
                
