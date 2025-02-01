import pandas as pd
from datetime import time
from data_handler import DataHandler
from strat_bollinger_band import BollingerBandStrat
from portfolio import Portfolio
from execution import SimulatedExecutionHandler
import matplotlib.pyplot as plt
import datetime as datetime

class BacktestEngine:
    def __init__(self, data_handler, strategy, portfolio, execution_handler, start_time=time(17,30), end_time=time(19,30), eod_time=time(21,00)):
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        self.start_time = start_time
        self.end_time = end_time
        self.eod_time = eod_time
        self.trades = []

    def run_backtest(self):
        # load the data
        df = self.data_handler.load_data()
        # add indicators to data
        df = self.strategy.calculate_indicators(df)

        df['equity'] = float('nan')

        current_day = None

        days_starting_cash = self.portfolio.cash
        days_ending_cash = None
        # iterate over every row in the dataframe
        for timestamp, row in df.iterrows():

            current_time = timestamp.time()

            if (current_day != timestamp.day):
                current_day = timestamp.day
                print("<"*15, "New Day", ">"*15)
                print("Starting Cash: ", days_starting_cash)

            print("-"*10, timestamp, "-"*10)
            # get the current state of the portfolio
            position = self.portfolio.get_current_positions()
            print("Current Holdings: " , position)

            
            # need at least an initial amount of rows to calculate indicators
            if (pd.isna(row['SMA'])):
                equity = self.portfolio.update_equity(row)
                df.loc[timestamp, 'equity'] = equity
                continue

            

            current_dt = datetime.datetime.combine(datetime.date.today(), current_time)
            eod_dt = datetime.datetime.combine(datetime.date.today(), self.eod_time)
            diff = eod_dt - current_dt

            if (abs(diff) <= datetime.timedelta(minutes=5) and (current_dt < eod_dt)):
                print("Trade: Flattening Before Close")
                self.portfolio.flatten_positions(row)
                equity = self.portfolio.update_equity(row)
                df.loc[timestamp, 'equity'] = equity
                print("<"*15, "End of Day", ">"*15)
                print("Ending Cash: ", self.portfolio.cash)
                print("Days P/L: ", self.portfolio.cash - days_starting_cash)
                days_starting_cash = self.portfolio.cash
                continue

            if not (self.start_time <= current_time <= self.end_time):
                equity = self.portfolio.update_equity(row)
                df.loc[timestamp, 'equity'] = equity
                continue

            signal = self.strategy.generate_signals(row, position)

            if signal is not None:
                if signal == "BUY_LONG":
                    fill_quantity, fill_price = self.execution_handler.execute_order("BUY", 4, row['SPY_close'])
                    print(f"Trade: Opening Long, {fill_quantity} @ {fill_price}")
                    self.portfolio.update_on_fill("SPY", "BUY", fill_quantity, fill_price)
                    print(f"Now Holding: {self.portfolio.holdings}")

                elif signal == "SELL_SHORT":
                    fill_quantity, fill_price = self.execution_handler.execute_order("SELL", 4, row['SPY_close'])
                    print(f"Trade: Opening Short, {fill_quantity} @ {fill_price}")
                    self.portfolio.update_on_fill("SPY", "SELL", fill_quantity, fill_price)
                    print(f"Now Holding: {self.portfolio.holdings}")

                elif signal == "PARTIAL_PROFIT_LONG":
                    fill_quantity, fill_price = self.execution_handler.execute_order("SELL", 1, row['SPY_close'])
                    print(f"Trade: Selling Profit, {fill_quantity} @ {fill_price}")
                    self.portfolio.update_on_fill("SPY", "SELL", fill_quantity, fill_price)
                    print(f"Now Holding: {self.portfolio.holdings}")

                elif signal == "PARTIAL_PROFIT_SHORT":
                    fill_quantity, fill_price = self.execution_handler.execute_order("BUY", 1, row['SPY_close'])
                    print(f"Trade: Buying Profit, {fill_quantity} @ {fill_price}")
                    self.portfolio.update_on_fill("SPY", "BUY", fill_quantity, fill_price)
                    print(f"Now Holding: {self.portfolio.holdings}")

                elif signal == "CLOSE_LONG":
                    print("Trade: Flattening Long")
                    self.portfolio.flatten_positions(row)
                    print(f"Now Holding: {self.portfolio.holdings}")

                elif signal == "CLOSE_SHORT":
                    print("Trade: Flattening Short")
                    self.portfolio.flatten_positions(row)
                    print(f"Now Holding: {self.portfolio.holdings}")

                self.trades.append((timestamp, signal, row['SPY_close']))

            equity = self.portfolio.update_equity(row)
            df.loc[timestamp, 'equity'] = equity
        
        return df, self.trades
    
    def plot_equity(self, df):
        plt.plot(df.index, df['equity'])
        plt.show()
            