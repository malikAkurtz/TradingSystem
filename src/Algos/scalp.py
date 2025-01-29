import numpy as np
import pandas as pd
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, time, timedelta
import pytz

from config import API_KEY, API_SECRET, BASE_URL
from alpaca.trading.client import TradingClient

trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

symbol = "SPY"
request_params = StockBarsRequest(symbol_or_symbols="SPY",
                                            timeframe = TimeFrame(3, TimeFrameUnit.Minute),
                                            start=datetime.utcnow() - timedelta(days=1))

def within_trade_window(input_time):
    est = pytz.timezone("US/Eastern")

    start_time = datetime.time(12, 30)
    end_time = datetime.time(2, 30)

    input_time_est = input_time.astimezone(est)

    return start_time <= input_time_est <= end_time

class Order:
    def __init__(self, side, quantity, price):
        self.side = side
        self.quantity = quantity
        self.price = price

def calculate_SMA(prices, period):
    return (sum(prices) / len(prices))

def calculate_STD(prices, sma_value):
    for ()

def calculate_BB(prices, sma_value, standard_deviations = 2):
    upper_BB = sma_value + standard_deviations * ()
    return



def strat():
    traded_today = False
    # if were in the window in which we can open trades
    if (within_trade_window(trading_client.get_clock())):
        
        # if we havent already opened a position
        if (not trading_client.get_all_positions()):

            bars = data_client.get_stock_bars(request_params)

            candles = pd.DataFrame(bars["SPY"])

def main():
    bars = data_client.get_stock_bars(request_params)
    candles = pd.DataFrame(bars["SPY"])

    return print(candles)

main()