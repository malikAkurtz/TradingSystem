import numpy as np
import pandas as pd
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import datetime

from config import API_KEY, API_SECRET, BASE_URL
from alpaca.trading.client import TradingClient

trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

symbol = "SPY"

request_params = StockBarsRequest(symbol_or_symbols="SPY",
                                  timeframe = TimeFrame(3, TimeFrameUnit.Minute),
                                  start=datetime.datetime.utcnow() - datetime.timedelta(days=1))

bars = data_client.get_stock_bars(request_params)

df = pd.DataFrame(bars["SPY"])

def main():
    print(df[3])

main()