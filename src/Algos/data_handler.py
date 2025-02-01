import pandas as pd
from config import API_KEY, API_SECRET, BASE_URL
from alpaca.data import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest

class DataHandler:
    def __init__(self, data_source_file_name):
        self.data = None
        self.file_name = data_source_file_name
        self.stock_data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

    def fetch_historical_data(self, start_time, end_time, symbols, timeframe):

        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start = start_time,
            end=end_time,
            feed="iex"
        )

        bars = self.stock_data_client.get_stock_bars(request_params=request_params)

        return bars.df
    
    def save_to_csv(self, df):
        df.to_csv(self.file_name)


    def load_data(self):
        self.data = pd.read_csv(self.file_name, parse_dates=['timestamp'])
        self.data.set_index('timestamp', inplace=True)
        # Might need to implement some kind of data clean up / organization in the future here
        return self.data
    
    def get_data(self):
        return self.data
        