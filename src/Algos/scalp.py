from config import API_KEY, API_SECRET, BASE_URL
from alpaca.trading.client import TradingClient

trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

def main():
    

