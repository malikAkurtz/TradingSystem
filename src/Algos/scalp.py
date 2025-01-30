import numpy as np
import math
import pandas as pd
import asyncio
from alpaca.data import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, time, timedelta
import pytz
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.live import StockDataStream, CryptoDataStream

from config import API_KEY, API_SECRET, BASE_URL
from alpaca.trading.client import TradingClient

# Initialize Alpaca Clients
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
stock_data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
crypto_data_client = CryptoHistoricalDataClient(API_KEY, API_SECRET)


# Initialize Time Constants
EST = pytz.timezone("US/Eastern")
LUNCH_START_TIME = time(12, 30)
LUNCH_END_TIME = time(14, 30)
EOD_TIME = time(16, 0)

#Strategy Hyper-Parameters
TRACKING_SYMBOL = "BTC"
LONG_SYMBOL = "BTC"
SHORT_SYMBOL = "BTC"
TIMEFRAME = TimeFrame(1, TimeFrameUnit.Minute)
BB_PERIOD = 20
BB_STD = 2
POSITION = None
UNITS = 0

# Function to get historical data for calculating indicators
def fetch_recent_data():
    end_time = datetime.now(pytz.utc)
    start_time = end_time - timedelta(days=1)

    request_params = StockBarsRequest(
        symbol_or_symbols=TRACKING_SYMBOL,
        timeframe = TIMEFRAME,
        start=start_time,
        end=end_time
        )
    bars = data_client.get_stock_bars(request_params)
    return pd.DataFrame(bars[TRACKING_SYMBOL])


def calculate_Bollinger_Bands(df):
    df["SMA"] = df["close"].rolling(window=BB_PERIOD).mean()
    df["STD"] = df["close"].rolling(window=BB_PERIOD).std();
    df["Upper_BB"] = df["SMA"] + (BB_STD * df["STD"])
    df["Lower_BB"] = df["SMA"] - (BB_STD * df["STD"])

    return df

def check_trade_signals(df):
    global POSITION
    global UNITS
    made_trade = False
    # STILL NEED TO FLATTEN AT MARKET CLOSE
    # Get latest candle
    latest_candle = df.iloc[-1]

    # If its not in the range to open a trade, or we already traded today, return
    # if (not (LUNCH_START_TIME <= datetime.now(EST).time() <= LUNCH_END_TIME) or made_trade):
    #     return
    
    # If we arent already in a position
    if POSITION == None:
        # If the latest bar closes above the upper bollinger band
        if latest_candle["close"] > latest_candle["Upper_BB"]:
            # Buy leveraged SPY
            print(f"LONG ENTRY: {LONG_SYMBOL} @ {latest_candle['close']}")
            execute_trade(LONG_SYMBOL, "BUY")
            POSITION = "long"
            UNITS = 4
        # If the latest bar closes below the lower bollinger band
        elif latest_candle["close"] < latest_candle["Lower_BB"]:
            # Buy leveraged inverse SPY
            print(f"SHORT ENTRY: {SHORT_SYMBOL} @ {latest_candle['close']}")
            execute_trade(SHORT_SYMBOL, "BUY")
            POSITION = "short"
            UNITS = 4

    # If we are in a position
    if POSITION == "long":
        # If the latest bar closes above the upper bollinger band
        if latest_candle["close"] > latest_candle["Upper_BB"]:
            # Sell 1/4 of position
            print(f"TAKING PROFIT: {LONG_SYMBOL} @ {latest_candle['close']}")
            execute_trade(LONG_SYMBOL, "SELL", 1)
            UNITS -= 1
        # If the latest bar closes below the lower bollinger band
        elif latest_candle["close"] < latest_candle["Lower_BB"]:
            # Flatten position
            print(f"STOPPED OUT {LONG_SYMBOL} @ {latest_candle['close']}")
            execute_trade(LONG_SYMBOL, "SELL", UNITS)
            UNITS = 0


    if POSITION == "short":
        # If the latest bar closes below the lower bollinger band
        if latest_candle["close"] < latest_candle["Lower_BB"]:
            # Sell 1/4 of position
            print(f"TAKING PROFIT: {SHORT_SYMBOL} @ {latest_candle['close']}")
            execute_trade(SHORT_SYMBOL, "SELL", 1)
            UNITS -= 1
        # If the latest bar closes above the upper bollinger band
        elif latest_candle["close"] > latest_candle["Upper_BB"]:
            # Flatten position
            print(f"STOPPED OUT {SHORT_SYMBOL} @ {latest_candle['close']}")
            execute_trade(SHORT_SYMBOL, "SELL", UNITS)
            UNITS = 0


    if (UNITS == 0):
        POSITION = None
        made_trade = True

def execute_trade(symbol, side, quantity=4):
    order = MarketOrderRequest(
        symbol = symbol,
        qty = quantity,
        side = OrderSide.BUY if side == "BUY" else OrderSide.Sell,
        time_in_force = TimeInForce.DAY
    )

    trading_client.submit_order(order)

async def stream_data():
    stock_stream = StockDataStream(API_KEY, API_SECRET)

    async def handle_bar_update(data):
        df = fetch_recent_data()
        df = calculate_Bollinger_Bands(df)
        print(df)
        check_trade_signals(df)

    stock_stream.subscribe_bars(handle_bar_update, TRACKING_SYMBOL)
    await stock_stream._run_forever()

async def main():
    await stream_data()

if __name__ == "__main__":
    try:
        print("Starting event loop...")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down gracefully...")