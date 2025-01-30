import numpy as np
import math
import pandas as pd
import asyncio
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, time, timedelta
import pytz
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.live import StockDataStream

from config import API_KEY, API_SECRET, BASE_URL
from alpaca.trading.client import TradingClient

# Initialize Alpaca Clients
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)


# Initialize Time Constants
EST = pytz.timezone("US/Eastern")
LUNCH_START_TIME = time(12, 30)
LUNCH_END_TIME = time(14, 30)
EOD_TIME = time(16, 0)

#Strategy Hyper-Parameters
SYMBOL = "SPY"
TIMEFRAME = TimeFrame(3, TimeFrameUnit.Minute)
BB_PERIOD = 20
BB_STD = 2
POSITION = None
UNITS = 0

# Function to get historical data for calculating indicators
def fetch_recent_data():
    end_time = datetime.now(pytz.utc)
    start_time = end_time - timedelta(days=1)

    request_params = StockBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe = TIMEFRAME,
        start=start_time,
        end=end_time
        )
    bars = data_client.get_stock_bars(request_params)
    return pd.DataFrame(bars[SYMBOL])


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

    # Get latest candle
    latest_candle = df.iloc[-1]

    # If its not in the range to open a trade, or we already traded today, return
    if (not (LUNCH_START_TIME <= datetime.now(EST).time() <= LUNCH_END_TIME) or made_trade):
        return
    
    # If we arent already in a position
    if POSITION == None:
        # If the latest bar closes above the upper bollinger band
        if latest_candle["close"] > latest_candle["Upper_BB"]:
            # Buy leveraged SPY
            print(f"LONG ENTRY: {'UPRO'} @ {latest_candle['close']}")
            execute_trade("UPRO", "BUY")
            POSITION = "long"
            UNITS = 4
        # If the latest bar closes below the lower bollinger band
        elif latest_candle["close"] < latest_candle["Lower_BB"]:
            # Buy leveraged inverse SPY
            print(f"SHORT ENTRY: {'SPXU'} @ {latest_candle['close']}")
            execute_trade("SPXU", "BUY")
            POSITION = "short"
            UNITS = 4

    # If we are in a position
    if POSITION == "long":
        # If the latest bar closes above the upper bollinger band
        if latest_candle["close"] > latest_candle["Upper_BB"]:
            # Sell 1/4 of position
            print(f"TAKING PROFIT: {'UPRO'} @ {latest_candle['close']}")
            execute_trade("UPRO", "SELL", 1)
            UNITS -= 1
        # If the latest bar closes below the lower bollinger band
        elif latest_candle["close"] < latest_candle["Lower_BB"]:
            # Flatten position
            print(f"STOPPED OUT {'UPRO'} @ {latest_candle['close']}")
            execute_trade("UPRO", "SELL", UNITS)
            UNITS = 0


    if POSITION == "short":
        # If the latest bar closes below the lower bollinger band
        if latest_candle["close"] < latest_candle["Lower_BB"]:
            # Sell 1/4 of position
            print(f"TAKING PROFIT: {'SPXU'} @ {latest_candle['close']}")
            execute_trade("SPXU", "SELL", 1)
            UNITS -= 1
        # If the latest bar closes above the upper bollinger band
        elif latest_candle["close"] > latest_candle["Upper_BB"]:
            # Flatten position
            print(f"STOPPED OUT {'SPXU'} @ {latest_candle['close']}")
            execute_trade("SPXU", "SELL", UNITS)
            UNITS = 0


    if (UNITS == 0):
        POSITION = None
        made_trade = True

def execute_trade(symbol, side, quantity=4):
    order = MarketOrderRequest(
        symbol = SYMBOL,
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

    stock_stream.subscribe_bars(handle_bar_update, SYMBOL)
    await stock_stream._run_forever()

async def main():
    await stream_data()

if __name__ == "__main__":
    try:
        print("Starting event loop...")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down gracefully...")