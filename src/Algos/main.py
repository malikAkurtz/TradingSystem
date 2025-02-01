import pandas as pd
from data_handler import DataHandler
from strat_bollinger_band import BollingerBandStrat
from portfolio import Portfolio
from execution import SimulatedExecutionHandler
from backtest_engine import BacktestEngine

from datetime import datetime
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import pytz

def main():
    data_handler = DataHandler("/Users/malikkurtz/Coding/TradingSystem/src/Algos/data_csv")

    start_time = datetime(2025, 1, 16, tzinfo=pytz.utc)
    end_time = datetime.now(pytz.utc)
    time_frame = TimeFrame(3, TimeFrameUnit.Minute)
    symbols = ["SPY", "SPXL", "SPXS"]

    df = data_handler.fetch_historical_data(start_time, end_time, symbols[0], time_frame)
    df.reset_index(inplace=True)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    df.drop(['symbol', 'open', 'high', 'low', 'volume', 'trade_count', 'vwap'], axis=1, inplace=True)

    df.rename(columns={'close' : symbols[0] + "_" + "close"}, inplace=True)

    # for symbol in symbols[1::]:
    #     symbol_df = data_handler.fetch_historical_data(start_time, end_time, symbol, time_frame)
    #     symbol_df.reset_index(inplace=True)
    #     symbol_df.drop_duplicates(subset='timestamp', keep='last', inplace=True)
    #     symbol_df.set_index('timestamp', inplace=True)
    #     symbol_df.sort_index(inplace=True)
    #     symbol_df.drop(['symbol', 'open', 'high', 'low', 'volume', 'trade_count', 'vwap'], axis=1, inplace=True)
    #     symbol_df.rename(columns={'close' : symbol + "_" + "close"}, inplace=True)

    #     df = pd.concat([df, symbol_df], axis=1, join="outer")

    df.ffill(inplace=True)

    data_handler.save_to_csv(df)

    strategy = BollingerBandStrat(20, 2)
    portfolio = Portfolio(100_000)
    execution_handler = SimulatedExecutionHandler()

    backtest_engine = BacktestEngine(data_handler=data_handler, strategy=strategy, portfolio=portfolio, execution_handler=execution_handler)

    output_df, trades = backtest_engine.run_backtest()
    print(output_df)

    print(f"Final equity: {output_df.iloc[-1]['equity']:.2f}")
    print(f"Number of trades: {len(trades)}")

    backtest_engine.plot_price_action(output_df, "2025-01-16 14:30:00+00:00", "2025-01-16 20:57:00+00:00")

if __name__ == "__main__":
    main()