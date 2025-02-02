import pandas as pd
from datetime import time
import pytz
from data_handler import DataHandler
from strat_bollinger_band import BollingerBandStrat
from portfolio import Portfolio
from execution import SimulatedExecutionHandler
import matplotlib.pyplot as plt
import datetime as datetime

class BacktestEngine:
    def __init__(self, data_handler, strategy, portfolio, execution_handler):
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        self.eod_time = time(21,00)
        self.trades = []

    def run_backtest(self):
        # Load & set up data
        df = self.data_handler.load_data()
        df = self.strategy.calculate_indicators(df)
        df['equity'] = float('nan')

        current_day = None
        days_starting_cash = self.portfolio.cash

        for timestamp, row in df.iterrows():
            # Detect a new day
            if current_day != timestamp.day:
                current_day = timestamp.day
                # Print an easy-to-see day header
                print("\n", "="*25, f"NEW DAY {timestamp.date()}", "="*25)
                print(f"  Starting Cash: {days_starting_cash:,.2f}\n")

            # Build a concise log line
            # Example: "2025-01-16 18:21 | Holdings={'SPY': 0}"
            ts_est = timestamp.tz_convert(pytz.timezone("US/Eastern"))
            dt_string = ts_est.strftime("%Y-%m-%d %H:%M")
            log_parts = [f"{dt_string}", f"Holdings={self.portfolio.holdings}"]

            # Generate signals
            signal = self.strategy.generate_signals(timestamp, row, self.portfolio.get_current_positions())

            # If there's a signal, add it to the log line
            if signal is not None:
                log_parts.append(f"Signal={signal}")

            # Print this single log line
            print(" | ".join(log_parts))

            # Execute the signal if it exists
            if signal is not None:
                if signal == "BUY_LONG":
                    fill_qty, fill_price = self.execution_handler.execute_order("BUY", 100, row['SPY_close'])
                    self.portfolio.update_on_fill("SPY", "BUY", fill_qty, fill_price)
                    # Print a second line for the trade, if desired
                    print(f"     Executed BUY_LONG ({fill_qty} @ {fill_price}) -> Holdings: {self.portfolio.holdings}")

                elif signal == "SELL_SHORT":
                    fill_qty, fill_price = self.execution_handler.execute_order("SELL", 100, row['SPY_close'])
                    self.portfolio.update_on_fill("SPY", "SELL", fill_qty, fill_price)
                    print(f"     Executed SELL_SHORT ({fill_qty} @ {fill_price}) -> Holdings: {self.portfolio.holdings}")

                elif signal == "PARTIAL_PROFIT_LONG":
                    fill_qty, fill_price = self.execution_handler.execute_order("SELL", 25, row['SPY_close'])
                    self.portfolio.update_on_fill("SPY", "SELL", fill_qty, fill_price)
                    print(f"     Partial LONG Sell ({fill_qty} @ {fill_price}) -> Holdings: {self.portfolio.holdings}")

                elif signal == "PARTIAL_PROFIT_SHORT":
                    fill_qty, fill_price = self.execution_handler.execute_order("BUY", 25, row['SPY_close'])
                    self.portfolio.update_on_fill("SPY", "BUY", fill_qty, fill_price)
                    print(f"     Partial SHORT Buy ({fill_qty} @ {fill_price}) -> Holdings: {self.portfolio.holdings}")

                elif signal == "CLOSE_LONG":
                    print("     Flattening Long...")
                    self.portfolio.flatten_positions(row)
                    print(f"     Holdings after flatten: {self.portfolio.holdings}")

                elif signal == "CLOSE_SHORT":
                    print("     Flattening Short...")
                    self.portfolio.flatten_positions(row)
                    print(f"     Holdings after flatten: {self.portfolio.holdings}")

                # Add to the trades list
                self.trades.append((timestamp, signal, row['SPY_close']))

            # Check if we're near end-of-day
            current_dt = datetime.datetime.combine(timestamp.date(), timestamp.time())
            eod_dt = datetime.datetime.combine(timestamp.date(), self.eod_time)
            diff = eod_dt - current_dt

            if abs(diff) <= datetime.timedelta(minutes=5) and current_dt < eod_dt:
                print("\n", ">"*20, "EOD Flatten / Summary", "<"*20)
                print(f"  Ending Cash: {self.portfolio.cash:,.2f}")
                print(f"  Day's P/L:   {self.portfolio.cash - days_starting_cash:,.2f}")
                days_starting_cash = self.portfolio.cash

            # Update equity
            equity = self.portfolio.update_equity(row)
            df.loc[timestamp, 'equity'] = equity

        return df, self.trades
    
    def save_equity_plot(self, df):
        plt.figure(figsize=(12,6))
        plt.plot(df.index, df['equity'])
        plt.savefig("equityplot.png")

    def save_price_action_plot(self, df, start_timestamp, end_timestamp):
        plt.figure(figsize=(12,6))
        sub_df = df.loc[start_timestamp:end_timestamp]

        time_axis = sub_df.index
        close_data = sub_df["SPY_close"]
        lower_bb = sub_df["Lower_BB"]
        upper_bb = sub_df["Upper_BB"]

        # Create boolean masks
        below_mask = close_data < lower_bb
        above_mask = close_data > upper_bb
        middle_mask = ~(below_mask | above_mask)  # in between the bands

        # Plot each region in a different color
        # 'linewidth=0' (or 'ls="none"') means no line connecting the markers
        plt.plot(time_axis[below_mask], close_data[below_mask],
                marker='x', color='red', linewidth=0)
        plt.plot(time_axis[above_mask], close_data[above_mask],
                marker='x', color='green', linewidth=0)
        # If you want to plot the “middle” region as well:
        plt.plot(time_axis[middle_mask], close_data[middle_mask],
                marker='x', color='blue', linewidth=0)

        # Now plot the BB lines themselves
        plt.plot(time_axis, sub_df["Upper_BB"], label="Upper_BB", color='black')
        plt.plot(time_axis, sub_df["Lower_BB"], label="Lower_BB", color='black')

        # Optionally shade the 17:30–19:30 window each day
        unique_days = time_axis.normalize().unique()
        for day in unique_days:
            start_range = day + pd.Timedelta(hours=17, minutes=30)
            end_range = day + pd.Timedelta(hours=19, minutes=30)
            plt.axvspan(start_range, end_range, color='yellow', alpha=0.2)

        plt.grid(True)
        plt.legend()
        plt.savefig("priceplot.png")


                