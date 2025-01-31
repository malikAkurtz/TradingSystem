from .data_handler import DataHandler
from .strat_bollinger_band import BollingerBandStrat
from .portfolio import Portfolio
from .execution import SimulatedExecutionHandler
from backtest_engine import BacktestEngine

def main():
    
    data_hander = DataHandler(data_path="bruh")
    strategy = BollingerBandStrat(20, 2)
    portfolio = Portfolio(100_000)
    execution_handler = SimulatedExecutionHandler()

    backtest_engine = BacktestEngine(data_handler=data_hander, strategy=strategy, portfolio=portfolio, execution_handler=execution_handler)

    equity_curve, trades = backtest_engine.run_backtest()

    print(f"Final equity: {equity_curve[-1]:.2f}")
    print(f"Number of trades: {len(trades)}")

if __name__ == "__main__":
    main()