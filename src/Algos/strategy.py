from abc import ABC, abstractmethod
import pandas as pd

class Strategy(ABC):
    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame):
        pass

    def generate_signals(self, row, current_position):
        pass