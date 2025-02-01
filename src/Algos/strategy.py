from abc import ABC, abstractmethod
import pandas as pd

class Strategy(ABC):
    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame):
        pass
    
    @abstractmethod
    def generate_signals(self, row, current_position):
        pass