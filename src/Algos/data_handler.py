import pandas as pd
from datetime import datetime

class DataHandler:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.data_path, parse_dates=['timestamp'])
        self.data.set_index('timestamp', inplace=True)
        # Might need to implement some kind of data clean up / organization in the future here
        return self.data
    
    def get_data(self):
        if self.data == None:
            raise ValueError("Call load_data() first")
        else:
            return self.data