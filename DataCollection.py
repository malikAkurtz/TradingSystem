import requests
from datetime import datetime, timezone, timedelta
import pandas as pd
import time
import sqlite3

db_file = "Financial_Data.db"

con = sqlite3.connect(db_file)
cur = con.cursor()

url = "https://api.coingecko.com/api/v3/simple/price"

params = {"ids": "ripple", "vs_currencies": "usd"}
cur.execute("""
        CREATE TABLE IF NOT EXISTS xrp_prices (
                   date TEXT PRIMARY KEY,
                   closing_price REAL)
        """)



while (True):
    cur_utc = datetime.utcnow().timestamp()
    timestamp = datetime.fromtimestamp(cur_utc, tz = timezone.utc) - timedelta(hours=5)

    if (timestamp.second == 0 or timestamp.second == 30):
        response = requests.get(url, params=params).json()
        xrp_price = response["ripple"]["usd"]
        cur.execute("""
        INSERT OR REPLACE INTO xrp_prices (date, closing_price) VALUES (?, ?)""", (timestamp.strftime('%Y-%m-%d %H:%M:%S'), xrp_price))
        con.commit()
        print("XRP Price:", xrp_price)
        print("This candle just closed")
        time.sleep(1)


