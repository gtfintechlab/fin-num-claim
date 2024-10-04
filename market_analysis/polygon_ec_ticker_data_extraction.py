import datetime
import pandas as pd
import numpy as np
from polygon import RESTClient
import glob

key=""
folderName = './stock_price_data/'

df = pd.read_excel("master_EC.xlsx", usecols=["ticker"])

df = df.drop_duplicates(ignore_index=True)

ticker_list = df['ticker'].tolist()

def ts_to_datetime(ts) -> str:
    return datetime.datetime.fromtimestamp(ts / 1000.0).strftime('%Y-%m-%d %H:%M')


client = RESTClient(api_key=key)
from_ = "2017-01-01"
to = "2023-05-31"
for ticker in ticker_list:
    resp = client.get_aggs(ticker=ticker, multiplier=1, timespan="day", from_=from_, to=to, adjusted=True)

    dates = []
    opening = []
    closing = []
    highest = []
    lowest = []
    v = []  
    for result in resp:
        dt = ts_to_datetime(result.timestamp)
        dates.append(dt)
        opening.append(result.open)
        highest.append(result.high)
        lowest.append(result.low)
        closing.append(result.close)
        v.append(result.transactions)
        
    df = pd.DataFrame(np.transpose([dates, opening, closing, highest, lowest, v]), columns = ['Date','Open','Close','High','Low','Volume'])
    df.to_excel(folderName + ticker + '.xlsx', index=False)
    print(ticker,' completed.')

