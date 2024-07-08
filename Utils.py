import pandas as pd

def raw_data_process(data):
    data.columns = ['date', 'close', 'open', 'high', 'low', 'volume', 'return']
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data.sort_index(inplace=True)
    data = data.replace(',', '', regex=True)
    data = data.replace('K', '', regex=True)
    data = data.replace('%', '', regex=True)
    data = data.astype(float)
    data['return'] = data['return'] / 100
    return data
