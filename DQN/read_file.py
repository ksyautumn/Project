import pandas as pd

def read_file(file_name):
    data = pd.read_csv(file_name)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    date_split = '2016-01-01'
    train = data[:date_split]
    test = data[date_split:]
    return [train, test, date_split]