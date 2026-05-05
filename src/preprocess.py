import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    return df

def preprocess(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df