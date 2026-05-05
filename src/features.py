def create_features(df):
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()
    df['Volatility'] = df['Close'].rolling(5).std()

    # 🎯 Regression target (% change next day)
    df['Target'] = df['Close'].pct_change().shift(-1) * 100

    df.dropna(inplace=True)

    X = df[['Open', 'High', 'Low', 'Close', 'Volume',
            'MA_5', 'MA_10', 'Volatility']]

    y = df['Target']

    return X, y