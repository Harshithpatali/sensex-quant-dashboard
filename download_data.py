import yfinance as yf
import pandas as pd

# Download data
df = yf.download("^BSESN", start="2015-01-01", end="2025-01-01")

# 🔥 Remove MultiIndex (important step)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Reset index
df.reset_index(inplace=True)

# Select only required columns
df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

# Save to CSV
df.to_csv("data/stock_data.csv", index=False)

print("✅ Clean SENSEX data saved!")