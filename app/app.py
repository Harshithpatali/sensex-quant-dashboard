import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os

# Path fix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import predict

st.set_page_config(layout="wide")
st.title("📈 Quant Trading Dashboard (SENSEX AI)")

# -----------------------
# SIDEBAR CONTROLS
# -----------------------
threshold = st.sidebar.slider("Prediction Threshold (%)", 0.01, 1.0, 0.2)
stop_loss = st.sidebar.slider("Stop Loss (%)", 0.5, 5.0, 2.0) / 100
take_profit = st.sidebar.slider("Take Profit (%)", 1.0, 10.0, 4.0) / 100

# -----------------------
# LOAD DATA
# -----------------------
df = yf.download("^BSESN", period="6mo")

# Fix multi-index columns
df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

df.reset_index(inplace=True)

# -----------------------
# FEATURE ENGINEERING
# -----------------------
df['MA_5'] = df['Close'].rolling(5).mean()
df['MA_10'] = df['Close'].rolling(10).mean()
df['Volatility'] = df['Close'].rolling(5).std()
df.dropna(inplace=True)

# -----------------------
# INITIALIZE
# -----------------------
capital = 100000
position = 0
entry_price = 0

equity = []
trades = []
buy_points = []
sell_points = []

# -----------------------
# BACKTEST LOOP
# -----------------------
for i in range(10, len(df)-1):
    row = df.iloc[i]

    inp = [
        row['Open'], row['High'], row['Low'], row['Close'],
        row['Volume'], row['MA_5'], row['MA_10'], row['Volatility']
    ]

    pred = float(predict(inp)['XGBoost (%)'])
    price = float(row['Close'])

    # BUY
    if pred > threshold and position == 0:
        position = 1
        entry_price = price

        trades.append({
            "Type": "BUY",
            "Price": price,
            "PnL (%)": None
        })

        buy_points.append((row['Date'], price))

    # SELL
    if position == 1:
        change = (price - entry_price) / entry_price

        if change <= -stop_loss or change >= take_profit or pred < -threshold:
            capital *= (price / entry_price)
            position = 0

            trades.append({
                "Type": "SELL",
                "Price": price,
                "PnL (%)": change * 100
            })

            sell_points.append((row['Date'], price))

    # Equity tracking
    if position == 1:
        equity.append(capital * (price / entry_price))
    else:
        equity.append(capital)

# -----------------------
# SAFETY FIX (NO CRASH)
# -----------------------
if len(equity) == 0:
    equity = [capital]

# -----------------------
# DASHBOARD
# -----------------------
col1, col2 = st.columns(2)

# 📊 PRICE CHART + SIGNALS
with col1:
    st.subheader("📊 Price Chart with Signals")

    fig, ax = plt.subplots()
    ax.plot(df['Date'].iloc[10:-1], df['Close'].iloc[10:-1], label="Price")

    if buy_points:
        bx, by = zip(*buy_points)
        ax.scatter(bx, by, marker="^", label="BUY")

    if sell_points:
        sx, sy = zip(*sell_points)
        ax.scatter(sx, sy, marker="v", label="SELL")

    ax.legend()
    st.pyplot(fig)

# 📈 EQUITY CURVE
with col2:
    st.subheader("📈 Equity Curve")

    fig2, ax2 = plt.subplots()
    ax2.plot(equity)
    st.pyplot(fig2)

# -----------------------
# METRICS (SAFE)
# -----------------------
if len(equity) > 1:
    returns = (equity[-1] / equity[0] - 1) * 100
    drawdown = np.array(equity) / np.maximum.accumulate(equity) - 1
    final_capital = equity[-1]
else:
    returns = 0
    drawdown = [0]
    final_capital = capital

col3, col4, col5 = st.columns(3)
col3.metric("Return (%)", f"{returns:.2f}")
col4.metric("Max Drawdown (%)", f"{min(drawdown)*100:.2f}")
col5.metric("Final Capital", f"{final_capital:.0f}")

# -----------------------
# TRADE TABLE
# -----------------------
st.subheader("📜 Trades")

df_trades = pd.DataFrame(trades)

if not df_trades.empty:
    df_trades["PnL (%)"] = df_trades["PnL (%)"].round(2)

st.dataframe(df_trades)