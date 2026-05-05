import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from preprocess import load_data, preprocess
from predict import predict

# -----------------------------
# CONFIG
# -----------------------------
INITIAL_CAPITAL = 100000

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "stock_data.csv")

# -----------------------------
# LOAD DATA
# -----------------------------
df = load_data(DATA_PATH)
df = preprocess(df)

# Feature engineering (same as training)
df['MA_5'] = df['Close'].rolling(5).mean()
df['MA_10'] = df['Close'].rolling(10).mean()
df['Volatility'] = df['Close'].rolling(5).std()

df.dropna(inplace=True)

# -----------------------------
# STRATEGY ENGINE
# -----------------------------
def run_strategy(threshold=0.5, stop_loss=0.02, take_profit=0.04):
    capital = INITIAL_CAPITAL
    position = 0
    entry_price = 0

    equity_curve = []
    trades = []

    for i in range(10, len(df) - 1):

        row = df.iloc[i]

        input_data = [
            row['Open'], row['High'], row['Low'], row['Close'],
            row['Volume'], row['MA_5'], row['MA_10'], row['Volatility']
        ]

        pred = predict(input_data)['XGBoost (%)']
        price = row['Close']

        # -----------------------------
        # ENTRY LOGIC
        # -----------------------------
        if pred > threshold and position == 0:
            position = 1
            entry_price = price
            trades.append({
                "Type": "BUY",
                "Price": price,
                "Index": i
            })

        # -----------------------------
        # EXIT LOGIC
        # -----------------------------
        if position == 1:
            change = (price - entry_price) / entry_price

            if (change <= -stop_loss) or (change >= take_profit) or (pred < -threshold):
                capital *= (price / entry_price)

                trades.append({
                    "Type": "SELL",
                    "Price": price,
                    "Index": i,
                    "PnL (%)": change * 100
                })

                position = 0

        # -----------------------------
        # EQUITY TRACKING
        # -----------------------------
        if position == 1:
            equity_curve.append(capital * (price / entry_price))
        else:
            equity_curve.append(capital)

    return np.array(equity_curve), trades


# -----------------------------
# METRICS
# -----------------------------
def compute_metrics(equity):
    returns = pd.Series(equity).pct_change().dropna()

    total_return = (equity[-1] / equity[0] - 1) * 100

    sharpe = 0
    if returns.std() != 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)

    drawdown = (equity / np.maximum.accumulate(equity) - 1)
    max_dd = drawdown.min() * 100

    return {
        "Return (%)": total_return,
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": max_dd
    }


# -----------------------------
# STRATEGY COMPARISON
# -----------------------------
def compare_strategies():
    strategies = {
        "Conservative": 0.3,
        "Balanced": 0.5,
        "Aggressive": 1.0
    }

    results = {}

    plt.figure(figsize=(10, 5))

    for name, threshold in strategies.items():
        equity, trades = run_strategy(threshold=threshold)
        metrics = compute_metrics(equity)

        results[name] = {
            "equity": equity,
            "metrics": metrics,
            "trades": trades
        }

        plt.plot(equity, label=f"{name} ({metrics['Return (%)']:.2f}%)")

    plt.title("📊 Strategy Comparison")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid()
    plt.show()

    print("\n📊 STRATEGY METRICS")
    for name, res in results.items():
        print(f"\n{name}")
        for k, v in res["metrics"].items():
            print(f"{k}: {v:.2f}")

    return results


# -----------------------------
# THRESHOLD OPTIMIZATION
# -----------------------------
def optimize_threshold():
    best_threshold = None
    best_return = -np.inf

    for t in np.arange(0.1, 1.5, 0.1):
        equity, _ = run_strategy(threshold=t)
        ret = (equity[-1] / equity[0] - 1)

        if ret > best_return:
            best_return = ret
            best_threshold = t

    print("\n🎯 OPTIMIZATION RESULT")
    print(f"Best Threshold: {best_threshold}")
    print(f"Best Return: {best_return * 100:.2f}%")

    return best_threshold


# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":

    print("🚀 Running Quant Backtest...\n")

    # Run strategy comparison
    results = compare_strategies()

    # Optimize threshold
    best_t = optimize_threshold()

    # Run best strategy
    print("\n📈 Running Best Strategy...\n")
    equity, trades = run_strategy(threshold=best_t)

    metrics = compute_metrics(equity)

    print("\n🏆 FINAL PERFORMANCE")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

    # Plot final equity curve
    plt.figure(figsize=(10, 5))
    plt.plot(equity)
    plt.title("🏆 Best Strategy Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    plt.grid()
    plt.show()

    # Show trade logs
    trades_df = pd.DataFrame(trades)
    print("\n📜 TRADE LOGS")
    print(trades_df.head(20))