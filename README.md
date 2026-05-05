# 📈 SENSEX Quant Trading Dashboard (PCA + Machine Learning)

An end-to-end **AI-powered stock analysis and trading dashboard** built using:

- 📊 Principal Component Analysis (PCA)
- 🤖 Machine Learning (Regression Models)
- 📈 Quantitative Trading Strategy
- 🌐 Streamlit Web App

This project simulates a **quant trading system on the SENSEX index** using predictive modeling and backtesting.

---

## 🚀 Live Demo

👉 https://2rtz2h527sbpfmdqks7c8q.streamlit.app/

---

## 🔥 Features

### 📊 Data
- Live SENSEX data using `yfinance`
- Ticker: `^BSESN`

### 🧠 Machine Learning
- PCA for dimensionality reduction  
- Models:
  - Random Forest Regressor
  - Linear Regression
  - XGBoost Regressor  

### 🔮 Prediction
- Predicts **next-day % price movement**

### 📈 Trading Strategy
- BUY → predicted return > threshold  
- SELL → stop-loss / take-profit / negative signal  
- Uses rule-based decision system for trade execution  

### 💰 Backtesting Engine
- Simulates trades over historical data  
- Tracks portfolio performance  

### 📊 Dashboard
- Price chart with BUY/SELL markers  
- Equity curve visualization  
- Trade logs  
- Key metrics:
  - Return (%)
  - Max Drawdown (%)
  - Final Capital  

---

## 🧠 How It Works

This system follows a real-world ML pipeline:
