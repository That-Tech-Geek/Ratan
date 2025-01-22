import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import streamlit as st

# Function to fetch data from Yahoo Finance
def get_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    returns = data.pct_change().dropna()
    return data, returns

# Function to fetch the 10-year Treasury yield from Yahoo Finance
def get_risk_free_rate():
    tnx = yf.Ticker("^TNX")
    data = tnx.history(period="1d")
    if not data.empty:
        risk_free_rate = data['Close'].iloc[-1] / 100  # Convert to decimal (e.g., 3% -> 0.03)
        return risk_free_rate
    else:
        return 0.03  # Default risk-free rate if data is unavailable

# Function to calculate WACC
def calculate_wacc(equity, debt, equity_cost, debt_cost, tax_rate):
    total = equity + debt
    if total > 0:
        wacc = (equity / total) * equity_cost + (debt / total) * debt_cost * (1 - tax_rate)
        return wacc
    return 0.0

# Function to calculate ROIC
def calculate_roic(net_income, debt, equity):
    invested_capital = debt + equity
    if invested_capital > 0:
        roic = net_income / invested_capital
        return roic
    return 0.0

# Function to prepare data for model training
def prepare_data(returns, lookback):
    X, y = [], []
    for i in range(len(returns) - lookback):
        X.append(returns.iloc[i:i + lookback].values)
        y.append(returns.iloc[i + lookback].values)
    return np.array(X), np.array(y)

# Function to train the model
def train_model(X_train, y_train, mse_threshold):
    model = LinearRegression()
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
    y_train_scaled = scaler.fit_transform(y_train)

    while True:
        model.fit(X_train_scaled, y_train_scaled)
        predictions = model.predict(X_train_scaled)
        mse = np.mean((predictions - y_train_scaled) ** 2)
        if mse <= mse_threshold:
            break

    return model, scaler

# Streamlit App
def main():
    st.title("Dynamic Portfolio Risk Hedging")

    tickers = st.text_input("Enter tickers (comma-separated):", "AAPL, MSFT, TSLA")
    start_date = st.date_input("Start Date:")
    end_date = st.date_input("End Date:")
    lookback = st.slider("Lookback Period:", min_value=1, max_value=30, value=5)
    mse_threshold = st.slider("MSE Threshold (in %):", min_value=0.1, max_value=5.0, value=0.1)

    if st.button("Run Analysis"):
        tickers_list = [ticker.strip() for ticker in tickers.split(",")]

        try:
            data, returns = get_data(tickers_list, start_date, end_date)
            risk_free_rate = get_risk_free_rate()

            st.write(f"Risk-Free Rate: {risk_free_rate:.2%}")

            X, y = prepare_data(returns, lookback)

            try:
                model, scaler = train_model(X, y, mse_threshold / 100)
                st.success("Model trained successfully with MSE below threshold!")
            except Exception as e:
                st.error(f"Error in model training: {e}")

            st.line_chart(data)

            # Display WACC and ROIC calculations (for analysis, not user display)
            equity = st.number_input("Equity (in USD):", min_value=0.0, step=1.0)
            debt = st.number_input("Debt (in USD):", min_value=0.0, step=1.0)
            equity_cost = st.number_input("Cost of Equity (in %):", min_value=0.0, step=0.1) / 100
            debt_cost = st.number_input("Cost of Debt (in %):", min_value=0.0, step=0.1) / 100
            tax_rate = st.number_input("Tax Rate (in %):", min_value=0.0, step=0.1) / 100
            net_income = st.number_input("Net Income (in USD):", min_value=0.0, step=1.0)

            if equity > 0 and debt > 0 and equity_cost > 0 and debt_cost > 0 and tax_rate > 0 and net_income > 0:
                wacc = calculate_wacc(equity, debt, equity_cost, debt_cost, tax_rate)
                roic = calculate_roic(net_income, debt, equity)
                st.write(f"WACC: {wacc:.2%}")
                st.write(f"ROIC: {roic:.2%}")

        except Exception as e:
            st.error(f"Error fetching or processing data: {e}")

if __name__ == "__main__":
    main()
