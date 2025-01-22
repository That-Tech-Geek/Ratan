import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# Function to fetch data from yfinance
def get_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    returns = data.pct_change().dropna()
    return data, returns

# Function to calculate portfolio metrics
def calculate_metrics(returns, risk_free_rate=0.02):
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    volatilities = np.sqrt(np.diag(cov_matrix))
    sharpe_ratios = (mean_returns - risk_free_rate) / volatilities
    return mean_returns, volatilities, sharpe_ratios

# Prepare data for LSTM
def prepare_data(returns, lookback=30):
    X, y = [], []
    for i in range(lookback, len(returns)):
        X.append(returns[i-lookback:i].values)
        y.append(returns[i].values)
    return np.array(X), np.array(y)

# Define LSTM model using PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Train model until MSE threshold
def train_model(model, X_train, y_train, mse_threshold=0.001, learning_rate=0.001, epochs=100):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if loss.item() <= mse_threshold:
            break
    return model

# Dynamic hedging
def hedge_portfolio(model, X_test, scaler, original_returns):
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    predictions = model(X_test_tensor).detach().numpy()
    predicted_returns = scaler.inverse_transform(predictions)
    hedge_ratios = predicted_returns / original_returns.iloc[-len(predicted_returns):].values
    return hedge_ratios

# Stress testing
def stress_test(returns, shocks=(-0.1, -0.2, -0.3)):
    stress_results = {}
    for shock in shocks:
        stressed_returns = returns + shock
        stress_results[f'Shock {shock*100:.0f}%'] = stressed_returns.mean(axis=0)
    return pd.DataFrame(stress_results)

# Streamlit app
st.title("Dynamic Portfolio Risk Hedging AI with PyTorch")

# Sidebar inputs
st.sidebar.header("Portfolio Settings")
tickers = st.sidebar.text_input("Enter Tickers (comma-separated)", value="AAPL,MSFT,GOOGL,AMZN").split(",")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-01-01"))
lookback = st.sidebar.slider("Lookback Period (days)", 10, 60, 30)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=2.0) / 100
mse_threshold = st.sidebar.number_input("MSE Threshold (%)", min_value=0.01, max_value=1.0, value=0.1) / 100

# Fetch data
st.subheader("Fetching Data")
with st.spinner("Downloading data..."):
    data, returns = get_data(tickers, start_date, end_date)
    st.write(f"Data fetched for {len(tickers)} tickers.")
    st.line_chart(data)

# Portfolio metrics
st.subheader("Portfolio Metrics")
mean_returns, volatilities, sharpe_ratios = calculate_metrics(returns, risk_free_rate)
metrics_df = pd.DataFrame({
    "Mean Returns": mean_returns,
    "Volatility": volatilities,
    "Sharpe Ratio": sharpe_ratios
}, index=tickers)
st.table(metrics_df)

# Data preparation
scaler = MinMaxScaler()
scaled_returns = scaler.fit_transform(returns)
X, y = prepare_data(pd.DataFrame(scaled_returns, columns=returns.columns), lookback)
X_train, y_train = X[:-100], y[:-100]
X_test, y_test = X[-100:], y[-100:]

# Train AI model
st.subheader("Training AI Model")
with st.spinner("Training the model..."):
    input_size = len(tickers)
    hidden_size = 64
    output_size = len(tickers)
    model = LSTMModel(input_size, hidden_size, output_size)
    trained_model = train_model(model, X_train, y_train, mse_threshold)
st.success("Model trained successfully!")

# Hedge ratios
st.subheader("Dynamic Hedging Recommendations")
hedge_ratios = hedge_portfolio(trained_model, X_test, scaler, returns)
hedge_df = pd.DataFrame(hedge_ratios, columns=tickers, index=data.index[-len(hedge_ratios):])
st.line_chart(hedge_df)

# Stress test
st.subheader("Stress Test Results")
stress_results = stress_test(returns)
st.table(stress_results)

# Footer
st.sidebar.info("Created by an AI Assistant for Investment Banking Insights")
