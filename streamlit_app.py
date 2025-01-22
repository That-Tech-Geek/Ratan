import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Define the neural network model for regression
class PortfolioRiskHedgingModel(nn.Module):
    def __init__(self, input_size):
        super(PortfolioRiskHedgingModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)  # Regression model with one output (for price prediction)

    def forward(self, x):
        return self.fc(x)

# Function to fetch historical stock data
def get_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    returns = data.pct_change().dropna()
    return data, returns

# Function to prepare the data for training (create X and y)
def prepare_data(returns, lookback):
    X = []
    y = []
    
    for i in range(lookback, len(returns)):
        X.append(returns.iloc[i - lookback:i].values.flatten())  # Collecting 'lookback' periods of returns
        y.append(returns.iloc[i].values)  # Target is the return at time 'i'
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape y to ensure it's a 2D array (batch_size, 1)
    y = y.reshape(-1, 1)
    
    return X, y

# Train the model with early stopping based on MSE threshold
def train_model(model, X_train, y_train, mse_threshold=0.001, num_epochs=100):
    criterion = nn.MSELoss()  # Mean Squared Error loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    # Ensure y_train_tensor is reshaped to match output dimension (batch_size, 1)
    if y_train_tensor.ndimension() == 1:
        y_train_tensor = y_train_tensor.view(-1, 1)  # Reshapes to (batch_size, 1)
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()  # Zero the gradients

        outputs = model(X_train_tensor)  # Model output (predicted returns)
        
        # Check the shapes before calculating loss
        print(f"Epoch {epoch+1}: Shape of outputs: {outputs.shape}")
        print(f"Epoch {epoch+1}: Shape of y_train_tensor: {y_train_tensor.shape}")

        loss = criterion(outputs, y_train_tensor)  # Calculate the loss
        loss.backward()  # Backpropagation to compute gradients
        optimizer.step()  # Update model parameters

        mse = loss.item()  # Get the MSE from the loss
        if mse < mse_threshold:
            print(f"Stopping early at epoch {epoch+1} due to MSE < {mse_threshold}")
            break

        print(f"Epoch {epoch+1}: Loss = {mse}")

    return model

# Streamlit app setup
def main():
    st.title("Dynamic Portfolio Risk Hedging AI")

    tickers = st.text_input("Enter Stock Tickers (comma separated)", "AAPL,GOOG,MSFT")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2021-01-01"))
    lookback = st.slider("Lookback Period (Days)", min_value=10, max_value=100, value=30)
    mse_threshold = st.slider("MSE Threshold", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001)
    
    if tickers and start_date and end_date:
        tickers = tickers.split(",")
        data, returns = get_data(tickers, start_date, end_date)
        
        # Prepare data for training
        X, y = prepare_data(returns, lookback)
        
        # Normalize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Initialize and train the model
        model = PortfolioRiskHedgingModel(input_size=X_scaled.shape[1])  # Input size is the number of features
        trained_model = train_model(model, X_scaled, y, mse_threshold)

        # Display the results
        st.write(f"Model trained with MSE threshold of {mse_threshold}")
        st.write(f"Trained Model: {trained_model}")
        
        # Optionally, you can use the model to make predictions on new data here.

if __name__ == "__main__":
    main()
