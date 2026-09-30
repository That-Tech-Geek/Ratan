"""Ratan: Streamlit portfolio risk and valuation dashboard."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def get_price_data(
    tickers: list[str], start_date: date, end_date: date
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download adjusted closing prices and calculate daily returns."""
    if not tickers:
        raise ValueError("Enter at least one ticker.")
    if start_date >= end_date:
        raise ValueError("Start date must be before end date.")

    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date + timedelta(days=1),
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise ValueError("No market data was returned. Check the tickers and date range.")

    prices = raw["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    prices.columns = [str(column).upper() for column in prices.columns]
    prices = prices.dropna(how="all").dropna(axis=1, how="all")

    if prices.empty:
        raise ValueError("No usable closing prices were returned.")

    returns = prices.pct_change(fill_method=None).dropna(how="all")
    return prices, returns


def get_risk_free_rate() -> float:
    """Return the latest 10-year US Treasury yield as a decimal."""
    history = yf.Ticker("^TNX").history(period="5d", auto_adjust=False)
    if history.empty or "Close" not in history:
        raise ValueError("Unable to retrieve the 10-year Treasury yield.")

    close = history["Close"].dropna()
    if close.empty:
        raise ValueError("The Treasury yield series contained no usable values.")

    return float(close.iloc[-1]) / 100.0


def calculate_wacc(
    equity: float,
    debt: float,
    cost_of_equity: float,
    cost_of_debt: float,
    tax_rate: float,
) -> float:
    """Calculate weighted average cost of capital."""
    total_capital = equity + debt
    if total_capital <= 0:
        raise ValueError("Equity plus debt must be greater than zero.")

    return (
        equity / total_capital * cost_of_equity
        + debt / total_capital * cost_of_debt * (1 - tax_rate)
    )


def calculate_roic(net_income: float, debt: float, equity: float) -> float:
    """Calculate the simplified ROIC used by this dashboard."""
    invested_capital = debt + equity
    if invested_capital <= 0:
        raise ValueError("Debt plus equity must be greater than zero.")
    return net_income / invested_capital


def prepare_data(returns: pd.DataFrame, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Create rolling return windows for a simple next-period regression."""
    if lookback < 1:
        raise ValueError("Lookback must be at least one period.")
    clean = returns.dropna()
    if len(clean) <= lookback:
        raise ValueError("Not enough complete observations for the selected lookback.")

    values = clean.to_numpy(dtype=float)
    x = np.array([values[i : i + lookback].ravel() for i in range(len(values) - lookback)])
    y = values[lookback:]
    return x, y


def train_model(
    x: np.ndarray, y: np.ndarray
) -> tuple[LinearRegression, StandardScaler, StandardScaler, float]:
    """Fit a bounded linear model and return its in-sample MSE."""
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaled = x_scaler.fit_transform(x)
    y_scaled = y_scaler.fit_transform(y)

    model = LinearRegression()
    model.fit(x_scaled, y_scaled)
    predictions = model.predict(x_scaled)
    mse = float(np.mean((predictions - y_scaled) ** 2))

    return model, x_scaler, y_scaler, mse


def main() -> None:
    st.set_page_config(page_title="Ratan", page_icon="📈", layout="wide")
    st.title("Ratan")
    st.caption("Portfolio risk, market data, and basic valuation analytics.")

    with st.sidebar:
        st.header("Market data")
        ticker_input = st.text_input("Tickers", "AAPL, MSFT, TSLA")
        default_start = date.today() - timedelta(days=365)
        start_date = st.date_input("Start date", default_start)
        end_date = st.date_input("End date", date.today())
        lookback = st.slider("Return lookback", 1, 30, 5)

    tickers = [ticker.strip().upper() for ticker in ticker_input.split(",") if ticker.strip()]

    try:
        prices, returns = get_price_data(tickers, start_date, end_date)
    except Exception as exc:
        st.error(str(exc))
        return

    left, right = st.columns(2)
    with left:
        st.subheader("Price history")
        st.line_chart(prices)
    with right:
        st.subheader("Daily returns")
        st.line_chart(returns)

    try:
        risk_free_rate = get_risk_free_rate()
        st.metric("10Y Treasury yield", f"{risk_free_rate:.2%}")
    except Exception as exc:
        st.warning(f"Risk-free rate unavailable: {exc}")

    st.divider()
    st.subheader("Return model")

    try:
        x, y = prepare_data(returns, lookback)
        _, _, _, mse = train_model(x, y)
        st.metric("In-sample standardized MSE", f"{mse:.4f}")
        st.caption("This is a descriptive fit metric, not a trading signal or forecast guarantee.")
    except Exception as exc:
        st.warning(f"Model unavailable: {exc}")

    st.divider()
    st.subheader("Capital efficiency")

    col1, col2 = st.columns(2)
    with col1:
        equity = st.number_input("Equity (USD)", min_value=0.0, value=1_000_000.0, step=10_000.0)
        debt = st.number_input("Debt (USD)", min_value=0.0, value=250_000.0, step=10_000.0)
        cost_of_equity = st.number_input("Cost of equity (%)", min_value=0.0, value=10.0, step=0.1) / 100
        cost_of_debt = st.number_input("Cost of debt (%)", min_value=0.0, value=6.0, step=0.1) / 100
        tax_rate = st.number_input("Tax rate (%)", min_value=0.0, max_value=100.0, value=25.0, step=0.5) / 100

    with col2:
        net_income = st.number_input("Net income (USD)", min_value=0.0, value=150_000.0, step=10_000.0)
        try:
            wacc = calculate_wacc(equity, debt, cost_of_equity, cost_of_debt, tax_rate)
            roic = calculate_roic(net_income, debt, equity)
            st.metric("WACC", f"{wacc:.2%}")
            st.metric("ROIC", f"{roic:.2%}")
        except ValueError as exc:
            st.warning(str(exc))


if __name__ == "__main__":
    main()
