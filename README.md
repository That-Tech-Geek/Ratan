# Ratan

A small Streamlit dashboard for market-data analysis, return modelling, and basic capital-efficiency calculations.

## What it does

- Downloads adjusted market prices with Yahoo Finance.
- Calculates daily returns and displays price/return charts.
- Shows the latest 10-year US Treasury yield when available.
- Fits a simple linear regression to rolling return windows.
- Calculates WACC and a simplified ROIC from user-supplied inputs.

This is an analytical prototype, not an execution or investment-advice system.

## Run locally

Requires Python 3.11+.

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Project structure

```
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── streamlit_app.py
```

## Notes

Yahoo Finance is an external data source and may be unavailable or delayed. The return model reports an in-sample fit metric only; it should not be interpreted as a validated trading strategy.
