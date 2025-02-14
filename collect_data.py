import os
import pandas as pd
from functions import fetch_stock_data, preprocess_data

# Define stock ticker and date range
TICKER = "AAPL"
START_DATE = "2020-01-01"
WINDOW_SIZE = 30

# Fetch the latest stock data
print("Fetching stock data...")
data = fetch_stock_data(TICKER, START_DATE)

# Preprocess data
print("Preprocessing data...")
X, y, scaler = preprocess_data(data, window_size=WINDOW_SIZE)

# Save preprocessed data
print("Saving preprocessed data...")
os.makedirs("data", exist_ok=True)
pd.DataFrame(X.reshape(X.shape[0], -1)).to_csv("data/X.csv", index=False)
pd.DataFrame(y).to_csv("data/y.csv", index=False)
pd.DataFrame(data).to_csv("data/raw_data.csv", index=True)

print("Data collection and preprocessing completed successfully!")