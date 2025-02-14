import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def fetch_stock_data(ticker, start_date):
    """
    Fetches historical stock data for a given ticker symbol.
    """
    data = yf.download(ticker, start=start_date)
    return data

def preprocess_data(data, window_size=30):
    """
    Prepares stock price data using a rolling window approach.
    Scales the data and creates sequences for training.
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']])

    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i+window_size])
        y.append(scaled_data[i+window_size])
    
    return np.array(X), np.array(y), scaler

def build_lstm_model(window_size):
    """
    Builds an LSTM model for stock price prediction.
    """
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(window_size, 1)),
        LSTM(units=50),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def evaluate_model(y_true, y_pred):
    """
    Evaluates model performance using RMSE.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse