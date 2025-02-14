import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from functions import build_lstm_model, evaluate_model

# Load preprocessed data
print("Loading preprocessed data...")
X = pd.read_csv("data/X.csv").values
y = pd.read_csv("data/y.csv").values

# Reshape for LSTM input
WINDOW_SIZE = 30
X = X.reshape(X.shape[0], WINDOW_SIZE, 1)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build and train the LSTM model
print("Building and training the LSTM model...")
model = build_lstm_model(WINDOW_SIZE)
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
print("Evaluating model performance...")
y_pred = model.predict(X_test)
rmse = evaluate_model(y_test, y_pred)
print(f"Model RMSE: {rmse}")

# Save the trained model
print("Saving trained model...")
os.makedirs("models", exist_ok=True)
model.save("models/stock_prediction_model.h5")

print("Training completed successfully!")