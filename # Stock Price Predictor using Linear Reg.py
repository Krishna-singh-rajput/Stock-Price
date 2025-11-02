# Stock Price Predictor using Linear Regression
# Author: Krishna Singh

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import yfinance as yf

# Step 2: Download Historical Stock Data (e.g., Reliance Industries)
stock_symbol = "RELIANCE.NS"   # You can change to any stock like "AAPL", "TCS.NS", etc.
data = yf.download(stock_symbol, start="2020-01-01", end="2024-12-31")
data.reset_index(inplace=True)

# Step 3: Prepare Data
data['Prediction'] = data[['Close']].shift(-1)
X = np.array(data[['Open', 'High', 'Low', 'Volume']])
y = np.array(data['Prediction'])
X = X[:-1]
y = y[:-1]

# Step 4: Split Data for Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Test and Evaluate Model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f" Mean Absolute Error: {mae:.2f}")
print(f" R² Score: {r2:.2f}")

# Step 7: Predict Next Day Price
next_day = np.array(data[['Open', 'High', 'Low', 'Volume']].iloc[-1]).reshape(1, -1)
predicted_price = model.predict(next_day)
print(f"\n Predicted Next Day Closing Price for {stock_symbol}: ₹{predicted_price[0]:.2f}")

# Step 8: Visualization
plt.figure(figsize=(10,5))
plt.plot(y_test[-50:], label='Actual Price', linewidth=2)
plt.plot(predictions[-50:], label='Predicted Price', linestyle='--')
plt.title(f'Stock Price Prediction for {stock_symbol}')
plt.xlabel('Days')
plt.ylabel('Price (INR)')
plt.legend()
plt.show()
