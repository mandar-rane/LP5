import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

df = pd.read_csv('GOOGL.csv')

df.head()

prices = df['Close'].values.reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0, 1))

scaled_prices = scaler.fit_transform(prices)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length -1):
        X.append(data[i:(i+seq_length), 0])
        y.append(data[i+seq_length, 0])
    return np.array(X), np.array(y)

seq_length = 20

X, y = create_sequences(scaled_prices, seq_length)

train_size = int(len(X)*0.8)
test_size = len(X) - train_size

X_train, X_test = X[0:train_size], X[train_size: len(X)]
y_train, y_test = y[0:train_size], y[train_size: len(y)] 

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(units=50),
    Dense(units=1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=1)
predictions = model.predict(X_test)

plt.figure(figsize=(10,6))
plt.plot(y_test, label='True Stock Price')
plt.plot(predictions, label='Predicted Stock Price')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Google Stock Price Prediction')
plt.legend()
plt.show()