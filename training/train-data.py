import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import ta
import joblib

# ------------------------------
# Load data from CSV
# ------------------------------
df = pd.read_csv("EURUSDm_historical_data.csv")

# Convert and clean time data
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# Ensure proper column names (adjust based on your CSV structure)
required_columns = {
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'tick_volume': 'Volume'
}

# Rename columns to match MT5 structure (if needed)
df = df.rename(columns={
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'tick_volume'
})

# ------------------------------
# Feature Engineering (same as before)
# ------------------------------
df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
df['ema_10'] = ta.trend.ema_indicator(df['close'], window=10)
df['rsi'] = ta.momentum.rsi(df['close'], window=14)
df['macd'] = ta.trend.macd(df['close'])
df['macd_signal'] = ta.trend.macd_signal(df['close'])
df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
df['boll_upper'] = ta.volatility.bollinger_hband(df['close'], window=20)
df['boll_lower'] = ta.volatility.bollinger_lband(df['close'], window=20)
df['spread'] = df['close'] - df['open']
df['volume_delta'] = df['tick_volume'].diff()
df['doji'] = (abs(df['open'] - df['close']) <= (df['high'] - df['low']) * 0.1).astype(int)
df['bullish_engulfing'] = ((df['close'].shift(1) < df['open'].shift(1)) &
                           (df['close'] > df['open']) &
                           (df['close'] > df['open'].shift(1)) &
                           (df['open'] < df['close'].shift(1))).astype(int)
df['bearish_engulfing'] = ((df['close'].shift(1) > df['open'].shift(1)) &
                           (df['close'] < df['open']) &
                           (df['close'] < df['open'].shift(1)) &
                           (df['open'] > df['close'].shift(1))).astype(int)

# Target variable: 1 if next candle's close is higher, else 0
df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
df.dropna(inplace=True)

# ------------------------------
# Data Preprocessing
# ------------------------------
features = ['sma_20', 'ema_10', 'rsi', 'macd', 'macd_signal', 'atr',
            'boll_upper', 'boll_lower', 'spread', 'volume_delta',
            'doji', 'bullish_engulfing', 'bearish_engulfing']

# Split data (this will work with any amount of data)
split_index = int(len(df) * 2/3)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[features])
X_test = scaler.transform(test_df[features])
y_train = train_df['target']
y_test = test_df['target']

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# ------------------------------
# Model Architecture
# ------------------------------
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(len(features),)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Removed early stopping to ensure training goes for 100 epochs
history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)


# Save the model
model.save("atlas_trading_model.keras")

# ------------------------------
# Simulation
# ------------------------------
test_dates = test_df.index
test_data = test_df[['open', 'close']].values
predictions = model.predict(X_test).flatten()

# Set starting capital to $100
capital = 100.0
risk_per_trade = 0.02  # You can adjust this as needed
capital_history = [capital]
trade_signals = []

for i in range(len(predictions)-1):
    if predictions[i] > 0.55:
        entry_price = test_data[i+1][0]  # use open price of next candle for entry
        exit_price = test_data[i+1][1]   # use close price of next candle for exit
        position_size = (capital * risk_per_trade) / (entry_price * 0.0001)
        pnl = position_size * (exit_price - entry_price)
        capital += pnl
        trade_signals.append((test_dates[i+1], entry_price))
    capital_history.append(capital)

# ------------------------------
# Visualization
# ------------------------------
plt.figure(figsize=(14,7))
plt.plot(test_dates[:len(capital_history)], capital_history, label='Capital Growth')
plt.title('Trading Strategy Performance')
plt.xlabel('Date')
plt.ylabel('Capital ($)')
plt.grid(True)
plt.legend()
plt.show()

print(f"Starting Capital: $100")
print(f"Final Capital: ${capital_history[-1]:.2f}")
print(f"Total Return: {((capital_history[-1]/100)-1)*100:.2f}%")
