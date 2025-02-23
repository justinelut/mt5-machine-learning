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
csv_file = "./EURUSDm_historical_data.csv"  # Replace with your CSV file path
df = pd.read_csv(csv_file)

# Convert and clean time data
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# Rename columns to match expected format
df = df.rename(columns={
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'tick_volume'
})

# ------------------------------
# Feature Engineering: Compute all 13 features
# ------------------------------
def process_features(df):
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
    df.dropna(inplace=True)
    return df

df = process_features(df)

# ------------------------------
# Use only the last 10 days of data
# ------------------------------
last_10_df = df[df.index >= df.index.max() - pd.Timedelta(days=10)]

# ------------------------------
# Load Model & Scaler
# ------------------------------
model = keras.models.load_model("./atlas_trading_model.keras")
scaler = joblib.load("./scaler.pkl")

# ------------------------------
# Predict Trades on Last 10 Days Data
# ------------------------------
features = ['sma_20', 'ema_10', 'rsi', 'macd', 'macd_signal', 'atr', 
            'boll_upper', 'boll_lower', 'spread', 'volume_delta', 
            'doji', 'bullish_engulfing', 'bearish_engulfing']
X_live = scaler.transform(last_10_df[features].values)
predictions = model.predict(X_live).flatten()
last_10_df['prediction'] = predictions

# ------------------------------
# Simulate Trading on Last 10 Days
# ------------------------------
# For simulation, we'll use the 'open' and 'close' prices from last_10_df
test_dates = last_10_df.index
test_data = last_10_df[['open', 'close']].values

# Set starting capital to $100
capital = 100.0
risk_per_trade = 0.02  # 2% risk per trade (this is a simplistic measure)
capital_history = [capital]
trade_signals = []  # Each element: (timestamp, entry_price, exit_price, trade_type, pnl)

# Simulate trades:
# - If prediction > 0.55, then BUY: use next candle's open for entry and close for exit.
# - If prediction < 0.45, then SELL: use next candle's open for entry and close for exit.
# - If prediction is between 0.45 and 0.55, no trade is taken.
for i in range(len(predictions)-1):
    if predictions[i] > 0.55:
        trade_type = 'BUY'
        entry_price = test_data[i+1][0]   # open price of next candle
        exit_price = test_data[i+1][1]    # close price of next candle
        position_size = (capital * risk_per_trade) / (entry_price * 0.0001)  # Simplistic position sizing
        pnl = position_size * (exit_price - entry_price)
        capital += pnl
        trade_signals.append((test_dates[i+1], entry_price, exit_price, trade_type, pnl))
    elif predictions[i] < 0.45:
        trade_type = 'SELL'
        entry_price = test_data[i+1][0]   # open price of next candle for short entry
        exit_price = test_data[i+1][1]    # close price of next candle for short exit
        position_size = (capital * risk_per_trade) / (entry_price * 0.0001)
        pnl = position_size * (entry_price - exit_price)
        capital += pnl
        trade_signals.append((test_dates[i+1], entry_price, exit_price, trade_type, pnl))
    # If prediction is between 0.45 and 0.55, no trade is made
    capital_history.append(capital)

# ------------------------------
# Visualization
# ------------------------------
plt.figure(figsize=(14,7))
plt.plot(test_dates[:len(capital_history)], capital_history, label='Capital Growth', color='purple')
plt.title('Trading Strategy Performance (Last 10 Days)')
plt.xlabel('Date')
plt.ylabel('Capital ($)')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(14,7))
plt.plot(last_10_df.index, last_10_df['close'], label='Market Price', color='blue')
# Plot buy and sell signals
buy_times = [sig[0] for sig in trade_signals if sig[3]=='BUY']
buy_prices = [sig[1] for sig in trade_signals if sig[3]=='BUY']
sell_times = [sig[0] for sig in trade_signals if sig[3]=='SELL']
sell_prices = [sig[2] for sig in trade_signals if sig[3]=='SELL']
plt.scatter(buy_times, buy_prices, marker='^', color='green', label='Buy Signal')
plt.scatter(sell_times, sell_prices, marker='v', color='red', label='Sell Signal')
plt.title('Market Price & Trading Signals (Last 10 Days)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

print(f"Starting Capital: $100")
print(f"Final Capital: ${capital:.2f}")
print(f"Total Return: {((capital/100)-1)*100:.2f}%")

print("Trade Signals:")
for sig in trade_signals:
    print(sig)
