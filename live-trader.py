import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import time
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import ta
import joblib

# ------------------------------
# Initialize MT5 Connection
# ------------------------------
if not mt5.initialize():
    print("MT5 initialization failed, error code =", mt5.last_error())
    quit()

# ------------------------------
# Parameters & Initialization
# ------------------------------
symbol = "EURUSDm"              # Use a valid symbol
timeframe = mt5.TIMEFRAME_M5      # 5-minute bars
initial_capital = 100.0
capital = initial_capital
risk_per_trade = 0.02            # 2% risk per trade (simplistic measure)
trade_signals = []               # Store trade details
capital_history = []             # Record capital after each trade

# Trading session parameters
session_duration = dt.timedelta(hours=2)
session_start = dt.datetime.now()
session_end = session_start + session_duration

# ------------------------------
# Feature Engineering Function (13 features)
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
    df['doji'] = (abs(df['open'] - df['close']) <= (df['high'] - df['low'])*0.1).astype(int)
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

# ------------------------------
# Load Model & Scaler
# ------------------------------
model = keras.models.load_model("./atlas_trading_model.keras")
scaler = joblib.load("./scaler.pkl")

# ------------------------------
# Live Trading Loop
# ------------------------------
print("Starting live trading session...")
while dt.datetime.now() < session_end and capital >= initial_capital * 0.7:
    # Retrieve the latest 17 days of data for feature calculations
    live_end_time = dt.datetime.now()
    live_start_time = live_end_time - dt.timedelta(days=17)
    rates = mt5.copy_rates_range(symbol, timeframe, live_start_time, live_end_time)
    
    if rates is None or len(rates) == 0:
        print("No data retrieved. Waiting for next update...")
        time.sleep(60)
        continue

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = process_features(df)
    
    # Ensure there is data after processing
    if df.empty:
        print("No valid data after feature processing. Waiting...")
        time.sleep(60)
        continue

    features_list = ['sma_20', 'ema_10', 'rsi', 'macd', 'macd_signal', 'atr', 
                     'boll_upper', 'boll_lower', 'spread', 'volume_delta', 
                     'doji', 'bullish_engulfing', 'bearish_engulfing']
    
    X_live = scaler.transform(df[features_list].values)
    predictions = model.predict(X_live).flatten()
    
    # Use the latest available prediction for decision making
    latest_pred = predictions[-1]
    latest_candle = df.iloc[-1]
    latest_open = latest_candle['open']
    latest_close = latest_candle['close']
    
    # Simulate trade decision based on prediction:
    if latest_pred > 0.55:
        trade_type = 'BUY'
        # In live trading, you would send an order here.
        # For simulation, we use the latest candle's open as entry and close as exit.
        entry_price = latest_open
        exit_price = latest_close
        position_size = (capital * risk_per_trade) / (entry_price * 0.0001)
        pnl = position_size * (exit_price - entry_price)
        capital += pnl
        trade_signals.append((df.index[-1], entry_price, exit_price, trade_type, pnl))
        print(f"[{df.index[-1]}] Executed BUY trade: Entry={entry_price:.5f}, Exit={exit_price:.5f}, PnL={pnl:.2f}, Capital={capital:.2f}")
        
    elif latest_pred < 0.45:
        trade_type = 'SELL'
        entry_price = latest_open
        exit_price = latest_close
        position_size = (capital * risk_per_trade) / (entry_price * 0.0001)
        pnl = position_size * (entry_price - exit_price)
        capital += pnl
        trade_signals.append((df.index[-1], entry_price, exit_price, trade_type, pnl))
        print(f"[{df.index[-1]}] Executed SELL trade: Entry={entry_price:.5f}, Exit={exit_price:.5f}, PnL={pnl:.2f}, Capital={capital:.2f}")
    else:
        print(f"[{df.index[-1]}] No trade executed. Prediction={latest_pred:.4f}")

    capital_history.append(capital)
    
    # Sleep until next candle (assuming 5-minute bars)
    time.sleep(300)

# Check why the loop ended
if capital < initial_capital * 0.7:
    print("Terminating trading: Capital fell below 70% of initial.")
else:
    print("Trading session ended after 2 hours.")

# ------------------------------
# Visualization (Optional)
# ------------------------------
plt.figure(figsize=(14,7))
plt.plot(capital_history, label='Capital History', color='purple')
plt.title('Live Trading Capital History (Simulated)')
plt.xlabel('Trade Count')
plt.ylabel('Capital ($)')
plt.grid(True)
plt.legend()
plt.show()

print("Trade Signals:")
for sig in trade_signals:
    print(sig)

mt5.shutdown()
