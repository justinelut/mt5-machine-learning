import pandas as pd
import numpy as np
import tensorflow as tf
import ta
import joblib
import time
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ------------------------------
# Initialize MT5 Connection
# ------------------------------
def initialize_mt5():
    if not mt5.initialize():
        print("MT5 initialization failed, error code =", mt5.last_error())
        quit()
    else:
        print("MT5 initialized successfully")

# ------------------------------
# Feature Engineering (13 features)
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
# Prediction Function
# ------------------------------
def predict_trades(df):
    features_list = [
        'sma_20', 'ema_10', 'rsi', 'macd', 'macd_signal', 'atr',
        'boll_upper', 'boll_lower', 'spread', 'volume_delta',
        'doji', 'bullish_engulfing', 'bearish_engulfing'
    ]
    
    # Ensure we have enough data
    if len(df) < 20:
        print("Not enough data for prediction")
        return None
    
    # Scale features
    try:
        X = scaler.transform(df[features_list].values)
    except ValueError as e:
        print("Scaling error:", e)
        return None
    
    # Predict
    df['prediction'] = model.predict(X).flatten()
    return df

# ------------------------------
# Profit Calculation
# ------------------------------
def calculate_profit(df, starting_capital=1000, lot_size=0.1):
    df['position'] = np.where(df['prediction'] > 0.55, 1, 
                             np.where(df['prediction'] < 0.45, -1, 0))
    
    # Calculate returns
    df['returns'] = df['close'].pct_change() * df['position'].shift(1)
    df['strategy'] = (df['returns'] + 1).cumprod()
    df['capital'] = starting_capital * df['strategy']
    return df

# ------------------------------
# Trading Execution
# ------------------------------
def execute_trades(symbol="EURUSDm", backtest_days=None, lot_size=0.1, show_plot=False):
    initialize_mt5()
    
    try:
        if backtest_days is not None:
            # Backtesting mode
            print(f"Running backtest for {backtest_days} days...")
            timeframe = mt5.TIMEFRAME_M5
            num_bars = backtest_days * 288  # 288 5-min bars per day
            
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
            if rates is None or len(rates) == 0:
                print("Failed to fetch historical data")
                return
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
            
            # Process and predict
            df = process_features(df)
            df = predict_trades(df)
            
            if df is not None:
                df = calculate_profit(df, starting_capital=1000, lot_size=lot_size)
                print("\nBacktest Results:")
                print(f"Final Capital: ${df['capital'].iloc[-1]:.2f}")
                print(f"Profit: {((df['capital'].iloc[-1]-1000)/1000*100):.2f}%")
                
                if show_plot:
                    plt.figure(figsize=(12,6))
                    plt.plot(df['time'], df['capital'])
                    plt.title("Backtest Capital Curve")
                    plt.show()
    
    finally:
        mt5.shutdown()
        print("MT5 connection closed")

# ------------------------------
# Load Model & Scaler
# ------------------------------
model = tf.keras.models.load_model("./atlas_trading_model.keras")
scaler = joblib.load("./scaler.pkl")

# Run backtest
execute_trades(symbol="EURUSDm", backtest_days=7, lot_size=0.1, show_plot=True)