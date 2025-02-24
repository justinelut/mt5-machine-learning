import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import datetime as dt
import time
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import ta
import joblib
import matplotlib.pyplot as plt
from pprint import pprint

# ------------------------------
# Initialize MT5 Connection
# ------------------------------
def init_mt5():
    # Replace with your MT5 credentials
    MT5_ACCOUNT = 208552812
    MT5_PASSWORD = "Ch!L$ea#1.Shant"
    MT5_SERVER = "ExnessKE-MT5Trial9"
    
    if not mt5.initialize():
        print("MT5 initialization failed, error code =", mt5.last_error())
        quit()
        
    print("MT5 initialized successfully!")
    
    authorized = mt5.login(login=MT5_ACCOUNT, password=MT5_PASSWORD, server=MT5_SERVER)
    if not authorized:
        print(f"Login failed: {mt5.last_error()}")
        mt5.shutdown()
        quit()
    
    print(f"Connected to account #{MT5_ACCOUNT}")
    account_info = mt5.account_info()
    print(f"Account Balance: {account_info.balance:.2f}")
    print(f"Free Margin: {account_info.margin_free:.2f}")
    return True

# ------------------------------
# Parameters & Configuration
# ------------------------------
SYMBOL = "EURUSDm"
TIMEFRAME = mt5.TIMEFRAME_M5
RISK_PERCENT = 0.02  # 2% risk per trade
MIN_CONFIDENCE_BUY = 0.55
MIN_CONFIDENCE_SELL = 0.45
DATA_WINDOW_DAYS = 10  # Match backtester's 10-day window

# ------------------------------
# Feature Engineering (Same as Backtester)
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
    return df.dropna()

# ------------------------------
# Trading Utilities (Enhanced)
# ------------------------------
def calculate_position_size(entry_price, stop_loss_pips):
    account = mt5.account_info()
    if not account:
        return 0.0
    
    risk_amount = account.balance * RISK_PERCENT
    pip_value = (entry_price * 0.0001)  # For EURUSD
    position_size = risk_amount / (stop_loss_pips * pip_value)
    
    # Get symbol info
    symbol_info = mt5.symbol_info(SYMBOL)
    if not symbol_info:
        return 0.0
    
    # Apply lot size constraints
    position_size = np.clip(position_size, 
                           symbol_info.volume_min, 
                           symbol_info.volume_max)
    return round(position_size, 2)

def execute_trade(signal_type, entry_price):
    symbol_info = mt5.symbol_info(SYMBOL)
    if not symbol_info:
        return None
    
    # Calculate dynamic stop loss (2% price movement)
    stop_loss_pips = 20  # Fixed SL like backtester's 1 candle
    take_profit_pips = 40  # Fixed TP like backtester's 2 candles
    
    lot_size = calculate_position_size(entry_price, stop_loss_pips)
    if lot_size <= 0:
        return None
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_BUY if signal_type == 'BUY' else mt5.ORDER_TYPE_SELL,
        "price": entry_price,
        "sl": entry_price - (stop_loss_pips * 0.0001) if signal_type == 'BUY' else entry_price + (stop_loss_pips * 0.0001),
        "tp": entry_price + (take_profit_pips * 0.0001) if signal_type == 'BUY' else entry_price - (take_profit_pips * 0.0001),
        "deviation": 20,
        "magic": 202404,
        "comment": "LiveTrade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    
    result = mt5.order_send(request)
    return result

# ------------------------------
# Live Trading Core
# ------------------------------
def live_trading():
    # Initialize connections
    if not init_mt5():
        return
    
    # Load model components
    model = keras.models.load_model("atlas_trading_model.keras")
    scaler = joblib.load("scaler.pkl")
    
    print("\n=== Live Trading Started ===")
    print(f"Symbol: {SYMBOL}")
    print(f"Timeframe: M5")
    print(f"Risk per Trade: {RISK_PERCENT*100}%")
    
    last_bar_time = None
    trade_active = False
    capital_history = [mt5.account_info().balance]
    
    while True:
        try:
            # Get fresh data
            end_time = dt.datetime.now()
            start_time = end_time - dt.timedelta(days=DATA_WINDOW_DAYS)
            rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, start_time, end_time)
            
            if rates is None or len(rates) < 100:
                print("Data retrieval failed, retrying...")
                time.sleep(5)
                continue
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Check for new candle
            current_bar_time = df.index[-1]
            if current_bar_time == last_bar_time:
                time.sleep(1)
                continue
            last_bar_time = current_bar_time
            
            # Process features
            df_processed = process_features(df.copy())
            if df_processed.empty:
                continue
            
            # Prepare prediction input
            features = ['sma_20', 'ema_10', 'rsi', 'macd', 'macd_signal', 
                       'atr', 'boll_upper', 'boll_lower', 'spread', 
                       'volume_delta', 'doji', 'bullish_engulfing', 'bearish_engulfing']
            X_live = scaler.transform(df_processed[features].tail(1))
            
            # Get prediction
            prediction = model.predict(X_live, verbose=0)[0][0]
            current_price = mt5.symbol_info_tick(SYMBOL).ask
            
            print(f"\n[{dt.datetime.now()}] New Candle:")
            print(f"Price: {current_price:.5f}")
            print(f"Prediction: {prediction:.4f}")
            
            # Check existing positions
            positions = mt5.positions_get(symbol=SYMBOL)
            if positions and len(positions) > 0:
                print("Position already open, skipping...")
                capital_history.append(mt5.account_info().balance)
                time.sleep(1)
                continue
            
            # Generate trade signal (matching backtester logic)
            if prediction > MIN_CONFIDENCE_BUY:
                signal = 'BUY'
                entry_price = mt5.symbol_info_tick(SYMBOL).ask
            elif prediction < MIN_CONFIDENCE_SELL:
                signal = 'SELL'
                entry_price = mt5.symbol_info_tick(SYMBOL).bid
            else:
                print("No valid signal")
                time.sleep(1)
                continue
            
            # Execute trade
            result = execute_trade(signal, entry_price)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"\n=== {signal} Order Executed ===")
                print(f"Entry Price: {entry_price:.5f}")
                print(f"Volume: {result.volume:.2f} lots")
                print(f"SL: {result.sl:.5f}")
                print(f"TP: {result.tp:.5f}")
                capital_history.append(mt5.account_info().balance)
                
                # Plot update
                plt.figure(figsize=(10,5))
                plt.plot(capital_history)
                plt.title('Live Capital Growth')
                plt.xlabel('Trade Count')
                plt.ylabel('Balance')
                plt.pause(0.01)
                
            time.sleep(1)

        except Exception as e:
            print(f"Error: {str(e)}")
            time.sleep(5)

if __name__ == "__main__":
    live_trading()
    mt5.shutdown()
    print("Trading Terminated")