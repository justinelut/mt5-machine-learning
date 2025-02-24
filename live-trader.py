import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress TensorFlow warnings
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
from pprint import pprint

# ------------------------------
# Initialize MT5 Connection
# ------------------------------
def init_mt5():
    # Replace these with your MT5 account credentials
    MT5_ACCOUNT = 208552812
    MT5_PASSWORD = "Ch!L$ea#1.Shant"
    MT5_SERVER = "ExnessKE-MT5Trial9"
    
    if not mt5.initialize():
        print("MT5 initialization failed, error code =", mt5.last_error())
        quit()
        
    print("MT5 initialized successfully!")
    
    # Attempt login
    authorized = mt5.login(login=MT5_ACCOUNT, password=MT5_PASSWORD, server=MT5_SERVER)
    if not authorized:
        print(f"Login failed: {mt5.last_error()}")
        mt5.shutdown()
        quit()
    
    print(f"Connected to account #{MT5_ACCOUNT}")
    print(f"Account Balance: {mt5.account_info().balance}")
    return True

# ------------------------------
# Parameters & Configuration
# ------------------------------
SYMBOL = "EURUSDm"
TIMEFRAME = mt5.TIMEFRAME_M5
RISK_PERCENT = 0.02  # 2% risk per trade
MAX_SPREAD = 2.0  # Max allowed spread in pips
TAKE_PROFIT_PIPS = 20
STOP_LOSS_PIPS = 10

# ------------------------------
# Feature Engineering
# ------------------------------
def process_features(df):
    # Technical indicators calculation
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
    return df.dropna()

# ------------------------------
# Trading Utilities
# ------------------------------
def get_symbol_data(symbol):
    """Retrieve and validate symbol information"""
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Symbol {symbol} not found")
        return None
    if not symbol_info.visible:
        print(f"Symbol {symbol} is not visible, trying to switch on")
        if not mt5.symbol_select(symbol, True):
            print(f"Failed to select {symbol}")
            return None
    return symbol_info

def get_current_tick(symbol):
    """Get current tick data with error handling"""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"Failed to get tick data for {symbol}: {mt5.last_error()}")
        return None
    return tick

def calculate_position_size(symbol, risk_percent, stop_loss_pips):
    """Calculate proper position size based on risk management"""
    symbol_info = mt5.symbol_info(symbol)
    account_balance = mt5.account_info().balance
    risk_amount = account_balance * risk_percent
    
    # Calculate pip value
    pip_value = mt5.market_book(symbol).data[0].price * 0.0001
    
    # Calculate position size
    position_size = risk_amount / (stop_loss_pips * pip_value)
    return round(position_size, 2)

def send_trade_order(order_type, symbol, lot_size, sl_pips, tp_pips):
    """Execute trade with proper order handling"""
    symbol_info = mt5.symbol_info(symbol)
    point = symbol_info.point
    price = mt5.symbol_info_tick(symbol).ask if order_type == 'BUY' else mt5.symbol_info_tick(symbol).bid
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_BUY if order_type == 'BUY' else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": price - sl_pips * point if order_type == 'BUY' else price + sl_pips * point,
        "tp": price + tp_pips * point if order_type == 'BUY' else price - tp_pips * point,
        "deviation": 20,
        "magic": 202404,
        "comment": "AI-Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed: {result.comment}")
        return None
    return result

# ------------------------------
# Main Trading Logic
# ------------------------------
def main():
    # Initialize MT5 connection
    if not init_mt5():
        return
    
    # Load AI model and scaler
    model = keras.models.load_model("atlas_trading_model.keras")
    scaler = joblib.load("scaler.pkl")
    
    print("\n=== Trading Bot Started ===")
    print(f"Symbol: {SYMBOL}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Risk Management: {RISK_PERCENT*100}% per trade")
    
    last_bar_time = None
    while True:
        try:
            # Check symbol availability
            if not get_symbol_data(SYMBOL):
                time.sleep(10)
                continue
                
            # Get current market data
            rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1000)
            if rates is None:
                print("Failed to get market data")
                time.sleep(60)
                continue
                
            # Process data
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Check for new bar
            if df.index[-1] == last_bar_time:
                time.sleep(10)
                continue
            last_bar_time = df.index[-1]
            
            # Feature engineering
            df_processed = process_features(df.copy())
            if df_processed.empty:
                continue
                
            # Prepare AI input
            features = ['sma_20', 'ema_10', 'rsi', 'macd', 'macd_signal', 
                       'atr', 'boll_upper', 'boll_lower', 'spread', 
                       'volume_delta', 'doji', 'bullish_engulfing', 'bearish_engulfing']
            X_live = scaler.transform(df_processed[features].tail(1))
            
            # Get prediction
            prediction = model.predict(X_live, verbose=0)[0][0]
            confidence = abs(prediction - 0.5) * 2  # 0-1 confidence
            
            print(f"\n[{dt.datetime.now()}] New Bar Analysis:")
            print(f"Close Price: {df['close'].iloc[-1]:.5f}")
            print(f"AI Prediction: {prediction:.4f}")
            print(f"Confidence: {confidence:.2%}")
            
            # Check existing positions
            positions = mt5.positions_get(symbol=SYMBOL)
            if positions and len(positions) > 0:
                print(f"Existing positions open: {len(positions)}")
                continue
                
            # Trade execution logic
            if prediction > 0.55 and confidence > 0.3:
                trade_type = 'BUY'
            elif prediction < 0.45 and confidence > 0.3:
                trade_type = 'SELL'
            else:
                print("No valid trading signal")
                continue
                
            # Calculate position size
            lot_size = calculate_position_size(SYMBOL, RISK_PERCENT, STOP_LOSS_PIPS)
            print(f"\n=== Executing {trade_type} Order ===")
            print(f"Lot Size: {lot_size}")
            print(f"Stop Loss: {STOP_LOSS_PIPS} pips")
            print(f"Take Profit: {TAKE_PROFIT_PIPS} pips")
            
            # Execute trade
            result = send_trade_order(trade_type, SYMBOL, lot_size, STOP_LOSS_PIPS, TAKE_PROFIT_PIPS)
            if result:
                print(f"\nTrade Executed Successfully:")
                pprint(result._asdict())
                print(f"New Balance: {mt5.account_info().balance:.2f}")
                
            time.sleep(10)  # Check every 10 seconds
            
        except Exception as e:
            print(f"Critical Error: {str(e)}")
            time.sleep(60)

if __name__ == "__main__":
    main()
    mt5.shutdown()