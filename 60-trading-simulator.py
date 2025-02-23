import pandas as pd
import numpy as np
import tensorflow as tf
import ta
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ------------------------------
# Load Data from CSV
# ------------------------------
csv_file = "./EURUSDm_historical_data.csv"  # Replace with your actual CSV file
df = pd.read_csv(csv_file)

# Ensure correct column names
df.rename(columns={'tick_volume': 'volume'}, inplace=True)  # Fix tick_volume if needed

# Convert time column to datetime if necessary
if 'time' in df.columns:
    df['time'] = pd.to_datetime(df['time'])

# ------------------------------
# Load Model & Scaler
# ------------------------------
model = tf.keras.models.load_model("./atlas_trading_model.keras")
scaler = joblib.load("./scaler.pkl")


def process_features(df):
    """Generate indicators used in training"""
    try:
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['ema_10'] = ta.trend.ema_indicator(df['close'], window=10)
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['macd'] = ta.trend.macd(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        df['boll_upper'] = ta.volatility.bollinger_hband(df['close'], window=20)
        df['boll_lower'] = ta.volatility.bollinger_lband(df['close'], window=20)
        df['spread'] = df['close'] - df['open']
        df['volume_delta'] = df['volume'].diff()
        df['doji'] = (abs(df['open'] - df['close']) <= (df['high'] - df['low']) * 0.1).astype(int)
        df['bullish_engulfing'] = ((df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open']) & 
                                   (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))).astype(int)
        df['bearish_engulfing'] = ((df['close'].shift(1) > df['open'].shift(1)) & (df['close'] < df['open']) & 
                                   (df['close'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1))).astype(int)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error processing features: {e}")
        return None


def predict_trades(df):
    """Predict trades based on processed data"""
    features = ['sma_20', 'ema_10', 'rsi', 'macd', 'macd_signal', 'atr', 'boll_upper',
                'boll_lower', 'spread', 'volume_delta', 'doji', 'bullish_engulfing', 'bearish_engulfing']

    try:
        if df is None or df.empty:
            print("Error: DataFrame is empty or None")
            return None

        # Ensure feature columns exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"Missing features: {missing_features}")
            return None

        X_live = scaler.transform(df[features].values)  # Scale input features
        predictions = model.predict(X_live)

        df['prediction'] = predictions  # Append predictions to DataFrame
        return df
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None


def calculate_profit(df, starting_capital=1000, lot_size=1):
    """
    Simulate trading strategy based on model predictions.
    - Buy when prediction > 0.5
    - Sell when prediction < 0.5
    - Assume we buy/sell 1 lot
    """
    df['position'] = np.where(df['prediction'] > 0.5, 1, -1)  # 1 for buy, -1 for sell

    df['returns'] = df['close'].pct_change() * df['position'].shift(1)  # Daily returns
    df['cumulative_returns'] = (1 + df['returns']).cumprod()  # Cumulative performance
    df['capital'] = starting_capital * df['cumulative_returns']  # Capital growth

    # Print final capital and profit percentage
    final_capital = df['capital'].iloc[-1]
    profit_percentage = ((final_capital - starting_capital) / starting_capital) * 100
    print(f"Final Capital: ${final_capital:.2f}")
    print(f"Percentage Profit: {profit_percentage:.2f}%")

    return df


def plot_results(df):
    """Plot trading performance"""
    plt.figure(figsize=(12, 6))

    # Plot close price and signals
    plt.subplot(2, 1, 1)
    plt.plot(df['time'], df['close'], label="Close Price", color='blue')
    plt.scatter(df['time'][df['position'] == 1], df['close'][df['position'] == 1], marker='^', color='green', label="Buy Signal")
    plt.scatter(df['time'][df['position'] == -1], df['close'][df['position'] == -1], marker='v', color='red', label="Sell Signal")
    plt.title("Trading Signals on Price")
    plt.legend()

    # Plot capital growth
    plt.subplot(2, 1, 2)
    plt.plot(df['time'], df['capital'], color='black', label="Capital Growth")
    plt.title("Capital Growth Over Time")
    plt.legend()

    plt.tight_layout()
    plt.show()


# ------------------------------
# Run Processing, Predictions, Profit Calculation & Plotting
# ------------------------------
df = process_features(df)
if df is not None:
    df = predict_trades(df)
    if df is not None:
        df = calculate_profit(df, starting_capital=1000, lot_size=1)  # Adjust initial capital if needed
        print(df[['time', 'close', 'prediction', 'position', 'capital']].tail(10))  # Show last 10 results
        df.to_csv("predicted_trades.csv", index=False)  # Save results
        plot_results(df)  # Show results
