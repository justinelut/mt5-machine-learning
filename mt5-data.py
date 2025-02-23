import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

# === 1. Initialize MT5 Connection ===
if not mt5.initialize():
    print("❌ MT5 Initialization Failed! Please open MT5 and log in to your broker.")
    exit()

# === 2. Check if Connected to a Broker ===
account_info = mt5.account_info()
if account_info is None:
    print("❌ Not connected to a broker. Please check your MT5 connection.")
    mt5.shutdown()
    exit()
else:
    print(f"✅ Connected to MT5 with Account: {account_info.login}")

# === 3. Check Available Symbols ===
symbols = [s.name for s in mt5.symbols_get()]

symbol = "EURUSDm"  # Adjust symbol based on your broker
if symbol not in symbols:
    print(f"❌ {symbol} not found in available symbols. Check your broker's symbol format.")
    print("Here are some available symbols:", symbols[:10])  # Show first 10 symbols for debugging
    mt5.shutdown()
    exit()

# === 4. Define Date Range for Data Fetching ===
end_date = datetime.now()  # Today
start_date = end_date - timedelta(days=180)  # 6 Months Back

# === 5. Fetch 1-Minute Candlestick Data ===
timeframe = mt5.TIMEFRAME_M1  # 1-minute candles
rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)

# === 6. Check if Data is Fetched ===
if rates is None or len(rates) == 0:
    print(f"❌ No data fetched for {symbol}. Try adjusting the date range or checking broker permissions.")
    mt5.shutdown()
    exit()
else:
    print(f"✅ Successfully fetched {len(rates)} candlesticks for {symbol}")

# === 7. Convert Data to DataFrame ===
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')  # Convert timestamp to readable format

# === 8. Save to CSV for Machine Learning ===
csv_filename = f"{symbol}_historical_data.csv"
df.to_csv(csv_filename, index=False)
print(f"✅ Data saved to {csv_filename}")

# === 9. Print Summary ===
print(df.head())

# === 10. Close MT5 Connection ===
mt5.shutdown()
