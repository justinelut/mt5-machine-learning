import MetaTrader5 as mt5

# Initialize MT5
if not mt5.initialize():
    print("❌ MT5 Initialization Failed!")
    exit()

# Get all available symbols
symbols = mt5.symbols_get()
symbol_names = [symbol.name for symbol in symbols]

# Filter symbols that contain USD, EUR, or GBP
usd_pairs = [s for s in symbol_names if "USD" in s]
eur_pairs = [s for s in symbol_names if "EUR" in s]
gbp_pairs = [s for s in symbol_names if "GBP" in s]

# Print results
print("\n✅ Available USD Pairs:", usd_pairs)
print("\n✅ Available EUR Pairs:", eur_pairs)
print("\n✅ Available GBP Pairs:", gbp_pairs)

# Shutdown MT5 connection
mt5.shutdown()
