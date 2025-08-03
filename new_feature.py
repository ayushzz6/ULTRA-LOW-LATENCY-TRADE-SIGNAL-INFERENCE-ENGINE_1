import pandas as pd
from tqdm import tqdm
from feature_engineering import FeatureEngineer  


columns = [
    "trade_id",     # Unique trade ID
    "price",        # Trade price
    "qty",          # Quantity traded
    "quote_qty",    # Quote quantity
    "timestamp",    # Timestamp (in milliseconds)
    "is_buyer_maker",  # Whether buyer is the market maker
    "is_best_match"    # Whether it's the best match
]

df = pd.read_csv(r"C:\downloads\ULTRA_LOW_LATENCY_TRADE_SIGNAL_ENGINE\BTCUSDT-trades-2025-08-02.csv", header=None, names=columns)

# save the DataFrame to a csv file
df.to_csv(r"C:\downloads\ULTRA_LOW_LATENCY_TRADE_SIGNAL_ENGINE\BTCUSDT-trades-2025-08-02.csv", index=False)

df = pd.read_csv(r"C:\downloads\ULTRA_LOW_LATENCY_TRADE_SIGNAL_ENGINE\BTCUSDT-trades-2025-08-02.csv")

# Convert microsecond timestamp to human-readable string
df['timestamp_str'] = pd.to_datetime(df['timestamp'], unit='us').dt.strftime('%Y-%m-%d %H:%M:%S.%f')

# Initialize feature engineer with window=5
fe = FeatureEngineer(window=5)

# Store feature DataFrames
features_list = []

# Loop through each row simulating real-time processing
for _, row in tqdm(df.iterrows(), total=len(df)):
    price = float(row['price'])
    volume = float(row['qty'])
    timestamp_str = row['timestamp_str']

    fe.update(price=price, volume=volume, timestamp_str=timestamp_str)
    features = fe.compute_features()

    if features is not None:
        features_list.append(features)

# Combine all computed features
final_features_df = pd.concat(features_list, ignore_index=True)

# Save or inspect results
final_features_df.to_csv("computed_features_from_trades.csv", index=False)

