#pip install websockets aiohttp python-binance pandas numpy
import asyncio
import json
import websockets
import logging
import time
from feature_engineering import FeatureEngineer
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize FeatureEngineer
fe = FeatureEngineer(window=5)

async def stream_binance(symbol="btcusdt"):
    url = f"wss://stream.binance.com:9443/ws/{symbol}@trade"
    async with websockets.connect(url) as ws:
        logging.info(f"Connected to Binance WebSocket for {symbol}")

        async for message in ws:
            try:
                data = json.loads(message)

                # Extract trade data
                price = float(data['p'])
                volume = float(data['q'])
                timestamp = datetime.fromtimestamp(data['T'] / 1000).strftime("%Y-%m-%d %H:%M:%S")

                # Update features
                fe.update(price, volume, timestamp)
                features_df = fe.compute_features()

                if features_df is not None:
                    print(" Real-time features:")
                    print(features_df)

            except Exception as e:
                logging.error(f"Error processing message: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        asyncio.run(stream_binance())
    except KeyboardInterrupt:
        logging.info("Stream stopped by user.")






