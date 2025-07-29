import numpy as np
import pandas as pd
import time
from datetime import datetime
from collections import deque
import logging
import os

# Setup logger
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "feature_engineering.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class FeatureEngineer:
    def __init__(self, window=5):
        self.window = window
        self.prices = deque(maxlen=window)
        self.volumes = deque(maxlen=window)
        self.timestamps_str = deque(maxlen=window)   # Human-readable
        self.timestamps_float = deque(maxlen=window) # Unix timestamps
        self.vwap_numerator = deque(maxlen=window)
        self.vwap_denominator = deque(maxlen=window)
        logging.info("Initialized FeatureEngineer with window size: %d", window)

    def update(self, price: float, volume: float, timestamp_str: str):
        try:
            self.prices.append(float(price))
            self.volumes.append(float(volume))
            self.timestamps_str.append(timestamp_str)
            self.timestamps_float.append(time.time())

            self.vwap_numerator.append(price * volume)
            self.vwap_denominator.append(volume)

            logging.info(f"Updated with price: {price}, volume: {volume}, timestamp: {timestamp_str}")

        except Exception as e:
            logging.error(f"Error in update(): {e}")

    def compute_features(self):
        try:
            if len(self.prices) < self.window:
                logging.warning("Not enough data to compute features.")
                return None

            price_now = self.prices[-1]
            return_over_5 = (self.prices[-1] - self.prices[0]) / (self.prices[0] + 1e-9)
            rolling_mean = np.mean(self.prices)
            rolling_std = np.std(self.prices)

            vwap = sum(self.vwap_numerator) / (sum(self.vwap_denominator) + 1e-9)
            vwap_diff = vwap - price_now

            volume_sum = sum(self.volumes)

            # RSI calculation
            diffs = np.diff(self.prices)

            gains = [d for d in diffs if d > 0]
            losses = [-d for d in diffs if d < 0]

            avg_gain = np.mean(gains) if gains else 1e-9
            avg_loss = np.mean(losses) if losses else 1e-9

            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))


            tick_intervals = np.diff(self.timestamps_float)
            tick_inter_time = np.mean(tick_intervals) if len(tick_intervals) > 0 else 0

            features = {
                "timestamp": self.timestamps_str[-1],
                "price_now": price_now,
                "return_5": return_over_5,
                "rolling_mean_5": rolling_mean,
                "rolling_std_5": rolling_std,
                "vwap_diff": vwap_diff,
                "volume_sum_5": volume_sum,
                "rsi_5": rsi,
                "tick_inter_arrival_time": tick_inter_time
            }

            logging.info("Feature vector computed successfully.")
            return pd.DataFrame([features])

        except Exception as e:
            logging.error(f"Error in compute_features(): {e}")
            return None
