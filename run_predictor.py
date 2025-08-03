import sys
import os
import asyncio
import json
import websockets
import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime
import joblib
from feature_engineering import FeatureEngineer
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = r"C:\downloads\ULTRA_LOW_LATENCY_TRADE_SIGNAL_ENGINE\model.onnx"
SCALER_PATH = r"C:\downloads\ULTRA_LOW_LATENCY_TRADE_SIGNAL_ENGINE\preprocessor.joblib"

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "predictor.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

try:
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider" ])
    input_name = session.get_inputs()[0].name
    scaler = joblib.load(SCALER_PATH)
    logging.info(" Model and scaler loaded successfully.")
except Exception as e:
    logging.exception(" Failed to load model or scaler.")
    raise e

fe = FeatureEngineer(window=5)

def measure_latency(start, label=""):
    elapsed_ms = (time.perf_counter() - start) * 1000
    logging.info(f" {label} latency: {elapsed_ms:.2f} ms")
    print(f"{label} latency: {elapsed_ms:.2f} ms")

def predict(features_df: pd.DataFrame) -> int:
    try:
        # Extract relevant columns
        features = features_df[[
            "price_now", "return_5", "rolling_mean_5", "rolling_std_5",
            "vwap_diff", "volume_sum_5", "rsi_5", "tick_inter_arrival_time"
        ]]

        # 
        scale_start = time.perf_counter()
        scaled_features = scaler.transform(features.astype(np.float32))
        measure_latency(scale_start, " Scaling")

        #  Inference 
        infer_start = time.perf_counter()
        prediction = session.run(None, {input_name: scaled_features})[0]
        measure_latency(infer_start, " ONNX Inference")

        predicted_class = int(prediction[0])
        logging.info(f" Prediction: {predicted_class}")
        return predicted_class

    except Exception as e:
        logging.exception(" Prediction error")
        return -1

async def stream_binance(symbol="btcusdt"):
    url = f"wss://stream.binance.com:9443/ws/{symbol}@trade"

    try:
        async with websockets.connect(url) as ws:
            logging.info(f" Connected to Binance WebSocket for {symbol}")

            async for message in ws:
                try:
                    full_start = time.perf_counter()   # calculate end-to-end latency
                    
                    data = json.loads(message)
                    price = float(data['p'])
                    volume = float(data['q'])
                    timestamp = datetime.fromtimestamp(data['T'] / 1000).strftime("%Y-%m-%d %H:%M:%S")

                    #  Feature Engineering
                    fe.update(price, volume, timestamp)   # Update features after each message and store the data internally
                    features_df = fe.compute_features()   # Compute features after each message  like rolling mean, std, RSI, etc.

                    if features_df is not None:
                        logging.info(f" Features: {features_df.to_dict(orient='records')[0]}")
                        signal = predict(features_df)
                        print(f" Trade Signal: {signal}")
                        print(features_df)
                        measure_latency(full_start, " End-to-End")

                except Exception as inner_e:
                    logging.exception(" Error processing WebSocket message")

    except Exception as conn_e:
        logging.exception("WebSocket connection error")


if __name__ == "__main__":
    try:
        asyncio.run(stream_binance())
    except KeyboardInterrupt:
        logging.info(" Interrupted by user.")
    except Exception as e:
        logging.exception(" Critical error in main loop")