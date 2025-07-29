import asyncio
import threading
import json
import websockets
import time
from datetime import datetime
from feature_engineering import FeatureEngineer
import pandas as pd
import joblib
import onnxruntime as ort
import numpy as np
import os

MODEL_PATH = "model.onnx"
SCALER_PATH = "preprocessor.pkl"

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
scaler = joblib.load(SCALER_PATH)
fe = FeatureEngineer(window=5)

latest_signal = {"signal": None, "features": {}, "timestamp": None}

def predict(features_df: pd.DataFrame) -> int:
    features = features_df[[
        "price_now", "return_5", "rolling_mean_5", "rolling_std_5",
        "vwap_diff", "volume_sum_5", "rsi_5", "tick_inter_arrival_time"
    ]]
    scaled_features = scaler.transform(features.astype(np.float32))
    prediction = session.run(None, {input_name: scaled_features})[0]
    return int(prediction[0])

async def stream_binance(symbol="btcusdt"):
    url = f"wss://stream.binance.com:9443/ws/{symbol}@trade"
    async with websockets.connect(url) as ws:
        async for message in ws:
            data = json.loads(message)
            price = float(data['p'])
            volume = float(data['q'])
            timestamp = datetime.fromtimestamp(data['T'] / 1000).strftime("%Y-%m-%d %H:%M:%S")
            fe.update(price, volume, timestamp)
            features_df = fe.compute_features()
            if features_df is not None:
                signal = predict(features_df)
                latest_signal["signal"] = signal
                latest_signal["features"] = features_df.to_dict(orient="records")[0]
                latest_signal["timestamp"] = timestamp

def run_asyncio_loop():
    asyncio.run(stream_binance())

def start_predictor():
    t = threading.Thread(target=run_asyncio_loop)
    t.daemon = True
    t.start()
