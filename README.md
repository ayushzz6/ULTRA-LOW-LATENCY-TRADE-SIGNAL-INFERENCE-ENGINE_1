ultra_latency_predictor/

run_predictor.py              # Main real-time loop
binance_stream.py             # Live WebSocket client
feature_engineering.py        # Real-time features (e.g., RSI)
model.onnx                    # Placeholder for trained ONNX model
modeltrain.ipynb  # Jupyter for training & backtesting
requirements.txt
README.md


| Feature                 | Type         |
| ----------------------- | ------------ |
| Price now               | Price-based  |
| Return over 5 ticks     | Price-based  |
| Rolling mean (5)        | Price-based  |
| Rolling std (5)         | Volatility   |
| VWAP - Price            | Volume/Flow  |
| Volume sum (5 ticks)    | Volume-based |
| RSI (5)                 | Momentum     |
| Order book imbalance    | Order book   | # we will not use it for high latency 
| Spread                  | Order book   | # we will not use it for high latency 
| Tick inter-arrival time | Time-based   |


IT is a high-performance inference engine designed for real-time financial signal prediction at ultra-low latency. It integrates live WebSocket data ingestion from Binance, high-frequency feature engineering, ONNX-optimized model inference, and latency profiling tools — all optimized to run under 20 milliseconds end-to-end on commodity hardware.

Key Features---
Live streaming from Binance WebSocket (per-tick updates)

Optimized feature engineering for volatility, momentum, and flow

ONNX Runtime inference low latency 

Latency profiling at each processing stage (feature, scale, predict)

Modular design (supports XGBoost, LightGBM, CatBoost)

End-to-end extensibility for deployment to trading systems or agents

Designed for---

High-frequency trading (HFT)

Latency-sensitive signal generation

Tick-level backtesting or reinforcement learning

Streaming inference for LOB/market-making bots

Stage              ---   Average Latency 

Feature Extraction   ---   7–10 ms        
Scaling              ---   2–4 ms          
ONNX Inference       ---   0.6–1.0 ms   
Total End-to-End     ---   15–20 ms 


Classification Report:
               precision    recall  f1-score   support

           0       0.73      0.69      0.71     84118
           1       0.77      0.81      0.79     59657
           2       0.78      0.81      0.79     60225

    accuracy                           0.76    204000
   macro avg       0.76      0.77      0.76    204000
weighted avg       0.76      0.76      0.76    204000

confusion matrix:
 [[57815 13474 12829]
 [10763 48233   661]
 [10928   683 48614]]
ROC AUC Score: 0.8994181909422815
