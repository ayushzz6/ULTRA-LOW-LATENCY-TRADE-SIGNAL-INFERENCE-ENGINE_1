It is a high-performance inference engine designed for real-time financial signal prediction at ultra-low latency. It integrates live WebSocket data ingestion from Binance, high-frequency feature engineering, ONNX-optimized model inference, and latency profiling tools  all optimized to run under 10-15 milliseconds end-to-end on commodity hardware.

An ultra-low-latency trade signal inference engine for real-time crypto trading
Achieves sub-15ms end-to-end prediction latency using ONNX, WebSocket streaming, and optimized feature pipelines.

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


Key Features---
Live streaming from Binance WebSocket (per-tick updates)
Optimized feature engineering for volatility, momentum, and flow
ONNX Runtime inference low latency 
Latency profiling at each processing stage (feature, scale, predict)
Modular design (supports XGBoost, LightGBM, CatBoost)
End-to-end extensibility for deployment to trading systems or agents


Model Details

Model: XGBoost (multi:softprob)
Classes: Buy(1) / Sell(2) / Hold(0) (3 classes)
Loss Function: Multi-class Log Loss (mlogloss)

Features Used:

price_now
return_5
rolling_mean_5
rolling_std_5
vwap_diff
volume_sum_5
rsi_5
tick_inter_arrival_time
Real-time feature engineering is performed with a window size of 5 trades.

Designed for---
High-frequency trading (HFT)
Latency-sensitive signal generation
Tick-level backtesting or reinforcement learning
Streaming inference for LOB/market-making bots

Stage              ---   Average Latency 

Feature Extraction   ---   3-7 ms        
Scaling              ---   1–4 ms          
ONNX Inference       ---   0.3–1.0 ms   
Total End-to-End     ---   5-12 ms 


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

 Future Work
 Integrate with FIX/REST trading gateways
 GPU-based inference for batching (if needed)
 Real PnL backtests using tick-by-tick replay
 Docker + FastAPI wrapper for deployable microservice
 Optional agent-based decision layer with RL triggers