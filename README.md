# FraudShield AI: End-to-End Detection Pipeline 🚨

FraudShield AI is a real-time fraud detection system that moves beyond static model training. It features a high-performance FastAPI backend, a live Data Streamer, and an interactive Streamlit Dashboard for financial monitoring. Raw data source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Key Features

    Cost-Sensitive Learning: Utilizes XGBoost with SMOTE+Tomek Links to handle extreme class imbalance (0.17% fraud).

    Sliding Threshold Optimization: Implements a dynamic risk engine that lowers detection thresholds for high-value transactions to maximize "Money Saved."

    Production-Ready API: A FastAPI interface with input validation and error handling.

    Live Monitoring: A real-time dashboard tracking fraud probability trends and system latency.

<p align="center">
  <img src="https://github.com/frankraDIUM/FraudShield-AI/blob/main/live_dashboard.png" />
</p>

 ## System Architecture

The project is split into three decoupled services:

    The Brain (FastAPI): Hosts the XGBoost model and provides a /predict endpoint.

    The Producer (Python/Requests): Simulates a live bank feed by streaming transactions from the test set.

    The Monitor (Streamlit): Visualizes the live stream, probability curves, and total financial savings.

## Model Performance & Insights

Based on the Time Analysis, the model identified a significant correlation between late-night transactions (Hours 2-4 AM) and fraudulent activity.
Metric	Result
- Fraud Recall - 88%
- Precision- 67%
- Avg. Latency - ~3.5 ms
- Net Model Value	Positive (Saved > Admin Costs)

<p align="center">
  <img src="https://github.com/frankraDIUM/FraudShield-AI/blob/main/time_analysis.png" />
</p>
--- 

## Technical Challenges & Solutions

1. Addressing Extreme Class Imbalance (0.17% Fraud)

    The Problem: With only 492 fraud cases out of 284,807 transactions, a "dumb" model could achieve 99.83% accuracy just by guessing "Normal" every time, while catching zero fraud.

    The Solution: I implemented SMOTE + Tomek Links. SMOTE creates synthetic fraud examples to balance the classes, while Tomek Links removes "noisy" data points where Normal and Fraud overlap, creating a cleaner decision boundary for the XGBoost model.

2. Financial-First Optimization (Sliding Thresholds)

    The Problem: A standard 0.5 probability threshold treats a $1.00 transaction and a $10,000.00 transaction with the same level of risk.

    The Solution: I developed a Sliding Threshold Engine.

        High-Value Transactions (>$5.00 scaled): Threshold drops to 0.25 (High Sensitivity). We’d rather have a false alarm than lose a large sum.

        Low-Value Transactions: Threshold stays at 0.55 (Low Sensitivity) to reduce "customer friction" and administrative costs for small amounts.

3. Real-Time Architecture & Latency

    The Problem: Fraud detection is useless if it takes 10 seconds to respond at a checkout counter.

    The Solution: By using FastAPI and an optimized XGBoost model, the system achieves an average inference latency of ~3.5ms, making it suitable for high-frequency financial environments.

## How to Run
1. Prerequisites

Ensure you have the required libraries:
Bash

pip install fastapi uvicorn streamlit xgboost scikit-learn pandas plotly requests

2. Launch the Brain (API)
Bash

uvicorn app:app --reload

3. Launch the Monitor (Dashboard)
Bash

streamlit run dashboard.py

4. Start the Feed (Streamer)
Bash

python streamer.py

🛠️ Tech Stack

    Language: Python 3.13

    ML Frameworks: XGBoost, Scikit-Learn, Imbalanced-Learn

    API: FastAPI, Pydantic, Uvicorn

    Dashboard: Streamlit, Plotly

    Data Handling: Pandas, NumPy, Joblib

## Future Works
- Integrating a database (PostgreSQL) to store transaction history.

- Adding Docker containers for one-click deployment.

- Implementing an automated re-training trigger when model drift is detected.
