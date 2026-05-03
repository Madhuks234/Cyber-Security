# 🏧 ATM Fraud Detection System
### Cybersecurity Project — Python ML Pipeline

---

## 📋 Project Overview

A complete **machine-learning-based ATM fraud detection system** built with Python.
It detects fraudulent ATM transactions using behavioral analytics, velocity checks,
and ensemble models — and fires real-time colour-coded risk alerts.

---

## 🗂️ Project Structure

```
atm_fraud_detection/
│
├── main.py              ← Entry point — run this
├── data_generator.py    ← Synthetic ATM transaction generator
├── preprocessing.py     ← Feature engineering & scaling
├── models.py            ← ML model training & evaluation
├── visualizations.py    ← Charts & dashboards
├── alert_engine.py      ← Real-time fraud alert engine
├── requirements.txt     ← Python dependencies
└── README.md            ← This file
```

---

## ⚙️ Installation

```bash
# 1. Clone / download the project
cd atm_fraud_detection

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Project

```bash
# Full pipeline  (recommended for first run)
python main.py

# Quick demo  (~1 000 transactions, fast)
python main.py --quick

# Skip chart generation  (faster, text output only)
python main.py --no-plots

# Live alert feed only  (requires prior run)
python main.py --live-feed

# Custom dataset size
python main.py --n-legit 5000 --n-fraud 300
```

---

## 🤖 Machine Learning Models

| Model                | Why Used                                              |
|----------------------|-------------------------------------------------------|
| Random Forest        | Handles imbalanced data; robust to outliers           |
| Gradient Boosting    | High accuracy; captures complex fraud patterns        |
| Logistic Regression  | Fast, interpretable baseline                          |
| SVM (RBF kernel)     | Effective in high-dimensional space                   |
| K-Nearest Neighbors  | Instance-based; catches local anomalies               |

Best model is selected automatically by **F1 Score** (fraud detection priority).

---

## 🔍 Fraud Detection Features

| Feature                  | Description                              |
|--------------------------|------------------------------------------|
| `amount`                 | Withdrawal amount                        |
| `hour`                   | Hour of transaction (night = high risk)  |
| `pin_failures`           | Number of failed PIN attempts            |
| `velocity_1h`            | Transactions in the past 1 hour          |
| `velocity_24h`           | Transactions in the past 24 hours        |
| `distance_from_home_km`  | Geographic displacement                  |
| `is_night`               | Flag: transaction between 10 PM – 5 AM  |
| `is_high_amount`         | Flag: withdrawal > $1000                 |
| `is_far_from_home`       | Flag: distance > 100 km                  |
| `is_high_velocity`       | Flag: > 5 transactions in 1 hour         |
| `amount_balance_ratio`   | Withdrawal as fraction of balance        |

---

## 📊 Output Files

| File                   | Description                        |
|------------------------|------------------------------------|
| `atm_transactions.csv` | Generated transaction dataset      |
| `fraud_model.pkl`      | Saved trained model                |
| `alert_log.json`       | Real-time alert history            |
| `charts/`              | 12 visualisation PNGs              |

---

## 🚨 Alert Risk Levels

| Level  | Threshold | Action                                         |
|--------|-----------|------------------------------------------------|
| 🔴 HIGH   | ≥ 75%  | Block transaction — Notify cardholder          |
| 🟡 MEDIUM | ≥ 50%  | Hold — Send OTP to registered mobile           |
| 🔵 LOW    | ≥ 30%  | Flag for review — Log and monitor              |
| 🟢 OK     | < 30%  | Approve — No suspicious activity               |

---

## 📈 Expected Performance (10 000 transactions, 5% fraud)

| Metric        | Typical Value |
|---------------|---------------|
| F1 Score      | 0.85 – 0.95   |
| ROC AUC       | 0.92 – 0.98   |
| Recall        | 0.80 – 0.95   |
| Precision     | 0.80 – 0.95   |

---

## 📚 Cybersecurity Concepts Covered

- **Behavioural Analytics** — detecting anomalies in transaction patterns
- **Velocity Checks** — flagging rapid successive withdrawals
- **Geographic Risk** — distance-from-home displacement
- **Time-based Risk** — night-time transaction scoring
- **Ensemble Learning** — combining multiple models for robustness
- **Real-time Alerting** — immediate response to suspicious activity
- **Model Explainability** — feature importance for audit trails

---

## 👨‍💻 Author

Built as a Cybersecurity final project demonstrating practical application of
machine learning for financial fraud prevention.
