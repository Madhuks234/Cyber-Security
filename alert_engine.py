"""
ATM Fraud Detection - Real-Time Alert System
=============================================
Simulates a live ATM feed and generates risk alerts.
"""

import time, random, json
from datetime import datetime


# ─── Risk Thresholds ──────────────────────────────────────────────────────────
THRESHOLDS = {
    "HIGH":   0.75,   # Immediate block recommended
    "MEDIUM": 0.50,   # Additional verification required
    "LOW":    0.30,   # Monitor and log
}

ALERT_COLORS = {
    "HIGH":   "\033[91m",   # Red
    "MEDIUM": "\033[93m",   # Yellow
    "LOW":    "\033[96m",   # Cyan
    "OK":     "\033[92m",   # Green
    "RESET":  "\033[0m",
}


class AlertEngine:
    """
    Wraps a trained model and fires risk alerts for every incoming transaction.
    """

    def __init__(self, trainer, preprocessor, threshold: float = 0.5):
        self.trainer       = trainer
        self.preprocessor  = preprocessor
        self.threshold     = threshold
        self.alert_log     = []

    # ── Core ──────────────────────────────────────────────────────────────────

    def evaluate(self, transaction: dict) -> dict:
        """
        Evaluate a single transaction dict.

        Returns an alert record with:
        - fraud_probability
        - risk_level  (HIGH / MEDIUM / LOW / OK)
        - recommended_action
        - timestamp
        """
        X = self.preprocessor.transform_single(transaction)
        result = self.trainer.predict(X)

        prob       = float(result["fraud_probability"][0])
        risk_level = self._classify_risk(prob)
        action     = self._recommend_action(risk_level)

        alert = {
            "timestamp":         datetime.now().isoformat(timespec="seconds"),
            "transaction_id":    transaction.get("transaction_id", "N/A"),
            "amount":            transaction.get("amount", 0),
            "atm_location":      transaction.get("atm_location", "Unknown"),
            "fraud_probability": round(prob, 4),
            "risk_level":        risk_level,
            "recommended_action": action,
        }
        self.alert_log.append(alert)
        return alert

    def _classify_risk(self, prob: float) -> str:
        if prob >= THRESHOLDS["HIGH"]:   return "HIGH"
        if prob >= THRESHOLDS["MEDIUM"]: return "MEDIUM"
        if prob >= THRESHOLDS["LOW"]:    return "LOW"
        return "OK"

    def _recommend_action(self, risk: str) -> str:
        return {
            "HIGH":   "🚫 BLOCK TRANSACTION — Notify cardholder immediately",
            "MEDIUM": "⚠️  HOLD FOR VERIFICATION — Send OTP to registered mobile",
            "LOW":    "👀 FLAG FOR REVIEW — Log and continue monitoring",
            "OK":     "✅ APPROVE — No suspicious activity detected",
        }[risk]

    # ── Simulation ────────────────────────────────────────────────────────────

    def simulate_live_feed(self, n: int = 20, delay: float = 0.3):
        """
        Simulate `n` incoming ATM transactions and print colour-coded alerts.
        """
        from data_generator import (
            generate_transaction, ATM_LOCATIONS, CARD_TYPES
        )

        print("\n" + "="*65)
        print("  🏧  LIVE ATM FRAUD DETECTION FEED  (press Ctrl+C to stop)")
        print("="*65)

        flagged = 0
        for i in range(1, n + 1):
            # Inject a fraud transaction every ~5 transactions for demo realism
            is_fraud_sim = (random.random() < 0.25)
            txn = generate_transaction(i, is_fraud=is_fraud_sim)
            alert = self.evaluate(txn)

            color  = ALERT_COLORS.get(alert["risk_level"], "")
            reset  = ALERT_COLORS["RESET"]
            risk   = alert["risk_level"]
            prob   = alert["fraud_probability"]

            print(
                f"{color}[{alert['timestamp']}]  "
                f"TXN#{alert['transaction_id']:>5}  "
                f"${alert['amount']:>8.2f}  "
                f"{alert['atm_location']:<20}  "
                f"Risk: {risk:<6}  P(fraud)={prob:.3f}{reset}"
            )
            if risk in ("HIGH", "MEDIUM"):
                print(f"  ↳  {alert['recommended_action']}")
                flagged += 1

            time.sleep(delay)

        print("\n" + "-"*65)
        print(f"  Feed complete.  {n} transactions processed.  "
              f"{flagged} flagged for review.")
        print("-"*65)

    def export_log(self, path: str = "alert_log.json"):
        """Dump the alert log to JSON."""
        with open(path, "w") as f:
            json.dump(self.alert_log, f, indent=2)
        print(f"[Alert] Log saved → {path}")

    def get_summary(self) -> dict:
        """Return a count breakdown of risk levels."""
        from collections import Counter
        counts = Counter(a["risk_level"] for a in self.alert_log)
        return dict(counts)


if __name__ == "__main__":
    print("AlertEngine loaded.  Run main.py to see the live demo.")
