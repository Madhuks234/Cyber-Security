"""
ATM Fraud Detection - Machine Learning Models
=============================================
Trains, evaluates and compares multiple fraud-detection classifiers.
"""

import numpy as np
import pandas as pd
import pickle, os, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection   import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.svm               import SVC
from sklearn.neighbors         import KNeighborsClassifier
from sklearn.metrics           import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve,
)


# ─── Model Registry ───────────────────────────────────────────────────────────
def _build_models() -> dict:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=5,
            learning_rate=0.1, random_state=42,
        ),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42,
        ),
        "SVM": SVC(
            kernel="rbf", class_weight="balanced",
            probability=True, random_state=42,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5, n_jobs=-1,
        ),
    }


# ─── Core Class ───────────────────────────────────────────────────────────────
class FraudModelTrainer:
    """Trains multiple classifiers and picks the best one."""

    def __init__(self):
        self.models          = _build_models()
        self.trained_models  = {}
        self.results         = {}
        self.best_model_name = None
        self.best_model      = None
        self.feature_names   = []

    # ── Training ──────────────────────────────────────────────────────────────

    def train_all(self, X: pd.DataFrame, y: pd.Series,
                  test_size: float = 0.2) -> dict:
        """
        Split data, train every model, evaluate, and return a
        dictionary of metric tables.
        """
        self.feature_names = list(X.columns)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        print("\n" + "="*60)
        print("  MODEL TRAINING & EVALUATION")
        print("="*60)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        best_f1 = -1

        for name, model in self.models.items():
            print(f"\n[Training] {name} …", end=" ", flush=True)
            model.fit(X_train, y_train)
            self.trained_models[name] = model

            # Predictions
            y_pred      = model.predict(X_test)
            y_prob      = model.predict_proba(X_test)[:, 1]

            # Core metrics
            prec   = precision_score(y_test, y_pred, zero_division=0)
            rec    = recall_score(y_test, y_pred, zero_division=0)
            f1     = f1_score(y_test, y_pred, zero_division=0)
            auc    = roc_auc_score(y_test, y_prob)
            cv_f1  = cross_val_score(model, X_train, y_train,
                                     cv=cv, scoring="f1").mean()
            cm     = confusion_matrix(y_test, y_pred)

            self.results[name] = {
                "precision":   round(prec, 4),
                "recall":      round(rec,  4),
                "f1_score":    round(f1,   4),
                "roc_auc":     round(auc,  4),
                "cv_f1":       round(cv_f1,4),
                "y_test":      y_test,
                "y_pred":      y_pred,
                "y_prob":      y_prob,
                "confusion_matrix": cm,
                "X_test":      X_test,
            }

            print(f"F1={f1:.4f}  AUC={auc:.4f}  CV-F1={cv_f1:.4f}")

            if f1 > best_f1:
                best_f1              = f1
                self.best_model_name = name
                self.best_model      = model

        print(f"\n✅ Best Model: {self.best_model_name} "
              f"(F1={self.results[self.best_model_name]['f1_score']:.4f})")
        return self.results

    # ── Evaluation Report ─────────────────────────────────────────────────────

    def print_full_report(self):
        """Print classification report and confusion matrix for best model."""
        res = self.results[self.best_model_name]
        print("\n" + "="*60)
        print(f"  FULL REPORT — {self.best_model_name}")
        print("="*60)
        print(classification_report(res["y_test"], res["y_pred"],
                                    target_names=["Legitimate", "Fraud"]))
        print("Confusion Matrix:")
        cm = res["confusion_matrix"]
        print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
        print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    def get_summary_table(self) -> pd.DataFrame:
        """Return a tidy DataFrame comparing all models."""
        rows = []
        for name, r in self.results.items():
            rows.append({
                "Model":      name,
                "Precision":  r["precision"],
                "Recall":     r["recall"],
                "F1 Score":   r["f1_score"],
                "ROC AUC":    r["roc_auc"],
                "CV F1":      r["cv_f1"],
            })
        return pd.DataFrame(rows).sort_values("F1 Score", ascending=False)

    # ── Feature Importance ────────────────────────────────────────────────────

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importances for tree-based models."""
        model = self.best_model
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_[0])
        else:
            return pd.DataFrame()

        return (
            pd.DataFrame({"Feature": self.feature_names, "Importance": imp})
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> dict:
        """Run best model on a feature matrix."""
        y_pred = self.best_model.predict(X)
        y_prob = self.best_model.predict_proba(X)[:, 1]
        return {"prediction": y_pred, "fraud_probability": y_prob}

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = "model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[Model] Saved → {path}")

    @staticmethod
    def load(path: str = "model.pkl") -> "FraudModelTrainer":
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    from data_generator  import generate_dataset
    from preprocessing   import ATMPreprocessor

    df   = generate_dataset(5000, 250)
    prep = ATMPreprocessor()
    X, y = prep.fit_transform(df)

    trainer = FraudModelTrainer()
    trainer.train_all(X, y)
    trainer.print_full_report()

    print("\nModel Comparison:")
    print(trainer.get_summary_table().to_string(index=False))

    print("\nTop-10 Features:")
    print(trainer.get_feature_importance().head(10).to_string(index=False))
