"""
ATM Fraud Detection - Visualizations
=====================================
Generates all charts: EDA, model metrics, feature importance, and alerts.
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix, precision_recall_curve

# ─── Global Style ─────────────────────────────────────────────────────────────
PALETTE   = {
    "primary":   "#1A73E8",
    "danger":    "#EA4335",
    "success":   "#34A853",
    "warning":   "#FBBC05",
    "dark":      "#202124",
    "light":     "#F8F9FA",
    "accent":    "#AB47BC",
}
plt.rcParams.update({
    "figure.facecolor":  PALETTE["light"],
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#DADCE0",
    "axes.grid":         True,
    "grid.color":        "#EEEEEE",
    "grid.linestyle":    "--",
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
})

OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save(fig, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Chart] Saved → {path}")
    return path


# ─── 1. EDA ───────────────────────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Class Distribution – ATM Transactions", fontsize=15, fontweight="bold")

    counts = df["is_fraud"].value_counts()
    labels = ["Legitimate", "Fraud"]
    colors = [PALETTE["success"], PALETTE["danger"]]

    # Bar chart
    axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.5)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 30, f"{v:,}", ha="center", fontsize=12, fontweight="bold")
    axes[0].set_title("Transaction Count")
    axes[0].set_ylabel("Count")

    # Pie chart
    axes[1].pie(counts.values, labels=labels, colors=colors,
                autopct="%1.1f%%", startangle=140,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Fraud Rate")

    return _save(fig, "01_class_distribution.png")


def plot_amount_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Withdrawal Amount Analysis", fontsize=15, fontweight="bold")

    legit = df[df["is_fraud"] == 0]["amount"]
    fraud = df[df["is_fraud"] == 1]["amount"]

    # Overlapping histograms
    axes[0].hist(legit, bins=50, alpha=0.6, color=PALETTE["success"], label="Legitimate")
    axes[0].hist(fraud, bins=50, alpha=0.6, color=PALETTE["danger"],  label="Fraud")
    axes[0].set_title("Amount Distribution (Histogram)")
    axes[0].set_xlabel("Amount ($)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # Box plot
    bp = axes[1].boxplot([legit, fraud], labels=["Legitimate", "Fraud"],
                         patch_artist=True, notch=True)
    for patch, color in zip(bp["boxes"], [PALETTE["success"], PALETTE["danger"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_title("Amount Distribution (Box Plot)")
    axes[1].set_ylabel("Amount ($)")

    return _save(fig, "02_amount_distribution.png")


def plot_hourly_pattern(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("Hourly Transaction Pattern", fontsize=15, fontweight="bold")

    hourly = df.groupby(["hour", "is_fraud"]).size().unstack(fill_value=0)
    x = np.arange(24)
    width = 0.4

    ax.bar(x - width/2, hourly.get(0, 0), width, label="Legitimate",
           color=PALETTE["success"], alpha=0.8)
    ax.bar(x + width/2, hourly.get(1, 0), width, label="Fraud",
           color=PALETTE["danger"], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:02d}:00" for h in x], rotation=45, ha="right")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Transaction Count")
    ax.legend()

    # Shade night hours
    ax.axvspan(-0.5, 5.5, alpha=0.05, color="navy", label="Night (0–5)")
    ax.axvspan(21.5, 23.5, alpha=0.05, color="navy")

    return _save(fig, "03_hourly_pattern.png")


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5,
                ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=15, fontweight="bold", pad=15)

    return _save(fig, "04_correlation_heatmap.png")


def plot_velocity_analysis(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Transaction Velocity Analysis", fontsize=15, fontweight="bold")

    for ax, col, label in zip(
        axes,
        ["velocity_1h", "velocity_24h"],
        ["Transactions in Last 1 Hour", "Transactions in Last 24 Hours"],
    ):
        legit = df[df["is_fraud"] == 0][col]
        fraud = df[df["is_fraud"] == 1][col]
        ax.hist(legit, bins=20, alpha=0.6, color=PALETTE["success"], label="Legitimate", density=True)
        ax.hist(fraud, bins=20, alpha=0.6, color=PALETTE["danger"],  label="Fraud",      density=True)
        ax.set_title(label)
        ax.set_xlabel("Count")
        ax.set_ylabel("Density")
        ax.legend()

    return _save(fig, "05_velocity_analysis.png")


# ─── 2. Model Evaluation ──────────────────────────────────────────────────────

def plot_model_comparison(summary_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold")

    metrics = ["F1 Score", "ROC AUC", "Precision", "Recall"]
    x = np.arange(len(summary_df))
    colors = [PALETTE["primary"], PALETTE["accent"],
              PALETTE["success"], PALETTE["warning"]]

    width = 0.2
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        axes[0].bar(x + i * width, summary_df[metric], width,
                    label=metric, color=color, alpha=0.85)

    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(summary_df["Model"], rotation=20, ha="right")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Metrics by Model")
    axes[0].legend(loc="lower right", fontsize=9)

    # CV F1 bar
    bars = axes[1].barh(summary_df["Model"], summary_df["CV F1"],
                        color=PALETTE["primary"], alpha=0.8)
    for bar, val in zip(bars, summary_df["CV F1"]):
        axes[1].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                     f"{val:.4f}", va="center", fontsize=10)
    axes[1].set_xlim(0, 1.1)
    axes[1].set_xlabel("Cross-Validated F1 Score")
    axes[1].set_title("5-Fold CV F1 Score")
    axes[1].invert_yaxis()

    return _save(fig, "06_model_comparison.png")


def plot_confusion_matrix(cm: np.ndarray, model_name: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted Legit", "Predicted Fraud"],
                yticklabels=["Actual Legit", "Actual Fraud"],
                linewidths=1, linecolor="white", ax=ax,
                annot_kws={"size": 16, "weight": "bold"})
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")

    # Annotate quadrants
    tn, fp, fn, tp = cm.ravel()
    ax.text(0.5, -0.08, f"TN={tn}  FP={fp}  FN={fn}  TP={tp}",
            ha="center", va="top", transform=ax.transAxes, fontsize=10, color="#555")

    return _save(fig, "07_confusion_matrix.png")


def plot_roc_curves(results: dict):
    fig, ax = plt.subplots(figsize=(9, 7))
    colors  = [PALETTE["primary"], PALETTE["danger"], PALETTE["success"],
                PALETTE["warning"], PALETTE["accent"]]

    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(res["y_test"], res["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})",
                color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    return _save(fig, "08_roc_curves.png")


def plot_precision_recall(results: dict):
    fig, ax = plt.subplots(figsize=(9, 7))
    colors  = [PALETTE["primary"], PALETTE["danger"], PALETTE["success"],
                PALETTE["warning"], PALETTE["accent"]]

    for (name, res), color in zip(results.items(), colors):
        prec, rec, _ = precision_recall_curve(res["y_test"], res["y_prob"])
        ax.plot(rec, prec, label=f"{name} (F1={res['f1_score']:.3f})",
                color=color, linewidth=2)

    ax.set_title("Precision–Recall Curves", fontsize=14, fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right", fontsize=10)

    return _save(fig, "09_precision_recall.png")


def plot_feature_importance(importance_df: pd.DataFrame, model_name: str):
    top_n = importance_df.head(15)
    fig, ax = plt.subplots(figsize=(10, 7))

    bars = ax.barh(top_n["Feature"], top_n["Importance"],
                   color=PALETTE["primary"], alpha=0.85)
    for bar, val in zip(bars, top_n["Importance"]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)

    ax.set_title(f"Top-15 Feature Importances — {model_name}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.invert_yaxis()

    return _save(fig, "10_feature_importance.png")


def plot_fraud_risk_score_distribution(y_true, y_prob):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist([y_prob[y_true == 0], y_prob[y_true == 1]],
            bins=40, alpha=0.7,
            color=[PALETTE["success"], PALETTE["danger"]],
            label=["Legitimate", "Fraud"])
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5,
               label="Default Threshold (0.5)")
    ax.set_title("Fraud Risk Score Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Fraud Probability")
    ax.set_ylabel("Count")
    ax.legend()

    return _save(fig, "11_risk_score_distribution.png")


# ─── 3. Dashboard Summary ─────────────────────────────────────────────────────

def plot_dashboard(df: pd.DataFrame, summary_df: pd.DataFrame,
                   best_model_name: str, best_res: dict):
    """4-panel executive dashboard."""
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(PALETTE["dark"])
    fig.suptitle("ATM Fraud Detection — Executive Dashboard",
                 fontsize=20, fontweight="bold", color="white", y=0.97)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Panel 1 – Fraud rate gauge (pie)
    ax1 = fig.add_subplot(gs[0, 0])
    counts = df["is_fraud"].value_counts()
    ax1.pie([counts.get(0, 0), counts.get(1, 0)],
            labels=["Legitimate", "Fraud"],
            colors=[PALETTE["success"], PALETTE["danger"]],
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
            textprops={"color": "white"})
    ax1.set_facecolor(PALETTE["dark"])
    ax1.set_title("Fraud Rate", color="white")

    # Panel 2 – Model F1 scores
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#2D2D2D")
    bar_colors = [PALETTE["danger"] if n == best_model_name else PALETTE["primary"]
                  for n in summary_df["Model"]]
    ax2.barh(summary_df["Model"], summary_df["F1 Score"],
             color=bar_colors, alpha=0.9)
    ax2.set_xlim(0, 1.1)
    ax2.set_title("Model F1 Scores", color="white")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#555")

    # Panel 3 – Confusion matrix
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#2D2D2D")
    cm = best_res["confusion_matrix"]
    im = ax3.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     fontsize=18, fontweight="bold", color="white")
    ax3.set_xticks([0, 1]); ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["Pred: Legit", "Pred: Fraud"], color="white")
    ax3.set_yticklabels(["Act: Legit", "Act: Fraud"], color="white", rotation=90, va="center")
    ax3.set_title(f"Confusion Matrix\n({best_model_name})", color="white")

    # Panel 4 – Hourly fraud count
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.set_facecolor("#2D2D2D")
    hourly_fraud = df[df["is_fraud"] == 1].groupby("hour").size()
    ax4.bar(hourly_fraud.index, hourly_fraud.values,
            color=PALETTE["danger"], alpha=0.8)
    ax4.set_title("Fraud by Hour of Day", color="white")
    ax4.set_xlabel("Hour", color="white")
    ax4.set_ylabel("Fraud Count", color="white")
    ax4.tick_params(colors="white")
    ax4.axvspan(-0.5, 5.5, alpha=0.15, color="yellow")
    ax4.axvspan(21.5, 23.5, alpha=0.15, color="yellow")
    for spine in ax4.spines.values():
        spine.set_edgecolor("#555")

    # Panel 5 – Key KPIs
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#2D2D2D")
    ax5.axis("off")
    kpis = [
        ("Total Transactions",  f"{len(df):,}"),
        ("Fraud Cases",         f"{df['is_fraud'].sum():,}"),
        ("Fraud Rate",          f"{df['is_fraud'].mean():.2%}"),
        ("Best Model",          best_model_name),
        ("F1 Score",            f"{best_res['f1_score']:.4f}"),
        ("ROC AUC",             f"{best_res['roc_auc']:.4f}"),
        ("Recall (Fraud)",      f"{best_res['recall']:.4f}"),
        ("Precision (Fraud)",   f"{best_res['precision']:.4f}"),
    ]
    for i, (label, value) in enumerate(kpis):
        ax5.text(0.02, 0.93 - i * 0.12, label + ":",
                 color="#AAAAAA", fontsize=10, transform=ax5.transAxes)
        ax5.text(0.98, 0.93 - i * 0.12, value,
                 color="white", fontsize=10, fontweight="bold",
                 ha="right", transform=ax5.transAxes)
    ax5.set_title("Key Performance Indicators", color="white")

    return _save(fig, "00_dashboard.png")


if __name__ == "__main__":
    from data_generator import generate_dataset
    df = generate_dataset(1000, 50)
    plot_class_distribution(df)
    plot_amount_distribution(df)
    plot_hourly_pattern(df)
    print("Charts generated in ./charts/")
