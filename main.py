"""
╔══════════════════════════════════════════════════════════════╗
║          ATM FRAUD DETECTION SYSTEM                          ║
║          Cybersecurity Project — Python ML Pipeline          ║
║          Author : Your Name                                  ║
║          Version: 1.0.0                                      ║
╚══════════════════════════════════════════════════════════════╝

MODULES
-------
  data_generator.py   — Synthetic ATM transaction data
  preprocessing.py    — Feature engineering & scaling
  models.py           — ML model training & evaluation
  visualizations.py   — Charts & dashboards
  alert_engine.py     — Real-time fraud alerting

HOW TO RUN
----------
  python main.py                     # Full pipeline
  python main.py --no-plots          # Skip chart generation
  python main.py --quick             # 1000-transaction quick demo
  python main.py --live-feed         # Live alert simulation only
"""

import os, sys, time, warnings, argparse
warnings.filterwarnings("ignore")

# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="ATM Fraud Detection — Cybersecurity ML Pipeline"
    )
    p.add_argument("--no-plots",  action="store_true",
                   help="Skip chart generation (faster)")
    p.add_argument("--quick",     action="store_true",
                   help="Use 1000 transactions for a quick demo")
    p.add_argument("--live-feed", action="store_true",
                   help="Show live alert simulation only")
    p.add_argument("--n-legit",   type=int, default=9500,
                   help="Number of legitimate transactions (default 9500)")
    p.add_argument("--n-fraud",   type=int, default=500,
                   help="Number of fraud transactions (default 500)")
    return p.parse_args()


# ─── Banner ───────────────────────────────────────────────────────────────────

BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║   ___  ________  _____ ______       ________            ║
║  |\  \|\   __  \|\   _ \  _   \    |\  _____\           ║
║  \ \  \ \  \|\  \ \  \\\__\ \  \   \ \  \__/            ║
║   \ \  \ \   __  \ \  \\|__| \  \   \ \   __\           ║
║    \ \  \ \  \ \  \ \  \    \ \  \   \ \  \_|           ║
║     \ \__\ \__\ \__\ \__\    \ \__\   \ \__\            ║
║      \|__|\|__|\|__|\|__|     \|__|    \|__|            ║
║                                                          ║
║        ATM FRAUD DETECTION — CYBERSECURITY SYSTEM        ║
╚══════════════════════════════════════════════════════════╝
"""


# ─── Pipeline Steps ───────────────────────────────────────────────────────────

def step_data_generation(n_legit: int, n_fraud: int):
    from data_generator import generate_dataset
    print("\n[STEP 1/5] Generating ATM Transaction Data …")
    df = generate_dataset(n_legit, n_fraud)
    df.to_csv("atm_transactions.csv", index=False)
    print(f"  ✓ Dataset saved → atm_transactions.csv  "
          f"({len(df):,} rows, {df['is_fraud'].mean():.1%} fraud)")
    return df


def step_preprocessing(df):
    from preprocessing import ATMPreprocessor
    print("\n[STEP 2/5] Feature Engineering & Preprocessing …")
    preprocessor = ATMPreprocessor()
    X, y = preprocessor.fit_transform(df)
    print(f"  ✓ {len(X.columns)} features ready.  "
          f"Class split: {int((y==0).sum())} legit / {int((y==1).sum())} fraud")
    return X, y, preprocessor


def step_training(X, y):
    from models import FraudModelTrainer
    print("\n[STEP 3/5] Training & Evaluating Models …")
    trainer = FraudModelTrainer()
    results = trainer.train_all(X, y)
    trainer.print_full_report()

    # Print comparison table
    print("\n  ── Model Comparison Table ──")
    print(trainer.get_summary_table().to_string(index=False))

    # Save trained model
    trainer.save("fraud_model.pkl")
    return trainer


def step_visualizations(df, trainer):
    import visualizations as viz
    print("\n[STEP 4/5] Generating Visualizations …")

    # EDA
    viz.plot_class_distribution(df)
    viz.plot_amount_distribution(df)
    viz.plot_hourly_pattern(df)
    viz.plot_correlation_heatmap(df)
    viz.plot_velocity_analysis(df)

    # Model evaluation
    summary = trainer.get_summary_table()
    viz.plot_model_comparison(summary)

    best_res = trainer.results[trainer.best_model_name]
    viz.plot_confusion_matrix(best_res["confusion_matrix"], trainer.best_model_name)
    viz.plot_roc_curves(trainer.results)
    viz.plot_precision_recall(trainer.results)

    fi = trainer.get_feature_importance()
    if not fi.empty:
        viz.plot_feature_importance(fi, trainer.best_model_name)

    viz.plot_fraud_risk_score_distribution(
        best_res["y_test"].values, best_res["y_prob"]
    )

    # Executive dashboard
    viz.plot_dashboard(df, summary, trainer.best_model_name, best_res)

    charts = os.listdir("charts")
    print(f"  ✓ {len(charts)} charts saved in ./charts/")


def step_live_feed(trainer, preprocessor):
    from alert_engine import AlertEngine
    print("\n[STEP 5/5] Starting Live Alert Simulation …")
    engine = AlertEngine(trainer, preprocessor, threshold=0.5)
    engine.simulate_live_feed(n=25, delay=0.25)
    engine.export_log("alert_log.json")

    summary = engine.get_summary()
    print("\n  Alert Summary:")
    for level, count in sorted(summary.items()):
        print(f"    {level:<8}: {count}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print(BANNER)
    t0 = time.time()

    # Quick mode
    if args.quick:
        args.n_legit, args.n_fraud = 950, 50

    if args.live_feed:
        # Try loading existing model; train fresh if not found
        try:
            from models import FraudModelTrainer
            from preprocessing import ATMPreprocessor
            trainer = FraudModelTrainer.load("fraud_model.pkl")
            preprocessor = ATMPreprocessor()
            df = step_data_generation(500, 25)
            _, _, preprocessor = step_preprocessing(df)
        except FileNotFoundError:
            df = step_data_generation(args.n_legit, args.n_fraud)
            X, y, preprocessor = step_preprocessing(df)
            trainer = step_training(X, y)
        step_live_feed(trainer, preprocessor)
        return

    # ── Full Pipeline ─────────────────────────────────────────────────────────
    df                    = step_data_generation(args.n_legit, args.n_fraud)
    X, y, preprocessor   = step_preprocessing(df)
    trainer               = step_training(X, y)

    if not args.no_plots:
        step_visualizations(df, trainer)

    step_live_feed(trainer, preprocessor)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  🏁  Pipeline complete in {elapsed:.1f}s")
    print(f"  📁  Outputs:")
    print(f"       atm_transactions.csv  — Raw dataset")
    print(f"       fraud_model.pkl       — Trained model")
    print(f"       alert_log.json        — Alert history")
    if not args.no_plots:
        print(f"       charts/               — {len(os.listdir('charts'))} visualizations")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
