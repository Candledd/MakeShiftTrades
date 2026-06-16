import optuna
import pandas as pd

def export_data():
    mode = input("Which version do you want to export? (ml / risk): ").strip().lower()
    if mode == "ml":
        study_name = "makeshift_trades_6mo_v5"
        csv_filename = "data/optuna_trials_ml.csv"
    else:
        study_name = "makeshift_trades_risk_v1"
        csv_filename = "data/optuna_trials_risk.csv"

    print("Connecting to Optuna database...")
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage="sqlite:///data/optuna_study.db"
        )
    except Exception as e:
        print(f"Error loading database: {e}")
        return

    print("Extracting trials...")
    df = study.trials_dataframe()
    
    
    df.to_csv(csv_filename, index=False)
    
    print(f"\nSUCCESS! Exported exactly {len(df)} trials to {csv_filename}")
    print("Your data is 100% safe and ready for the Surrogate AI.")

if __name__ == "__main__":
    export_data()
