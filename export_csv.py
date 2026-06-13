import optuna
import pandas as pd

def export_data():
    print("Connecting to Optuna database...")
    try:
        study = optuna.load_study(
            study_name="makeshift_trades_6mo",
            storage="sqlite:///optuna_study.db"
        )
    except Exception as e:
        print(f"Error loading database: {e}")
        return

    print("Extracting trials...")
    df = study.trials_dataframe()
    
    csv_filename = "optuna_trials_data.csv"
    df.to_csv(csv_filename, index=False)
    
    print(f"\nSUCCESS! Exported exactly {len(df)} trials to {csv_filename}")
    print("Your data is 100% safe and ready for the Surrogate AI.")

if __name__ == "__main__":
    export_data()