import pandas as pd
import optuna
from optuna.distributions import FloatDistribution, IntDistribution
import os

def migrate_csv_to_db():
    mode = input("Which version do you want to import? (ml / risk): ").strip().lower()
    if mode == "ml":
        study_name = "makeshift_trades_6mo_v5"
        csv_file = "data/optuna_trials_ml.csv"
        distributions = {
            "MR_BB_STD": FloatDistribution(1.5, 3.0, step=0.1),
            "TP_BB_STD": FloatDistribution(1.5, 3.0, step=0.1),
            "MB_DONCHIAN_PERIOD": IntDistribution(36, 72),
            "MR_BB_PERIOD": IntDistribution(10, 40),
            "MR_RSI_PERIOD": IntDistribution(5, 20),
            "TP_BB_PERIOD": IntDistribution(10, 40),
            "MB_COMPRESSION_THRESHOLD": IntDistribution(50, 65, step=5)
        }
    else:
        study_name = "makeshift_trades_risk_v1"
        csv_file = "data/optuna_trials_risk.csv"
        distributions = {
            "MAX_POSITION_PCT": FloatDistribution(1.0, 10.0, step=0.5),
            "MAX_RISK_PCT": FloatDistribution(0.01, 0.10, step=0.01),
            "RISK_TIER_EQUITY_PCT": FloatDistribution(0.01, 0.10, step=0.01),
            "TP_STOP_MULT": FloatDistribution(1.0, 3.0, step=0.1),
            "TP_MIN_RR": FloatDistribution(1.0, 2.0, step=0.1),
            "MR_STOP_MULT": FloatDistribution(1.0, 3.0, step=0.1),
            "MR_MIN_RR": FloatDistribution(0.5, 2.0, step=0.1)
        }

    if not os.path.exists(csv_file):
        print(f"File {csv_file} not found. Nothing to import.")
        return

    df = pd.read_csv(csv_file)
    if df.empty:
        print("CSV is empty.")
        return

    # Create/load the study
    study = optuna.create_study(
        study_name=study_name,
        storage="sqlite:///data/optuna_study.db",
        direction="maximize",
        load_if_exists=True
    )
    trials_added = 0
    for _, row in df.iterrows():
        # Only import completed trials
        if 'state' in row and row['state'] != 'COMPLETE':
            continue
            
        # Extract params and user attrs
        params = {}
        user_attrs = {}
        
        for col in df.columns:
            if col.startswith("params_"):
                p_name = col.replace("params_", "")
                if pd.notna(row[col]) and p_name in distributions:
                    params[p_name] = row[col]
            elif col.startswith("user_attrs_"):
                attr_name = col.replace("user_attrs_", "")
                if pd.notna(row[col]):
                    user_attrs[attr_name] = row[col]

        # Ensure we have all necessary params for a valid trial
        if not params:
            continue
            
        value = row['value'] if 'value' in row and pd.notna(row['value']) else None
        if value is None:
            continue

        # Create the FrozenTrial
        trial = optuna.trial.create_trial(
            params=params,
            distributions={k: distributions[k] for k in params.keys()},
            value=value,
            user_attrs=user_attrs
        )
        
        study.add_trial(trial)
        trials_added += 1

    print(f"Successfully migrated {trials_added} historical trials into the persistent SQLite database!")

if __name__ == "__main__":
    migrate_csv_to_db()
