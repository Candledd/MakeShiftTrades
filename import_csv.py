import pandas as pd
import optuna
from optuna.distributions import FloatDistribution, IntDistribution
import os

def migrate_csv_to_db():
    csv_file = "optuna_trials_data.csv"
    if not os.path.exists(csv_file):
        print(f"File {csv_file} not found. Nothing to import.")
        return

    df = pd.read_csv(csv_file)
    if df.empty:
        print("CSV is empty.")
        return

    # Create/load the study
    study = optuna.create_study(
        study_name="makeshift_trades_optimization",
        storage="sqlite:///optuna_study.db",
        direction="maximize",
        load_if_exists=True
    )

    # Distributions mapped exactly to ml_optimizer.py
    distributions = {
        "MR_BB_STD": FloatDistribution(1.5, 3.0, step=0.1),
        "MR_RSI_OVERSOLD": FloatDistribution(20.0, 40.0, step=1.0),
        "MR_RSI_OVERBOUGHT": FloatDistribution(60.0, 80.0, step=1.0),
        "TP_BB_STD": FloatDistribution(1.5, 3.0, step=0.1),
        "MB_DONCHIAN_PERIOD": IntDistribution(10, 40),
        "MR_STOP_MULT": FloatDistribution(1.0, 4.0, step=0.1),
        "TP_STOP_MULT": FloatDistribution(1.0, 4.0, step=0.1),
        "TP_PULLBACK_BUFFER": FloatDistribution(1.000, 1.010, step=0.001),
        "MB_ADX_THRESHOLD": FloatDistribution(15.0, 35.0, step=1.0),
        "MR_BB_PERIOD": IntDistribution(10, 40),
        "MR_RSI_PERIOD": IntDistribution(5, 20),
        "TP_BB_PERIOD": IntDistribution(10, 40),
        "MB_FALSE_BREAKOUT_BARS": IntDistribution(1, 5)
    }

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
