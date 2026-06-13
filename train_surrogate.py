import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def train_surrogate():
    csv_file = "optuna_trials_data.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. You need to run ml_optimizer.py first.")
        return

    print("Loading overnight simulation data...")
    df = pd.read_csv(csv_file)
    
    # Filter for completed trials
    if 'state' in df.columns:
        df = df[df['state'] == 'COMPLETE']

    param_cols = [c for c in df.columns if c.startswith('params_')]
    
    if len(param_cols) == 0:
        print("No parameter columns found in the CSV.")
        return
        
    if len(df) < 10:
        print(f"Warning: Only {len(df)} rows found. XGBoost works best with 100+ rows.")
        
    X = df[param_cols].copy()
    X.columns = [c.replace('params_', '') for c in X.columns]
    
    # Extract individual targets
    y_mr = df['user_attrs_MR_PnL'].fillna(0).copy()
    y_tp = df['user_attrs_TP_PnL'].fillna(0).copy()
    y_mb = df['user_attrs_MB_PnL'].fillna(0).copy()

    # Split the data into train and test sets
    X_train, X_test, mr_train, mr_test, tp_train, tp_test, mb_train, mb_test = train_test_split(
        X, y_mr, y_tp, y_mb, test_size=0.2, random_state=42
    )

    # Dynamically scale tree depth based on how much data we collected overnight
    # More data = we can safely allow deeper, more complex interactions (5-way or 6-way)
    optimal_depth = 5
    if len(X_train) > 1600:
        optimal_depth = 6
        
    print(f"Training 'Ensemble of Specialists' XGBoost Models on {len(X_train)} samples (Auto-Depth: {optimal_depth})...")
    
    # Train 3 separate models
    model_mr = xgb.XGBRegressor(n_estimators=100, max_depth=optimal_depth, learning_rate=0.1, random_state=42)
    model_tp = xgb.XGBRegressor(n_estimators=100, max_depth=optimal_depth, learning_rate=0.1, random_state=42)
    model_mb = xgb.XGBRegressor(n_estimators=100, max_depth=optimal_depth, learning_rate=0.1, random_state=42)

    model_mr.fit(X_train, mr_train)
    model_tp.fit(X_train, tp_train)
    model_mb.fit(X_train, mb_train)

    # Evaluate the models
    print(f"Mean Reversion Model R^2 Score: {r2_score(mr_test, model_mr.predict(X_test)):.2f}")
    print(f"Trend Pullback Model R^2 Score: {r2_score(tp_test, model_tp.predict(X_test)):.2f}")
    print(f"Momentum Breakout Model R^2 Score: {r2_score(mb_test, model_mb.predict(X_test)):.2f}")

    # Generate SHAP Values for the Overall Portfolio (using MR as the primary proxy for the plot)
    print("\nGenerating SHAP explanations for Mean Reversion (Primary Alpha Driver)...")
    explainer = shap.TreeExplainer(model_mr)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=300)
    print("-> Saved visual parameter breakdown to 'shap_summary.png'")
    
    print("\nSimulating 100,000 virtual backtests via the Neural Surrogate...")
    
    random_samples = {}
    for col in X.columns:
        low, high = X[col].min(), X[col].max()
        if X[col].dtype.kind in 'i':
            random_samples[col] = np.random.randint(int(low), int(high)+1, 100000)
        else:
            random_samples[col] = np.random.uniform(float(low), float(high), 100000)
            
    X_virt = pd.DataFrame(random_samples)
    
    # Predict individually and sum for Total PnL
    X_virt['Predicted_MR'] = model_mr.predict(X_virt[X.columns])
    X_virt['Predicted_TP'] = model_tp.predict(X_virt[X.columns])
    X_virt['Predicted_MB'] = model_mb.predict(X_virt[X.columns])
    X_virt['Predicted_PnL'] = X_virt['Predicted_MR'] + X_virt['Predicted_TP'] + X_virt['Predicted_MB']
    
    best_virtual = X_virt.sort_values(by='Predicted_PnL', ascending=False).head(5)
    print("\nTop 5 AI-Predicted 'Sweet Spots' (Strategy Breakdown):")
    
    display_cols = ['Predicted_PnL', 'Predicted_MR', 'Predicted_TP', 'Predicted_MB'] + list(X.columns)
    print(best_virtual[display_cols].to_string(index=False))
    
    import json
    export_list = []
    for _, row in best_virtual.iterrows():
        # Only export the actual parameters plus the Predicted PnL metric
        clean_row = {k: v for k, v in row.to_dict().items() if not k.startswith("Predicted_") or k == "Predicted_PnL"}
        export_list.append(clean_row)
        
    with open("surrogate_top_5.json", "w") as f:
        json.dump(export_list, f, indent=4)
        
    print("\n-> Automatically exported these Top 5 predictions to 'surrogate_top_5.json'!")
    print("-> Run 'python verify_predictions.py' to backtest them instantly.")

if __name__ == "__main__":
    train_surrogate()