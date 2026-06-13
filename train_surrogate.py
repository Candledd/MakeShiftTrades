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

    # Identify inputs (parameters) and output (target PnL)
    param_cols = [c for c in df.columns if c.startswith('params_')]
    target_col = 'value'
    
    if len(param_cols) == 0:
        print("No parameter columns found in the CSV.")
        return
        
    if len(df) < 10:
        print(f"Warning: Only {len(df)} rows found. XGBoost works best with 100+ rows.")
        
    X = df[param_cols].copy()
    # Rename columns to drop the 'params_' prefix for readability in SHAP charts
    X.columns = [c.replace('params_', '') for c in X.columns]
    y = df[target_col].copy()

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training XGBoost Surrogate Model on {len(X_train)} samples...")
    model = xgb.XGBRegressor(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.1, 
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"Model R^2 Score (Predictive Accuracy): {r2:.2f}")

    print("\n--- Feature Importance (What actually matters) ---")
    importances = model.feature_importances_
    features = X.columns
    for feat, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
        print(f"{feat:25s}: {imp:.4f} ({imp*100:.1f}%)")

    # Generate SHAP Values
    print("\nGenerating SHAP explanations...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Plot SHAP Summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=300)
    print("-> Saved visual parameter breakdown to 'shap_summary.png'")
    
    # Optional: Predict 100,000 Random Combinations Instantly
    print("\nSimulating 100,000 virtual backtests via the Neural Surrogate...")
    
    # Generate random combinations between min and max bounds observed in the dataset
    random_samples = {}
    for col in X.columns:
        low, high = X[col].min(), X[col].max()
        if X[col].dtype.kind in 'i':  # integer parameter
            random_samples[col] = np.random.randint(int(low), int(high)+1, 100000)
        else: # float parameter
            random_samples[col] = np.random.uniform(float(low), float(high), 100000)
            
    X_virt = pd.DataFrame(random_samples)
    y_virt_pred = model.predict(X_virt)
    X_virt['Predicted_PnL'] = y_virt_pred
    
    best_virtual = X_virt.sort_values(by='Predicted_PnL', ascending=False).head(5)
    print("\nTop 5 AI-Predicted 'Sweet Spots' (Did Optuna miss these?):")
    print(best_virtual.to_string(index=False))
    
    # Export the top 5 to a JSON file so verify_predictions.py can automatically run them
    import json
    export_list = []
    for _, row in best_virtual.iterrows():
        export_list.append(row.to_dict())
        
    with open("surrogate_top_5.json", "w") as f:
        json.dump(export_list, f, indent=4)
        
    print("\n-> Automatically exported these Top 5 predictions to 'surrogate_top_5.json'!")
    print("-> Run 'python verify_predictions.py' to backtest them instantly.")

if __name__ == "__main__":
    train_surrogate()