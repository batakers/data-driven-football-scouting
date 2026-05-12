import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def ensure_dirs():
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)

def evaluate_model(y_true, y_pred, positions, prefix):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    ape = np.abs((y_true - y_pred) / y_true)
    median_ape = np.median(ape) * 100
    
    metrics = {
        'Model': prefix,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Median_APE_Pct': median_ape
    }
    
    results_df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'position': positions
    })
    
    for pos in results_df['position'].unique():
        pos_df = results_df[results_df['position'] == pos]
        if len(pos_df) > 0:
            pos_mae = mean_absolute_error(pos_df['y_true'], pos_df['y_pred'])
            metrics[f'MAE_{pos}'] = pos_mae
            
    return metrics

def plot_feature_importance(model, features, title, filename):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        idx = np.argsort(importances)[-20:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(idx)), importances[idx], align='center')
        plt.yticks(range(len(idx)), [features[i] for i in idx])
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f"outputs/figures/{filename}")
        plt.close()

def main():
    print("Starting Modeling Phase...")
    ensure_dirs()
    
    df = pd.read_csv("data/processed/featured_players.csv", low_memory=False)
    
    base_features = [
        'age_at_valuation', 'minutes_last_season', 
        'goals_per_90_ls', 'assists_per_90_ls', 'cards_per_90_ls'
    ]
    
    pos_cols = [col for col in df.columns if col.startswith('pos_')]
    foot_cols = [col for col in df.columns if col.startswith('foot_')]
    
    features_a = base_features + pos_cols + foot_cols # Performance-Only
    features_b = features_a + ['previous_market_value'] # Market-Aware
    
    X_a = df[features_a]
    X_b = df[features_b].fillna({'previous_market_value': 0})
    
    # Target transform: log1p
    y = np.log1p(df['target_market_value'])
    positions = df['position_group_raw'].values
    
    indices = np.arange(len(df))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_a_train, X_a_test = X_a.iloc[idx_train], X_a.iloc[idx_test]
    X_b_train, X_b_test = X_b.iloc[idx_train], X_b.iloc[idx_test]
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]
    pos_test = positions[idx_test]
    
    results = []
    
    # Baseline Model (Ridge)
    ridge_a = Ridge(alpha=1.0)
    ridge_a.fit(X_a_train.fillna(0), y_train)
    y_pred_ridge_a = ridge_a.predict(X_a_test.fillna(0))
    results.append(evaluate_model(np.expm1(y_test), np.expm1(y_pred_ridge_a), pos_test, "Baseline_Ridge_PerfOnly"))
    
    # Model 1: Random Forest
    rf_a = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_a.fit(X_a_train.fillna(0), y_train)
    y_pred_rf_a = rf_a.predict(X_a_test.fillna(0))
    results.append(evaluate_model(np.expm1(y_test), np.expm1(y_pred_rf_a), pos_test, "RF_PerfOnly"))
    
    # Model 2: XGBoost (Model A - Performance Only)
    xgb_a = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
    xgb_a.fit(X_a_train, y_train)
    y_pred_xgb_a = xgb_a.predict(X_a_test)
    results.append(evaluate_model(np.expm1(y_test), np.expm1(y_pred_xgb_a), pos_test, "XGB_PerfOnly (Model A)"))
    plot_feature_importance(xgb_a, features_a, "Feature Importance - Model A (Perf Only)", "feature_importance_model_A.png")
    
    # Model 2: XGBoost (Model B - Market Aware)
    xgb_b = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
    xgb_b.fit(X_b_train, y_train)
    y_pred_xgb_b = xgb_b.predict(X_b_test)
    results.append(evaluate_model(np.expm1(y_test), np.expm1(y_pred_xgb_b), pos_test, "XGB_MarketAware (Model B)"))
    plot_feature_importance(xgb_b, features_b, "Feature Importance - Model B (Market Aware)", "feature_importance_model_B.png")
    
    eval_df = pd.DataFrame(results)
    eval_df.to_csv("outputs/model_evaluation.csv", index=False)
    print("\nModel Evaluation Summary:")
    print(eval_df[['Model', 'MAE', 'R2', 'Median_APE_Pct']])
    
    # Save the Models
    joblib.dump(xgb_a, "outputs/models/performance_only_model.pkl")
    joblib.dump(xgb_b, "outputs/models/market_aware_model.pkl")
    
    # Generate Predictions for ALL players using Model A (for scouting)
    df['predicted_log_value'] = xgb_a.predict(X_a)
    df['predicted_value'] = np.expm1(df['predicted_log_value'])
    df['undervalued_pct'] = (df['predicted_value'] - df['target_market_value']) / df['target_market_value']
    
    preds_df = df[['player_id', 'name', 'position_group_raw', 'age_at_valuation', 'current_club_id',
                   'minutes_last_season', 'goals_per_90_ls', 'assists_per_90_ls', 'cards_per_90_ls',
                   'target_market_value', 'predicted_value', 'undervalued_pct',
                   'current_club_domestic_competition_id', 'current_club_name',
                   'league_tier', 'league_tier_label', 'league_tier_short',
                   'contract_expiry_date', 'contract_months_remaining',
                   'contract_status', 'contract_status_label', 'contract_badge',
                   'form_trend', 'form_trend_goals', 'form_trend_assists', 'form_trend_minutes',
                   'form_mins_a', 'form_mins_b']]
    preds_df.to_csv("outputs/predictions_per_player.csv", index=False)
    
    print("\nModeling complete! Evaluation saved to outputs/model_evaluation.csv")
    print("Predictions saved to outputs/predictions_per_player.csv")

if __name__ == '__main__':
    main()
