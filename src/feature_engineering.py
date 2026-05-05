import pandas as pd
from pathlib import Path
import numpy as np

def main():
    print("Starting Feature Engineering...")
    df = pd.read_csv("data/processed/cleaned_players.csv", low_memory=False)
    
    # 4.1 Position Group Mapping
    pos_map = {
        'Attack': 'Forward',
        'Midfield': 'Midfielder',
        'Defender': 'Defender',
        'Goalkeeper': 'Goalkeeper'
    }
    df['position_group_raw'] = df['position'].map(pos_map).fillna('Other')
    df['position_group_dummy'] = df['position_group_raw'] # Duplicate for dummies
    
    # One-hot encode position groups
    df = pd.get_dummies(df, columns=['position_group_dummy'], prefix='pos')
    pos_cols = [col for col in df.columns if col.startswith('pos_')]
    for col in pos_cols:
        df[col] = df[col].astype(int)
        
    # One-hot encode preferred foot
    df = pd.get_dummies(df, columns=['foot'], prefix='foot')
    foot_cols = [col for col in df.columns if col.startswith('foot_')]
    for col in foot_cols:
        df[col] = df[col].astype(int)
    
    # 4.2 Per 90 Metrics for Last Season
    # If minutes is 0, setting per_90 to 0 avoids infinity
    df['goals_per_90_ls'] = np.where(df['minutes_last_season'] > 0, 
                                     (df['goals_last_season'] / df['minutes_last_season']) * 90, 0)
    
    df['assists_per_90_ls'] = np.where(df['minutes_last_season'] > 0, 
                                       (df['assists_last_season'] / df['minutes_last_season']) * 90, 0)
                                              
    df['cards_per_90_ls'] = np.where(df['minutes_last_season'] > 0, 
                                     ((df['yellow_cards_last_season'] + df['red_cards_last_season']) / df['minutes_last_season']) * 90, 0)
    
    # 4.3 Previous Market Value
    # Kept as 'previous_market_value' natively, nulls will be handled in modeling
    
    output_path = Path("data/processed/featured_players.csv")
    df.to_csv(output_path, index=False)
    print(f"Feature Engineering complete! Shape: {df.shape}")
    print(f"Saved to: {output_path}")

if __name__ == '__main__':
    main()
