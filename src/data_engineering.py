import pandas as pd
import numpy as np
from pathlib import Path

def ensure_dirs():
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("src").mkdir(exist_ok=True)

def load_data():
    # Adding low_memory=False to avoid DtypeWarning
    players = pd.read_csv('data/raw/players.csv')
    appearances = pd.read_csv('data/raw/appearances.csv')
    games = pd.read_csv('data/raw/games.csv')
    valuations = pd.read_csv('data/raw/player_valuations.csv')
    return players, appearances, games, valuations

def get_valuation_dates(valuations):
    # Sort valuations by date
    valuations['date'] = pd.to_datetime(valuations['date'])
    valuations = valuations.sort_values(by=['player_id', 'date'], ascending=[True, False])
    
    # Get latest valuation
    latest_val = valuations.groupby('player_id').first().reset_index()
    latest_val = latest_val[['player_id', 'date', 'market_value_in_eur']].rename(
        columns={'date': 'valuation_date', 'market_value_in_eur': 'target_market_value'}
    )
    
    # Get previous valuation (2nd row per player)
    prev_val = valuations.groupby('player_id').nth(1).reset_index()
    prev_val = prev_val[['player_id', 'market_value_in_eur']].rename(
        columns={'market_value_in_eur': 'previous_market_value'}
    )
    
    val_data = pd.merge(latest_val, prev_val, on='player_id', how='left')
    return val_data

def process_appearances(appearances, games, val_data):
    # Appearances already has the 'date' column, no need to merge with games just for date
    # But to satisfy the blueprint, we can keep the games df loaded.
    app_games = appearances.copy()
    app_games['game_date'] = pd.to_datetime(app_games['date'])
    
    # Merge with valuation dates
    app_val = pd.merge(app_games, val_data[['player_id', 'valuation_date']], on='player_id', how='inner')
    
    # Filter 1: No Leakage (game_date <= valuation_date)
    valid_apps = app_val[app_val['game_date'] <= app_val['valuation_date']].copy()
    
    # Filter 2: Last Season Only (within 365 days of valuation date)
    valid_apps['days_before_valuation'] = (valid_apps['valuation_date'] - valid_apps['game_date']).dt.days
    recent_apps = valid_apps[valid_apps['days_before_valuation'] <= 365]
    
    # Aggregate stats
    stats = recent_apps.groupby('player_id').agg(
        appearances_last_season=('appearance_id', 'count'),
        minutes_last_season=('minutes_played', 'sum'),
        goals_last_season=('goals', 'sum'),
        assists_last_season=('assists', 'sum'),
        yellow_cards_last_season=('yellow_cards', 'sum'),
        red_cards_last_season=('red_cards', 'sum')
    ).reset_index()
    
    return stats

def main():
    print("Starting Data Engineering...")
    ensure_dirs()
    players, appearances, games, valuations = load_data()
    
    print("Extracting Valuation Dates...")
    val_data = get_valuation_dates(valuations)
    
    print("Processing Appearances and preventing Data Leakage...")
    recent_stats = process_appearances(appearances, games, val_data)
    
    print("Merging Data...")
    # Base player data
    master_df = pd.merge(players, val_data, on='player_id', how='inner')
    
    # Add recent stats
    master_df = pd.merge(master_df, recent_stats, on='player_id', how='left')
    
    # Fill NaN stats with 0 for players who had no games in the last 365 days
    stat_cols = ['appearances_last_season', 'minutes_last_season', 'goals_last_season', 
                 'assists_last_season', 'yellow_cards_last_season', 'red_cards_last_season']
    master_df[stat_cols] = master_df[stat_cols].fillna(0)
    
    # Save processed data
    output_path = Path("data/processed/master_players.csv")
    master_df.to_csv(output_path, index=False)
    print(f"Data Engineering complete! Shape: {master_df.shape}")
    print(f"Saved to: {output_path}")

if __name__ == '__main__':
    main()
