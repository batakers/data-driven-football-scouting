import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.excluded_players import EXCLUDED_PLAYERS

def main():
    print("Starting Data Cleaning...")
    df = pd.read_csv("data/processed/master_players.csv", low_memory=False)
    
    # Calculate exact Age at valuation date
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
    df['valuation_date'] = pd.to_datetime(df['valuation_date'], errors='coerce')
    
    # Calculate age in years accurately
    df['age_at_valuation'] = (df['valuation_date'] - df['date_of_birth']).dt.days / 365.25
    
    # Drop rows where target or age is missing
    df = df.dropna(subset=['target_market_value', 'age_at_valuation'])
    
    # Filter out inactive players (e.g. valuation date is very old, or last_season is old)
    df = df[df['last_season'] >= 2023].copy()

    # Exclude retired/inactive players from the active pipeline
    if EXCLUDED_PLAYERS:
        before = len(df)
        df = df[~df['name'].isin(EXCLUDED_PLAYERS)].copy()
        excluded_count = before - len(df)
        if excluded_count > 0:
            print(f"  Excluded {excluded_count} player(s): {', '.join(EXCLUDED_PLAYERS[:5])}")
    
    # Handle missing values
    # Impute height by position median
    df['height_in_cm'] = df.groupby('position')['height_in_cm'].transform(lambda x: x.fillna(x.median()))
    
    # Fill remaining missing categorical
    df['foot'] = df['foot'].fillna('unknown')
    
    # Drop essential nulls
    df = df.dropna(subset=['position', 'current_club_id'])
    
    output_path = Path("data/processed/cleaned_players.csv")
    df.to_csv(output_path, index=False)
    print(f"Data Cleaning complete! Cleaned shape: {df.shape}")
    print(f"Saved to: {output_path}")

if __name__ == '__main__':
    main()
