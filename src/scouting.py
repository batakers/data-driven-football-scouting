import pandas as pd
from pathlib import Path

def main():
    print("Starting Scouting Phase (Undervalued Candidates Detection)...")
    
    # Load predictions from Model A (Performance-Only)
    df = pd.read_csv("outputs/predictions_per_player.csv")
    
    # Filter Logic for Undervalued Candidates
    # Added target_market_value >= 500000 to prevent exploding undervalued_pct
    scouting_criteria = (
        (df['minutes_last_season'] >= 900) &
        (df['target_market_value'] <= 20000000) &
        (df['target_market_value'] >= 500000) & 
        (df['age_at_valuation'] <= 25) &
        (df['undervalued_pct'] > 0)
    )
    
    candidates = df[scouting_criteria].copy()
    
    # Sort by undervalued_pct descending
    candidates = candidates.sort_values(by='undervalued_pct', ascending=False)
    candidates['undervalued_pct_formatted'] = (candidates['undervalued_pct'] * 100).round(1).astype(str) + '%'
    
    # Save Overall Candidates
    candidates.to_csv("outputs/undervalued_candidates_overall.csv", index=False)
    print(f"Found {len(candidates)} initial undervalued candidates.")
    
    # Generate Shortlists
    Path("outputs/shortlists").mkdir(parents=True, exist_ok=True)
    
    # 1. Top 20 overall
    candidates.head(20).to_csv("outputs/shortlists/top_20_overall.csv", index=False)
    
    # 2. Top 10 per position
    positions = ['Forward', 'Midfielder', 'Defender', 'Goalkeeper']
    for pos in positions:
        pos_df = candidates[candidates['position_group_raw'] == pos].head(10)
        pos_df.to_csv(f"outputs/shortlists/top_10_{pos.lower()}.csv", index=False)
        
    # 3. Top 5 under €5M
    under_5m = candidates[candidates['target_market_value'] <= 5000000].head(5)
    under_5m.to_csv("outputs/shortlists/top_5_under_5m.csv", index=False)
    
    # 4. Top 5 U21
    u21 = candidates[candidates['age_at_valuation'] <= 21].head(5)
    u21.to_csv("outputs/shortlists/top_5_u21.csv", index=False)
    
    print("Scouting analysis complete! Shortlists saved in outputs/shortlists/")

if __name__ == '__main__':
    main()
