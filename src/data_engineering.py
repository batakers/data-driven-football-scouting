import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ensure src/ is importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.league_tiers import enrich_dataframe as enrich_league_tier

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


def process_form_windows(appearances, val_data):
    """
    Phase 11 — Split-window form features.

    Splits the 365-day window into two halves:
      Window A (recent):  0–180 days before valuation  → recent form
      Window B (earlier): 181–365 days before valuation → earlier form

    Computes per-90 metrics for each window and derives trend signals.
    Only players with ≥450 minutes in BOTH windows get a form trend;
    others receive 'Insufficient Data'.
    """
    app_games = appearances.copy()
    app_games['game_date'] = pd.to_datetime(app_games['date'])

    app_val = pd.merge(
        app_games,
        val_data[['player_id', 'valuation_date']],
        on='player_id',
        how='inner',
    )

    # No leakage
    app_val = app_val[app_val['game_date'] <= app_val['valuation_date']].copy()
    app_val['days_before'] = (app_val['valuation_date'] - app_val['game_date']).dt.days

    # Window A: 0–180 days (recent)
    win_a = app_val[app_val['days_before'] <= 180].groupby('player_id').agg(
        form_mins_a=('minutes_played', 'sum'),
        form_goals_a=('goals', 'sum'),
        form_assists_a=('assists', 'sum'),
        form_cards_a=('yellow_cards', 'sum'),
    ).reset_index()

    # Window B: 181–365 days (earlier)
    win_b = app_val[
        (app_val['days_before'] > 180) & (app_val['days_before'] <= 365)
    ].groupby('player_id').agg(
        form_mins_b=('minutes_played', 'sum'),
        form_goals_b=('goals', 'sum'),
        form_assists_b=('assists', 'sum'),
        form_cards_b=('yellow_cards', 'sum'),
    ).reset_index()

    form = pd.merge(win_a, win_b, on='player_id', how='outer').fillna(0)

    MIN_WINDOW_MINS = 450  # minimum minutes per window for reliable per-90

    # Per-90 for each window (avoid division by zero)
    for window, mins_col in [('a', 'form_mins_a'), ('b', 'form_mins_b')]:
        mins = form[mins_col].replace(0, np.nan)
        form[f'form_goals_p90_{window}'] = (form[f'form_goals_{window}'] / mins * 90).fillna(0)
        form[f'form_assists_p90_{window}'] = (form[f'form_assists_{window}'] / mins * 90).fillna(0)
        form[f'form_cards_p90_{window}'] = (form[f'form_cards_{window}'] / mins * 90).fillna(0)

    # Trend deltas (A minus B — positive = improving)
    sufficient = (form['form_mins_a'] >= MIN_WINDOW_MINS) & (form['form_mins_b'] >= MIN_WINDOW_MINS)

    form['form_trend_goals'] = np.where(
        sufficient,
        form['form_goals_p90_a'] - form['form_goals_p90_b'],
        np.nan,
    )
    form['form_trend_assists'] = np.where(
        sufficient,
        form['form_assists_p90_a'] - form['form_assists_p90_b'],
        np.nan,
    )
    form['form_trend_minutes'] = np.where(
        sufficient & (form['form_mins_b'] > 0),
        form['form_mins_a'] / form['form_mins_b'],
        np.nan,
    )

    # Classify overall form trend
    def _classify_form(row):
        if pd.isna(row['form_trend_goals']) or pd.isna(row['form_trend_minutes']):
            return 'Insufficient Data'
        # Rising: goals improving OR minutes ratio > 1.15 (getting more game time)
        goals_up = row['form_trend_goals'] > 0.05
        mins_up = row['form_trend_minutes'] > 1.15
        goals_down = row['form_trend_goals'] < -0.05
        mins_down = row['form_trend_minutes'] < 0.85
        if goals_up or mins_up:
            return 'Rising'
        if goals_down or mins_down:
            return 'Declining'
        return 'Stable'

    form['form_trend'] = form.apply(_classify_form, axis=1)

    # Keep only the columns needed downstream
    keep_cols = [
        'player_id',
        'form_mins_a', 'form_mins_b',
        'form_goals_p90_a', 'form_goals_p90_b',
        'form_assists_p90_a', 'form_assists_p90_b',
        'form_trend_goals', 'form_trend_assists', 'form_trend_minutes',
        'form_trend',
    ]
    return form[keep_cols]

def main():
    print("Starting Data Engineering...")
    ensure_dirs()
    players, appearances, games, valuations = load_data()
    
    print("Extracting Valuation Dates...")
    val_data = get_valuation_dates(valuations)
    
    print("Processing Appearances and preventing Data Leakage...")
    recent_stats = process_appearances(appearances, games, val_data)

    # ── Phase 11: Form Window Features ─────────────────────────────────────
    print("Computing form trend windows (Phase 11)...")
    form_stats = process_form_windows(appearances, val_data)
    # ───────────────────────────────────────────────────────────────────────

    print("Merging Data...")
    # Base player data
    master_df = pd.merge(players, val_data, on='player_id', how='inner')
    
    # Add recent stats
    master_df = pd.merge(master_df, recent_stats, on='player_id', how='left')

    # ── Phase 11: merge form features ──────────────────────────────────────
    master_df = pd.merge(master_df, form_stats, on='player_id', how='left')
    master_df['form_trend'] = master_df['form_trend'].fillna('Insufficient Data')
    form_counts = master_df['form_trend'].value_counts()
    for trend, count in form_counts.items():
        print(f"  Form trend '{trend}': {count:,} players")
    # ───────────────────────────────────────────────────────────────────────
    
    # Fill NaN stats with 0 for players who had no games in the last 365 days
    stat_cols = ['appearances_last_season', 'minutes_last_season', 'goals_last_season', 
                 'assists_last_season', 'yellow_cards_last_season', 'red_cards_last_season']
    master_df[stat_cols] = master_df[stat_cols].fillna(0)

    # ── Phase 8: League Tier Enrichment ────────────────────────────────────
    # Assign league_tier, league_tier_label, league_tier_short based on the
    # player's current domestic competition. This is done here so all
    # downstream processed files (cleaned, featured, predictions) inherit it.
    print("Enriching league tier context...")
    master_df = enrich_league_tier(master_df, competition_col="current_club_domestic_competition_id")
    tier_counts = master_df["league_tier"].value_counts().sort_index()
    for tier, count in tier_counts.items():
        print(f"  Tier {tier}: {count:,} players")
    # ───────────────────────────────────────────────────────────────────────

    # Save processed data
    output_path = Path("data/processed/master_players.csv")
    master_df.to_csv(output_path, index=False)
    print(f"Data Engineering complete! Shape: {master_df.shape}")
    print(f"Saved to: {output_path}")

if __name__ == '__main__':
    main()
