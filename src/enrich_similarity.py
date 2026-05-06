import pandas as pd
import numpy as np
import os
import re
from thefuzz import fuzz, process
import unicodedata
import json

# Paths
KAGGLE_PATH = "data/raw/Top 5 League Football Player Stats (2017-2025)/Top5_League_Players_2017to2024_dataset.csv"
TM_PLAYERS_PATH = "data/processed/featured_players.csv"
TM_PREDS_PATH = "outputs/predictions_per_player.csv"
OUTPUT_PATH = "data/processed/enriched_similarity_players.csv"
REPORT_PATH = "outputs/matching_report.csv"
AUDIT_PATH = "outputs/matching_audit.csv"

# Configuration
FUZZY_THRESHOLD_AUTO = 95
FUZZY_THRESHOLD_COMPATIBLE = 90
FUZZY_THRESHOLD_REVIEW = 85
TEAM_MATCH_THRESHOLD = 80

POSITION_MAP = {
    "Forward": ["FW", "FW,MF", "MF,FW"],
    "Midfielder": ["MF", "MF,FW", "FW,MF", "DF,MF", "MF,DF"],
    "Defender": ["DF", "DF,MF", "MF,DF"],
    "Goalkeeper": ["GK"]
}

def normalize_string(s):
    if pd.isna(s):
        return ""
    # Normalize unicode (remove accents)
    s = unicodedata.normalize('NFKD', str(s)).encode('ASCII', 'ignore').decode('ASCII')
    # Lowercase and remove special characters
    s = s.lower().strip()
    s = re.sub(r'[^a-z0-9\s]', '', s)
    return s

def get_season_from_date(date_str):
    if pd.isna(date_str):
        return None
    try:
        # Expected YYYY-MM-DD
        year = int(date_str[:4])
        month = int(date_str[5:7])
        
        # Season boundary approx July 1st
        if month >= 7:
            return f"{str(year)[2:]}{str(year+1)[2:]}"
        else:
            return f"{str(year-1)[2:]}{str(year)[2:]}"
    except:
        return None

def get_previous_season(season_str):
    if not season_str or len(season_str) != 4:
        return None
    try:
        y1 = int(season_str[:2])
        y2 = int(season_str[2:])
        return f"{y1-1:02d}{y2-1:02d}"
    except:
        return None

def is_position_compatible(tm_pos, kaggle_pos):
    if tm_pos not in POSITION_MAP:
        return False
    allowed = POSITION_MAP[tm_pos]
    # Kaggle pos can be comma-separated
    k_parts = [p.strip() for p in str(kaggle_pos).split(',')]
    for kp in k_parts:
        if kp in allowed:
            return True
    return False

def team_similarity_score(norm_tm_team, norm_kaggle_team):
    if not norm_tm_team or not norm_kaggle_team:
        return 0
    return max(
        fuzz.ratio(norm_tm_team, norm_kaggle_team),
        fuzz.partial_ratio(norm_tm_team, norm_kaggle_team),
        fuzz.token_set_ratio(norm_tm_team, norm_kaggle_team),
    )

def append_rejection_reason(existing_reason, reason):
    if not reason:
        return existing_reason
    if pd.isna(existing_reason) or str(existing_reason).strip() == "":
        return reason
    reasons = [r.strip() for r in str(existing_reason).split("|") if r.strip()]
    if reason not in reasons:
        reasons.append(reason)
    return "|".join(reasons)

def demote_match(df, mask, reason, status="review_required"):
    if not mask.any():
        return
    df.loc[mask, "match_status"] = status
    df.loc[mask, "review_required"] = True
    df.loc[mask, "match_rejection_reason"] = df.loc[mask, "match_rejection_reason"].apply(
        lambda value: append_rejection_reason(value, reason)
    )

def truthy_series(series):
    return series.astype("string").str.strip().str.lower().isin(["true", "1", "yes"])

def apply_release_gates(df_audit):
    """Remove matches that should not be accepted into the enriched pool."""
    df = df_audit.copy()

    for col, default in {
        "review_required": False,
        "duplicate_conflict": False,
        "duplicate_resolution": "",
        "match_rejection_reason": "",
        "team_match_score": 0,
    }.items():
        if col not in df.columns:
            df[col] = default

    matched_mask = df["match_status"] == "matched"

    if "is_future_season_used" in df.columns:
        future_mask = matched_mask & (df["is_future_season_used"] == True)
        demote_match(df, future_mask, "future_season_used", status="unmatched")

    if "position_compatible" in df.columns:
        position_mask = (df["match_status"] == "matched") & (df["position_compatible"] != True)
        demote_match(df, position_mask, "position_mismatch", status="review_required")

    fuzzy_review_mask = (df["match_status"] == "matched") & (df["match_method"] == "fuzzy_review")
    demote_match(df, fuzzy_review_mask, "fuzzy_review", status="review_required")

    if "kaggle_index" in df.columns:
        matched_with_index = df[(df["match_status"] == "matched") & df["kaggle_index"].notna()].copy()
    else:
        matched_with_index = pd.DataFrame()
    if not matched_with_index.empty:
        method_rank = {
            "exact_name_team_season": 4,
            "normalized_name_team_season": 3,
            "fuzzy_high_confidence": 2,
            "fuzzy_review": 1,
        }
        matched_with_index["_resolution_score"] = (
            truthy_series(matched_with_index["position_compatible"]).astype(int) * 100000
            + truthy_series(matched_with_index["club_match"]).astype(int) * 10000
            + matched_with_index["team_match_score"].fillna(0).astype(float) * 100
            + matched_with_index["match_score"].fillna(0).astype(float) * 10
            + matched_with_index["match_method"].map(method_rank).fillna(0).astype(float)
            - matched_with_index["age_diff"].fillna(99).astype(float)
        )

        for _, group in matched_with_index.groupby("kaggle_index"):
            if group["player_id"].nunique() <= 1:
                continue
            keep_idx = group.sort_values("_resolution_score", ascending=False).index[0]
            drop_idx = group.index.difference([keep_idx])
            df.loc[group.index, "duplicate_conflict"] = True
            df.loc[keep_idx, "duplicate_resolution"] = "kept_best_match"
            df.loc[drop_idx, "duplicate_resolution"] = "dropped_conflicting_match"
            demote_match(df, df.index.isin(drop_idx), "duplicate_kaggle_assignment", status="review_required")

    df["enriched_available"] = df["match_status"] == "matched"
    return df

def safe_rate(numerator, denominator):
    return numerator / denominator if denominator else 0

def build_matching_report(df_audit):
    matched = df_audit[df_audit["match_status"] == "matched"].copy()
    total_players = len(df_audit)
    total_matched = len(matched)
    eligible = df_audit[df_audit["target_league"].notna()]
    eligible_matched = eligible[eligible["match_status"] == "matched"]

    rows = [
        {
            "section": "overall",
            "segment": "all_players",
            "numerator": total_matched,
            "denominator": total_players,
            "rate": safe_rate(total_matched, total_players),
            "description": "Accepted enriched matches across the full Transfermarkt pool",
        },
        {
            "section": "eligible_top5",
            "segment": "top5_transfermarkt_players",
            "numerator": len(eligible_matched),
            "denominator": len(eligible),
            "rate": safe_rate(len(eligible_matched), len(eligible)),
            "description": "Accepted enriched matches among Transfermarkt rows mapped to Top 5 leagues",
        },
    ]

    for league, group in eligible.groupby("target_league"):
        matched_count = (group["match_status"] == "matched").sum()
        rows.append({
            "section": "league",
            "segment": league,
            "numerator": int(matched_count),
            "denominator": int(len(group)),
            "rate": safe_rate(int(matched_count), int(len(group))),
            "description": "Accepted enriched matches by Transfermarkt league denominator",
        })

    for method, count in matched["match_method"].value_counts().sort_index().items():
        rows.append({
            "section": "method",
            "segment": method,
            "numerator": int(count),
            "denominator": int(total_matched),
            "rate": safe_rate(int(count), int(total_matched)),
            "description": "Share of accepted enriched matches by match method",
        })

    status_counts = df_audit["match_status"].value_counts()
    for status, count in status_counts.sort_index().items():
        rows.append({
            "section": "status",
            "segment": status,
            "numerator": int(count),
            "denominator": int(total_players),
            "rate": safe_rate(int(count), int(total_players)),
            "description": "Audit row distribution by final match status",
        })

    quality_gates = {
        "future_leakage_accepted": len(matched[matched["is_future_season_used"] == True]),
        "position_mismatch_accepted": len(matched[matched["position_compatible"] != True]),
        "duplicate_conflicting_kaggle_assignment": 0,
    }
    if "kaggle_index" in matched.columns:
        quality_gates["duplicate_conflicting_kaggle_assignment"] = int(
            matched[matched["kaggle_index"].notna()]
            .groupby("kaggle_index")["player_id"]
            .nunique()
            .gt(1)
            .sum()
        )
    for gate, count in quality_gates.items():
        rows.append({
            "section": "quality_gate",
            "segment": gate,
            "numerator": int(count),
            "denominator": int(total_matched),
            "rate": safe_rate(int(count), int(total_matched)),
            "description": "Release gate violation count among accepted enriched matches",
        })

    return pd.DataFrame(rows)

def main():
    print("Loading datasets...")
    df_tm = pd.read_csv(TM_PLAYERS_PATH)
    df_preds = pd.read_csv(TM_PREDS_PATH)
    
    # Load Kaggle with specific settings for the format observed
    df_kaggle = pd.read_csv(KAGGLE_PATH, sep=';', decimal=',', encoding='utf-8', encoding_errors='ignore')
    
    # Merge TM basic data with predictions
    cols_to_use = df_preds.columns.difference(df_tm.columns).tolist() + ['player_id']
    df_pool = df_tm.merge(df_preds[cols_to_use], on='player_id', how='left')
    
    # Normalize names and teams
    print("Normalizing strings...")
    if 'name' not in df_pool.columns:
        df_pool['name'] = df_pool['first_name'].fillna('') + ' ' + df_pool['last_name'].fillna('')
    
    df_pool['norm_name'] = df_pool['name'].apply(normalize_string)
    df_pool['norm_club'] = df_pool['current_club_name'].apply(normalize_string)
    
    df_kaggle['norm_player'] = df_kaggle['player'].apply(normalize_string)
    df_kaggle['norm_team'] = df_kaggle['team'].apply(normalize_string)
    
    # Season alignment
    df_pool['valuation_season'] = df_pool['valuation_date'].apply(get_season_from_date)
    df_pool['match_season'] = df_pool['valuation_season'].apply(get_previous_season)
    
    # League Mapping
    comp_to_league = {
        'GB1': 'ENG-Premier League',
        'ES1': 'ESP-La Liga',
        'L1': 'GER-Bundesliga',
        'IT1': 'ITA-Serie A',
        'FR1': 'FRA-Ligue 1'
    }
    df_pool['target_league'] = df_pool['current_club_domestic_competition_id'].map(comp_to_league)
    
    # Pre-index Kaggle
    kaggle_sl_dict = {}
    for idx, row in df_kaggle.iterrows():
        key = (str(row['season']), row['league'])
        if key not in kaggle_sl_dict:
            kaggle_sl_dict[key] = []
        kaggle_sl_dict[key].append(idx)

    kaggle_ns_dict = {}
    for idx, row in df_kaggle.iterrows():
        key = (row['norm_player'], str(row['season']))
        if key not in kaggle_ns_dict:
            kaggle_ns_dict[key] = []
        kaggle_ns_dict[key].append(idx)

    print("Starting multi-stage matching with validation...")
    audit_results = []
    
    for idx, row in df_pool.iterrows():
        tm_id = row['player_id']
        tm_name = row['name']
        tm_norm_name = row['norm_name']
        tm_club = row['current_club_name']
        tm_norm_club = row['norm_club']
        tm_pos = row['position_group_raw']
        tm_season = row['match_season']
        tm_val_season = row['valuation_season']
        tm_league = row['target_league']
        
        match_status = 'unmatched'
        match_method = 'none'
        match_score = 0
        found_idx = None
        pos_compatible = False
        club_match = False
        team_match_score_value = 0
        season_aligned = False
        is_future_season_used = False
        review_required = False
        is_ambiguous = False
        match_rejection_reason = ""
        
        if not tm_season:
            audit_results.append({
                'player_id': tm_id,
                'tm_name': tm_name,
                'tm_club': tm_club,
                'tm_position': tm_pos,
                'tm_season': tm_season,
                'target_league': tm_league,
                'match_status': 'unmatched',
                'match_method': 'no_season',
                'match_score': 0,
                'duplicate_conflict': False,
                'duplicate_resolution': '',
                'match_rejection_reason': 'no_season'
            })
            continue

        # Stage 1: Exact Name + Season
        key = (tm_norm_name, tm_season)
        if key in kaggle_ns_dict:
            candidates = kaggle_ns_dict[key]
            
            best_team_score = -1
            best_c_idx = None
            
            for c_idx in candidates:
                c_team = df_kaggle.loc[c_idx, 'norm_team']
                t_score = team_similarity_score(tm_norm_club, c_team)
                if t_score > best_team_score:
                    best_team_score = t_score
                    best_c_idx = c_idx
            
            if best_team_score >= 80:
                found_idx = best_c_idx
                match_method = 'exact_name_team_season'
                match_score = 100
            elif len(candidates) == 1 and best_team_score >= 50:
                # Allow some leeway if name is unique in that season
                found_idx = best_c_idx
                match_method = 'exact_name_team_season'
                match_score = 90
            else:
                is_ambiguous = True
                # Don't auto-match if team is very different


        # Stage 2: Fuzzy Matching within Season + League
        if found_idx is None and tm_league:
            sl_key = (tm_season, tm_league)
            if sl_key in kaggle_sl_dict:
                sl_indices = kaggle_sl_dict[sl_key]
                names_to_match = df_kaggle.loc[sl_indices, 'norm_player'].tolist()
                
                # Get candidates
                fuzzy_results = process.extractBests(tm_norm_name, names_to_match, scorer=fuzz.ratio, limit=2)
                
                if fuzzy_results:
                    best_match, score = fuzzy_results[0][0], fuzzy_results[0][1]
                    best_idx = sl_indices[names_to_match.index(best_match)]
                    
                    if len(fuzzy_results) > 1:
                        if score - fuzzy_results[1][1] < 3:
                            is_ambiguous = True
                    
                    if score >= FUZZY_THRESHOLD_AUTO:
                        found_idx = best_idx
                        match_method = 'fuzzy_high_confidence'
                        match_score = score
                    elif score >= FUZZY_THRESHOLD_REVIEW:
                        found_idx = best_idx
                        match_method = 'fuzzy_review'
                        match_score = score
                        review_required = True

        # Post-match Validation
        if found_idx is not None:
            k_row = df_kaggle.loc[found_idx]
            k_pos = k_row['pos_']
            k_team = k_row['team']
            k_norm_team = k_row['norm_team']
            k_season = str(k_row['season'])
            
            pos_compatible = is_position_compatible(tm_pos, k_pos)
            team_match_score_value = team_similarity_score(tm_norm_club, k_norm_team)
            club_match = team_match_score_value >= TEAM_MATCH_THRESHOLD
            season_aligned = (k_season == tm_season)
            
            # Check for future leakage
            try:
                tm_val_year = int(tm_val_season[:2])
                k_year = int(k_season[:2])
                if k_year >= tm_val_year:
                    is_future_season_used = True
            except:
                pass

            # Final Acceptance Logic
            if is_future_season_used:
                match_status = 'unmatched'
                match_rejection_reason = append_rejection_reason(match_rejection_reason, "future_season_used")
            elif not pos_compatible:
                match_status = 'review_required'
                review_required = True
                match_rejection_reason = append_rejection_reason(match_rejection_reason, "position_mismatch")
            elif match_score >= FUZZY_THRESHOLD_AUTO:
                match_status = 'matched'
            elif match_score >= FUZZY_THRESHOLD_COMPATIBLE and pos_compatible and club_match:
                match_status = 'matched'
            elif review_required:
                match_status = 'review_required'
                match_rejection_reason = append_rejection_reason(match_rejection_reason, "fuzzy_review")
            else:
                match_status = 'unmatched'

            audit_results.append({
                'player_id': tm_id,
                'tm_name': tm_name,
                'tm_club': tm_club,
                'tm_position': tm_pos,
                'tm_season': tm_season,
                'target_league': tm_league,
                'kaggle_player': k_row['player'],
                'kaggle_squad': k_team,
                'kaggle_position': k_pos,
                'kaggle_season': k_season,
                'match_status': match_status,
                'match_method': match_method,
                'match_score': match_score,
                'season_aligned': season_aligned,
                'club_match': club_match,
                'team_match_score': team_match_score_value,
                'position_compatible': pos_compatible,
                'age_diff': abs(row['age_at_valuation'] - k_row['age_']),
                'is_ambiguous': is_ambiguous,
                'review_required': review_required,
                'is_future_season_used': is_future_season_used,
                'kaggle_index': found_idx,
                'duplicate_conflict': False,
                'duplicate_resolution': '',
                'match_rejection_reason': match_rejection_reason
            })
        else:
            audit_results.append({
                'player_id': tm_id,
                'tm_name': tm_name,
                'tm_club': tm_club,
                'tm_position': tm_pos,
                'tm_season': tm_season,
                'target_league': tm_league,
                'match_status': 'unmatched',
                'match_method': 'none',
                'match_score': 0,
                'duplicate_conflict': False,
                'duplicate_resolution': '',
                'match_rejection_reason': ''
            })

    df_audit = pd.DataFrame(audit_results)
    df_audit = apply_release_gates(df_audit)
    
    # Merge stats for 'matched' players
    print("Merging statistical data...")
    results_final = []
    for _, a_row in df_audit.iterrows():
        res = a_row.to_dict()
        if a_row['match_status'] == 'matched':
            k_idx = int(a_row['kaggle_index'])
            k_row = df_kaggle.loc[k_idx]
            for col in df_kaggle.columns:
                if col not in ['player', 'team', 'season', 'norm_player', 'norm_team', 'league', 'pos_', 'age_']:
                    res[f"adv_{col}"] = k_row[col]
        results_final.append(res)
    
    df_final_enriched = pd.DataFrame(results_final)
    
    # Final Pool Merge
    df_output = df_pool.merge(
        df_final_enriched.drop(columns=['tm_name', 'tm_club', 'tm_position', 'tm_season'], errors='ignore'),
        on='player_id',
        how='left'
    )
    
    # Save Outputs
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_output.to_csv(OUTPUT_PATH, index=False)
    df_audit.to_csv(AUDIT_PATH, index=False)
    
    # Generate Report
    print("Generating matching report...")
    df_report = build_matching_report(df_audit)
    df_report.to_csv(REPORT_PATH, index=False)
    
    overall_row = df_report[(df_report["section"] == "overall") & (df_report["segment"] == "all_players")].iloc[0]
    top5_row = df_report[(df_report["section"] == "eligible_top5")].iloc[0]
    league_breakdown = df_report[df_report["section"] == "league"].set_index("segment")["rate"].to_dict()
    print("\nMatch Rate by League:")
    for l, r in league_breakdown.items():
        print(f" - {l}: {r:.1%}")
        
    print(f"\nEnrichment complete. Overall match rate: {overall_row['rate']:.2%}")
    print(f"Eligible Top 5 match rate: {top5_row['rate']:.2%}")
    print(f"Audit saved to {AUDIT_PATH}")
    print(f"Report saved to {REPORT_PATH}")

if __name__ == "__main__":
    main()
