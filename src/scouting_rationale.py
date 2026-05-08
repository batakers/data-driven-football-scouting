import numpy as np
import pandas as pd

def normalize_value_gap(value):
    """Normalize value gap to percentage scale. Handles both ratio (0-50) and percent (50+) formats."""
    if pd.isna(value):
        return 0
    value = float(value)
    # If stored as ratio, convert to percent
    if value < 50:
        return value * 100
    return value

def classify_value_gap(undervalued_pct):
    if pd.isna(undervalued_pct):
        return "Unknown value gap"
    gap_pct = normalize_value_gap(undervalued_pct)
    if gap_pct >= 300:
        return "Very large value gap"
    elif gap_pct >= 100:
        return "Large value gap"
    elif gap_pct >= 50:
        return "Moderate value gap"
    else:
        return "Small value gap"

def classify_age_signal(age):
    if pd.isna(age):
        return "Age unknown"
    if age <= 21:
        return "High development upside"
    elif age <= 25:
        return "Within recruitment age window"
    else:
        return "Older than default youth-scouting profile"

def classify_minutes_signal(minutes):
    if pd.isna(minutes):
        return "Minutes unknown"
    if minutes >= 1800:
        return "Trusted with significant minutes"
    elif minutes >= 900:
        return "Meets scouting minutes threshold"
    else:
        return "Limited sample size"

def build_scouting_signals(row):
    signals = []
    
    # Minutes
    mins = row.get("minutes_last_season", 0)
    mins_msg = classify_minutes_signal(mins)
    signals.append({"Signal": "Playing Time", "Player Evidence": f"{mins:.0f} minutes", "Benchmark / Context": "Above shortlist threshold", "Scout Interpretation": mins_msg})
    
    # Age
    age = row.get("age_at_valuation", 0)
    age_msg = classify_age_signal(age)
    signals.append({"Signal": "Age Profile", "Player Evidence": f"{age:.1f}", "Benchmark / Context": "Within target", "Scout Interpretation": age_msg})
    
    # Gap
    gap = row.get("undervalued_pct", 0)
    gap_msg = classify_value_gap(gap)
    signals.append({"Signal": "Value Gap", "Player Evidence": f"{gap*100:+.0f}%", "Benchmark / Context": "High", "Scout Interpretation": gap_msg})
    
    # Output based on position
    pos = row.get("position_group_raw", "")
    goals = row.get("goals_per_90_ls", 0)
    assists = row.get("assists_per_90_ls", 0)
    
    if pos == "Forward":
        if goals >= 0.3:
            signals.append({"Signal": "Attacking Output", "Player Evidence": f"{goals:.2f} goals / 90", "Benchmark / Context": "Strong for role", "Scout Interpretation": "Strong scoring output"})
        elif assists >= 0.2:
            signals.append({"Signal": "Creative Output", "Player Evidence": f"{assists:.2f} assists / 90", "Benchmark / Context": "Strong for role", "Scout Interpretation": "Adds creative value"})
    elif pos == "Midfielder":
        if assists >= 0.15:
            signals.append({"Signal": "Creative Output", "Player Evidence": f"{assists:.2f} assists / 90", "Benchmark / Context": "Strong for role", "Scout Interpretation": "Strong creative output"})
    return pd.DataFrame(signals)

def build_scout_checks(row):
    return [
        "⚠️ Review league strength and competition level",
        "⚠️ Check tactical role usage on video",
        "⚠️ Review injury history",
        "⚠️ Check contract length and transfer feasibility"
    ]

def build_rationale_summary(player_name, evidence_df):
    return f"**{player_name}** appears undervalued based on recent playing time and performance metrics. The model estimates a higher value than his current market price, creating a potential recruitment opportunity worth investigating further."

