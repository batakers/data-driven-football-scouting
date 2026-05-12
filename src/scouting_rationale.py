"""
Scouting Rationale  -  Phase 10
==============================
Generates personalized, player-specific scouting notes that reference actual
data points rather than generic templates.

Every function in this module produces output that is unique per player.
The rationale layer is intentionally non-technical: it translates model
outputs and player evidence into scout-friendly language.

Design principles:
  - Every sentence must reference at least one concrete data point from the row.
  - No two players with different data should produce identical rationale text.
  - League tier and contract context are woven in when available (Phase 8 & 9).
  - Confidence level reflects data completeness, not model accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(value, default: float = 0.0) -> float:
    try:
        v = float(value)
        return default if np.isnan(v) else v
    except (TypeError, ValueError):
        return default


def _safe_str(value, default: str = "N/A") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    s = str(value).strip()
    return s if s else default


def _pct(ratio: float) -> str:
    """Format a ratio (e.g. 0.45) as a percentage string (e.g. '+45%')."""
    return f"{ratio * 100:+.0f}%"


def _eur(value: float) -> str:
    """Format a euro value compactly (e.g. €4.2M, €850K)."""
    if value >= 1_000_000:
        return f"€{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"€{value / 1_000:.0f}K"
    return f"€{value:.0f}"


# ---------------------------------------------------------------------------
# Classification helpers (used by signals and rationale)
# ---------------------------------------------------------------------------

def normalize_value_gap(value) -> float:
    """Normalize value gap to percentage scale. Handles ratio (0-50) and percent (50+)."""
    if pd.isna(value):
        return 0.0
    v = float(value)
    return v * 100 if v < 50 else v


def classify_value_gap(undervalued_pct) -> str:
    if pd.isna(undervalued_pct):
        return "Unknown value gap"
    gap_pct = normalize_value_gap(undervalued_pct)
    if gap_pct >= 300:
        return "Very large value gap"
    if gap_pct >= 100:
        return "Large value gap"
    if gap_pct >= 50:
        return "Moderate value gap"
    return "Small value gap"


def classify_age_signal(age) -> str:
    if pd.isna(age):
        return "Age unknown"
    if age <= 21:
        return "High development upside"
    if age <= 25:
        return "Within recruitment age window"
    return "Older than default youth-scouting profile"


def classify_minutes_signal(minutes) -> str:
    if pd.isna(minutes):
        return "Minutes unknown"
    if minutes >= 2700:
        return "Starter-level minutes  -  strong sample"
    if minutes >= 1800:
        return "Trusted with significant minutes"
    if minutes >= 900:
        return "Meets scouting minutes threshold"
    return "Limited sample size"


# ---------------------------------------------------------------------------
# Confidence level
# ---------------------------------------------------------------------------

def compute_confidence_level(row: dict | pd.Series) -> str:
    """
    Assess data completeness and return a confidence level.

    High    -  enriched stats + Tier 1-2 + contract known + >1800 mins
    Medium  -  basic stats + any tier + >900 mins
    Low     -  limited data points
    """
    mins = _safe_float(row.get("minutes_last_season", 0))
    tier = _safe_float(row.get("league_tier", 4))
    contract = _safe_str(row.get("contract_status", "Unknown"))
    enriched = str(row.get("enriched_available", "false")).strip().lower() in {"true", "1", "yes"}
    form_trend = _safe_str(row.get("form_trend", "Insufficient Data"))

    score = 0
    if mins >= 1800:
        score += 2
    elif mins >= 900:
        score += 1

    if tier <= 2:
        score += 2
    elif tier == 3:
        score += 1

    if contract not in ("Unknown", ""):
        score += 1

    if enriched:
        score += 2

    # Phase 11: form trend adds confidence when available
    if form_trend in ("Rising", "Stable", "Declining"):
        score += 1

    if score >= 7:
        return "High"
    if score >= 3:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Scouting signals table (evidence table in dashboard)
# ---------------------------------------------------------------------------

def build_scouting_signals(row: dict | pd.Series) -> pd.DataFrame:
    """
    Build a DataFrame of scouting signals with player-specific evidence.
    Each row references a concrete data point from the player record.
    """
    signals = []

    # ── Playing Time ────────────────────────────────────────────────────────
    mins = _safe_float(row.get("minutes_last_season", 0))
    mins_label = classify_minutes_signal(mins)
    benchmark = (
        "Top 10% of shortlist" if mins >= 2700
        else "Above shortlist threshold" if mins >= 1800
        else "Meets minimum threshold"
    )
    signals.append({
        "Signal": "Playing Time",
        "Player Evidence": f"{mins:,.0f} minutes",
        "Benchmark / Context": benchmark,
        "Scout Interpretation": mins_label,
    })

    # ── Age Profile ─────────────────────────────────────────────────────────
    age = _safe_float(row.get("age_at_valuation", 0))
    age_label = classify_age_signal(age)
    age_context = (
        "U21  -  peak development window" if age <= 21
        else "U25  -  prime recruitment target" if age <= 25
        else "Senior profile  -  proven experience"
    )
    signals.append({
        "Signal": "Age Profile",
        "Player Evidence": f"{age:.1f} years old",
        "Benchmark / Context": age_context,
        "Scout Interpretation": age_label,
    })

    # ── Value Gap ───────────────────────────────────────────────────────────
    gap = _safe_float(row.get("undervalued_pct", 0))
    gap_pct = normalize_value_gap(gap)
    gap_label = classify_value_gap(gap)
    current_val = _safe_float(row.get("target_market_value", 0))
    predicted_val = _safe_float(row.get("predicted_value", 0))
    gap_evidence = (
        f"Estimated {_eur(predicted_val)} vs listed {_eur(current_val)}"
        if current_val > 0 and predicted_val > 0
        else f"{gap_pct:+.0f}% above market price"
    )
    signals.append({
        "Signal": "Value Gap",
        "Player Evidence": gap_evidence,
        "Benchmark / Context": f"{gap_pct:+.0f}% above market price",
        "Scout Interpretation": gap_label,
    })

    # ── League Tier ─────────────────────────────────────────────────────────
    tier = _safe_float(row.get("league_tier", 4))
    tier_short = _safe_str(row.get("league_tier_short", "Unknown"))
    league_name = _safe_str(row.get("current_club_domestic_competition_id", ""))
    tier_interp = {
        1: "Performance from an elite league  -  high confidence in data quality",
        2: "Performance from a strong league  -  generally translates well",
        3: "Performance from a competitive league  -  verify step-up adaptability",
        4: "Performance from a developing league  -  significant step-up risk",
    }.get(int(tier), "League context unknown")
    signals.append({
        "Signal": "League Context",
        "Player Evidence": f"{tier_short} ({league_name})" if league_name != "Unknown" else tier_short,
        "Benchmark / Context": f"Tier {int(tier)} of 4",
        "Scout Interpretation": tier_interp,
    })

    # ── Contract ────────────────────────────────────────────────────────────
    contract_status = _safe_str(row.get("contract_status", "Unknown"))
    contract_months = row.get("contract_months_remaining", None)
    if contract_status != "Unknown":
        if contract_months is not None and not pd.isna(contract_months):
            months_val = float(contract_months)
            contract_evidence = (
                f"{months_val:.0f} months remaining"
                if months_val >= 0
                else "Contract appears expired"
            )
        else:
            contract_evidence = contract_status
        contract_interp = {
            "Expiring": "Negotiation window open  -  potential free transfer or reduced fee",
            "Short":    "Club retains leverage but pre-contract approach may be feasible",
            "Medium":   "Transfer fee required  -  monitor for renewal talks",
            "Long":     "Significant fee expected  -  suitable for clubs with budget",
            "Expired":  "May be available on a free  -  verify current status",
        }.get(contract_status, "Contract status unknown")
        signals.append({
            "Signal": "Contract Situation",
            "Player Evidence": contract_evidence,
            "Benchmark / Context": contract_status,
            "Scout Interpretation": contract_interp,
        })

    # ── Phase 11: Form Trend ─────────────────────────────────────────────────
    form_trend = _safe_str(row.get("form_trend", "Insufficient Data"))
    trend_goals = row.get("form_trend_goals", None)
    trend_assists = row.get("form_trend_assists", None)
    trend_mins = row.get("form_trend_minutes", None)

    if form_trend not in ("Insufficient Data", "N/A", ""):
        # Build specific evidence string
        evidence_parts = []
        if trend_goals is not None and not pd.isna(trend_goals):
            direction = "+" if float(trend_goals) >= 0 else ""
            evidence_parts.append(f"goals/90 {direction}{float(trend_goals):.2f}")
        if trend_mins is not None and not pd.isna(trend_mins):
            mins_ratio = float(trend_mins)
            evidence_parts.append(f"minutes ratio {mins_ratio:.2f}x")
        form_evidence = ", ".join(evidence_parts) if evidence_parts else form_trend

        form_interp = {
            "Rising":   "Improving form  -  recent 6 months stronger than earlier 6 months",
            "Stable":   "Consistent form  -  performance level maintained across both windows",
            "Declining":"Declining form  -  recent 6 months weaker than earlier 6 months",
        }.get(form_trend, "Form trend unknown")

        signals.append({
            "Signal": "Form Trend",
            "Player Evidence": form_evidence,
            "Benchmark / Context": f"Recent vs earlier 6-month window",
            "Scout Interpretation": form_interp,
        })
    else:
        signals.append({
            "Signal": "Form Trend",
            "Player Evidence": "Insufficient data",
            "Benchmark / Context": "< 450 mins in one or both windows",
            "Scout Interpretation": "Not enough data in both windows to assess trend",
        })

    # ── Position-specific output ─────────────────────────────────────────────
    pos = _safe_str(row.get("position_group_raw", ""))
    goals = _safe_float(row.get("goals_per_90_ls", 0))
    assists = _safe_float(row.get("assists_per_90_ls", 0))
    cards = _safe_float(row.get("cards_per_90_ls", 0))

    if pos == "Forward":
        if goals >= 0.5:
            signals.append({
                "Signal": "Attacking Output",
                "Player Evidence": f"{goals:.2f} goals / 90",
                "Benchmark / Context": "Elite scoring rate for role",
                "Scout Interpretation": "Elite scoring output  -  top-tier finishing rate",
            })
        elif goals >= 0.3:
            signals.append({
                "Signal": "Attacking Output",
                "Player Evidence": f"{goals:.2f} goals / 90",
                "Benchmark / Context": "Strong for role",
                "Scout Interpretation": "Strong scoring output  -  consistent goal threat",
            })
        if assists >= 0.2:
            signals.append({
                "Signal": "Creative Output",
                "Player Evidence": f"{assists:.2f} assists / 90",
                "Benchmark / Context": "Strong for forward role",
                "Scout Interpretation": "Adds creative value beyond pure finishing",
            })

    elif pos == "Midfielder":
        if assists >= 0.2:
            signals.append({
                "Signal": "Creative Output",
                "Player Evidence": f"{assists:.2f} assists / 90",
                "Benchmark / Context": "Strong for midfield role",
                "Scout Interpretation": "Elite creative output  -  key chance creator",
            })
        elif assists >= 0.12:
            signals.append({
                "Signal": "Creative Output",
                "Player Evidence": f"{assists:.2f} assists / 90",
                "Benchmark / Context": "Above average for role",
                "Scout Interpretation": "Consistent creative contribution",
            })
        if goals >= 0.15:
            signals.append({
                "Signal": "Goal Contribution",
                "Player Evidence": f"{goals:.2f} goals / 90",
                "Benchmark / Context": "High for midfield role",
                "Scout Interpretation": "Box-to-box threat  -  arrives late into scoring positions",
            })

    elif pos == "Defender":
        if cards <= 0.1 and mins >= 900:
            signals.append({
                "Signal": "Disciplinary Record",
                "Player Evidence": f"{cards:.2f} cards / 90",
                "Benchmark / Context": "Clean disciplinary record",
                "Scout Interpretation": "Disciplined defender  -  low suspension risk",
            })
        elif cards >= 0.4:
            signals.append({
                "Signal": "Disciplinary Risk",
                "Player Evidence": f"{cards:.2f} cards / 90",
                "Benchmark / Context": "Above average for role",
                "Scout Interpretation": "Elevated card rate  -  review tackling style on video",
            })

    elif pos == "Goalkeeper":
        signals.append({
            "Signal": "Specialist Role",
            "Player Evidence": "Goalkeeper",
            "Benchmark / Context": "Specialist metrics unavailable",
            "Scout Interpretation": "Advanced GK metrics not available  -  prioritize video review",
        })

    return pd.DataFrame(signals)


# ---------------------------------------------------------------------------
# Scout checks (contextual, not generic)
# ---------------------------------------------------------------------------

def build_scout_checks(row: dict | pd.Series) -> list[str]:
    """
    Generate a list of contextual scout checks based on the player's actual data.
    Each check is specific to this player's situation, not a generic template.
    """
    checks = []

    pos = _safe_str(row.get("position_group_raw", ""))
    mins = _safe_float(row.get("minutes_last_season", 0))
    age = _safe_float(row.get("age_at_valuation", 0))
    tier = int(_safe_float(row.get("league_tier", 4)))
    contract_status = _safe_str(row.get("contract_status", "Unknown"))
    cards = _safe_float(row.get("cards_per_90_ls", 0))
    primary_role = _safe_str(row.get("primary_role", ""))
    enriched = str(row.get("enriched_available", "false")).strip().lower() in {"true", "1", "yes"}

    # ── League strength check ────────────────────────────────────────────────
    if tier >= 3:
        checks.append(
            "⚠️ Verify performance translates to higher competition level  -  "
            f"current league is Tier {tier} (lower-tier context)"
        )
    else:
        checks.append("✅ League context is strong  -  performance data is from a competitive environment")

    # ── Playing time check ───────────────────────────────────────────────────
    if mins < 900:
        checks.append(
            f"⚠️ Limited sample size ({mins:.0f} minutes)  -  "
            "verify whether limited minutes reflect injury, rotation, or form"
        )
    elif mins < 1350:
        checks.append(
            f"⚠️ Moderate sample ({mins:.0f} minutes)  -  "
            "check whether player is a regular starter or rotation option"
        )
    else:
        checks.append(
            f"✅ Sufficient playing time ({mins:.0f} minutes)  -  "
            "per-90 metrics are based on a reliable sample"
        )

    # ── Age-specific check ───────────────────────────────────────────────────
    if age <= 21:
        checks.append(
            f"⚠️ Young player ({age:.1f} years)  -  "
            "verify readiness for senior first-team role; check youth international record"
        )
    elif age > 28:
        checks.append(
            f"⚠️ Player is {age:.1f} years old  -  "
            "assess remaining peak years and resale value potential"
        )

    # ── Contract check ───────────────────────────────────────────────────────
    if contract_status == "Expiring":
        checks.append(
            "⚠️ Contract expiring within 12 months  -  "
            "confirm player's intentions and whether club will negotiate or let them leave"
        )
    elif contract_status == "Expired":
        checks.append(
            "⚠️ Contract appears expired  -  "
            "verify current registration status before approaching"
        )
    elif contract_status == "Unknown":
        checks.append(
            "⚠️ Contract information unavailable  -  "
            "verify contract length and transfer feasibility before approaching"
        )

    # ── Tactical role check ──────────────────────────────────────────────────
    if primary_role and primary_role not in ("N/A", "UNKNOWN", ""):
        checks.append(
            f"⚠️ Confirm tactical deployment as {primary_role} via video  -  "
            "role metadata is from Transfermarkt and may not reflect current usage"
        )
    else:
        checks.append(
            "⚠️ Check tactical role usage on video  -  "
            "role metadata not available for this player"
        )

    # ── Disciplinary check ───────────────────────────────────────────────────
    if cards >= 0.4:
        checks.append(
            f"⚠️ High card rate ({cards:.2f} / 90)  -  "
            "review tackling style and suspension risk before committing"
        )

    # ── Phase 11: Form trend check ───────────────────────────────────────────
    form_trend = _safe_str(row.get("form_trend", "Insufficient Data"))
    if form_trend == "Declining":
        checks.append(
            "⚠️ Form trend is declining  -  recent 6-month output is weaker than the earlier 6 months. "
            "Verify current fitness, tactical role, and whether this is a temporary dip or structural issue."
        )
    elif form_trend == "Rising":
        checks.append(
            "✅ Form trend is rising  -  recent 6-month output is stronger than the earlier 6 months. "
            "Positive momentum supports the scouting case."
        )
    elif form_trend == "Insufficient Data":
        checks.append(
            "⚠️ Form trend could not be calculated  -  insufficient minutes in one or both 6-month windows. "
            "Verify recent playing time and form through video or match reports."
        )

    # ── Data quality check ───────────────────────────────────────────────────
    if not enriched:
        checks.append(
            "⚠️ Advanced stats not available for this player  -  "
            "similarity and profile scores are based on basic metrics only"
        )

    # ── Goalkeeper specialist check ──────────────────────────────────────────
    if pos == "Goalkeeper":
        checks.append(
            "⚠️ Goalkeeper specialist metrics (save %, PSxG) are not available  -  "
            "prioritize video review and match report analysis"
        )

    # ── Always include injury check ──────────────────────────────────────────
    checks.append("⚠️ Review injury history  -  data does not capture time missed through injury")

    return checks


# ---------------------------------------------------------------------------
# Rationale summary (personalized narrative)
# ---------------------------------------------------------------------------

def build_rationale_summary(player_name: str, evidence_df: pd.DataFrame, row: dict | pd.Series | None = None) -> str:
    """
    Generate a personalized rationale paragraph for a player.
    References specific data points from the player record.
    No two players with different data should produce identical text.
    """
    if row is None:
        # Fallback: extract what we can from evidence_df
        return (
            f"**{player_name}** appears undervalued based on recent playing time and "
            "performance metrics. The model estimates a higher value than the current "
            "market price, creating a potential recruitment opportunity worth investigating."
        )

    # Extract key data points
    age = _safe_float(row.get("age_at_valuation", 0))
    mins = _safe_float(row.get("minutes_last_season", 0))
    gap = _safe_float(row.get("undervalued_pct", 0))
    gap_pct = normalize_value_gap(gap)
    pos = _safe_str(row.get("position_group_raw", "player"))
    primary_role = _safe_str(row.get("primary_role", ""))
    current_val = _safe_float(row.get("target_market_value", 0))
    predicted_val = _safe_float(row.get("predicted_value", 0))
    tier = int(_safe_float(row.get("league_tier", 4)))
    tier_short = _safe_str(row.get("league_tier_short", ""))
    contract_status = _safe_str(row.get("contract_status", "Unknown"))
    goals = _safe_float(row.get("goals_per_90_ls", 0))
    assists = _safe_float(row.get("assists_per_90_ls", 0))
    confidence = compute_confidence_level(row)

    # ── Role description ─────────────────────────────────────────────────────
    role_desc = primary_role if primary_role not in ("N/A", "UNKNOWN", "") else pos

    # ── Age framing ──────────────────────────────────────────────────────────
    if age <= 21:
        age_phrase = f"At {age:.0f} years old, {player_name} is in the early stages of senior development"
    elif age <= 25:
        age_phrase = f"At {age:.0f} years old, {player_name} is in the prime recruitment window"
    else:
        age_phrase = f"At {age:.0f} years old, {player_name} brings senior experience"

    # ── Playing time framing ─────────────────────────────────────────────────
    if mins >= 2700:
        mins_phrase = f"logging {mins:,.0f} minutes as a clear starter"
    elif mins >= 1800:
        mins_phrase = f"accumulating {mins:,.0f} minutes of first-team exposure"
    else:
        mins_phrase = f"with {mins:,.0f} minutes of recent playing time"

    # ── Value gap framing ────────────────────────────────────────────────────
    if current_val > 0 and predicted_val > 0:
        gap_phrase = (
            f"the performance-only model estimates a value of {_eur(predicted_val)}, "
            f"compared to a listed market price of {_eur(current_val)}  -  "
            f"a gap of {gap_pct:+.0f}%"
        )
    else:
        gap_phrase = f"the model estimates a value {gap_pct:+.0f}% above the current market price"

    # ── Performance highlight ────────────────────────────────────────────────
    perf_phrase = ""
    if pos == "Forward":
        if goals >= 0.3:
            perf_phrase = f" Attacking output of {goals:.2f} goals per 90 minutes supports the case."
        elif assists >= 0.2:
            perf_phrase = f" Creative contribution of {assists:.2f} assists per 90 adds further value."
    elif pos == "Midfielder":
        if assists >= 0.12:
            perf_phrase = f" Creative output of {assists:.2f} assists per 90 is above average for the role."
        elif goals >= 0.15:
            perf_phrase = f" Goal contribution of {goals:.2f} per 90 suggests a box-to-box profile."
    elif pos == "Defender":
        perf_phrase = " Defensive metrics are based on basic stats  -  video review is recommended."

    # ── League context ───────────────────────────────────────────────────────
    if tier == 1:
        league_phrase = f"Performance is from a {tier_short} league, providing high confidence in the data."
    elif tier == 2:
        league_phrase = f"Performance is from a {tier_short} league  -  generally translates well to top competition."
    elif tier == 3:
        league_phrase = f"Performance is from a {tier_short} league  -  adaptability to higher competition should be verified."
    else:
        league_phrase = f"Performance is from a {tier_short} league  -  significant step-up risk; contextual review is essential."

    # ── Contract framing ─────────────────────────────────────────────────────
    contract_phrase = ""
    if contract_status == "Expiring":
        contract_phrase = " Contract is expiring within 12 months, opening a potential negotiation window."
    elif contract_status == "Expired":
        contract_phrase = " Contract appears expired  -  player may be available on a free transfer."
    elif contract_status == "Short":
        contract_phrase = " Contract has 12–24 months remaining  -  a pre-contract approach may be feasible."

    # ── Phase 11: Form trend framing ─────────────────────────────────────────
    form_trend = _safe_str(row.get("form_trend", "Insufficient Data"))
    trend_goals = row.get("form_trend_goals", None)
    form_phrase = ""
    if form_trend == "Rising":
        if trend_goals is not None and not pd.isna(trend_goals) and float(trend_goals) > 0.05:
            form_phrase = f" Recent form is improving  -  goals/90 up {float(trend_goals):+.2f} vs the previous 6 months."
        else:
            form_phrase = " Recent form is improving  -  more playing time in the last 6 months."
    elif form_trend == "Declining":
        form_phrase = " Note: recent form shows a declining trend  -  verify current fitness and usage."
    elif form_trend == "Stable":
        form_phrase = " Form has been consistent across both halves of the season."

    # ── Confidence note ──────────────────────────────────────────────────────
    confidence_note = {
        "High":   "Data confidence is high  -  enriched stats and strong league context support this assessment.",
        "Medium": "Data confidence is medium  -  assessment is based on standard metrics.",
        "Low":    "Data confidence is low  -  limited data points; treat as an early-stage lead only.",
    }.get(confidence, "")

    # ── Assemble ─────────────────────────────────────────────────────────────
    summary = (
        f"**{player_name}** is a {role_desc} who warrants scouting review. "
        f"{age_phrase}, {mins_phrase}. "
        f"Based on recent statistical output, {gap_phrase}. "
        f"{perf_phrase} "
        f"{league_phrase}"
        f"{contract_phrase}"
        f"{form_phrase} "
        f"{confidence_note}"
    )

    return summary.strip()
