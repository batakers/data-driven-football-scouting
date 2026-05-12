"""
League Tier Mapping  -  Phase 8
==============================
Assigns a competitive tier to each domestic league based on UEFA coefficients,
transfer market activity, and general scouting consensus.

Tier 1  -  Elite (Top 5 European leagues)
Tier 2  -  Strong (Established European leagues with regular CL/EL participants)
Tier 3  -  Competitive (Mid-level European + strong non-European leagues)
Tier 4  -  Developing (Lower European + emerging global leagues)

This module is intentionally standalone so it can be imported by any pipeline
step without circular dependencies.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Core tier mapping  -  competition_id → tier (int)
# ---------------------------------------------------------------------------

LEAGUE_TIER_MAP: dict[str, int] = {
    # ── Tier 1: Elite ──────────────────────────────────────────────────────
    "GB1": 1,   # Premier League (England)
    "ES1": 1,   # La Liga (Spain)
    "L1":  1,   # Bundesliga (Germany)
    "IT1": 1,   # Serie A (Italy)
    "FR1": 1,   # Ligue 1 (France)

    # ── Tier 2: Strong ─────────────────────────────────────────────────────
    "NL1": 2,   # Eredivisie (Netherlands)
    "PO1": 2,   # Liga Portugal (Portugal)
    "BE1": 2,   # Jupiler Pro League (Belgium)
    "TR1": 2,   # Süper Lig (Turkey)
    "A1":  2,   # Austrian Bundesliga
    "SC1": 2,   # Scottish Premiership
    "C1":  2,   # Swiss Super League
    "RU1": 2,   # Russian Premier League
    "UKR1": 2,  # Ukrainian Premier League

    # ── Tier 3: Competitive ────────────────────────────────────────────────
    "GR1": 3,   # Super League Greece
    "DK1": 3,   # Superligaen (Denmark)
    "SE1": 3,   # Allsvenskan (Sweden)
    "NO1": 3,   # Eliteserien (Norway)
    "PL1": 3,   # Ekstraklasa (Poland)
    "RO1": 3,   # Romanian SuperLiga
    "SER1": 3,  # Serbian SuperLiga
    "TS1": 3,   # Czech First League
    "KR1": 3,   # Croatian SuperSport HNL
    "BRA1": 3,  # Brasileirão Série A
    "ARG1": 3,  # Liga Profesional (Argentina)
    "MEX1": 3,  # Liga MX (Mexico)
    "MLS1": 3,  # Major League Soccer (USA)
    "JAP1": 3,  # J1 League (Japan)
    "RSK1": 3,  # K League 1 (South Korea)
    "COL1": 3,  # Liga BetPlay (Colombia)

    # ── Tier 4: Developing ─────────────────────────────────────────────────
    "SA1":  4,  # Saudi Pro League
    "AUS1": 4,  # A-League Men (Australia)
}

TIER_LABELS: dict[int, str] = {
    1: "Tier 1  -  Elite",
    2: "Tier 2  -  Strong",
    3: "Tier 3  -  Competitive",
    4: "Tier 4  -  Developing",
}

TIER_SHORT_LABELS: dict[int, str] = {
    1: "Elite",
    2: "Strong",
    3: "Competitive",
    4: "Developing",
}

# Scout-facing warning messages per tier (used in rationale layer)
TIER_SCOUT_NOTES: dict[int, str] = {
    1: None,  # No warning needed  -  top league
    2: "Performance from a strong European league  -  generally translates well to top-5 competition.",
    3: "Performance from a competitive but lower-tier league  -  verify adaptability to higher competition level.",
    4: "Performance from a developing league  -  significant step-up risk; prioritize video and contextual review.",
}

DEFAULT_TIER = 4  # Unknown leagues treated as developing


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_tier(competition_id: str | None) -> int:
    """Return the numeric tier for a competition_id. Defaults to 4 if unknown."""
    if not competition_id or pd.isna(competition_id):
        return DEFAULT_TIER
    return LEAGUE_TIER_MAP.get(str(competition_id).strip(), DEFAULT_TIER)


def get_tier_label(competition_id: str | None) -> str:
    """Return the full tier label string."""
    return TIER_LABELS[get_tier(competition_id)]


def get_tier_short(competition_id: str | None) -> str:
    """Return the short tier label (e.g. 'Elite', 'Strong')."""
    return TIER_SHORT_LABELS[get_tier(competition_id)]


def get_scout_note(competition_id: str | None) -> str | None:
    """Return the scout-facing note for this league tier, or None for Tier 1."""
    return TIER_SCOUT_NOTES[get_tier(competition_id)]


def enrich_dataframe(
    df: pd.DataFrame,
    competition_col: str = "current_club_domestic_competition_id",
) -> pd.DataFrame:
    """
    Add league tier columns to a DataFrame in-place (returns copy).

    Adds:
        league_tier          int   (1-4)
        league_tier_label    str   ("Tier 1  -  Elite", ...)
        league_tier_short    str   ("Elite", "Strong", ...)

    Parameters
    ----------
    df : pd.DataFrame
        Any DataFrame that contains a competition_id column.
    competition_col : str
        Name of the column holding the competition/league ID.
    """
    out = df.copy()
    if competition_col not in out.columns:
        out["league_tier"] = DEFAULT_TIER
        out["league_tier_label"] = TIER_LABELS[DEFAULT_TIER]
        out["league_tier_short"] = TIER_SHORT_LABELS[DEFAULT_TIER]
        return out

    out["league_tier"] = out[competition_col].apply(get_tier)
    out["league_tier_label"] = out[competition_col].apply(get_tier_label)
    out["league_tier_short"] = out[competition_col].apply(get_tier_short)
    return out


def build_coverage_report(df: pd.DataFrame, competition_col: str = "current_club_domestic_competition_id") -> pd.DataFrame:
    """
    Return a summary DataFrame showing tier distribution in a dataset.
    Useful for validation and audit outputs.
    """
    if competition_col not in df.columns:
        return pd.DataFrame()

    enriched = enrich_dataframe(df, competition_col)
    report = (
        enriched.groupby(["league_tier", "league_tier_label"])
        .agg(player_count=("player_id", "count"))
        .reset_index()
        .sort_values("league_tier")
    )
    total = len(enriched)
    report["coverage_pct"] = (report["player_count"] / total * 100).round(2)
    return report


# ---------------------------------------------------------------------------
# CLI quick-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    test_ids = ["GB1", "NL1", "TR1", "BRA1", "SA1", "XYZ", None]
    print(f"{'competition_id':<20} {'tier':<6} {'label':<25} {'scout_note'}")
    print("-" * 90)
    for cid in test_ids:
        tier = get_tier(cid)
        label = get_tier_label(cid)
        note = get_scout_note(cid) or " - "
        print(f"{str(cid):<20} {tier:<6} {label:<25} {note[:60]}")
