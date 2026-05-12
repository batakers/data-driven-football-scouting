"""
Contract Enrichment  -  Phase 9
==============================
Adds contract awareness to the player pool by combining two sources:

  1. players.csv  → contract_expiration_date  (primary, already in pipeline)
  2. player_bio.csv → contract_until           (secondary, ISO datetime format)

For each player the pipeline resolves a single `contract_expiry_date`, then
computes months remaining relative to the player's valuation_date and assigns
a human-readable contract status label.

Contract Status Labels
-----------------------
  "Expiring"    -  < 12 months remaining  (urgent window)
  "Short"       -  12–24 months remaining
  "Medium"      -  24–36 months remaining
  "Long"        -  > 36 months remaining
  "Expired"     -  contract end date is before valuation date
  "Unknown"     -  no contract data available

Scout-facing urgency notes are also generated per status so the rationale
layer can surface them without extra logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTRACT_STATUS_LABELS = {
    "Expiring": "Expiring (<12 months)",
    "Short":    "Short (12–24 months)",
    "Medium":   "Medium (24–36 months)",
    "Long":     "Long (>36 months)",
    "Expired":  "Expired / Out of contract",
    "Unknown":  "Unknown",
}

CONTRACT_SCOUT_NOTES: dict[str, str | None] = {
    "Expiring": "Contract expires within 12 months  -  negotiation window is open. Club may accept lower fee or player could leave on a free.",
    "Short":    "Contract has 12–24 months remaining  -  club still holds leverage but a pre-contract approach may be feasible.",
    "Medium":   "Contract has 24–36 months remaining  -  transfer fee likely required. Monitor for renewal talks.",
    "Long":     "Long-term contract in place  -  significant transfer fee expected. Suitable for clubs with budget.",
    "Expired":  "Contract appears expired  -  player may be available on a free transfer. Verify current status.",
    "Unknown":  None,
}

CONTRACT_URGENCY_BADGE: dict[str, str] = {
    "Expiring": "⏰ Expiring",
    "Short":    "📅 Short",
    "Medium":   "📋 Medium",
    "Long":     "🔒 Long",
    "Expired":  "🆓 Free Agent?",
    "Unknown":  "❓ Unknown",
}

BIO_PATH = Path("data/raw/player_bio.csv")
OUTPUT_REPORT_PATH = Path("outputs/contract_enrichment_report.csv")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _parse_date(series: pd.Series) -> pd.Series:
    """Parse a date series that may contain ISO datetime strings or plain dates.
    Handles mixed timezone offsets (e.g. player_bio contract_until) by
    converting to UTC first, then stripping timezone info.
    """
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    # Strip timezone so all dates are naive (comparable with valuation_date)
    if hasattr(parsed, "dt") and parsed.dt.tz is not None:
        return parsed.dt.tz_localize(None)
    return parsed


def _months_between(start: pd.Series, end: pd.Series) -> pd.Series:
    """Return fractional months from start to end. Negative means end is in the past."""
    delta_days = (end - start).dt.days
    return delta_days / 30.44  # average days per month


def classify_contract_status(months_remaining: float | None) -> str:
    """Classify a months_remaining value into a contract status key."""
    if months_remaining is None or (isinstance(months_remaining, float) and np.isnan(months_remaining)):
        return "Unknown"
    if months_remaining < 0:
        return "Expired"
    if months_remaining < 12:
        return "Expiring"
    if months_remaining < 24:
        return "Short"
    if months_remaining < 36:
        return "Medium"
    return "Long"


def get_contract_scout_note(status: str) -> str | None:
    """Return the scout-facing note for a contract status key."""
    return CONTRACT_SCOUT_NOTES.get(status)


def get_contract_badge(status: str) -> str:
    """Return the urgency badge string for a contract status key."""
    return CONTRACT_URGENCY_BADGE.get(status, "❓ Unknown")


# ---------------------------------------------------------------------------
# Bio contract loader
# ---------------------------------------------------------------------------

def _load_bio_contracts() -> pd.DataFrame:
    """
    Load contract_until from player_bio.csv.
    Returns a DataFrame with columns: [player_id, bio_contract_expiry].
    Deduplicates by keeping the most recent (latest) contract_until per tmid.
    """
    if not BIO_PATH.exists():
        print(f"  Warning: {BIO_PATH} not found  -  skipping bio contract enrichment.")
        return pd.DataFrame(columns=["player_id", "bio_contract_expiry"])

    bio = pd.read_csv(BIO_PATH, usecols=["tmid", "contract_until"], low_memory=False)
    bio = bio.dropna(subset=["tmid", "contract_until"]).copy()
    bio["tmid"] = pd.to_numeric(bio["tmid"], errors="coerce").dropna().astype(int)
    bio["bio_contract_expiry"] = _parse_date(bio["contract_until"])
    bio = bio.dropna(subset=["bio_contract_expiry"])

    # Keep the latest contract_until per player (handles multi-row bio entries)
    bio = (
        bio.sort_values("bio_contract_expiry", ascending=False)
        .drop_duplicates(subset=["tmid"], keep="first")
        .rename(columns={"tmid": "player_id"})
    )[["player_id", "bio_contract_expiry"]]

    return bio


# ---------------------------------------------------------------------------
# Main enrichment function
# ---------------------------------------------------------------------------

def enrich_contract(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add contract awareness columns to a player DataFrame.

    Input requirements:
        - player_id          (int)
        - valuation_date     (str or datetime)  -  used as reference date
        - contract_expiration_date (str or datetime, optional)  -  from players.csv

    Added columns:
        - contract_expiry_date      datetime  resolved contract end date
        - contract_source           str       'players_csv' | 'player_bio' | 'unknown'
        - contract_months_remaining float     months from valuation_date to expiry
        - contract_status           str       Expiring / Short / Medium / Long / Expired / Unknown
        - contract_status_label     str       human-readable label
        - contract_scout_note       str|None  scout-facing urgency note
        - contract_badge            str       emoji badge for UI
    """
    out = df.copy()

    # Parse reference date (valuation_date)
    out["_ref_date"] = _parse_date(out["valuation_date"])

    # Source 1: contract_expiration_date from players.csv (already in pipeline)
    if "contract_expiration_date" in out.columns:
        out["_players_expiry"] = _parse_date(out["contract_expiration_date"])
    else:
        out["_players_expiry"] = pd.NaT

    # Source 2: player_bio.csv contract_until
    bio_contracts = _load_bio_contracts()
    if not bio_contracts.empty:
        out = out.merge(bio_contracts, on="player_id", how="left")
    else:
        out["bio_contract_expiry"] = pd.NaT

    # Resolve: prefer players.csv (more directly tied to the pipeline player record),
    # fall back to player_bio when players.csv is missing
    out["contract_expiry_date"] = out["_players_expiry"].combine_first(out["bio_contract_expiry"])

    # Track source for audit
    has_players = out["_players_expiry"].notna()
    has_bio = out["bio_contract_expiry"].notna()
    out["contract_source"] = "unknown"
    out.loc[has_bio, "contract_source"] = "player_bio"
    out.loc[has_players, "contract_source"] = "players_csv"  # players.csv wins

    # Compute months remaining
    out["contract_months_remaining"] = np.where(
        out["contract_expiry_date"].notna() & out["_ref_date"].notna(),
        _months_between(out["_ref_date"], out["contract_expiry_date"]),
        np.nan,
    )

    # Classify status
    out["contract_status"] = out["contract_months_remaining"].apply(classify_contract_status)
    out["contract_status_label"] = out["contract_status"].map(CONTRACT_STATUS_LABELS)
    out["contract_scout_note"] = out["contract_status"].map(CONTRACT_SCOUT_NOTES)
    out["contract_badge"] = out["contract_status"].map(CONTRACT_URGENCY_BADGE)

    # Drop internal working columns
    out = out.drop(columns=["_ref_date", "_players_expiry", "bio_contract_expiry"], errors="ignore")

    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_contract_enrichment(df: pd.DataFrame) -> bool:
    """Run basic sanity checks on the enriched DataFrame. Returns True if PASS."""
    errors = []

    # No negative months_remaining should be labelled as anything other than Expired/Unknown
    bad_status = df[
        (df["contract_months_remaining"] < 0) &
        (~df["contract_status"].isin(["Expired", "Unknown"]))
    ]
    if not bad_status.empty:
        errors.append(f"  {len(bad_status)} rows have negative months_remaining but wrong status")

    # contract_status must be one of the known keys
    valid_statuses = set(CONTRACT_STATUS_LABELS.keys())
    unknown_statuses = set(df["contract_status"].unique()) - valid_statuses
    if unknown_statuses:
        errors.append(f"  Unknown contract_status values: {unknown_statuses}")

    # contract_source must be one of the known values
    valid_sources = {"players_csv", "player_bio", "unknown"}
    if "contract_source" in df.columns:
        unknown_sources = set(df["contract_source"].unique()) - valid_sources
        if unknown_sources:
            errors.append(f"  Unknown contract_source values: {unknown_sources}")

    if errors:
        print("Contract Enrichment Validation: FAIL")
        for e in errors:
            print(e)
        return False

    print("Contract Enrichment Validation: PASS")
    coverage = df["contract_status"].ne("Unknown").sum()
    total = len(df)
    print(f"  Coverage: {coverage:,} / {total:,} = {coverage/total*100:.1f}%")
    print(f"  Status distribution:")
    for status, count in df["contract_status"].value_counts().items():
        label = CONTRACT_STATUS_LABELS.get(status, status)
        print(f"    {label}: {count:,}")
    return True


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Starting Contract Enrichment (Phase 9)...")

    master_path = Path("data/processed/master_players.csv")
    if not master_path.exists():
        print(f"Error: {master_path} not found. Run src/data_engineering.py first.")
        sys.exit(1)

    df = pd.read_csv(master_path, low_memory=False)
    print(f"  Loaded master_players: {len(df):,} rows")

    df = enrich_contract(df)

    # Validation
    print()
    validate_contract_enrichment(df)

    # Save back to master_players
    df.to_csv(master_path, index=False)
    print(f"\n  Saved enriched master_players to {master_path}")

    # Save coverage report
    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = (
        df.groupby(["contract_status", "contract_status_label", "contract_source"])
        .agg(player_count=("player_id", "count"))
        .reset_index()
        .sort_values(["contract_status", "contract_source"])
    )
    report.to_csv(OUTPUT_REPORT_PATH, index=False)
    print(f"  Coverage report saved to {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    main()
