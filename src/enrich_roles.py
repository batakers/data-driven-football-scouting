import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.role_mapping import (
    compatible_roles,
    foot_role_fit,
    map_position_to_role,
    role_family,
    side_preference,
    unique_roles,
)


PLAYER_BIO_PATH = "data/raw/player_bio.csv"
BASE_PLAYERS_PATH = "data/processed/enriched_similarity_players.csv"
OUTPUT_PATH = "data/processed/role_enriched_players.csv"
AUDIT_PATH = "outputs/role_enrichment_audit.csv"
REPORT_PATH = "outputs/role_enrichment_report.csv"

BIO_COLUMNS = [
    "tmid",
    "player_name",
    "height",
    "foot",
    "main_position",
    "position_group",
    "first_side_position",
    "second_side_position",
    "club",
    "club_id",
    "date_of_birth",
    "age",
]


def normalize_join_key(series):
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype("string")


def stringify_roles(roles):
    return ",".join(roles)


def build_role_features(row):
    primary_role = map_position_to_role(row.get("main_position"))
    roles = unique_roles(
        row.get("main_position"),
        row.get("first_side_position"),
        row.get("second_side_position"),
    )
    secondary_roles = [role for role in roles if role != primary_role]

    metadata_available = primary_role != "UNKNOWN"
    compatible = []
    if metadata_available:
        for role in roles:
            for compatible_role in compatible_roles(role):
                if compatible_role not in compatible:
                    compatible.append(compatible_role)

    return pd.Series(
        {
            "primary_role": primary_role,
            "secondary_roles": stringify_roles(secondary_roles),
            "role_tags": stringify_roles(roles),
            "compatible_roles": stringify_roles(compatible),
            "role_family": role_family(primary_role),
            "side_preference": side_preference(primary_role),
            "foot_role_fit": foot_role_fit(primary_role, row.get("foot")),
            "role_metadata_available": bool(metadata_available),
        }
    )


def build_report(audit):
    total = len(audit)
    matched = int(audit["role_metadata_available"].sum())
    missing_main_position = int(audit["main_position"].isna().sum())
    missing_foot = int(audit["foot"].isna().sum())
    duplicate_tmid = int(audit["tmid"].notna().sum() - audit["tmid"].dropna().nunique())

    rows = [
        {
            "section": "coverage",
            "segment": "total_similarity_players",
            "numerator": total,
            "denominator": total,
            "rate": 1.0 if total else 0.0,
        },
        {
            "section": "coverage",
            "segment": "role_metadata_matched",
            "numerator": matched,
            "denominator": total,
            "rate": matched / total if total else 0.0,
        },
        {
            "section": "coverage",
            "segment": "unmatched_role_metadata",
            "numerator": total - matched,
            "denominator": total,
            "rate": (total - matched) / total if total else 0.0,
        },
        {
            "section": "quality",
            "segment": "duplicate_tmid",
            "numerator": duplicate_tmid,
            "denominator": total,
            "rate": duplicate_tmid / total if total else 0.0,
        },
        {
            "section": "quality",
            "segment": "missing_main_position",
            "numerator": missing_main_position,
            "denominator": total,
            "rate": missing_main_position / total if total else 0.0,
        },
        {
            "section": "quality",
            "segment": "missing_foot",
            "numerator": missing_foot,
            "denominator": total,
            "rate": missing_foot / total if total else 0.0,
        },
    ]

    for section, column in [
        ("role_distribution", "primary_role"),
        ("foot_distribution", "foot"),
        ("position_group_distribution", "position_group"),
    ]:
        for segment, count in audit[column].fillna("UNKNOWN").value_counts().sort_index().items():
            rows.append(
                {
                    "section": section,
                    "segment": segment,
                    "numerator": int(count),
                    "denominator": total,
                    "rate": int(count) / total if total else 0.0,
                }
            )

    return pd.DataFrame(rows)


def main():
    print("Loading v1.3 similarity pool and Transfermarkt role metadata...")
    if not os.path.exists(BASE_PLAYERS_PATH):
        print(f"Error: {BASE_PLAYERS_PATH} not found. Run src/enrich_similarity.py first.")
        return False
    if not os.path.exists(PLAYER_BIO_PATH):
        print(f"Error: {PLAYER_BIO_PATH} not found.")
        return False

    players = pd.read_csv(BASE_PLAYERS_PATH)
    bio = pd.read_csv(PLAYER_BIO_PATH, usecols=BIO_COLUMNS, low_memory=False)

    players["_role_join_key"] = normalize_join_key(players["player_id"])
    bio["_role_join_key"] = normalize_join_key(bio["tmid"])

    duplicate_bio_tmid = bio[bio["_role_join_key"].duplicated(keep=False) & bio["_role_join_key"].notna()]
    if not duplicate_bio_tmid.empty:
        bio = bio.drop_duplicates("_role_join_key", keep="first").copy()

    bio_roles = pd.concat([bio, bio.apply(build_role_features, axis=1)], axis=1)

    merged = players.merge(
        bio_roles,
        on="_role_join_key",
        how="left",
        suffixes=("", "_bio"),
    )

    unmatched_mask = merged["tmid"].isna()
    unknown_role_mask = merged["primary_role"].isna() | (merged["primary_role"] == "UNKNOWN")
    merged["role_metadata_available"] = (~unmatched_mask) & (~unknown_role_mask)
    merged["primary_role"] = merged["primary_role"].fillna("UNKNOWN")
    merged["secondary_roles"] = merged["secondary_roles"].fillna("")
    merged["role_tags"] = merged["role_tags"].fillna("")
    merged["compatible_roles"] = merged["compatible_roles"].fillna("")
    merged["role_family"] = merged["role_family"].fillna("unknown")
    merged["side_preference"] = merged["side_preference"].fillna("unknown")
    merged["foot_role_fit"] = merged["foot_role_fit"].fillna(0.90)

    audit_cols = [
        "player_id",
        "name",
        "tmid",
        "player_name",
        "main_position",
        "position_group",
        "first_side_position",
        "second_side_position",
        "foot",
        "primary_role",
        "secondary_roles",
        "role_tags",
        "compatible_roles",
        "role_family",
        "side_preference",
        "foot_role_fit",
        "role_metadata_available",
    ]
    audit = merged[audit_cols].copy()
    audit["join_status"] = audit["tmid"].notna().map({True: "matched_tmid", False: "missing_tmid"})
    audit.loc[audit["primary_role"] == "UNKNOWN", "join_status"] = "unknown_role"

    output = merged.drop(columns=["_role_join_key"], errors="ignore")

    Path(os.path.dirname(OUTPUT_PATH)).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(AUDIT_PATH)).mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    audit.to_csv(AUDIT_PATH, index=False)
    build_report(audit).to_csv(REPORT_PATH, index=False)

    matched_count = int(output["role_metadata_available"].sum())
    print(f"Role enrichment complete. Coverage: {matched_count / len(output):.2%} ({matched_count}/{len(output)})")
    print(f"Output saved to {OUTPUT_PATH}")
    print(f"Audit saved to {AUDIT_PATH}")
    print(f"Report saved to {REPORT_PATH}")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
