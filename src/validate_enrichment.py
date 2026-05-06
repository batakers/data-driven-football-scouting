import pandas as pd
import os
import json
import sys

# Paths
AUDIT_PATH = "outputs/matching_audit.csv"
ENRICHED_PATH = "data/processed/enriched_similarity_players.csv"
VALIDATION_REPORT_PATH = "outputs/validation_report.json"

def bool_series(df, col):
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    series = df[col]
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype("string").str.strip().str.lower().isin(["true", "1", "yes"])

def build_summary(audit, enriched, errors):
    matched = audit[audit["match_status"] == "matched"].copy()
    eligible = audit[audit["target_league"].notna()] if "target_league" in audit.columns else audit.iloc[0:0]
    eligible_matched = eligible[eligible["match_status"] == "matched"]

    position_compatible = bool_series(audit, "position_compatible")
    future_used = bool_series(audit, "is_future_season_used")
    enriched_available = bool_series(enriched, "enriched_available")

    conflicting_duplicate_rows = 0
    if "kaggle_index" in matched.columns:
        conflicting_indexes = (
            matched[matched["kaggle_index"].notna()]
            .groupby("kaggle_index")["player_id"]
            .nunique()
        )
        conflict_keys = conflicting_indexes[conflicting_indexes > 1].index
        conflicting_duplicate_rows = int(matched["kaggle_index"].isin(conflict_keys).sum())

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "matched_count": int(len(matched)),
        "audit_count": int(len(audit)),
        "overall_match_rate": float(len(matched) / len(audit)) if len(audit) else 0.0,
        "eligible_top5_count": int(len(eligible)),
        "eligible_top5_matched_count": int(len(eligible_matched)),
        "eligible_top5_match_rate": float(len(eligible_matched) / len(eligible)) if len(eligible) else 0.0,
        "future_leakage_accepted": int((future_used & (audit["match_status"] == "matched")).sum()),
        "position_mismatch_accepted": int(((~position_compatible) & (audit["match_status"] == "matched")).sum()),
        "duplicate_conflicting_kaggle_assignment_rows": conflicting_duplicate_rows,
        "fuzzy_review_accepted": int(
            ((audit["match_status"] == "matched") & (audit["match_method"] == "fuzzy_review")).sum()
        ),
        "enriched_available_rows": int(enriched_available.sum()),
    }

def validate():
    print("Starting Enrichment Validation...")
    
    if not os.path.exists(AUDIT_PATH) or not os.path.exists(ENRICHED_PATH):
        print("Error: Audit or Enriched files not found. Run src/enrich_similarity.py first.")
        return False

    audit = pd.read_csv(AUDIT_PATH)
    enriched = pd.read_csv(ENRICHED_PATH)

    errors = []

    # 1. Unique player_id in enriched
    if not enriched["player_id"].is_unique:
        errors.append("Duplicate player_id in enriched dataset")

    # 2. Match Score range
    if not audit["match_score"].between(0, 100).all():
        errors.append("Invalid match scores found (outside 0-100)")

    # 3. Position mismatch check (for auto-accepted matches)
    position_compatible = bool_series(audit, "position_compatible")
    pos_mismatches = audit[(audit["match_status"] == "matched") & (~position_compatible)]
    if not pos_mismatches.empty:
        errors.append(f"Position mismatch accepted in {len(pos_mismatches)} matched players")

    # 4. Future leakage check
    future_leaks = audit[(audit["match_status"] == "matched") & bool_series(audit, "is_future_season_used")]
    if not future_leaks.empty:
        errors.append(f"Future season leakage found in {len(future_leaks)} players")

    # 5. Duplicate Kaggle Match check (one Kaggle row must not feed multiple players)
    matched = audit[audit["match_status"] == "matched"].copy()
    if "kaggle_index" in matched.columns:
        conflict_counts = (
            matched[matched["kaggle_index"].notna()]
            .groupby("kaggle_index")["player_id"]
            .nunique()
        )
        conflict_indexes = conflict_counts[conflict_counts > 1].index
        duplicate_conflicts = matched[matched["kaggle_index"].isin(conflict_indexes)]
    else:
        duplicate_conflicts = matched[
            matched.duplicated(subset=["kaggle_player", "kaggle_squad", "kaggle_season"], keep=False)
        ]
    if not duplicate_conflicts.empty:
        errors.append(
            f"Duplicate conflicting Kaggle row assignment: {len(duplicate_conflicts)} matched rows map to reused Kaggle rows"
        )

    # 6. Fuzzy review rows require manual review and must not be auto-accepted.
    fuzzy_review_accepted = audit[
        (audit["match_status"] == "matched") & (audit["match_method"] == "fuzzy_review")
    ]
    if not fuzzy_review_accepted.empty:
        errors.append(f"Fuzzy review rows accepted without manual review: {len(fuzzy_review_accepted)}")

    # 7. Enriched availability should align with accepted matches only
    if "enriched_available" in enriched.columns:
        enriched_available = bool_series(enriched, "enriched_available")
        status_matched = enriched["match_status"] == "matched"
        if not (enriched_available == status_matched).all():
            errors.append("enriched_available does not align with match_status == matched")

    adv_cols = [col for col in enriched.columns if col.startswith("adv_")]
    if adv_cols:
        non_matched_with_adv = enriched[
            (enriched["match_status"] != "matched") & enriched[adv_cols].notna().any(axis=1)
        ]
        if not non_matched_with_adv.empty:
            errors.append(f"Advanced stats present on {len(non_matched_with_adv)} non-matched rows")

    # 8. Stats Sanity Check
    # Check a few key stats
    sanity_checks = {
        'adv_Playing Time_Min': (0, 4000),
        'adv_Expected_xG': (0, 50),
        'adv_Standard_Sh': (0, 300)
    }
    
    for col, (min_v, max_v) in sanity_checks.items():
        if col in enriched.columns:
            invalid = enriched[enriched[col].notna() & (~enriched[col].between(min_v, max_v))]
            if not invalid.empty:
                errors.append(f"Sanity check failed for {col}: {len(invalid)} values outside ({min_v}, {max_v})")

    # Final Output
    if not errors:
        print("All Enrichment Validations Passed!")
        print(f" - Total Matched: {len(audit[audit['match_status'] == 'matched'])}")
        print(f" - Total Audit Rows: {len(audit)}")
    else:
        print("Enrichment Validation Failed:")
        for err in errors:
            print(f" - {err}")
        
    # Save a small validation summary report
    summary = build_summary(audit, enriched, errors)
    
    os.makedirs(os.path.dirname(VALIDATION_REPORT_PATH), exist_ok=True)
    with open(VALIDATION_REPORT_PATH, "w") as f:
        json.dump(summary, f, indent=4)
        
    return not errors

if __name__ == "__main__":
    sys.exit(0 if validate() else 1)
