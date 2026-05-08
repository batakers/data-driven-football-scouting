import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.role_mapping import COMPATIBLE_ROLE_THRESHOLD, ROLE_COMPATIBILITY, explicit_role_list
from src.similarity import PlayerSimilarity

setattr(sys.modules["__main__"], "PlayerSimilarity", PlayerSimilarity)


ROLE_DATA_PATH = "data/processed/role_enriched_players.csv"
SIMILARITY_ENGINE_PATH = "outputs/models/similarity_engine.pkl"
REPORT_PATH = "outputs/role_validation_report.json"


def bool_series(df, col):
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    series = df[col]
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype("string").str.strip().str.lower().isin(["true", "1", "yes"])


def validate_role_matrix(errors):
    for source_role, targets in ROLE_COMPATIBILITY.items():
        for target_role, score in targets.items():
            if not 0 <= score <= 1:
                errors.append(f"Role compatibility outside 0-1: {source_role}->{target_role}={score}")
            if source_role == "GK" and target_role != "GK" and score > 0:
                errors.append("GK is compatible with a non-GK role")
            if target_role == "GK" and source_role != "GK" and score > 0:
                errors.append("Non-GK role is compatible with GK")


def validate_engine_modes(errors):
    if not os.path.exists(SIMILARITY_ENGINE_PATH):
        errors.append("Similarity engine not found. Run src/similarity.py before role validation.")
        return {"exact_role_checks": 0, "compatible_role_checks": 0}

    engine = joblib.load(SIMILARITY_ENGINE_PATH)
    data = engine.data.copy()
    role_data = data[
        bool_series(data, "role_metadata_available")
        & data["primary_role"].notna()
        & (data["primary_role"] != "UNKNOWN")
    ]

    exact_checks = 0
    compatible_checks = 0
    for role, group in role_data.groupby("primary_role"):
        if len(group) < 2:
            continue
        target_id = group.iloc[0]["player_id"]

        exact = engine.find_similar(target_id, top_n=20, mode="basic", role_mode="exact")
        if not exact.empty:
            exact_checks += len(exact)
            invalid_exact = exact[
                ~exact.apply(
                    lambda row: role in explicit_role_list(row.get("primary_role"), row.get("role_tags")),
                    axis=1,
                )
            ]
            if not invalid_exact.empty:
                errors.append(f"Exact Role Mode returned candidates without explicit {role} role tag")

        compatible = engine.find_similar(
            target_id,
            top_n=20,
            mode="basic",
            role_mode="compatible",
            role_threshold=COMPATIBLE_ROLE_THRESHOLD,
        )
        if not compatible.empty:
            compatible_checks += len(compatible)
            invalid_scores = compatible[
                ~compatible["role_compatibility_score"].between(COMPATIBLE_ROLE_THRESHOLD, 1)
            ]
            if not invalid_scores.empty:
                errors.append(f"Compatible Role Mode returned candidates below threshold for {role}")

            invalid_gk = compatible[
                ((compatible["primary_role"] == "GK") & (role != "GK"))
                | ((compatible["primary_role"] != "GK") & (role == "GK"))
            ]
            if not invalid_gk.empty:
                errors.append("Compatible Role Mode returned GK/non-GK mismatch")

    if exact_checks == 0:
        errors.append("Exact Role Mode validation did not check any returned candidates")
    if compatible_checks == 0:
        errors.append("Compatible Role Mode validation did not check any returned candidates")

    return {
        "exact_role_checks": int(exact_checks),
        "compatible_role_checks": int(compatible_checks),
    }


def validate():
    print("Starting Role Validation...")
    errors = []

    if not os.path.exists(ROLE_DATA_PATH):
        print(f"Error: {ROLE_DATA_PATH} not found. Run src/enrich_roles.py first.")
        return False

    roles = pd.read_csv(ROLE_DATA_PATH)

    if not roles["player_id"].is_unique:
        errors.append("Duplicate player_id in role-enriched dataset")

    role_available = bool_series(roles, "role_metadata_available")
    valid_boolean_values = roles["role_metadata_available"].astype("string").str.lower().isin(
        ["true", "false", "1", "0", "yes", "no"]
    )
    if not valid_boolean_values.all():
        errors.append("role_metadata_available contains non-boolean values")

    invalid_primary_role = roles[
        role_available
        & (
            roles["primary_role"].isna()
            | (roles["primary_role"].astype("string").str.strip() == "")
            | (roles["primary_role"] == "UNKNOWN")
        )
    ]
    if not invalid_primary_role.empty:
        errors.append(f"Missing primary_role for {len(invalid_primary_role)} rows with role metadata")

    if "foot_role_fit" in roles.columns:
        invalid_foot_fit = roles[~roles["foot_role_fit"].between(0, 1)]
        if not invalid_foot_fit.empty:
            errors.append(f"foot_role_fit outside 0-1 for {len(invalid_foot_fit)} rows")
    else:
        errors.append("foot_role_fit column missing")

    validate_role_matrix(errors)
    engine_checks = validate_engine_modes(errors)

    summary = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "role_enriched_count": int(len(roles)),
        "role_metadata_available_count": int(role_available.sum()),
        "role_metadata_coverage": float(role_available.sum() / len(roles)) if len(roles) else 0.0,
        **engine_checks,
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(summary, f, indent=4)

    if errors:
        print("Role Validation Failed:")
        for error in errors:
            print(f" - {error}")
    else:
        print("All Role Validations Passed!")
        print(f" - Role Metadata Coverage: {summary['role_metadata_coverage']:.2%}")
        print(f" - Exact Role Checks: {summary['exact_role_checks']}")
        print(f" - Compatible Role Checks: {summary['compatible_role_checks']}")

    return not errors


if __name__ == "__main__":
    sys.exit(0 if validate() else 1)
