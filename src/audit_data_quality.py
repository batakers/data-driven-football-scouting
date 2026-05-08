import json
import os
import sys
from pathlib import Path

import pandas as pd


REPORT_PATH = Path("outputs/data_quality_audit_report.json")
ISSUES_PATH = Path("outputs/data_quality_issues.csv")
CONTEXT_AUDIT_PATH = Path("outputs/player_context_resolution_audit.csv")

DATASETS = {
    "shortlist": Path("outputs/undervalued_candidates_overall.csv"),
    "predictions": Path("outputs/predictions_per_player.csv"),
    "featured_players": Path("data/processed/featured_players.csv"),
    "role_enriched_players": Path("data/processed/role_enriched_players.csv"),
    "clubs": Path("data/raw/clubs.csv"),
    "competitions": Path("data/raw/competitions.csv"),
    "player_valuations": Path("data/raw/player_valuations.csv"),
    "appearances": Path("data/raw/appearances.csv"),
    "validation_report": Path("outputs/validation_report.json"),
    "role_validation_report": Path("outputs/role_validation_report.json"),
}

REQUIRED_COLUMNS = {
    "shortlist": [
        "player_id",
        "name",
        "position_group_raw",
        "age_at_valuation",
        "current_club_id",
        "minutes_last_season",
        "target_market_value",
        "predicted_value",
        "undervalued_pct",
    ],
    "predictions": [
        "player_id",
        "name",
        "position_group_raw",
        "age_at_valuation",
        "current_club_id",
        "minutes_last_season",
        "target_market_value",
        "predicted_value",
        "undervalued_pct",
    ],
    "featured_players": [
        "player_id",
        "name",
        "current_club_id",
        "current_club_name",
        "current_club_domestic_competition_id",
        "age_at_valuation",
        "position_group_raw",
        "minutes_last_season",
    ],
    "role_enriched_players": [
        "player_id",
        "primary_role",
        "role_tags",
        "foot",
        "role_metadata_available",
    ],
    "clubs": ["club_id", "name", "domestic_competition_id"],
    "competitions": ["competition_id", "name", "type"],
    "player_valuations": [
        "player_id",
        "date",
        "current_club_name",
        "current_club_id",
        "player_club_domestic_competition_id",
    ],
    "appearances": ["player_id", "date", "competition_id"],
}


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def missing_mask(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype("string").str.strip() == "")


def normalize_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype("string")


def sample_values(df: pd.DataFrame, cols: list[str], limit: int = 8) -> str:
    if df.empty:
        return ""
    available = [col for col in cols if col in df.columns]
    if not available:
        return ""
    rows = df[available].head(limit).astype("string").fillna("").to_dict("records")
    return json.dumps(rows, ensure_ascii=True)


def add_issue(
    issues: list[dict],
    section: str,
    severity: str,
    check: str,
    message: str,
    affected_rows: int,
    sample: str = "",
    recommendation: str = "",
) -> None:
    issues.append(
        {
            "section": section,
            "severity": severity,
            "check": check,
            "message": message,
            "affected_rows": int(affected_rows),
            "sample": sample,
            "recommendation": recommendation,
        }
    )


def combine_first_with_source(
    df: pd.DataFrame,
    target_col: str,
    source_col: str,
    source_name: str,
    source_marker_col: str,
) -> pd.DataFrame:
    if source_col not in df.columns:
        return df
    if target_col not in df.columns:
        df[target_col] = pd.NA
    if source_marker_col not in df.columns:
        df[source_marker_col] = pd.NA

    target_missing = missing_mask(df[target_col])
    source_available = ~missing_mask(df[source_col])
    fill_mask = target_missing & source_available
    df.loc[fill_mask, target_col] = df.loc[fill_mask, source_col]
    df.loc[fill_mask, source_marker_col] = source_name
    return df


def load_json_status(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    try:
        with path.open("r", encoding="utf-8") as f:
            return str(json.load(f).get("status", "UNKNOWN"))
    except (json.JSONDecodeError, OSError):
        return "INVALID_JSON"


def audit_required_columns(datasets: dict[str, pd.DataFrame], issues: list[dict]) -> dict:
    summary = {}
    for name, path in DATASETS.items():
        if path.suffix == ".json":
            summary[name] = {"path": str(path), "exists": path.exists()}
            continue

        exists = path.exists()
        df = datasets.get(name, pd.DataFrame())
        summary[name] = {
            "path": str(path),
            "exists": bool(exists),
            "rows": int(len(df)) if exists else 0,
            "columns": int(len(df.columns)) if exists else 0,
        }

        if not exists:
            add_issue(
                issues,
                "dataset_availability",
                "FAIL",
                f"{name}_exists",
                f"Required dataset is missing: {path}",
                1,
                recommendation="Regenerate the pipeline output or restore the raw data file.",
            )
            continue

        required = REQUIRED_COLUMNS.get(name, [])
        missing_cols = [col for col in required if col not in df.columns]
        if missing_cols:
            add_issue(
                issues,
                "schema",
                "FAIL",
                f"{name}_required_columns",
                f"{name} is missing required columns: {', '.join(missing_cols)}",
                len(missing_cols),
                recommendation="Update the upstream pipeline or dashboard contract before release.",
            )
    return summary


def audit_unique_keys(datasets: dict[str, pd.DataFrame], issues: list[dict]) -> None:
    for name in ["shortlist", "predictions", "featured_players", "role_enriched_players"]:
        df = datasets.get(name, pd.DataFrame())
        if df.empty or "player_id" not in df.columns:
            continue
        duplicated = df[df["player_id"].duplicated(keep=False)]
        if not duplicated.empty:
            add_issue(
                issues,
                "entity_keys",
                "FAIL",
                f"{name}_player_id_unique",
                f"{name} contains duplicate player_id values.",
                len(duplicated),
                sample_values(duplicated, ["player_id", "name", "primary_role"]),
                "Final player-level outputs should be one row per player_id.",
            )


def audit_pool_relationships(datasets: dict[str, pd.DataFrame], issues: list[dict]) -> None:
    shortlist = datasets.get("shortlist", pd.DataFrame())
    predictions = datasets.get("predictions", pd.DataFrame())
    featured = datasets.get("featured_players", pd.DataFrame())
    roles = datasets.get("role_enriched_players", pd.DataFrame())

    if not shortlist.empty and not predictions.empty and {"player_id"}.issubset(shortlist.columns) and {"player_id"}.issubset(predictions.columns):
        missing_pred = shortlist[~shortlist["player_id"].isin(predictions["player_id"])]
        if not missing_pred.empty:
            add_issue(
                issues,
                "pool_consistency",
                "WARN",
                "shortlist_subset_of_predictions",
                "Some shortlist players are not present in predictions_per_player.csv.",
                len(missing_pred),
                sample_values(missing_pred, ["player_id", "name"]),
                "Check whether shortlist and prediction outputs were generated from the same run.",
            )

    if not predictions.empty and not featured.empty and {"player_id"}.issubset(predictions.columns) and {"player_id"}.issubset(featured.columns):
        missing_featured = predictions[~predictions["player_id"].isin(featured["player_id"])]
        if not missing_featured.empty:
            add_issue(
                issues,
                "pool_consistency",
                "WARN",
                "predictions_have_featured_rows",
                "Some modelled players are not present in featured_players.csv.",
                len(missing_featured),
                sample_values(missing_featured, ["player_id", "name"]),
                "Rebuild featured_players or inspect dropped raw player rows.",
            )

    if not shortlist.empty and not roles.empty and {"player_id"}.issubset(shortlist.columns) and {"player_id"}.issubset(roles.columns):
        missing_roles = shortlist[~shortlist["player_id"].isin(roles["player_id"])]
        if not missing_roles.empty:
            add_issue(
                issues,
                "role_context",
                "WARN",
                "shortlist_has_role_rows",
                "Some shortlist players are not present in role_enriched_players.csv.",
                len(missing_roles),
                sample_values(missing_roles, ["player_id", "name"]),
                "Run src/enrich_roles.py after regenerating the model outputs.",
            )


def latest_by_player(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if df.empty or "player_id" not in df.columns or date_col not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    return (
        out.sort_values(date_col)
        .dropna(subset=["player_id"])
        .drop_duplicates("player_id", keep="last")
    )


def build_context_resolution(
    base_df: pd.DataFrame,
    source_name: str,
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if base_df.empty or "player_id" not in base_df.columns:
        return pd.DataFrame()

    base_cols = [
        "player_id",
        "name",
        "current_club_id",
        "current_club_name",
        "current_club_domestic_competition_id",
    ]
    ctx = base_df[[col for col in base_cols if col in base_df.columns]].copy()
    ctx = ctx.drop_duplicates("player_id")
    ctx["source_dataset"] = source_name

    if "current_club_name" not in ctx.columns:
        ctx["current_club_name"] = pd.NA
    if "current_club_domestic_competition_id" not in ctx.columns:
        ctx["current_club_domestic_competition_id"] = pd.NA
    if "current_club_id" not in ctx.columns:
        ctx["current_club_id"] = pd.NA

    ctx["club_source"] = pd.NA
    ctx.loc[~missing_mask(ctx["current_club_name"]), "club_source"] = source_name
    ctx["league_source"] = pd.NA
    ctx.loc[~missing_mask(ctx["current_club_domestic_competition_id"]), "league_source"] = source_name

    featured = datasets.get("featured_players", pd.DataFrame())
    if not featured.empty and "player_id" in featured.columns:
        featured_cols = [
            "player_id",
            "current_club_name",
            "current_club_domestic_competition_id",
            "current_club_id",
        ]
        feat = featured[[col for col in featured_cols if col in featured.columns]].drop_duplicates("player_id")
        feat = feat.rename(
            columns={
                "current_club_name": "featured_club_name",
                "current_club_domestic_competition_id": "featured_competition_id",
                "current_club_id": "featured_club_id",
            }
        )
        ctx = ctx.merge(feat, on="player_id", how="left")
        ctx = combine_first_with_source(ctx, "current_club_name", "featured_club_name", "featured_players", "club_source")
        ctx = combine_first_with_source(ctx, "current_club_domestic_competition_id", "featured_competition_id", "featured_players", "league_source")
        ctx = combine_first_with_source(ctx, "current_club_id", "featured_club_id", "featured_players", "club_id_source")

    clubs = datasets.get("clubs", pd.DataFrame())
    if not clubs.empty and {"club_id", "name"}.issubset(clubs.columns):
        clubs_lookup = clubs.copy()
        clubs_lookup["club_id_key"] = normalize_id(clubs_lookup["club_id"])
        club_cols = ["club_id_key", "name", "domestic_competition_id"]
        clubs_lookup = clubs_lookup[[col for col in club_cols if col in clubs_lookup.columns]].drop_duplicates("club_id_key")
        clubs_lookup = clubs_lookup.rename(
            columns={
                "name": "club_lookup_name",
                "domestic_competition_id": "club_lookup_competition_id",
            }
        )
        ctx["current_club_id_key"] = normalize_id(ctx["current_club_id"])
        ctx = ctx.merge(clubs_lookup, left_on="current_club_id_key", right_on="club_id_key", how="left")
        ctx = combine_first_with_source(ctx, "current_club_name", "club_lookup_name", "clubs", "club_source")
        ctx = combine_first_with_source(ctx, "current_club_domestic_competition_id", "club_lookup_competition_id", "clubs", "league_source")

    valuations = datasets.get("player_valuations", pd.DataFrame())
    if not valuations.empty:
        latest_val = latest_by_player(valuations, "date")
        val_cols = [
            "player_id",
            "current_club_name",
            "current_club_id",
            "player_club_domestic_competition_id",
        ]
        latest_val = latest_val[[col for col in val_cols if col in latest_val.columns]]
        latest_val = latest_val.rename(
            columns={
                "current_club_name": "valuation_club_name",
                "current_club_id": "valuation_club_id",
                "player_club_domestic_competition_id": "valuation_competition_id",
            }
        )
        ctx = ctx.merge(latest_val, on="player_id", how="left")
        ctx = combine_first_with_source(ctx, "current_club_name", "valuation_club_name", "player_valuations", "club_source")
        ctx = combine_first_with_source(ctx, "current_club_id", "valuation_club_id", "player_valuations", "club_id_source")
        ctx = combine_first_with_source(ctx, "current_club_domestic_competition_id", "valuation_competition_id", "player_valuations", "league_source")

    competitions = datasets.get("competitions", pd.DataFrame())
    domestic_competition_ids = set()
    competition_names = pd.DataFrame()
    if not competitions.empty and {"competition_id", "name"}.issubset(competitions.columns):
        if "type" in competitions.columns:
            domestic = competitions[competitions["type"].eq("domestic_league")].copy()
        else:
            domestic = competitions.copy()
        domestic_competition_ids = set(domestic["competition_id"].astype(str))
        competition_names = domestic[["competition_id", "name"]].drop_duplicates("competition_id")
        competition_names = competition_names.rename(columns={"name": "resolved_league_name"})

    appearances = datasets.get("appearances", pd.DataFrame())
    if not appearances.empty and {"player_id", "date", "competition_id"}.issubset(appearances.columns):
        app = appearances.copy()
        if domestic_competition_ids:
            app = app[app["competition_id"].astype(str).isin(domestic_competition_ids)].copy()
        latest_app = latest_by_player(app, "date")
        latest_app = latest_app[["player_id", "competition_id"]].rename(
            columns={"competition_id": "appearance_competition_id"}
        )
        ctx = ctx.merge(latest_app, on="player_id", how="left")
        ctx = combine_first_with_source(ctx, "current_club_domestic_competition_id", "appearance_competition_id", "appearances_domestic", "league_source")

    if not competition_names.empty:
        ctx = ctx.merge(
            competition_names,
            left_on="current_club_domestic_competition_id",
            right_on="competition_id",
            how="left",
        )
    else:
        ctx["resolved_league_name"] = pd.NA

    ctx["club_resolved"] = ~missing_mask(ctx["current_club_name"])
    ctx["league_resolved"] = ~missing_mask(ctx["current_club_domestic_competition_id"])
    ctx["context_status"] = "complete"
    ctx.loc[~ctx["club_resolved"] & ctx["league_resolved"], "context_status"] = "missing_club"
    ctx.loc[ctx["club_resolved"] & ~ctx["league_resolved"], "context_status"] = "missing_league"
    ctx.loc[~ctx["club_resolved"] & ~ctx["league_resolved"], "context_status"] = "missing_club_and_league"

    out_cols = [
        "source_dataset",
        "player_id",
        "name",
        "current_club_id",
        "current_club_name",
        "current_club_domestic_competition_id",
        "resolved_league_name",
        "club_source",
        "league_source",
        "context_status",
    ]
    return ctx[[col for col in out_cols if col in ctx.columns]].copy()


def audit_context_resolution(context_audit: pd.DataFrame, issues: list[dict]) -> dict:
    if context_audit.empty:
        add_issue(
            issues,
            "club_league_context",
            "FAIL",
            "context_audit_generated",
            "Could not generate player context resolution audit.",
            1,
            recommendation="Check player_id availability in shortlist and predictions outputs.",
        )
        return {}

    summary = {}
    for source, group in context_audit.groupby("source_dataset"):
        missing = group[group["context_status"] != "complete"]
        summary[source] = {
            "players": int(len(group)),
            "complete_context": int((group["context_status"] == "complete").sum()),
            "missing_context": int(len(missing)),
            "context_coverage": float((group["context_status"] == "complete").mean()) if len(group) else 0.0,
        }
        if not missing.empty:
            add_issue(
                issues,
                "club_league_context",
                "WARN",
                f"{source}_club_league_context_complete",
                f"{source} has players with unresolved Club / League context after all fallbacks.",
                len(missing),
                sample_values(missing, ["player_id", "name", "current_club_id", "context_status"]),
                "Inspect raw Transfermarkt club ids, valuation history, and domestic appearances for these players.",
            )
    return summary


def audit_orphan_club_ids(datasets: dict[str, pd.DataFrame], issues: list[dict]) -> None:
    clubs = datasets.get("clubs", pd.DataFrame())
    if clubs.empty or "club_id" not in clubs.columns:
        return
    known = set(normalize_id(clubs["club_id"]).dropna())

    for name in ["shortlist", "predictions", "featured_players"]:
        df = datasets.get(name, pd.DataFrame())
        if df.empty or "current_club_id" not in df.columns:
            continue
        ids = normalize_id(df["current_club_id"])
        missing = df[ids.notna() & ~ids.isin(known)]
        if not missing.empty:
            add_issue(
                issues,
                "club_league_context",
                "WARN",
                f"{name}_current_club_id_in_clubs",
                f"{name} contains current_club_id values not found in clubs.csv.",
                len(missing),
                sample_values(missing, ["player_id", "name", "current_club_id"]),
                "Use valuation and appearance fallbacks, or refresh clubs.csv so club ids resolve directly.",
            )


def audit_value_ranges(datasets: dict[str, pd.DataFrame], issues: list[dict]) -> None:
    for name in ["shortlist", "predictions", "featured_players"]:
        df = datasets.get(name, pd.DataFrame())
        if df.empty:
            continue

        if "age_at_valuation" in df.columns:
            age = pd.to_numeric(df["age_at_valuation"], errors="coerce")
            bad_age = df[age.isna() | (age < 15) | (age > 45)]
            if not bad_age.empty:
                add_issue(
                    issues,
                    "value_ranges",
                    "WARN",
                    f"{name}_age_reasonable",
                    f"{name} has age_at_valuation outside the expected 15-45 range or missing.",
                    len(bad_age),
                    sample_values(bad_age, ["player_id", "name", "age_at_valuation"]),
                    "Check date_of_birth and valuation_date parsing.",
                )

        if "minutes_last_season" in df.columns:
            minutes = pd.to_numeric(df["minutes_last_season"], errors="coerce")
            bad_minutes = df[minutes.isna() | (minutes < 0) | (minutes > 6000)]
            if not bad_minutes.empty:
                add_issue(
                    issues,
                    "value_ranges",
                    "WARN",
                    f"{name}_minutes_reasonable",
                    f"{name} has minutes_last_season outside the expected 0-6000 range or missing.",
                    len(bad_minutes),
                    sample_values(bad_minutes, ["player_id", "name", "minutes_last_season"]),
                    "Check season aggregation and duplicate match records.",
                )

        for value_col in ["target_market_value", "predicted_value"]:
            if value_col in df.columns:
                values = pd.to_numeric(df[value_col], errors="coerce")
                bad_values = df[values.isna() | (values <= 0)]
                if not bad_values.empty:
                    add_issue(
                        issues,
                        "value_ranges",
                        "WARN",
                        f"{name}_{value_col}_positive",
                        f"{name} has missing or non-positive {value_col}.",
                        len(bad_values),
                        sample_values(bad_values, ["player_id", "name", value_col]),
                        "Check market value target construction and prediction output generation.",
                    )


def audit_role_metadata(datasets: dict[str, pd.DataFrame], issues: list[dict]) -> dict:
    roles = datasets.get("role_enriched_players", pd.DataFrame())
    if roles.empty:
        return {}

    summary = {"rows": int(len(roles))}
    if "role_metadata_available" in roles.columns:
        available = roles["role_metadata_available"].astype("string").str.lower().isin(["true", "1", "yes"])
        summary["role_metadata_available"] = int(available.sum())
        summary["role_metadata_coverage"] = float(available.mean()) if len(roles) else 0.0

        if "primary_role" in roles.columns:
            missing_role = roles[available & (missing_mask(roles["primary_role"]) | roles["primary_role"].eq("UNKNOWN"))]
            if not missing_role.empty:
                add_issue(
                    issues,
                    "role_context",
                    "FAIL",
                    "available_role_has_primary_role",
                    "Rows with role_metadata_available=True have missing or UNKNOWN primary_role.",
                    len(missing_role),
                    sample_values(missing_role, ["player_id", "name", "primary_role", "role_tags"]),
                    "Re-run role enrichment or update Transfermarkt position mapping.",
                )
    return summary


def audit_validation_reports(issues: list[dict]) -> dict:
    enrichment_status = load_json_status(DATASETS["validation_report"])
    role_status = load_json_status(DATASETS["role_validation_report"])
    statuses = {
        "enrichment_validation_status": enrichment_status,
        "role_validation_status": role_status,
    }
    for label, status in statuses.items():
        if status != "PASS":
            add_issue(
                issues,
                "validation_reports",
                "FAIL" if status in {"FAIL", "INVALID_JSON"} else "WARN",
                label,
                f"{label} is {status}, expected PASS.",
                1,
                recommendation="Run the corresponding validation script before relying on dashboard outputs.",
            )
    return statuses


def run_audit() -> bool:
    print("Starting data quality audit...")
    datasets = {name: load_csv(path) for name, path in DATASETS.items() if path.suffix == ".csv"}
    issues: list[dict] = []

    dataset_summary = audit_required_columns(datasets, issues)
    audit_unique_keys(datasets, issues)
    audit_pool_relationships(datasets, issues)
    audit_orphan_club_ids(datasets, issues)
    audit_value_ranges(datasets, issues)
    role_summary = audit_role_metadata(datasets, issues)
    validation_statuses = audit_validation_reports(issues)

    context_frames = []
    for name in ["shortlist", "predictions"]:
        ctx = build_context_resolution(datasets.get(name, pd.DataFrame()), name, datasets)
        if not ctx.empty:
            context_frames.append(ctx)
    context_audit = pd.concat(context_frames, ignore_index=True) if context_frames else pd.DataFrame()
    context_summary = audit_context_resolution(context_audit, issues)

    fail_count = sum(1 for issue in issues if issue["severity"] == "FAIL")
    warn_count = sum(1 for issue in issues if issue["severity"] == "WARN")
    status = "FAIL" if fail_count else ("WARN" if warn_count else "PASS")

    report = {
        "status": status,
        "fail_count": int(fail_count),
        "warn_count": int(warn_count),
        "issue_count": int(len(issues)),
        "datasets": dataset_summary,
        "club_league_context": context_summary,
        "role_metadata": role_summary,
        "validation_reports": validation_statuses,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    pd.DataFrame(issues).to_csv(ISSUES_PATH, index=False)
    if not context_audit.empty:
        context_audit.to_csv(CONTEXT_AUDIT_PATH, index=False)

    print(f"Data Quality Audit: {status}")
    print(f" - Issues: {len(issues)} ({fail_count} fail, {warn_count} warn)")
    if context_summary:
        for source, summary in context_summary.items():
            print(
                f" - {source} Club/League coverage: "
                f"{summary['context_coverage']:.2%} "
                f"({summary['complete_context']}/{summary['players']})"
            )
    print(f" - Report: {REPORT_PATH}")
    print(f" - Issues: {ISSUES_PATH}")
    if not context_audit.empty:
        print(f" - Context audit: {CONTEXT_AUDIT_PATH}")

    return fail_count == 0


if __name__ == "__main__":
    sys.exit(0 if run_audit() else 1)
