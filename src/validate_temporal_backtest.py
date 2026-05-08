import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


OUTPUT_DIR = Path("outputs/temporal_validation")
REPORT_PATH = OUTPUT_DIR / "temporal_validation_report.json"
MODEL_PATH = Path("outputs/models/performance_only_model.pkl")

SNAPSHOT_DATES = [
    "2021-07-01",
    "2022-01-01",
    "2022-07-01",
    "2023-01-01",
    "2023-07-01",
    "2024-01-01",
    "2024-07-01",
    "2025-01-01",
]
EXPECTED_SNAPSHOT_DATES = sorted(set(SNAPSHOT_DATES))

REQUIRED_FILES = {
    "snapshots": OUTPUT_DIR / "temporal_backtest_snapshots.csv",
    "candidates": OUTPUT_DIR / "temporal_backtest_candidates.csv",
    "summary": OUTPUT_DIR / "temporal_backtest_summary.csv",
    "timeline_audit": OUTPUT_DIR / "valuation_timeline_audit.csv",
    "by_value_gap": OUTPUT_DIR / "success_rate_by_value_gap_bucket.csv",
    "by_position": OUTPUT_DIR / "success_rate_by_position.csv",
    "by_age_group": OUTPUT_DIR / "success_rate_by_age_group.csv",
    "by_market_value": OUTPUT_DIR / "success_rate_by_market_value_bucket.csv",
}

DATE_COLS = [
    "snapshot_date",
    "feature_window_start",
    "feature_window_end",
    "max_appearance_date_last_365",
    "current_valuation_date",
    "future_valuation_date_6m",
    "future_valuation_date_12m",
]


def add_issue(issues: list[dict], severity: str, check: str, message: str, count: int = 0) -> None:
    issues.append(
        {
            "severity": severity,
            "check": check,
            "message": message,
            "count": int(count),
        }
    )


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
    return df


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def validate_required_files(issues: list[dict]) -> dict[str, pd.DataFrame]:
    datasets = {}
    for name, path in REQUIRED_FILES.items():
        if not path.exists():
            add_issue(issues, "FAIL", "required_file", f"Missing required output: {path}")
            datasets[name] = pd.DataFrame()
        else:
            datasets[name] = load_csv(path)
    return datasets


def validate_model_features(issues: list[dict]) -> list[str]:
    if not MODEL_PATH.exists():
        add_issue(issues, "FAIL", "model_artifact", f"Missing model artifact: {MODEL_PATH}")
        return []

    model = joblib.load(MODEL_PATH)
    features = [str(col) for col in getattr(model, "feature_names_in_", [])]
    if not features:
        add_issue(issues, "FAIL", "model_features", "Model artifact does not expose feature_names_in_.")
        return []

    forbidden_tokens = ["future", "growth", "hit_", "positive_growth", "current_market_value"]
    bad_features = [feature for feature in features if any(token in feature for token in forbidden_tokens)]
    if bad_features:
        add_issue(
            issues,
            "FAIL",
            "future_values_not_features",
            f"Model features include future/outcome/current-value fields: {bad_features}",
            len(bad_features),
        )
    return features


def validate_snapshot_integrity(snapshots: pd.DataFrame, issues: list[dict]) -> None:
    if snapshots.empty:
        add_issue(issues, "FAIL", "snapshots_not_empty", "Snapshot output is empty.")
        return

    required_cols = {
        "player_id",
        "snapshot_date",
        "current_valuation_date",
        "current_market_value",
        "feature_window_end",
        "is_eligible_baseline",
        "is_scouting_lead",
    }
    missing = sorted(required_cols - set(snapshots.columns))
    if missing:
        add_issue(issues, "FAIL", "snapshot_schema", f"Snapshot output missing columns: {missing}", len(missing))
        return

    dupes = snapshots.duplicated(subset=["player_id", "snapshot_date"]).sum()
    if dupes:
        add_issue(issues, "FAIL", "duplicate_player_snapshot", "Duplicate player_id + snapshot_date rows found.", dupes)

    bad_current_dates = (snapshots["current_valuation_date"] > snapshots["snapshot_date"]).sum()
    if bad_current_dates:
        add_issue(
            issues,
            "FAIL",
            "current_valuation_leakage",
            "Rows use a current valuation after the snapshot date.",
            bad_current_dates,
        )

    bad_feature_end = (snapshots["feature_window_end"] > snapshots["snapshot_date"]).sum()
    if bad_feature_end:
        add_issue(
            issues,
            "FAIL",
            "feature_window_leakage",
            "Feature window end exceeds snapshot date.",
            bad_feature_end,
        )

    app_mask = snapshots["max_appearance_date_last_365"].notna()
    bad_app_dates = (snapshots.loc[app_mask, "max_appearance_date_last_365"] > snapshots.loc[app_mask, "snapshot_date"]).sum()
    if bad_app_dates:
        add_issue(
            issues,
            "FAIL",
            "appearance_leakage",
            "Recent appearance feature window includes matches after snapshot date.",
            bad_app_dates,
        )

    non_positive_values = (pd.to_numeric(snapshots["current_market_value"], errors="coerce") <= 0).sum()
    if non_positive_values:
        add_issue(
            issues,
            "FAIL",
            "current_value_positive",
            "Rows contain non-positive current market values.",
            non_positive_values,
        )

    present_snapshots = sorted(snapshots["snapshot_date"].dt.strftime("%Y-%m-%d").dropna().unique().tolist())
    missing_snapshots = sorted(set(EXPECTED_SNAPSHOT_DATES) - set(present_snapshots))
    if missing_snapshots:
        add_issue(
            issues,
            "FAIL",
            "configured_snapshots_present",
            f"Missing configured snapshots: {missing_snapshots}",
            len(missing_snapshots),
        )


def validate_future_windows(df: pd.DataFrame, issues: list[dict]) -> int:
    leakage_violations = 0
    windows = {
        "6m": (150, 240),
        "12m": (300, 450),
    }
    for label, (start_day, end_day) in windows.items():
        date_col = f"future_valuation_date_{label}"
        value_col = f"future_value_{label}"
        if date_col not in df.columns or value_col not in df.columns:
            add_issue(issues, "FAIL", f"future_window_{label}_schema", f"Missing {label} future outcome columns.")
            continue

        mask = df[date_col].notna()
        deltas = (df.loc[mask, date_col] - df.loc[mask, "snapshot_date"]).dt.days
        bad_window = ((deltas < start_day) | (deltas > end_day)).sum()
        bad_order = (deltas <= 0).sum()
        if bad_window:
            add_issue(
                issues,
                "FAIL",
                f"future_window_{label}",
                f"{label} future valuation dates fall outside the configured window.",
                bad_window,
            )
            leakage_violations += int(bad_window)
        if bad_order:
            add_issue(
                issues,
                "FAIL",
                f"future_after_snapshot_{label}",
                f"{label} future valuation dates are not after snapshot date.",
                bad_order,
            )
            leakage_violations += int(bad_order)
    return leakage_violations


def validate_growth_math(df: pd.DataFrame, issues: list[dict]) -> None:
    for label in ["6m", "12m"]:
        future_col = f"future_value_{label}"
        growth_eur_col = f"growth_{label}_eur"
        growth_pct_col = f"growth_{label}_pct"
        positive_col = f"positive_growth_{label}"
        hit_col = f"hit_{label}"

        needed = {future_col, growth_eur_col, growth_pct_col, positive_col, hit_col, "current_market_value"}
        missing = sorted(needed - set(df.columns))
        if missing:
            add_issue(issues, "FAIL", f"growth_math_{label}_schema", f"Missing growth columns: {missing}", len(missing))
            continue

        mask = df[future_col].notna() & (pd.to_numeric(df["current_market_value"], errors="coerce") > 0)
        if not mask.any():
            add_issue(issues, "FAIL", f"growth_math_{label}_available", f"No rows with {label} future values.")
            continue

        expected_eur = df.loc[mask, future_col] - df.loc[mask, "current_market_value"]
        expected_pct = expected_eur / df.loc[mask, "current_market_value"]
        bad_eur = (~np.isclose(df.loc[mask, growth_eur_col], expected_eur, rtol=1e-6, atol=1e-3)).sum()
        bad_pct = (~np.isclose(df.loc[mask, growth_pct_col], expected_pct, rtol=1e-6, atol=1e-6)).sum()
        if bad_eur:
            add_issue(issues, "FAIL", f"growth_eur_formula_{label}", f"{label} EUR growth formula mismatch.", bad_eur)
        if bad_pct:
            add_issue(issues, "FAIL", f"growth_pct_formula_{label}", f"{label} pct growth formula mismatch.", bad_pct)

        expected_positive = df.loc[mask, future_col] > df.loc[mask, "current_market_value"]
        expected_hit = df.loc[mask, future_col] >= df.loc[mask, "current_market_value"] * 1.25
        bad_positive = (bool_series(df.loc[mask, positive_col]) != expected_positive).sum()
        bad_hit = (bool_series(df.loc[mask, hit_col]) != expected_hit).sum()
        if bad_positive:
            add_issue(issues, "FAIL", f"positive_growth_flag_{label}", f"{label} positive growth flag mismatch.", bad_positive)
        if bad_hit:
            add_issue(issues, "FAIL", f"hit_flag_{label}", f"{label} 25 pct hit flag mismatch.", bad_hit)

        impossible_declines = (df.loc[mask, growth_pct_col] < -1.000001).sum()
        extreme_growth = (df.loc[mask, growth_pct_col] > 20).sum()
        if impossible_declines:
            add_issue(issues, "FAIL", f"growth_range_low_{label}", f"{label} growth below -100 pct found.", impossible_declines)
        if extreme_growth:
            add_issue(
                issues,
                "WARN",
                f"growth_range_high_{label}",
                f"{label} growth above 2000 pct found; inspect outliers.",
                extreme_growth,
            )


def validate_candidate_coverage(candidates: pd.DataFrame, summary: pd.DataFrame, issues: list[dict]) -> None:
    if candidates.empty:
        add_issue(issues, "FAIL", "candidate_rows", "Temporal backtest candidate output is empty.")
        return

    candidate_snapshots = sorted(candidates["snapshot_date"].dt.strftime("%Y-%m-%d").dropna().unique().tolist())
    missing_candidate_snapshots = sorted(set(EXPECTED_SNAPSHOT_DATES) - set(candidate_snapshots))
    if missing_candidate_snapshots:
        add_issue(
            issues,
            "FAIL",
            "candidate_snapshot_coverage",
            f"No candidates for configured snapshots: {missing_candidate_snapshots}",
            len(missing_candidate_snapshots),
        )

    for label in ["6m", "12m"]:
        available = candidates[f"future_value_{label}"].notna().sum() if f"future_value_{label}" in candidates.columns else 0
        if available == 0:
            add_issue(issues, "FAIL", f"candidate_outcome_{label}", f"No candidate rows have {label} outcomes.")
        elif available < 100:
            add_issue(
                issues,
                "WARN",
                f"candidate_outcome_{label}_sample",
                f"Only {available} candidate rows have {label} outcomes.",
                available,
            )

    if summary.empty:
        add_issue(issues, "FAIL", "summary_rows", "Temporal backtest summary is empty.")
        return

    expected_populations = {"Scouting Leads", "Eligible Baseline"}
    missing_populations = expected_populations - set(summary.get("population", pd.Series(dtype=str)).dropna().unique())
    if missing_populations:
        add_issue(
            issues,
            "FAIL",
            "summary_populations",
            f"Summary missing populations: {sorted(missing_populations)}",
            len(missing_populations),
        )


def validate_temporal_backtest() -> bool:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    issues: list[dict] = []
    datasets = validate_required_files(issues)
    model_features = validate_model_features(issues)

    snapshots = datasets.get("snapshots", pd.DataFrame())
    candidates = datasets.get("candidates", pd.DataFrame())
    summary = datasets.get("summary", pd.DataFrame())

    validate_snapshot_integrity(snapshots, issues)
    leakage_violations = validate_future_windows(snapshots, issues) if not snapshots.empty else 0
    validate_growth_math(snapshots, issues)
    validate_candidate_coverage(candidates, summary, issues)

    fail_count = sum(1 for issue in issues if issue["severity"] == "FAIL")
    warn_count = sum(1 for issue in issues if issue["severity"] == "WARN")
    status = "FAIL" if fail_count else ("WARN" if warn_count else "PASS")

    report = {
        "status": status,
        "fail_count": int(fail_count),
        "warn_count": int(warn_count),
        "issue_count": int(len(issues)),
        "snapshot_rows": int(len(snapshots)),
        "candidate_rows": int(len(candidates)),
        "future_6m_available": int(candidates["future_value_6m"].notna().sum()) if "future_value_6m" in candidates else 0,
        "future_12m_available": int(candidates["future_value_12m"].notna().sum()) if "future_value_12m" in candidates else 0,
        "configured_snapshots": EXPECTED_SNAPSHOT_DATES,
        "model_features": model_features,
        "leakage_violations": int(leakage_violations),
        "issues": issues,
    }

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"Temporal Backtest Validation: {status}")
    print(f" - Snapshot rows: {report['snapshot_rows']:,}")
    print(f" - Candidate rows: {report['candidate_rows']:,}")
    print(f" - 6M outcomes: {report['future_6m_available']:,}")
    print(f" - 12M outcomes: {report['future_12m_available']:,}")
    print(f" - Issues: {len(issues)} ({fail_count} fail, {warn_count} warn)")
    print(f" - Report: {REPORT_PATH}")
    return fail_count == 0


if __name__ == "__main__":
    sys.exit(0 if validate_temporal_backtest() else 1)
