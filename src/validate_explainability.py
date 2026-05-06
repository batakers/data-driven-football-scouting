import json
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("outputs/explainability")
PLAYER_OUTPUT_DIR = OUTPUT_DIR / "player_explanations"
VALIDATION_REPORT_PATH = OUTPUT_DIR / "explainability_validation_report.json"

IMPORTANCE_PATH = OUTPUT_DIR / "shap_feature_importance.csv"
CANDIDATES_PATH = OUTPUT_DIR / "top_hidden_gems_explanations.csv"
DETAILS_PATH = PLAYER_OUTPUT_DIR / "player_explanation_table.csv"
REPORT_PATH = OUTPUT_DIR / "explainability_report.json"

EXPECTED_PLOTS = [
    OUTPUT_DIR / "shap_summary_model_a.png",
    OUTPUT_DIR / "shap_bar_model_a.png",
    OUTPUT_DIR / "shap_summary_model_b.png",
    OUTPUT_DIR / "shap_bar_model_b.png",
]

MODEL_A = "Model A - Performance Only"
MODEL_B = "Model B - Market Aware"


def add_check(checks: list[dict], name: str, passed: bool, detail: str = "") -> None:
    checks.append(
        {
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        }
    )


def file_exists_and_nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def main() -> bool:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []

    required_files = [IMPORTANCE_PATH, CANDIDATES_PATH, DETAILS_PATH, REPORT_PATH, *EXPECTED_PLOTS]
    for path in required_files:
        add_check(
            checks,
            f"required_file:{path.as_posix()}",
            file_exists_and_nonempty(path),
            "file exists and is non-empty" if file_exists_and_nonempty(path) else "missing or empty",
        )

    if not all(file_exists_and_nonempty(path) for path in [IMPORTANCE_PATH, CANDIDATES_PATH, DETAILS_PATH, REPORT_PATH]):
        validation = {
            "status": "FAIL",
            "checks": checks,
        }
        VALIDATION_REPORT_PATH.write_text(json.dumps(validation, indent=2), encoding="utf-8")
        print("Explainability validation failed: required outputs are missing.")
        return False

    importance = pd.read_csv(IMPORTANCE_PATH)
    candidates = pd.read_csv(CANDIDATES_PATH)
    details = pd.read_csv(DETAILS_PATH)
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    add_check(
        checks,
        "report_status_pass",
        report.get("status") == "PASS",
        f"status={report.get('status')}",
    )
    add_check(
        checks,
        "target_space_log_value",
        report.get("target_space") == "log1p_market_value",
        f"target_space={report.get('target_space')}",
    )

    models_present = set(importance.get("model", []))
    add_check(
        checks,
        "both_models_present",
        {MODEL_A, MODEL_B}.issubset(models_present),
        f"models={sorted(models_present)}",
    )

    importance_values = pd.to_numeric(importance.get("mean_abs_shap_log_value"), errors="coerce")
    add_check(
        checks,
        "importance_values_valid",
        importance_values.notna().all() and (importance_values >= 0).all() and importance_values.sum() > 0,
        "mean_abs_shap_log_value is numeric, non-negative, and non-zero",
    )

    model_a_importance = importance[importance["model"] == MODEL_A].sort_values("rank")
    model_b_importance = importance[importance["model"] == MODEL_B].sort_values("rank")
    model_b_top_feature = model_b_importance.iloc[0]["feature"] if not model_b_importance.empty else None
    add_check(
        checks,
        "model_b_market_signal_dominant",
        model_b_top_feature == "previous_market_value",
        f"top_feature={model_b_top_feature}",
    )
    add_check(
        checks,
        "model_a_excludes_previous_market_value",
        "previous_market_value" not in set(model_a_importance.get("feature", [])),
        "Model A should remain performance-only",
    )

    required_candidate_cols = {
        "player_id",
        "Player Name",
        "Actual Value",
        "Predicted Value",
        "Undervaluation %",
        "Explanation Summary",
    }
    candidate_cols = set(candidates.columns)
    add_check(
        checks,
        "candidate_columns_present",
        required_candidate_cols.issubset(candidate_cols),
        f"missing={sorted(required_candidate_cols - candidate_cols)}",
    )
    add_check(
        checks,
        "candidate_explanations_nonempty",
        len(candidates) > 0 and candidates["Explanation Summary"].fillna("").str.len().gt(0).all(),
        f"rows={len(candidates)}",
    )

    contribution_values = pd.to_numeric(details.get("shap_contribution_log_value"), errors="coerce")
    add_check(
        checks,
        "local_contributions_valid",
        len(details) > 0 and contribution_values.notna().all() and contribution_values.abs().sum() > 0,
        f"rows={len(details)}",
    )

    expected_waterfalls = min(10, len(candidates))
    waterfall_count = len(list(PLAYER_OUTPUT_DIR.glob("player_*_waterfall.png")))
    add_check(
        checks,
        "waterfall_plots_present",
        waterfall_count >= expected_waterfalls,
        f"waterfall_count={waterfall_count}, expected_at_least={expected_waterfalls}",
    )

    status = "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL"
    validation = {
        "status": status,
        "checks": checks,
        "summary": {
            "importance_rows": int(len(importance)),
            "candidate_explanations": int(len(candidates)),
            "local_contribution_rows": int(len(details)),
            "waterfall_plots": int(waterfall_count),
            "model_b_top_feature": model_b_top_feature,
        },
    }
    VALIDATION_REPORT_PATH.write_text(json.dumps(validation, indent=2), encoding="utf-8")

    if status == "PASS":
        print("Explainability validation PASS")
        return True

    failed = [check for check in checks if check["status"] == "FAIL"]
    print("Explainability validation FAIL")
    for check in failed:
        print(f" - {check['check']}: {check['detail']}")
    return False


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
