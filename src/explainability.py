import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_PATH = Path("data/processed/featured_players.csv")
PREDICTIONS_PATH = Path("outputs/predictions_per_player.csv")
SCOUTING_CANDIDATES_PATH = Path("outputs/undervalued_candidates_overall.csv")
MODEL_A_PATH = Path("outputs/models/performance_only_model.pkl")
MODEL_B_PATH = Path("outputs/models/market_aware_model.pkl")
OUTPUT_DIR = Path("outputs/explainability")
PLAYER_OUTPUT_DIR = OUTPUT_DIR / "player_explanations"

TOP_N_CANDIDATES = 20
GLOBAL_SAMPLE_SIZE = 5000
LOCAL_FEATURE_COUNT = 6
RANDOM_STATE = 42
SUMMARY_MIN_ABS_CONTRIBUTION = 0.05


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLAYER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_base_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run the feature engineering pipeline first.")
    return pd.read_csv(DATA_PATH, low_memory=False)


def build_feature_matrices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    base_features = [
        "age_at_valuation",
        "minutes_last_season",
        "goals_per_90_ls",
        "assists_per_90_ls",
        "cards_per_90_ls",
    ]
    pos_cols = [col for col in df.columns if col.startswith("pos_")]
    foot_cols = [col for col in df.columns if col.startswith("foot_")]

    features_a = base_features + pos_cols + foot_cols
    features_b = features_a + ["previous_market_value"]

    missing_a = [col for col in features_a if col not in df.columns]
    missing_b = [col for col in features_b if col not in df.columns]
    if missing_a or missing_b:
        raise ValueError(f"Missing model feature columns: {sorted(set(missing_a + missing_b))}")

    x_a = df[features_a].copy()
    x_b = df[features_b].copy().fillna({"previous_market_value": 0})
    return x_a, x_b, features_a, features_b


def load_models():
    if not MODEL_A_PATH.exists() or not MODEL_B_PATH.exists():
        raise FileNotFoundError("Model artifacts not found. Run `python src/modeling.py` first.")
    return joblib.load(MODEL_A_PATH), joblib.load(MODEL_B_PATH)


def sample_frame(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()
    return df.sample(n=max_rows, random_state=RANDOM_STATE).copy()


def shap_values_for_model(model, x: pd.DataFrame) -> shap.Explanation:
    explainer = shap.TreeExplainer(model)
    return explainer(x)


def save_global_plots(
    shap_values: shap.Explanation,
    x: pd.DataFrame,
    model_key: str,
    model_label: str,
) -> pd.DataFrame:
    summary_path = OUTPUT_DIR / f"shap_summary_{model_key}.png"
    bar_path = OUTPUT_DIR / f"shap_bar_{model_key}.png"

    plt.figure()
    shap.summary_plot(shap_values, x, show=False, max_display=20)
    plt.title(f"SHAP Summary - {model_label}")
    plt.tight_layout()
    plt.savefig(summary_path, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.title(f"SHAP Feature Importance - {model_label}")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=160, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(shap_values.values).mean(axis=0)
    importance = pd.DataFrame(
        {
            "model": model_label,
            "feature": x.columns,
            "mean_abs_shap_log_value": mean_abs,
        }
    ).sort_values("mean_abs_shap_log_value", ascending=False)
    importance["rank"] = range(1, len(importance) + 1)
    return importance


def top_contributions(
    shap_values: np.ndarray,
    feature_names: list[str],
    positive: bool,
    limit: int = 2,
    min_abs: float = 0.0,
) -> list[tuple[str, float]]:
    pairs = list(zip(feature_names, shap_values))
    if positive:
        pairs = [pair for pair in pairs if pair[1] > min_abs]
        pairs.sort(key=lambda item: item[1], reverse=True)
    else:
        pairs = [pair for pair in pairs if pair[1] < -min_abs]
        pairs.sort(key=lambda item: item[1])
    return pairs[:limit]


def readable_feature(feature: str) -> str:
    replacements = {
        "age_at_valuation": "age",
        "minutes_last_season": "minutes played",
        "goals_per_90_ls": "goals per 90",
        "assists_per_90_ls": "assists per 90",
        "cards_per_90_ls": "cards per 90",
        "previous_market_value": "previous market value",
    }
    if feature in replacements:
        return replacements[feature]
    if feature.startswith("pos_"):
        return f"position: {feature.replace('pos_', '').lower()}"
    if feature.startswith("foot_"):
        return f"foot: {feature.replace('foot_', '').lower()}"
    return feature.replace("_", " ")


def feature_with_value(feature: str, contribution: float) -> str:
    return f"{readable_feature(feature)} ({contribution:+.3f})"


def explanation_summary(player_name: str, positives: list[tuple[str, float]], negative: tuple[str, float] | None) -> str:
    positive_text = ", ".join(readable_feature(feature) for feature, _ in positives) or "available performance features"
    summary = (
        f"{player_name} is flagged as undervalued mainly because these signals push "
        f"the performance-only model prediction higher: {positive_text}."
    )
    if negative is not None:
        summary += f" The strongest downward pressure is {readable_feature(negative[0])}."
    return summary


def save_local_plots(
    shap_values: shap.Explanation,
    x: pd.DataFrame,
    candidate_rows: list[tuple[int, int]],
) -> None:
    for source_idx, player_id in candidate_rows:
        explanation = shap_values[source_idx]

        waterfall_path = PLAYER_OUTPUT_DIR / f"player_{player_id}_waterfall.png"
        plt.figure()
        shap.plots.waterfall(explanation, max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(waterfall_path, dpi=160, bbox_inches="tight")
        plt.close()

        force_path = PLAYER_OUTPUT_DIR / f"player_{player_id}_force.html"
        force_plot = shap.plots.force(
            explanation.base_values,
            explanation.values,
            x.iloc[source_idx, :],
            matplotlib=False,
        )
        shap.save_html(str(force_path), force_plot)


def build_candidate_explanations(
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    x_a: pd.DataFrame,
    shap_values_a: shap.Explanation,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[int, int]]]:
    candidates = candidates.copy()
    candidates = candidates[candidates["undervalued_pct"] > 0].copy()
    candidates = candidates.sort_values("undervalued_pct", ascending=False).head(TOP_N_CANDIDATES)

    prediction_lookup = predictions.reset_index().rename(columns={"index": "source_index"})
    candidates = candidates.merge(
        prediction_lookup[["source_index", "player_id"]],
        on="player_id",
        how="inner",
        suffixes=("", "_prediction"),
    )

    summary_rows = []
    detail_rows = []
    candidate_plot_rows = []
    feature_names = list(x_a.columns)

    for _, row in candidates.iterrows():
        source_idx = int(row["source_index"])
        player_id = int(row["player_id"])
        candidate_plot_rows.append((source_idx, player_id))
        values = shap_values_a.values[source_idx]

        positives = top_contributions(
            values,
            feature_names,
            positive=True,
            limit=2,
            min_abs=SUMMARY_MIN_ABS_CONTRIBUTION,
        )
        negatives = top_contributions(
            values,
            feature_names,
            positive=False,
            limit=1,
            min_abs=SUMMARY_MIN_ABS_CONTRIBUTION,
        )
        negative = negatives[0] if negatives else None

        summary_rows.append(
            {
                "player_id": player_id,
                "Player Name": row["name"],
                "Actual Value": row["target_market_value"],
                "Predicted Value": row["predicted_value"],
                "Undervaluation %": row["undervalued_pct"],
                "Top Positive SHAP Feature": feature_with_value(*positives[0]) if positives else "N/A",
                "Second Positive SHAP Feature": feature_with_value(*positives[1]) if len(positives) > 1 else "N/A",
                "Top Negative SHAP Feature": feature_with_value(*negative) if negative else "N/A",
                "Explanation Summary": explanation_summary(row["name"], positives, negative),
            }
        )

        top_features = sorted(
            zip(feature_names, values),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:LOCAL_FEATURE_COUNT]
        for feature, contribution in top_features:
            detail_rows.append(
                {
                    "player_id": int(row["player_id"]),
                    "Player Name": row["name"],
                    "feature": feature,
                    "feature_readable": readable_feature(feature),
                    "feature_value": x_a.iloc[source_idx][feature],
                    "shap_contribution_log_value": contribution,
                    "direction": "pushes higher" if contribution > 0 else "pushes lower",
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows), candidate_plot_rows


def main() -> bool:
    ensure_dirs()

    print("Loading data and model artifacts...")
    df = load_base_data()
    model_a, model_b = load_models()
    x_a, x_b, features_a, features_b = build_feature_matrices(df)

    predictions = pd.read_csv(PREDICTIONS_PATH) if PREDICTIONS_PATH.exists() else pd.DataFrame()
    if predictions.empty:
        raise FileNotFoundError(f"{PREDICTIONS_PATH} not found. Run `python src/modeling.py` first.")
    candidates = (
        pd.read_csv(SCOUTING_CANDIDATES_PATH)
        if SCOUTING_CANDIDATES_PATH.exists()
        else predictions[predictions["undervalued_pct"] > 0].copy()
    )

    # Align prediction rows to featured_players rows. modeling.py writes predictions in the same row order.
    predictions = predictions.reset_index(drop=True)
    x_a = x_a.reset_index(drop=True)
    x_b = x_b.reset_index(drop=True)
    x_a.index = predictions.index
    x_b.index = predictions.index

    print("Generating global SHAP explanations...")
    x_a_sample = sample_frame(x_a, GLOBAL_SAMPLE_SIZE)
    x_b_sample = x_b.loc[x_a_sample.index]

    shap_a_global = shap_values_for_model(model_a, x_a_sample)
    shap_b_global = shap_values_for_model(model_b, x_b_sample)

    importance = pd.concat(
        [
            save_global_plots(shap_a_global, x_a_sample, "model_a", "Model A - Performance Only"),
            save_global_plots(shap_b_global, x_b_sample, "model_b", "Model B - Market Aware"),
        ],
        ignore_index=True,
    )
    importance.to_csv(OUTPUT_DIR / "shap_feature_importance.csv", index=False)

    print("Generating local explanations for top undervalued candidates...")
    shap_a_full = shap_values_for_model(model_a, x_a)
    candidate_summary, candidate_details, candidate_plot_rows = build_candidate_explanations(
        predictions=predictions,
        candidates=candidates,
        x_a=x_a,
        shap_values_a=shap_a_full,
    )
    candidate_summary.to_csv(OUTPUT_DIR / "top_hidden_gems_explanations.csv", index=False)
    candidate_details.to_csv(PLAYER_OUTPUT_DIR / "player_explanation_table.csv", index=False)

    save_local_plots(shap_a_full, x_a, candidate_plot_rows[:10])

    metadata = {
        "status": "PASS",
        "target_space": "log1p_market_value",
        "model_a_features": features_a,
        "model_b_features": features_b,
        "global_sample_size": int(len(x_a_sample)),
        "top_candidate_explanations": int(len(candidate_summary)),
        "note": "SHAP values explain model output in log-value space and should not be interpreted as direct euro contributions or causal effects.",
    }
    (OUTPUT_DIR / "explainability_report.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Explainability outputs saved to {OUTPUT_DIR}")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
