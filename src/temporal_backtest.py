import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path("outputs/temporal_validation")
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_PATH = Path("outputs/models/performance_only_model.pkl")

SNAPSHOT_DATES = pd.to_datetime(
    [
        "2021-07-01",
        "2022-01-01",
        "2022-07-01",
        "2023-01-01",
        "2023-07-01",
        "2024-01-01",
        "2024-07-01",
        "2025-01-01",
    ]
)

POSITION_MAP = {
    "Attack": "Forward",
    "Midfield": "Midfielder",
    "Defender": "Defender",
    "Goalkeeper": "Goalkeeper",
}

FEATURE_STAT_COLS = [
    "appearances_last_365",
    "minutes_last_365",
    "goals_last_365",
    "assists_last_365",
    "yellow_cards_last_365",
    "red_cards_last_365",
]

EXPORT_COLS = [
    "player_id",
    "name",
    "snapshot_date",
    "feature_window_start",
    "feature_window_end",
    "max_appearance_date_last_365",
    "current_valuation_date",
    "age_at_snapshot",
    "position_group_raw",
    "primary_role",
    "role_tags",
    "current_club_id",
    "current_club_name",
    "current_market_value",
    "predicted_value",
    "value_gap",
    "value_gap_bucket",
    "minutes_last_365",
    "goals_last_365",
    "assists_last_365",
    "cards_last_365",
    "goals_per_90_ls",
    "assists_per_90_ls",
    "cards_per_90_ls",
    "age_group",
    "market_value_bucket",
    "is_eligible_baseline",
    "is_scouting_lead",
    "future_valuation_date_6m",
    "future_value_6m",
    "growth_6m_eur",
    "growth_6m_pct",
    "positive_growth_6m",
    "hit_6m",
    "future_valuation_date_12m",
    "future_value_12m",
    "growth_12m_eur",
    "growth_12m_pct",
    "positive_growth_12m",
    "hit_12m",
]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    players = pd.read_csv("data/raw/players.csv", low_memory=False)
    valuations = pd.read_csv("data/raw/player_valuations.csv", low_memory=False)
    appearances = pd.read_csv("data/raw/appearances.csv", low_memory=False)

    role_path = Path("data/processed/role_enriched_players.csv")
    if role_path.exists():
        role_cols = ["player_id", "primary_role", "role_tags"]
        roles = pd.read_csv(role_path, usecols=lambda col: col in role_cols, low_memory=False)
        roles = roles.drop_duplicates(subset=["player_id"])
    else:
        roles = pd.DataFrame(columns=["player_id", "primary_role", "role_tags"])

    players["date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")
    valuations["date"] = pd.to_datetime(valuations["date"], errors="coerce")
    appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")

    valuations = valuations.dropna(subset=["player_id", "date", "market_value_in_eur"]).copy()
    valuations["market_value_in_eur"] = pd.to_numeric(valuations["market_value_in_eur"], errors="coerce")
    valuations = valuations[valuations["market_value_in_eur"] > 0].copy()

    return players, valuations, appearances, roles


def build_valuation_timeline_audit(valuations: pd.DataFrame, appearances: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "row_type": "overall",
            "snapshot_date": "ALL",
            "valuation_min_date": valuations["date"].min(),
            "valuation_max_date": valuations["date"].max(),
            "appearance_min_date": appearances["date"].min(),
            "appearance_max_date": appearances["date"].max(),
            "valuation_rows": len(valuations),
            "unique_valuation_dates": valuations["date"].nunique(),
            "valuation_players": valuations["player_id"].nunique(),
            "players_with_multiple_valuations": int((valuations.groupby("player_id").size() > 1).sum()),
            "current_players": np.nan,
            "future_6m_players": np.nan,
            "future_12m_players": np.nan,
            "recent_appearance_players": np.nan,
        }
    ]

    for snapshot_date in SNAPSHOT_DATES:
        rows.append(
            {
                "row_type": "snapshot",
                "snapshot_date": snapshot_date,
                "valuation_min_date": valuations["date"].min(),
                "valuation_max_date": valuations["date"].max(),
                "appearance_min_date": appearances["date"].min(),
                "appearance_max_date": appearances["date"].max(),
                "valuation_rows": len(valuations),
                "unique_valuation_dates": valuations["date"].nunique(),
                "valuation_players": valuations["player_id"].nunique(),
                "players_with_multiple_valuations": int((valuations.groupby("player_id").size() > 1).sum()),
                "current_players": valuations[valuations["date"] <= snapshot_date]["player_id"].nunique(),
                "future_6m_players": valuation_window_players(valuations, snapshot_date, 150, 240),
                "future_12m_players": valuation_window_players(valuations, snapshot_date, 300, 450),
                "recent_appearance_players": appearances[
                    (appearances["date"] >= snapshot_date - pd.Timedelta(days=365))
                    & (appearances["date"] <= snapshot_date)
                ]["player_id"].nunique(),
            }
        )

    return pd.DataFrame(rows)


def valuation_window_players(valuations: pd.DataFrame, snapshot_date: pd.Timestamp, start_day: int, end_day: int) -> int:
    return valuations[
        (valuations["date"] >= snapshot_date + pd.Timedelta(days=start_day))
        & (valuations["date"] <= snapshot_date + pd.Timedelta(days=end_day))
    ]["player_id"].nunique()


def latest_current_valuation(valuations: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    current = valuations[valuations["date"] <= snapshot_date].copy()
    current = (
        current.sort_values(["player_id", "date"])
        .drop_duplicates(subset=["player_id"], keep="last")
        .rename(
            columns={
                "date": "current_valuation_date",
                "market_value_in_eur": "current_market_value",
                "current_club_id": "valuation_current_club_id",
                "current_club_name": "valuation_current_club_name",
                "player_club_domestic_competition_id": "valuation_competition_id",
            }
        )
    )
    return current


def aggregate_recent_appearances(appearances: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    start_date = snapshot_date - pd.Timedelta(days=365)
    recent = appearances[(appearances["date"] >= start_date) & (appearances["date"] <= snapshot_date)].copy()
    if recent.empty:
        return pd.DataFrame(columns=["player_id"] + FEATURE_STAT_COLS + ["max_appearance_date_last_365"])

    stats = (
        recent.groupby("player_id")
        .agg(
            appearances_last_365=("appearance_id", "count"),
            minutes_last_365=("minutes_played", "sum"),
            goals_last_365=("goals", "sum"),
            assists_last_365=("assists", "sum"),
            yellow_cards_last_365=("yellow_cards", "sum"),
            red_cards_last_365=("red_cards", "sum"),
            max_appearance_date_last_365=("date", "max"),
        )
        .reset_index()
    )
    return stats


def nearest_future_valuation(
    valuations: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    start_day: int,
    end_day: int,
    label: str,
) -> pd.DataFrame:
    start_date = snapshot_date + pd.Timedelta(days=start_day)
    end_date = snapshot_date + pd.Timedelta(days=end_day)
    target_date = snapshot_date + pd.Timedelta(days=(start_day + end_day) // 2)

    future = valuations[(valuations["date"] >= start_date) & (valuations["date"] <= end_date)].copy()
    if future.empty:
        return pd.DataFrame(columns=["player_id", f"future_valuation_date_{label}", f"future_value_{label}"])

    future["days_from_target"] = (future["date"] - target_date).abs().dt.days
    future = (
        future.sort_values(["player_id", "days_from_target", "date"])
        .drop_duplicates(subset=["player_id"], keep="first")
        .rename(columns={"date": f"future_valuation_date_{label}", "market_value_in_eur": f"future_value_{label}"})
    )
    return future[["player_id", f"future_valuation_date_{label}", f"future_value_{label}"]]


def add_bucket_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["value_gap_bucket"] = pd.cut(
        out["value_gap"],
        bins=[-np.inf, 0, 0.5, 1.0, 3.0, np.inf],
        labels=["Not Undervalued", "Small: 0-50%", "Moderate: 50-100%", "Large: 100-300%", "Very Large: >300%"],
        right=True,
    ).astype("object")
    out["age_group"] = pd.cut(
        out["age_at_snapshot"],
        bins=[-np.inf, 21, 23, 25, np.inf],
        labels=["U21", "U23", "U25", "25+"],
        right=True,
    ).astype("object")
    out["market_value_bucket"] = pd.cut(
        out["current_market_value"],
        bins=[-np.inf, 500_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, np.inf],
        labels=[
            "Below EUR0.5M",
            "EUR0.5M-2M",
            "EUR2M-5M",
            "EUR5M-10M",
            "EUR10M-20M",
            "Above EUR20M",
        ],
        right=True,
    ).astype("object")
    return out


def build_snapshot_frame(
    players: pd.DataFrame,
    valuations: pd.DataFrame,
    appearances: pd.DataFrame,
    roles: pd.DataFrame,
    model,
    model_features: list[str],
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:
    current = latest_current_valuation(valuations, snapshot_date)
    stats = aggregate_recent_appearances(appearances, snapshot_date)

    base = players.merge(
        current[
            [
                col
                for col in [
                    "player_id",
                    "current_valuation_date",
                    "current_market_value",
                    "valuation_current_club_id",
                    "valuation_current_club_name",
                ]
                if col in current.columns
            ]
        ],
        on="player_id",
        how="inner",
    )
    base = base.merge(stats, on="player_id", how="left")
    base = base.merge(roles, on="player_id", how="left")

    base["snapshot_date"] = snapshot_date
    base["feature_window_start"] = snapshot_date - pd.Timedelta(days=365)
    base["feature_window_end"] = snapshot_date
    base["age_at_snapshot"] = (snapshot_date - base["date_of_birth"]).dt.days / 365.25
    base = base.dropna(subset=["age_at_snapshot", "current_market_value"]).copy()
    base = base[base["current_market_value"] > 0].copy()

    for col in FEATURE_STAT_COLS:
        if col not in base.columns:
            base[col] = 0
        base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0)

    base["max_appearance_date_last_365"] = pd.to_datetime(base["max_appearance_date_last_365"], errors="coerce")
    base["cards_last_365"] = base["yellow_cards_last_365"] + base["red_cards_last_365"]
    base["goals_per_90_ls"] = np.where(
        base["minutes_last_365"] > 0,
        (base["goals_last_365"] / base["minutes_last_365"]) * 90,
        0,
    )
    base["assists_per_90_ls"] = np.where(
        base["minutes_last_365"] > 0,
        (base["assists_last_365"] / base["minutes_last_365"]) * 90,
        0,
    )
    base["cards_per_90_ls"] = np.where(
        base["minutes_last_365"] > 0,
        (base["cards_last_365"] / base["minutes_last_365"]) * 90,
        0,
    )

    base["position_group_raw"] = base["position"].map(POSITION_MAP).fillna("Other")
    base["foot"] = base["foot"].fillna("unknown")
    base["age_at_valuation"] = base["age_at_snapshot"]
    base["minutes_last_season"] = base["minutes_last_365"]

    if "valuation_current_club_id" in base.columns:
        base["current_club_id"] = base["valuation_current_club_id"].combine_first(base.get("current_club_id"))
    if "valuation_current_club_name" in base.columns:
        base["current_club_name"] = base["valuation_current_club_name"].combine_first(base.get("current_club_name"))

    feature_frame = pd.get_dummies(base, columns=["position_group_raw", "foot"], prefix=["pos", "foot"])
    for col in model_features:
        if col not in feature_frame.columns:
            feature_frame[col] = 0

    base["predicted_log_value"] = model.predict(feature_frame[model_features].fillna(0))
    base["predicted_value"] = np.expm1(base["predicted_log_value"])
    base["value_gap"] = (base["predicted_value"] - base["current_market_value"]) / base["current_market_value"]

    for label, start_day, end_day in [("6m", 150, 240), ("12m", 300, 450)]:
        future = nearest_future_valuation(valuations, snapshot_date, start_day, end_day, label)
        base = base.merge(future, on="player_id", how="left")
        base[f"growth_{label}_eur"] = base[f"future_value_{label}"] - base["current_market_value"]
        base[f"growth_{label}_pct"] = base[f"growth_{label}_eur"] / base["current_market_value"]
        base[f"positive_growth_{label}"] = base[f"future_value_{label}"] > base["current_market_value"]
        base[f"hit_{label}"] = base[f"future_value_{label}"] >= base["current_market_value"] * 1.25

    base["is_eligible_baseline"] = (
        (base["minutes_last_365"] >= 900)
        & (base["current_market_value"] >= 500_000)
        & (base["current_market_value"] <= 20_000_000)
        & (base["age_at_snapshot"] <= 25)
    )
    base["is_scouting_lead"] = base["is_eligible_baseline"] & (base["value_gap"] > 0)

    base = add_bucket_columns(base)
    return base


def metric_row(df: pd.DataFrame, population: str, snapshot_date: str) -> dict:
    row = {
        "population": population,
        "snapshot_date": snapshot_date,
        "row_count": int(len(df)),
    }
    for label in ["6m", "12m"]:
        available = df[f"future_value_{label}"].notna()
        available_df = df[available]
        row[f"available_{label}"] = int(available.sum())
        row[f"positive_growth_rate_{label}"] = (
            float(available_df[f"positive_growth_{label}"].mean()) if not available_df.empty else np.nan
        )
        row[f"hit_rate_{label}_25pct"] = float(available_df[f"hit_{label}"].mean()) if not available_df.empty else np.nan
        row[f"median_growth_{label}_pct"] = (
            float(available_df[f"growth_{label}_pct"].median()) if not available_df.empty else np.nan
        )
        row[f"median_growth_{label}_eur"] = (
            float(available_df[f"growth_{label}_eur"].median()) if not available_df.empty else np.nan
        )
    return row


def build_summary(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    populations = {
        "Scouting Leads": snapshot_df[snapshot_df["is_scouting_lead"]].copy(),
        "Eligible Baseline": snapshot_df[snapshot_df["is_eligible_baseline"]].copy(),
    }
    for population, df in populations.items():
        rows.append(metric_row(df, population, "ALL"))
        for snapshot_date, group in df.groupby("snapshot_date"):
            rows.append(metric_row(group, population, str(pd.Timestamp(snapshot_date).date())))
    return pd.DataFrame(rows)


def grouped_success(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    populations = {
        "Scouting Leads": df[df["is_scouting_lead"]].copy(),
        "Eligible Baseline": df[df["is_eligible_baseline"]].copy(),
    }
    for population, pop_df in populations.items():
        for segment, segment_df in pop_df.groupby(group_col, dropna=False):
            segment_label = "Unknown" if pd.isna(segment) else str(segment)
            row = metric_row(segment_df, population, "ALL")
            row[group_col] = segment_label
            rows.append(row)
    return pd.DataFrame(rows)


def save_figures(candidates: pd.DataFrame, by_gap: pd.DataFrame, by_position: pd.DataFrame, by_age: pd.DataFrame) -> None:
    candidates_12m = candidates[candidates["growth_12m_pct"].notna()].copy()
    if not candidates_12m.empty:
        plt.figure(figsize=(9, 5))
        plt.hist((candidates_12m["growth_12m_pct"] * 100).clip(-100, 500), bins=35, color="#2563eb", alpha=0.82)
        plt.axvline(25, color="#f59e0b", linestyle="--", linewidth=2, label="25% hit threshold")
        plt.title("12M Future Value Growth Distribution")
        plt.xlabel("12M growth (%)")
        plt.ylabel("Historical scouting leads")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "future_growth_distribution.png", dpi=160)
        plt.close()

    plot_bar(
        by_gap[by_gap["population"] == "Scouting Leads"],
        "value_gap_bucket",
        "hit_rate_12m_25pct",
        "12M Hit Rate by Value Gap Bucket",
        "Value gap bucket",
        "12M hit rate",
        FIGURE_DIR / "hit_rate_by_value_gap_bucket.png",
    )
    plot_bar(
        by_position[by_position["population"] == "Scouting Leads"],
        "position_group_raw",
        "hit_rate_12m_25pct",
        "12M Hit Rate by Position",
        "Position",
        "12M hit rate",
        FIGURE_DIR / "hit_rate_by_position.png",
    )
    plot_bar(
        by_age[by_age["population"] == "Scouting Leads"],
        "age_group",
        "median_growth_12m_pct",
        "Median 12M Growth by Age Group",
        "Age group",
        "Median 12M growth",
        FIGURE_DIR / "median_growth_by_age_group.png",
    )


def plot_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        return
    plot_df[y_col] = plot_df[y_col] * 100
    plt.figure(figsize=(9, 5))
    plt.bar(plot_df[x_col].astype(str), plot_df[y_col], color="#0f766e", alpha=0.88)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(f"{ylabel} (%)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def export_outputs(snapshot_df: pd.DataFrame, valuations: pd.DataFrame, appearances: pd.DataFrame) -> None:
    audit_df = build_valuation_timeline_audit(valuations, appearances)
    candidates = snapshot_df[snapshot_df["is_scouting_lead"]].copy()
    summary = build_summary(snapshot_df)
    by_gap = grouped_success(snapshot_df, "value_gap_bucket")
    by_position = grouped_success(snapshot_df, "position_group_raw")
    by_age = grouped_success(snapshot_df, "age_group")
    by_market_value = grouped_success(snapshot_df, "market_value_bucket")

    audit_df.to_csv(OUTPUT_DIR / "valuation_timeline_audit.csv", index=False)
    snapshot_df[[col for col in EXPORT_COLS if col in snapshot_df.columns]].to_csv(
        OUTPUT_DIR / "temporal_backtest_snapshots.csv",
        index=False,
    )
    candidates[[col for col in EXPORT_COLS if col in candidates.columns]].to_csv(
        OUTPUT_DIR / "temporal_backtest_candidates.csv",
        index=False,
    )
    summary.to_csv(OUTPUT_DIR / "temporal_backtest_summary.csv", index=False)
    by_gap.to_csv(OUTPUT_DIR / "success_rate_by_value_gap_bucket.csv", index=False)
    by_position.to_csv(OUTPUT_DIR / "success_rate_by_position.csv", index=False)
    by_age.to_csv(OUTPUT_DIR / "success_rate_by_age_group.csv", index=False)
    by_market_value.to_csv(OUTPUT_DIR / "success_rate_by_market_value_bucket.csv", index=False)
    save_figures(candidates, by_gap, by_position, by_age)

    metadata = {
        "method": "fixed_model_retrospective_signal_audit",
        "model_path": str(MODEL_PATH),
        "snapshot_dates": [str(date.date()) for date in SNAPSHOT_DATES],
        "candidate_rows": int(len(candidates)),
        "snapshot_rows": int(len(snapshot_df)),
        "scouting_lead_definition": {
            "age_max": 25,
            "minutes_last_365_min": 900,
            "current_market_value_min": 500000,
            "current_market_value_max": 20000000,
            "value_gap_min_exclusive": 0,
        },
        "future_windows": {
            "6m": {"start_days": 150, "end_days": 240},
            "12m": {"start_days": 300, "end_days": 450},
        },
    }
    with (OUTPUT_DIR / "temporal_backtest_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def main() -> None:
    print("Starting Phase 7 temporal validation backtest...")
    ensure_dirs()
    players, valuations, appearances, roles = load_inputs()
    model = joblib.load(MODEL_PATH)
    model_features = [str(col) for col in getattr(model, "feature_names_in_", [])]
    if not model_features:
        raise ValueError("Model artifact does not expose feature_names_in_; cannot align snapshot features safely.")

    snapshot_frames = []
    for snapshot_date in SNAPSHOT_DATES:
        print(f"Building snapshot {snapshot_date.date()}...")
        snapshot_frames.append(
            build_snapshot_frame(players, valuations, appearances, roles, model, model_features, snapshot_date)
        )

    snapshot_df = pd.concat(snapshot_frames, ignore_index=True)
    export_outputs(snapshot_df, valuations, appearances)

    candidates = snapshot_df[snapshot_df["is_scouting_lead"]]
    print(f"Temporal backtest complete. Snapshot rows: {len(snapshot_df):,}")
    print(f"Historical scouting leads: {len(candidates):,}")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
