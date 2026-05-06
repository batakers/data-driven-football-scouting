import os
import sys
import html
from pathlib import Path
import joblib

import pandas as pd
import plotly.express as px
import streamlit as st

# Fix for ModuleNotFoundError: No module named 'src'
# Ensures the project root is in the path when running from the app/ folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib
import src.similarity
importlib.reload(src.similarity)
from src.similarity import PlayerSimilarity
from src.role_mapping import COMPATIBLE_ROLE_THRESHOLD


# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Football Talent Scouting Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# Theme Styling
# Uses Streamlit native CSS variables so it adapts to light/dark mode.
# =========================
PLOTLY_TEMPLATE = "plotly_dark"  # Keep charts dark for professional look
REFERENCE_LINE_COLOR = "#94a3b8"

st.markdown(
    """
<style>
.kpi-card {
    background-color: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.25);
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.10);
    min-height: 105px;
    margin-bottom: 25px;
}

.kpi-label {
    color: var(--text-color);
    opacity: 0.70;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 10px;
}

.kpi-value {
    color: var(--text-color);
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.15;
}

.methodology-box {
    background-color: rgba(59, 130, 246, 0.12);
    border: 1px solid rgba(59, 130, 246, 0.35);
    color: var(--text-color);
    padding: 0.85rem 1rem;
    border-radius: 8px;
    font-size: 0.92rem;
    margin-top: 15px;
    margin-bottom: 10px;
}

.disclaimer-box {
    background-color: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.45);
    color: var(--text-color);
    padding: 0.85rem 1rem;
    border-radius: 8px;
    font-size: 0.92rem;
    margin-top: 15px;
}

div[data-testid="stExpander"] {
    border-radius: 10px;
}
.advanced-stats-badge {
    background-color: #059669;
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 10px;
    display: inline-block;
}
.basic-stats-badge {
    background-color: #4b5563;
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 10px;
    display: inline-block;
}
.target-panel {
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.10), rgba(15, 23, 42, 0.02));
    border: 1px solid rgba(37, 99, 235, 0.24);
    border-radius: 16px;
    padding: 18px 20px;
    margin: 8px 0 16px 0;
}
.target-name {
    color: var(--text-color);
    font-size: 1.45rem;
    font-weight: 800;
    margin-bottom: 5px;
}
.target-summary {
    color: var(--text-color);
    opacity: 0.80;
    font-size: 0.96rem;
    margin-bottom: 10px;
}
.target-context {
    color: var(--text-color);
    opacity: 0.72;
    font-size: 0.88rem;
}
.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 6px 0 14px 0;
}
.profile-chip {
    border: 1px solid rgba(100, 116, 139, 0.28);
    background: rgba(148, 163, 184, 0.12);
    color: var(--text-color);
    border-radius: 999px;
    padding: 5px 10px;
    font-size: 0.84rem;
    font-weight: 600;
}
.data-chip {
    border: 1px solid rgba(37, 99, 235, 0.30);
    background: rgba(59, 130, 246, 0.10);
}
</style>
""",
    unsafe_allow_html=True,
)

def kpi_card(label: str, value: str) -> None:
    """Render a custom KPI card that works better in both light and dark mode."""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def methodology_box(message: str) -> None:
    st.markdown(
        f"""
        <div class="methodology-box">
            💡 <strong>Methodology:</strong> {message}
        </div>
        """,
        unsafe_allow_html=True,
    )

def disclaimer_box(message: str) -> None:
    st.markdown(
        f"""
        <div class="disclaimer-box">
            ⚠️ <strong>Disclaimer:</strong> {message}
        </div>
        """,
        unsafe_allow_html=True,
    )

def scalar_bool(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}

def format_stats_season(value) -> str:
    if pd.isna(value):
        return "N/A"
    raw = str(value).strip()
    try:
        raw = str(int(float(raw)))
    except ValueError:
        raw = raw.replace(".0", "")
    if len(raw) != 4:
        return "N/A"
    return f"20{raw[:2]}/20{raw[2:]}"

def format_role_tags(value) -> str:
    if pd.isna(value):
        return "N/A"
    tags = [tag.strip() for tag in str(value).split(",") if tag.strip()]
    return ", ".join(tags) if tags else "N/A"

def format_foot_profile(value, has_metadata: bool) -> str:
    if not has_metadata or pd.isna(value):
        return "Foot unknown"
    foot = str(value).strip().lower()
    if foot in {"left", "right"}:
        return f"{foot.title()}-footed"
    if foot == "both":
        return "Two-footed"
    return "Foot unknown"

def role_tags_are_redundant(primary_role: str, role_tags: str) -> bool:
    if not role_tags or role_tags == "N/A":
        return True
    tags = [tag.strip() for tag in role_tags.split(",") if tag.strip()]
    return len(tags) == 1 and tags[0].upper() == str(primary_role).upper()

def render_chip_row(labels: list[str], extra_class: str = "") -> str:
    class_name = f"profile-chip {extra_class}".strip()
    chips = "".join(
        f'<span class="{class_name}">{html.escape(str(label))}</span>'
        for label in labels
        if label and str(label).strip()
    )
    return f'<div class="chip-row">{chips}</div>'

def format_score(value) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.2f}%"

def format_match_method(value) -> str:
    if pd.isna(value):
        return "N/A"
    method = str(value).strip()
    if not method:
        return "N/A"
    return method.replace("_", " ").title()

def format_match_confidence(value, fallback: str = "N/A") -> str:
    if pd.isna(value):
        return fallback
    return f"{float(value):.0f}%"

def fuzzy_confidence_note(method, score) -> str | None:
    method_raw = "" if pd.isna(method) else str(method).strip().lower()
    if not method_raw.startswith("fuzzy"):
        return None
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = None
    if score_value is not None and score_value >= 99.5:
        return "100% is the fuzzy text-match score after validation gates; it does not mean the match method was exact."
    return "Fuzzy matches are accepted only after validation gates such as role, team, season, and duplicate-source checks."

def format_currency(value) -> str:
    if pd.isna(value):
        return "N/A"
    return f"€{float(value):,.0f}"

def format_percentage(value, decimals: int = 1) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.{decimals}f}%"

LEAGUE_LABELS = {
    "GB1": "Premier League",
    "ES1": "La Liga",
    "L1": "Bundesliga",
    "IT1": "Serie A",
    "FR1": "Ligue 1",
    "NL1": "Eredivisie",
    "PO1": "Liga Portugal",
    "BE1": "Jupiler Pro League",
    "TR1": "Super Lig",
    "SC1": "Scottish Premiership",
    "DK1": "Superligaen",
    "RU1": "Russian Premier League",
    "UKR1": "Ukrainian Premier League",
    "GR1": "Super League Greece",
    "SA1": "Saudi Pro League",
    "BRA1": "Brasileirao Serie A",
    "NO1": "Eliteserien",
    "MLS1": "MLS",
    "SER1": "Serbian SuperLiga",
    "RO1": "Romanian SuperLiga",
    "PL1": "Ekstraklasa",
    "SE1": "Allsvenskan",
    "A1": "Austrian Bundesliga",
    "ARG1": "Liga Profesional",
    "C1": "Championship",
    "AUS1": "A-League Men",
    "MEX1": "Liga MX",
    "JAP1": "J1 League",
    "KR1": "K League 1",
}

CLUB_SHORT_OVERRIDES = {
    "Association sportive de Monaco Football Club": "AS Monaco",
    "Brighton and Hove Albion Football Club": "Brighton",
    "Club Atletico de Madrid S.A.D.": "Atletico Madrid",
    "Futebol Clube do Porto": "FC Porto",
    "Futbol Club Barcelona": "Barcelona",
    "Olympique Gymnaste Club Nice Cote d'Azur": "OGC Nice",
    "Olympique Gymnaste Club Nice Côte d'Azur": "OGC Nice",
    "Reial Club Deportiu Espanyol de Barcelona S.A.D.": "Espanyol",
    "Real Madrid Club de Futbol": "Real Madrid",
    "Real Madrid Club de Fútbol": "Real Madrid",
    "The Celtic Football Club": "Celtic FC",
}

CLUB_NAME_SUFFIXES = [
    " Football Club",
    " Fútbol Club S.A.D.",
    " Fútbol Club S. A. D.",
    " Futbol Club S.A.D.",
    " Futbol Club S. A. D.",
    " Fútbol Club",
    " Futbol Club",
    " Club de Fútbol",
    " Club de Futbol",
    " S.A.D.",
    " S. A. D.",
]

def current_league_label(row: pd.Series) -> str:
    comp_id = row.get("current_club_domestic_competition_id", None)
    if pd.notna(comp_id) and str(comp_id).strip():
        comp = str(comp_id).strip()
        return LEAGUE_LABELS.get(comp, comp)

    for col in ["current_league", "target_league_x", "target_league_y", "target_league"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            value = str(row[col]).strip()
            return value.split("-", 1)[1] if "-" in value else value

    return "N/A"

def current_club_label(row: pd.Series) -> str:
    for col in ["current_club_name", "current_club", "club"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col]).strip()
    return "N/A"

def comparable_id(value) -> str:
    if pd.isna(value):
        return ""
    raw = str(value).strip()
    try:
        return str(int(float(raw)))
    except ValueError:
        return raw

def matching_current_club_id(row: pd.Series) -> bool:
    current_id = comparable_id(row.get("current_club_id", None))
    club_id = comparable_id(row.get("club_id", None))
    return bool(current_id and club_id and current_id == club_id)

def clean_club_display_name(club_name: str) -> str:
    if club_name == "N/A":
        return club_name

    cleaned = CLUB_SHORT_OVERRIDES.get(club_name, club_name)
    for suffix in CLUB_NAME_SUFFIXES:
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip()
            break

    if cleaned.startswith("The "):
        cleaned = cleaned[4:].strip()

    if len(cleaned) > 36:
        cleaned = f"{cleaned[:33].rstrip()}..."
    return cleaned

def current_club_display_label(row: pd.Series) -> str:
    full_club = current_club_label(row)
    short_club = row.get("club", None)
    if (
        full_club != "N/A"
        and pd.notna(short_club)
        and str(short_club).strip()
        and matching_current_club_id(row)
        and len(str(short_club).strip()) < len(full_club)
    ):
        return str(short_club).strip()
    return clean_club_display_name(full_club)

def club_league_label(row: pd.Series) -> str:
    club = current_club_display_label(row)
    league = current_league_label(row)
    if club == "N/A":
        return league
    if league == "N/A":
        return club
    return f"{club} · {league}"

def add_download_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    download_df = df.copy()
    download_df["current_club"] = download_df.apply(current_club_label, axis=1)
    download_df["current_league"] = download_df.apply(current_league_label, axis=1)
    return download_df

def add_overview_context_columns(df: pd.DataFrame, club_context: pd.DataFrame) -> pd.DataFrame:
    overview_df = df.copy()
    if club_context.empty or "player_id" not in overview_df.columns:
        overview_df["Club / League"] = overview_df.apply(club_league_label, axis=1)
        return overview_df

    context_cols = [
        "player_id",
        "current_club_name",
        "current_club_domestic_competition_id",
    ]
    available_cols = [col for col in context_cols if col in club_context.columns]
    overview_df = overview_df.merge(
        club_context[available_cols].drop_duplicates("player_id"),
        on="player_id",
        how="left",
        suffixes=("", "_context"),
    )

    for col in ["current_club_name", "current_club_domestic_competition_id"]:
        context_col = f"{col}_context"
        if context_col in overview_df.columns:
            if col in overview_df.columns:
                overview_df[col] = overview_df[col].combine_first(overview_df[context_col])
            else:
                overview_df[col] = overview_df[context_col]
            overview_df = overview_df.drop(columns=[context_col])

    overview_df["Club / League"] = overview_df.apply(club_league_label, axis=1)
    return overview_df

def build_overview_display_table(df: pd.DataFrame) -> pd.DataFrame:
    display_cols = [
        "name",
        "Club / League",
        "age_at_valuation",
        "position_group_raw",
        "minutes_last_season",
        "target_market_value",
        "predicted_value",
        "undervalued_pct",
    ]
    available_cols = [col for col in display_cols if col in df.columns]
    display_df = df[available_cols].copy()

    if "age_at_valuation" in display_df.columns:
        display_df["age_at_valuation"] = display_df["age_at_valuation"].round(1)
    if "minutes_last_season" in display_df.columns:
        display_df["minutes_last_season"] = display_df["minutes_last_season"].round(0).astype(int)
    if "target_market_value" in display_df.columns:
        display_df["target_market_value"] = display_df["target_market_value"].apply(format_currency)
    if "predicted_value" in display_df.columns:
        display_df["predicted_value"] = display_df["predicted_value"].apply(format_currency)
    if "undervalued_pct" in display_df.columns:
        display_df["undervalued_pct"] = display_df["undervalued_pct"].apply(format_percentage)

    return display_df.rename(
        columns={
            "name": "Player Name",
            "age_at_valuation": "Age",
            "position_group_raw": "Position",
            "minutes_last_season": "Minutes",
            "target_market_value": "Actual Value",
            "predicted_value": "Predicted Value",
            "undervalued_pct": "Undervaluation %",
        }
    )

def build_similarity_display_table(res_df: pd.DataFrame, target_position: str, mode: str = "basic") -> pd.DataFrame:
    """Formats and filters columns for the similarity results table based on player position."""
    table = res_df.copy()

    table["Age"] = table["age_at_valuation"].round(1)
    table["Age Diff"] = table["age_difference"].apply(lambda x: f"{x:+.1f} yrs")
    table["Height"] = table["height_in_cm"].apply(lambda x: f"{x:.0f} cm" if pd.notna(x) else "N/A")
    table["Minutes"] = table["minutes_last_season"].round(0).astype(int)

    table["Goals/90"] = table["goals_per_90_ls"].round(2)
    table["Assists/90"] = table["assists_per_90_ls"].round(2)
    table["Cards/90"] = table["cards_per_90_ls"].round(2)

    table["Market Value"] = table["target_market_value"].apply(lambda x: f"€{x:,.0f}")
    table["Value Diff"] = table["value_difference_vs_target"].apply(lambda x: f"€{x:+,.0f}")
    table["Predicted Value"] = table["predicted_value"].apply(lambda x: f"€{x:,.0f}")
    table["Stat Score"] = table.get("statistical_score", table["similarity_score"]).apply(format_score)
    role_score = table["role_compatibility_score"] if "role_compatibility_score" in table.columns else pd.Series(0, index=table.index)
    final_score = table["final_similarity_score"] if "final_similarity_score" in table.columns else table["similarity_score"]
    table["Role Score"] = role_score.apply(format_score)
    table["Final Score"] = final_score.apply(format_score)
    table["Primary Role"] = table["primary_role"].fillna("UNKNOWN") if "primary_role" in table.columns else "UNKNOWN"
    role_match = table["role_matching_mode"] if "role_matching_mode" in table.columns else pd.Series("broad", index=table.index)
    table["Role Match"] = role_match.fillna("broad").str.replace("_", " ").str.title()
    table["Foot"] = table["foot"].fillna("N/A").str.title() if "foot" in table.columns else "N/A"
    table["Club / League"] = table.apply(club_league_label, axis=1)

    table["Cheaper?"] = table["is_cheaper"].apply(lambda x: "✅ Yes" if x else "❌ No")
    table["Younger?"] = table["is_younger"].apply(lambda x: "✅ Yes" if x else "❌ No")
    table["Undervalued?"] = table["is_undervalued"].apply(lambda x: "💎 Yes" if x else "— No")

    base_cols = ["name", "Club / League", "Primary Role", "Role Match", "Foot", "Age", "Age Diff"]
    value_cols = [
        "Market Value", "Value Diff", "Predicted Value", "Stat Score",
        "Role Score", "Final Score", "Cheaper?", "Younger?", "Undervalued?"
    ]

    if mode == "enriched":
        # Add position-specific advanced columns
        if target_position == "Forward":
            table["xG"] = table["adv_Expected_xG"].round(2)
            table["SoT%"] = table["adv_Standard_SoT%"].round(1)
            table["SCA90"] = table["adv_SCA_SCA90"].round(2)
            table["Take-On%"] = table["adv_Take-Ons_Succ%"].round(1)
            adv_cols = ["xG", "SoT%", "SCA90", "Take-On%"]
        elif target_position == "Midfielder":
            table["Key Passes"] = table["adv_KP_"].round(2)
            table["PrgP"] = table["adv_Progression_PrgP"].round(2)
            table["Tackles W"] = table["adv_Tackles_TklW"].round(2)
            table["Int"] = table["adv_Int_"].round(2)
            adv_cols = ["Key Passes", "PrgP", "Tackles W", "Int"]
        elif target_position == "Defender":
            table["Blocks"] = table["adv_Blocks_Blocks"].round(2)
            table["Clearance"] = table["adv_Clr_"].round(2)
            table["Aerial%"] = table["adv_Aerial Duels_Won%"].round(1)
            table["Tkl%"] = table["adv_Challenges_Tkl%"].round(1)
            adv_cols = ["Blocks", "Clearance", "Aerial%", "Tkl%"]
        else: # Goalkeeper
            table["Pass Cmp%"] = table["adv_Total_Cmp%"].round(1)
            table["Recoveries"] = table["adv_Performance_Recov"].round(2)
            table["Aerial Win%"] = table["adv_Aerial Duels_Won%"].round(1)
            adv_cols = ["Pass Cmp%", "Recoveries", "Aerial Win%"]
        
        selected_cols = base_cols + adv_cols + value_cols
    else:
        # Basic mode
        if target_position == "Goalkeeper":
            selected_cols = base_cols + ["Height", "Minutes"] + value_cols
        elif target_position == "Defender":
            selected_cols = base_cols + ["Height", "Minutes", "Cards/90"] + value_cols
        else: # Forward / Midfielder
            selected_cols = base_cols + ["Goals/90", "Assists/90", "Cards/90"] + value_cols

    final_table = table[selected_cols].copy()
    final_table = final_table.rename(columns={"name": "Player Name"})
    return final_table


# =========================
# Resource Loading
# =========================
@st.cache_data
def load_scouting_data() -> pd.DataFrame:
    """Load scouting output for the overview tab."""
    candidate_paths = [
        os.path.join("outputs", "undervalued_candidates_overall.csv"),
        os.path.join("outputs", "hidden_gems_overall.csv"),
    ]
    for file_path in candidate_paths:
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
    return pd.DataFrame()

@st.cache_data
def load_current_club_context() -> pd.DataFrame:
    """Load current Transfermarkt club context for display-only dashboard enrichment."""
    context_path = os.path.join("data", "processed", "featured_players.csv")
    if not os.path.exists(context_path):
        return pd.DataFrame()

    context_cols = [
        "player_id",
        "current_club_name",
        "current_club_domestic_competition_id",
    ]
    df_context = pd.read_csv(context_path)
    df_context = df_context[[col for col in context_cols + ["current_club_id"] if col in df_context.columns]].copy()

    clubs_path = os.path.join("data", "raw", "clubs.csv")
    if os.path.exists(clubs_path) and "current_club_id" in df_context.columns:
        clubs = pd.read_csv(clubs_path)
        club_cols = ["club_id", "name", "domestic_competition_id"]
        clubs = clubs[[col for col in club_cols if col in clubs.columns]].drop_duplicates("club_id")
        df_context = df_context.merge(
            clubs,
            left_on="current_club_id",
            right_on="club_id",
            how="left",
        )
        if "name" in df_context.columns:
            df_context["current_club_name"] = df_context["current_club_name"].combine_first(df_context["name"])
        if "domestic_competition_id" in df_context.columns:
            df_context["current_club_domestic_competition_id"] = (
                df_context["current_club_domestic_competition_id"]
                .combine_first(df_context["domestic_competition_id"])
            )

    return df_context[[col for col in context_cols if col in df_context.columns]].copy()

def load_similarity_engine():
    """Load the similarity search model."""
    engine_path = os.path.join("outputs", "models", "similarity_engine.pkl")
    if os.path.exists(engine_path):
        return joblib.load(engine_path)
    return None


# =========================
# Header
# =========================
st.title("⚽ Football Talent Scouting Dashboard")

tab_scouting, tab_similarity = st.tabs(["💎 Scouting Overview", "🔍 Similarity Search"])

# =========================
# TAB 1: SCOUTING OVERVIEW
# =========================
with tab_scouting:
    df = load_scouting_data()
    club_context_df = load_current_club_context()
    
    if df.empty:
        st.error("No scouting data found. Please run the pipeline first.")
    else:
        # Sidebar Filters
        st.sidebar.header("📊 Global Shortlist Filters")
        st.sidebar.info("These filters apply to the **Scouting Overview** candidates. In **Similarity Search**, these filters apply to the alternatives found, not the target player.")
        
        # We reuse the existing sidebar filters for the overview
        positions = sorted(df["position_group_raw"].dropna().unique().tolist())
        selected_positions = st.sidebar.multiselect("Position Group", positions, default=positions)
        
        min_age, max_age = int(df["age_at_valuation"].min()), int(df["age_at_valuation"].max())
        selected_age = st.sidebar.slider("Age Range", min_age, max_age, (min_age, 25))
        
        min_val, max_val = int(df["target_market_value"].min()), int(df["target_market_value"].max())
        selected_val = st.sidebar.slider("Max Market Value (€)", min_val, max_val, min(max(5000000, min_val), max_val), step=500000)
        
        min_mins, max_mins = int(df["minutes_last_season"].min()), int(df["minutes_last_season"].max())
        selected_mins = st.sidebar.slider("Minimum Minutes Played", min_mins, max_mins, min(max(900, min_mins), max_mins))
        
        # Apply Filters
        filtered_df = df[
            (df["position_group_raw"].isin(selected_positions)) &
            (df["age_at_valuation"].between(selected_age[0], selected_age[1])) &
            (df["target_market_value"] <= selected_val) &
            (df["minutes_last_season"] >= selected_mins)
        ].copy()
        
        filtered_df = filtered_df.sort_values(by="undervalued_pct", ascending=False)
        
        # Metrics
        st.markdown("### 📊 Key Metrics")
        k1, k2, k3, k4 = st.columns(4)
        with k1: kpi_card("Candidates Found", f"{len(filtered_df):,}")
        with k2: kpi_card("Average Age", f"{filtered_df['age_at_valuation'].mean():.1f} yrs" if not filtered_df.empty else "N/A")
        with k3: kpi_card("Average Undervaluation %", f"{filtered_df['undervalued_pct'].mean()*100:.1f}%" if not filtered_df.empty else "N/A")
        with k4: kpi_card("Median Actual Value", f"€{filtered_df['target_market_value'].median():,.0f}" if not filtered_df.empty else "N/A")
        
        methodology_box("This overview uses the performance-only model to identify players whose recent statistical output is higher than their current market value baseline.")
        st.divider()
        
        if not filtered_df.empty:
            overview_df = add_overview_context_columns(filtered_df, club_context_df)
            st.markdown("### 📈 Visual Analysis")
            c1, c2 = st.columns(2)
            
            with c1:
                fig1 = px.scatter(
                    filtered_df, x="age_at_valuation", y="undervalued_pct",
                    color="position_group_raw", size="target_market_value", hover_name="name",
                    title="Age vs Undervaluation", template=PLOTLY_TEMPLATE,
                    labels={
                        "age_at_valuation": "Age",
                        "undervalued_pct": "Undervaluation %",
                        "position_group_raw": "Position",
                        "target_market_value": "Actual Market Value (€)",
                        "name": "Player Name",
                    },
                )
                fig1.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig1, use_container_width=True)
                
            with c2:
                fig2 = px.scatter(
                    filtered_df, x="target_market_value", y="predicted_value",
                    color="position_group_raw", hover_name="name",
                    title="Actual vs Predicted Market Value", template=PLOTLY_TEMPLATE,
                    labels={
                        "target_market_value": "Actual Market Value (€)",
                        "predicted_value": "Predicted Market Value (€)",
                        "position_group_raw": "Position",
                        "name": "Player Name",
                    },
                )
                max_l = max(filtered_df["target_market_value"].max(), filtered_df["predicted_value"].max())
                fig2.add_shape(type="line", line=dict(dash="dash", color=REFERENCE_LINE_COLOR), x0=0, y0=0, x1=max_l, y1=max_l)
                st.plotly_chart(fig2, use_container_width=True)
                st.caption("Points above the dashed line indicate players predicted to be worth more than their current market value.")
            
            st.divider()
            st.markdown("### 📋 Top Undervalued Candidates")
            
            display_df = build_overview_display_table(overview_df)
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            download_df = add_download_context_columns(overview_df)
            st.download_button(
                label="💾 Download Filtered Shortlist",
                data=download_df.to_csv(index=False),
                file_name="filtered_undervalued_shortlist.csv",
                mime="text/csv",
            )
            disclaimer_box("Statistical leads should be validated by professional human scouting.")
        else:
            st.warning("Adjust filters to see candidates.")


# =========================
# TAB 2: SIMILARITY SEARCH
# =========================
with tab_similarity:
    engine = load_similarity_engine()
    
    if engine is None:
        st.error("Similarity engine not found. Please run `python src/similarity.py` first.")
    else:
        st.markdown("### 🔍 Find Statistical Twin")
        st.markdown("Search for players with similar performance profiles and find cheaper or younger alternatives.")
        # Player Selection
        all_players = engine.data.sort_values("name")
        player_options = {row["name"]: row["player_id"] for _, row in all_players.iterrows()}
        
        target_player_name = st.selectbox(
            "Select Target Player to find 'Statistical Twins'",
            options=list(player_options.keys()),
            index=list(player_options.keys()).index("Aaron Ramsdale") if "Aaron Ramsdale" in player_options else 0
        )
        target_player_id = player_options[target_player_name]
        
        target_data = engine.data[engine.data['player_id'] == target_player_id].iloc[0]
        
        # Defensive check for enrichment metadata (v1.3 migration)
        if 'enriched_available' in target_data:
            has_enriched = scalar_bool(target_data['enriched_available'])
        elif 'match_status' in target_data:
            has_enriched = target_data['match_status'] == 'matched'
        else:
            # Fallback if engine artifact is outdated
            has_enriched = False
            st.sidebar.error("⚠️ Similarity data is outdated. Please re-run 'python src/similarity.py' to enable enriched mode.")
        has_role_metadata = scalar_bool(target_data.get('role_metadata_available', False))

        # Search Settings
        col_s1, col_s2, col_s3 = st.columns([1, 1, 1])
        with col_s1:
            search_mode = st.radio(
                "Search Mode",
                ["Statistical Twins", "Recruitment Alternatives"],
                index=1,
                horizontal=True,
                help="Statistical Twins: Purely similar profiles. Recruitment Alternatives: Similar profiles who are YOUNGER and CHEAPER than the target."
            )
        with col_s2:
            role_mode_label = st.selectbox(
                "Role Matching Mode",
                ["Compatible Roles", "Exact Role", "Broad Position Group"],
                index=0,
                help="Compatible Roles uses the v1.4 role compatibility matrix. Exact Role requires the same primary role. Broad Position Group falls back to the v1.3 position-group behavior."
            )
            role_mode_map = {
                "Compatible Roles": "compatible",
                "Exact Role": "exact",
                "Broad Position Group": "broad",
            }
            role_mode_value = role_mode_map[role_mode_label]
        with col_s3:
            if has_enriched:
                sim_engine_mode = st.selectbox(
                    "Similarity Model",
                    ["Enriched (Advanced Stats)", "Basic (Core Stats)"],
                    index=0,
                    help="Enriched mode uses advanced metrics (xG, Key Passes, Clearances, etc.) from Top 5 leagues. Basic mode uses only core stats (Goals, Assists, Cards)."
                )
                actual_mode = "enriched" if "Enriched" in sim_engine_mode else "basic"
                st.markdown('<span class="advanced-stats-badge">Advanced Stats Available</span>', unsafe_allow_html=True)
            else:
                st.selectbox("Similarity Model", ["Basic (Core Stats)"], disabled=True)
                actual_mode = "basic"
                st.markdown('<span class="basic-stats-badge">Core Stats Only</span>', unsafe_allow_html=True)

        # Show target profile
        st.markdown("#### 👤 Target Profile")
        target_role = target_data.get('primary_role', 'UNKNOWN') if has_role_metadata else "UNKNOWN"
        target_tags = format_role_tags(target_data.get('role_tags', None))
        target_foot = format_foot_profile(target_data.get('foot', None), has_role_metadata)
        role_chips = [f"Role: {target_role}", target_foot]
        if not role_tags_are_redundant(str(target_role), target_tags):
            role_chips.insert(1, f"Tags: {target_tags}")
        role_metadata_label = "Metadata available" if has_role_metadata else "Broad-position fallback"
        role_chips.append(role_metadata_label)
        match_method = target_data.get('match_method', 'N/A' if has_enriched else 'none')
        match_score = target_data.get('match_score', 0)
        match_method_label = format_match_method(match_method)
        match_confidence_label = format_match_confidence(match_score) if has_enriched else "N/A"
        stats_season_label = format_stats_season(target_data.get('kaggle_season', None))
        similarity_mode_label = (
            "Enriched" if has_enriched and actual_mode == "enriched"
            else "Basic" if has_enriched
            else "Basic Fallback"
        )
        if has_enriched and actual_mode == "enriched":
            data_context = f"Using Enriched advanced stats from {stats_season_label}. Player metadata matched via {match_method_label} validation."
        elif has_enriched:
            data_context = f"Using Basic core stats by selection. Enriched stats are available from {stats_season_label} but not used in this search."
        else:
            data_context = "Using Basic Fallback with core stats only. No accepted external enrichment matched for this target."

        st.markdown(
            f"""
            <div class="target-panel">
                <div class="target-name">{html.escape(str(target_player_name))}</div>
                <div class="target-summary">
                    {html.escape(str(target_data['position_group_raw']))} · Primary role: {html.escape(str(target_role))} · {html.escape(target_foot)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        p1, p2, p3, p4 = st.columns(4)
        with p1: kpi_card("Position", target_data['position_group_raw'])
        with p2: kpi_card("Age", f"{target_data['age_at_valuation']:.1f}")
        with p3: kpi_card("Market Value", f"€{target_data['target_market_value']:,.0f}")
        with p4: kpi_card("Minutes Played", f"{target_data['minutes_last_season']:.0f}")

        st.markdown("##### Role Profile")
        st.markdown(render_chip_row(role_chips), unsafe_allow_html=True)

        st.markdown("##### Data Context")
        data_chips = [
            f"{similarity_mode_label} mode",
            "Advanced stats available" if has_enriched else "Core stats only",
            f"Stats season: {stats_season_label}",
        ]
        st.markdown(render_chip_row(data_chips, "data-chip"), unsafe_allow_html=True)
        st.caption(data_context)

        with st.expander("View data matching details"):
            details_df = pd.DataFrame(
                [
                    {"Field": "Similarity Mode", "Value": similarity_mode_label},
                    {"Field": "Match Method", "Value": match_method_label},
                    {"Field": "Match Confidence", "Value": match_confidence_label},
                    {"Field": "Stats Season", "Value": stats_season_label},
                ]
            )
            st.dataframe(details_df, use_container_width=True, hide_index=True)
            st.caption("Match confidence refers to the player metadata matching process between datasets. It is not the similarity score.")

            fuzzy_note = fuzzy_confidence_note(match_method, match_score)
            if fuzzy_note:
                st.caption(fuzzy_note)
            if has_enriched and actual_mode == "basic":
                st.info("Using Basic Mode by selection. Advanced stats are available for this target but are not used in this search.")
            elif not has_enriched:
                st.info("Using Basic Fallback (Core Stats only). No accepted external enrichment matched for this target.")

        
        st.divider()
        
        # Similarity Search
        st.caption(f"Top alternatives for **{target_player_name}** using **{actual_mode.upper()}** statistical mode and **{role_mode_label}** role logic.")
        if role_mode_value == "broad":
            st.warning("⚠️ Broad Position Group uses v1.3 fallback logic and may produce less role-specific matches.")
        if not has_role_metadata:
            st.info("Role metadata is not available for this target, so the engine falls back to broad position-group matching.")
        
        # Get a larger pool first (50) to allow for recruitment filtering
        similar_df = engine.find_similar(
            target_player_id,
            top_n=50,
            same_position=True,
            mode=actual_mode,
            role_mode=role_mode_value,
            role_threshold=COMPATIBLE_ROLE_THRESHOLD,
        )
            
        if similar_df is not None and not similar_df.empty:
            # Apply Recruitment Filter if selected
            if search_mode == "Recruitment Alternatives":
                # Filter for younger AND cheaper
                similar_df = similar_df[
                    (similar_df["is_younger"]) & 
                    (similar_df["is_cheaper"])
                ].copy()
            
            # Take final top 10 after filtering
            similar_df = similar_df.head(10)
            
            if similar_df.empty:
                st.warning(f"No candidates found in '{search_mode}' mode. Try switching to 'Statistical Twins' or disabling strict position filtering.")
            else:
                # Format for display using the new position-aware function
                target_position = target_data["position_group_raw"]
                final_display = build_similarity_display_table(
                    res_df=similar_df,
                    target_position=target_position,
                    mode=actual_mode
                )
                
                st.dataframe(final_display, use_container_width=True, hide_index=True)
                with st.expander("How to read this table"):
                    st.markdown(
                        "- **Club / League** uses current Transfermarkt club and competition context when available; enriched stats still follow the displayed stats season.\n"
                        "- **Stat Score** measures performance-profile similarity before role weighting.\n"
                        "- **Role Score** measures tactical compatibility. 100% means the same role; lower scores indicate compatible adjacent roles.\n"
                        "- **Final Score** combines statistical similarity, role compatibility, and preferred-foot fit."
                    )
                    if actual_mode == "enriched":
                        st.markdown("- **Enriched mode** uses Kaggle Top 5 League advanced stats and includes only accepted external matches.")
                    elif target_position == "Goalkeeper":
                        st.markdown("- **Basic goalkeeper mode** relies mainly on age, height, and playing time because detailed save metrics are not available.")
                    elif target_position == "Defender":
                        st.markdown("- **Basic defender mode** uses available profile and discipline metrics; detailed defensive actions are only available in Enriched mode.")
                    else:
                        st.markdown("- **Basic attacking mode** uses core contribution metrics: Goals/90, Assists/90, and Cards/90.")
                
                # Download
                st.download_button(
                    label=f"💾 Download Alternatives for {target_player_name}",
                    data=add_download_context_columns(similar_df).to_csv(index=False),
                    file_name=f"alternatives_{target_player_name.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No similar players found for the selected role mode. Try Compatible Roles or Broad Position Group.")

        st.divider()
        methodology_box("Similarity search supports statistical and role layers. **Enriched Mode** (v1.3) uses advanced Top 5 League metrics matched using season-safe enrichment. **Role-Aware Mode** (v1.4) adds Transfermarkt role metadata, compatible-role filtering, and a small foot/side-fit bonus. Market value and predicted valuation are used strictly for post-match filtering and comparison, never for calculating similarity.")
        disclaimer_box("Candidates identified here are statistical leads based on predictive modeling of recent performance and should be interpreted as a starting point for professional human scouting.")
