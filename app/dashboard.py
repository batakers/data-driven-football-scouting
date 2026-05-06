import os
import sys
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

def format_score(value) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.2f}%"

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
    table["Statistical Score"] = table.get("statistical_score", table["similarity_score"]).apply(format_score)
    role_score = table["role_compatibility_score"] if "role_compatibility_score" in table.columns else pd.Series(0, index=table.index)
    final_score = table["final_similarity_score"] if "final_similarity_score" in table.columns else table["similarity_score"]
    table["Role Score"] = role_score.apply(format_score)
    table["Final Match Score"] = final_score.apply(format_score)
    table["Primary Role"] = table["primary_role"].fillna("UNKNOWN") if "primary_role" in table.columns else "UNKNOWN"
    role_match = table["role_matching_mode"] if "role_matching_mode" in table.columns else pd.Series("broad", index=table.index)
    table["Role Match"] = role_match.fillna("broad").str.replace("_", " ").str.title()
    table["Foot"] = table["foot"].fillna("N/A").str.title() if "foot" in table.columns else "N/A"

    table["Cheaper?"] = table["is_cheaper"].apply(lambda x: "✅ Yes" if x else "❌ No")
    table["Younger?"] = table["is_younger"].apply(lambda x: "✅ Yes" if x else "❌ No")
    table["Undervalued?"] = table["is_undervalued"].apply(lambda x: "💎 Yes" if x else "— No")

    base_cols = ["name", "Primary Role", "Role Match", "Foot", "Age", "Age Diff"]
    value_cols = [
        "Market Value", "Value Diff", "Predicted Value", "Statistical Score",
        "Role Score", "Final Match Score", "Cheaper?", "Younger?", "Undervalued?"
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
            table["PassCmp%"] = table["adv_Total_Cmp%"].round(1)
            table["Recov"] = table["adv_Performance_Recov"].round(2)
            table["Aerial%"] = table["adv_Aerial Duels_Won%"].round(1)
            adv_cols = ["PassCmp%", "Recov", "Aerial%"]
        
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
        with k3: kpi_card("Avg Undervaluation %", f"{filtered_df['undervalued_pct'].mean()*100:.1f}%" if not filtered_df.empty else "N/A")
        with k4: kpi_card("Median Market Value", f"€{filtered_df['target_market_value'].median():,.0f}" if not filtered_df.empty else "N/A")
        
        methodology_box("This dashboard identifies players whose statistical performance exceeds their market value.")
        st.divider()
        
        if not filtered_df.empty:
            st.markdown("### 📈 Visual Analysis")
            c1, c2 = st.columns(2)
            
            with c1:
                fig1 = px.scatter(
                    filtered_df, x="age_at_valuation", y="undervalued_pct",
                    color="position_group_raw", size="target_market_value", hover_name="name",
                    title="Age vs Undervaluation Ratio", template=PLOTLY_TEMPLATE
                )
                st.plotly_chart(fig1, use_container_width=True)
                
            with c2:
                fig2 = px.scatter(
                    filtered_df, x="target_market_value", y="predicted_value",
                    color="position_group_raw", hover_name="name",
                    title="Actual vs Predicted Value", template=PLOTLY_TEMPLATE
                )
                max_l = max(filtered_df["target_market_value"].max(), filtered_df["predicted_value"].max())
                fig2.add_shape(type="line", line=dict(dash="dash", color=REFERENCE_LINE_COLOR), x0=0, y0=0, x1=max_l, y1=max_l)
                st.plotly_chart(fig2, use_container_width=True)
            
            st.divider()
            st.markdown("### 📋 Shortlist")
            
            display_cols = ['name', 'age_at_valuation', 'position_group_raw', 'minutes_last_season', 'target_market_value', 'predicted_value', 'undervalued_pct']
            display_df = filtered_df[display_cols].copy()
            display_df['age_at_valuation'] = display_df['age_at_valuation'].round(1)
            display_df['target_market_value'] = display_df['target_market_value'].apply(lambda x: f"€{x:,.0f}")
            display_df['predicted_value'] = display_df['predicted_value'].apply(lambda x: f"€{x:,.0f}")
            display_df['undervalued_pct'] = display_df['undervalued_pct'].apply(lambda x: f"{x*100:.1f}%")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
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
        p1, p2, p3, p4 = st.columns(4)
        with p1: kpi_card("Position", target_data['position_group_raw'])
        with p2: kpi_card("Age", f"{target_data['age_at_valuation']:.1f}")
        with p3: kpi_card("Market Value", f"€{target_data['target_market_value']:,.0f}")
        with p4: kpi_card("Minutes Played", f"{target_data['minutes_last_season']:.0f}")

        r1, r2, r3, r4, r5 = st.columns(5)
        with r1: kpi_card("Primary Role", target_data.get('primary_role', 'UNKNOWN'))
        with r2: kpi_card("Role Tags", format_role_tags(target_data.get('role_tags', None)))
        with r3: kpi_card("Foot", str(target_data.get('foot', 'N/A')).title() if has_role_metadata else "N/A")
        with r4: kpi_card("Role Mode", role_mode_label)
        with r5: kpi_card("Role Metadata", "Available" if has_role_metadata else "Fallback")
        
        # Enrichment Metadata Display
        if has_enriched and actual_mode == "enriched":
            m1, m2, m3, m4 = st.columns(4)
            with m1: st.metric("Similarity Mode", "Enriched")
            with m2: st.metric("Match Method", target_data.get('match_method', 'N/A').replace('_', ' ').title())
            with m3: st.metric("Match Confidence", f"{target_data.get('match_score', 0):.0f}%")
            with m4: st.metric("Stats Season", format_stats_season(target_data.get('kaggle_season', None)))
        elif actual_mode == "basic":
            m1, m2, m3, m4 = st.columns(4)
            with m1: st.metric("Similarity Mode", "Basic" if has_enriched else "Basic Fallback")
            with m2: st.metric("Match Method", target_data.get('match_method', 'none').replace('_', ' ').title())
            with m3: st.metric("Match Confidence", f"{target_data.get('match_score', 0):.0f}%" if has_enriched else "N/A")
            with m4: st.metric("Stats Season", format_stats_season(target_data.get('kaggle_season', None)))
            if has_enriched:
                st.info("ℹ️ Using Basic Mode by selection. Advanced stats are available for this target but are not used in this search.")
            else:
                st.info("ℹ️ Using Basic Fallback (Core Stats only). No accepted external enrichment matched for this target.")

        
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
                st.caption("Role Score measures tactical compatibility between the target role and candidate role. 100% means the same role; lower scores indicate compatible adjacent roles.")
                
                # Position-specific captions
                if actual_mode == "enriched":
                    st.caption("Advanced similarity uses Kaggle Top 5 League stats. Only players matched with the external dataset are included in this mode.")
                else:
                    if target_position == "Goalkeeper":
                        st.caption("Goalkeeper similarity is based mainly on age, height, and playing time because detailed save or clean-sheet metrics are not available in basic mode.")
                    elif target_position == "Defender":
                        st.caption("Defender similarity is based on available profile and discipline metrics. Detailed defensive actions are only available in Enriched mode.")
                    else:
                        st.caption("Attacking similarity uses core contribution metrics: Goals/90, Assists/90, and Cards/90.")
                
                # Download
                st.download_button(
                    label=f"💾 Download Alternatives for {target_player_name}",
                    data=similar_df.to_csv(index=False),
                    file_name=f"alternatives_{target_player_name.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No similar players found for the selected role mode. Try Compatible Roles or Broad Position Group.")

        st.divider()
        methodology_box("Similarity search supports statistical and role layers. **Enriched Mode** (v1.3) uses advanced Top 5 League metrics matched using season-safe enrichment. **Role-Aware Mode** (v1.4) adds Transfermarkt role metadata, compatible-role filtering, and a small foot/side-fit bonus. Market value and predicted valuation are used strictly for post-match filtering and comparison, never for calculating similarity.")
        disclaimer_box("Candidates identified here are statistical leads based on predictive modeling of recent performance and should be interpreted as a starting point for professional human scouting.")
