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

from src.similarity import PlayerSimilarity


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

def build_similarity_display_table(res_df: pd.DataFrame, target_position: str) -> pd.DataFrame:
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
    table["Match Score"] = (table["similarity_score"] * 100).apply(lambda x: f"{x:.2f}%")

    table["Cheaper?"] = table["is_cheaper"].apply(lambda x: "✅ Yes" if x else "❌ No")
    table["Younger?"] = table["is_younger"].apply(lambda x: "✅ Yes" if x else "❌ No")
    table["Undervalued?"] = table["is_undervalued"].apply(lambda x: "💎 Yes" if x else "— No")

    base_cols = ["name", "Age", "Age Diff"]
    value_cols = ["Market Value", "Value Diff", "Predicted Value", "Match Score", "Cheaper?", "Younger?", "Undervalued?"]

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

@st.cache_resource
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
        
        # Search Settings
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            same_pos_toggle = st.checkbox("Strict: Same Position Group Only", value=True)
        with col_s2:
            search_mode = st.radio(
                "Search Mode",
                ["Statistical Twins", "Recruitment Alternatives"],
                index=1, # Default to Recruitment for scouting use case
                horizontal=True,
                help="Statistical Twins: Purely similar profiles. Recruitment Alternatives: Similar profiles who are YOUNGER and CHEAPER than the target."
            )
        
        # Show target profile
        target_data = engine.data[engine.data['player_id'] == target_player_id].iloc[0]
        
        st.markdown("#### 👤 Target Profile")
        p1, p2, p3, p4 = st.columns(4)
        with p1: kpi_card("Position", target_data['position_group_raw'])
        with p2: kpi_card("Age", f"{target_data['age_at_valuation']:.1f}")
        with p3: kpi_card("Market Value", f"€{target_data['target_market_value']:,.0f}")
        with p4: kpi_card("Minutes Played", f"{target_data['minutes_last_season']:.0f}")
        
        st.divider()
        
        # Similarity Search
        st.caption(f"Top alternatives for **{target_player_name}** based on statistical performance profile.")
        if not same_pos_toggle:
            st.warning("⚠️ Disabling strict position filtering may produce tactically less relevant matches.")
        
        # Get a larger pool first (50) to allow for recruitment filtering
        similar_df = engine.find_similar(target_player_id, top_n=50, same_position=same_pos_toggle)
            
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
                    target_position=target_position
                )
                
                st.dataframe(final_display, use_container_width=True, hide_index=True)
                
                # Position-specific captions
                if target_position == "Goalkeeper":
                    st.caption("Goalkeeper similarity is based mainly on age, height, and playing time because detailed save or clean-sheet metrics are not available.")
                elif target_position == "Defender":
                    st.caption("Defender similarity is based on available profile and discipline metrics. Detailed defensive actions such as tackles, interceptions, and clearances are not available.")
                else:
                    st.caption("Attacking similarity uses available recent contribution metrics: Goals/90, Assists/90, and Cards/90.")
                
                # Download
                st.download_button(
                    label=f"💾 Download Alternatives for {target_player_name}",
                    data=similar_df.to_csv(index=False),
                    file_name=f"alternatives_{target_player_name.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
        else:
            if same_pos_toggle:
                st.info("No similar players found within the same position group. Try unchecking 'Strict: Same Position Group Only'.")
            else:
                st.info("No similar players found. Try selecting another target player.")

        st.divider()
        methodology_box("Similarity uses position-aware feature weighting based on available recent features. Forwards and midfielders rely more on attacking contribution metrics, while defenders and goalkeepers rely more on age, height, and disciplinary profile due to limited defensive/GK-specific data. Market value is used only for comparison, not for similarity calculation.")
        disclaimer_box("Candidates identified here are statistical leads based on predictive modeling of recent performance and should be interpreted as a starting point for professional human scouting.")