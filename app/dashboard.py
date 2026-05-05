import os
import joblib

import pandas as pd
import plotly.express as px
import streamlit as st
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
}

.disclaimer-box {
    background-color: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.45);
    color: var(--text-color);
    padding: 0.85rem 1rem;
    border-radius: 8px;
    font-size: 0.92rem;
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
        # Sidebar Filters (Moved into the tab logic or remains global)
        st.sidebar.header("🔍 Global Filters")
        
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
        all_players = sorted(engine.data['name'].tolist())
        target_player = st.selectbox("Select a Player to find alternatives:", all_players, index=0)
        
        if target_player:
            # Show Target Profile
            target_data = engine.data[engine.data['name'] == target_player].iloc[0]
            
            st.markdown("#### 👤 Target Profile")
            p1, p2, p3, p4 = st.columns(4)
            with p1: kpi_card("Position", target_data['position_group_raw'])
            with p2: kpi_card("Age", f"{target_data['age_at_valuation']:.1f}")
            with p3: kpi_card("Market Value", f"€{target_data['target_market_value']:,.0f}")
            with p4: kpi_card("Minutes Played", f"{target_data['minutes_last_season']:.0f}")
            
            st.divider()
            
            # Similarity Search
            st.markdown("#### 💎 Top 10 Similar Alternatives")
            st.caption(f"Players in the same position group with the most similar playing style to **{target_player}**.")
            
            similar_df = engine.find_similar(target_player, top_n=10)
            
            if similar_df is not None and not similar_df.empty:
                # Format for display
                res_df = similar_df[['name', 'age_at_valuation', 'minutes_last_season', 'target_market_value', 'predicted_value', 'undervalued_pct', 'similarity_score', 'is_cheaper', 'is_younger']].copy()
                
                # Add highlighting logic for table display
                res_df['Age'] = res_df['age_at_valuation'].round(1)
                res_df['Similarity %'] = (res_df['similarity_score'] * 100).round(1).astype(str) + "%"
                res_df['Market Value'] = res_df['target_market_value'].apply(lambda x: f"€{x:,.0f}")
                res_df['Predicted Value'] = res_df['predicted_value'].apply(lambda x: f"€{x:,.0f}")
                res_df['Undervaluation %'] = (res_df['undervalued_pct'] * 100).round(1).astype(str) + "%"
                
                # Flag columns
                res_df['Cheaper?'] = res_df['is_cheaper'].apply(lambda x: "✅" if x else "❌")
                res_df['Younger?'] = res_df['is_younger'].apply(lambda x: "✅" if x else "❌")
                res_df['Undervalued?'] = res_df['is_undervalued'].apply(lambda x: "💎" if x else "⚪")
                
                final_display = res_df[['name', 'Age', 'Market Value', 'Predicted Value', 'Undervaluation %', 'Similarity %', 'Cheaper?', 'Younger?', 'Undervalued?']]
                final_display.columns = ['Player Name', 'Age', 'Market Value', 'Predicted Value', 'Undervaluation %', 'Match Score', 'Cheaper?', 'Younger?', 'Undervalued?']
                
                st.dataframe(final_display, use_container_width=True, hide_index=True)
                
                # Download
                st.download_button(
                    label=f"💾 Download Alternatives for {target_player}",
                    data=similar_df.to_csv(index=False),
                    file_name=f"alternatives_{target_player.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No similar players found within the same position group.")

        st.divider()
        methodology_box("Similarity is calculated using performance metrics (Goals, Assists, Cards per 90) and Age. Market value and predicted undervaluation are used only for comparison, not for similarity calculation.")
        disclaimer_box("Candidates identified here are statistical leads based on predictive modeling and should be interpreted as a starting point for professional human scouting."
)