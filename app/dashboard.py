import os

import pandas as pd
import plotly.express as px
import streamlit as st


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
# Header
# =========================
st.title("⚽ Transfermarkt Talent Scouting Dashboard")

st.markdown(
    """
Welcome to the Data-Driven Talent Scouting Dashboard!  
This tool explores **initial undervalued candidates** identified by our **Performance-Only XGBoost Model**.  
Use the filters on the left to find players whose statistical output significantly exceeds their current market value.
"""
)


# =========================
# Load Data
# =========================
@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load scouting output.

    The first path is the current dashboard default.
    The fallback paths make the dashboard more robust if output filenames change.
    """
    candidate_paths = [
        os.path.join("outputs", "undervalued_candidates_overall.csv"),
        os.path.join("outputs", "hidden_gems_overall.csv"),
        os.path.join("outputs", "shortlists", "top_20_overall.csv"),
    ]

    for file_path in candidate_paths:
        if os.path.exists(file_path):
            return pd.read_csv(file_path)

    return pd.DataFrame()


df = load_data()

if df.empty:
    st.error(
        "No scouting data found. Please run the pipeline first to generate "
        "`outputs/undervalued_candidates_overall.csv`."
    )
    st.stop()


# =========================
# Basic Column Validation
# =========================
required_columns = [
    "name",
    "age_at_valuation",
    "position_group_raw",
    "minutes_last_season",
    "target_market_value",
    "predicted_value",
    "undervalued_pct",
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(
        "The scouting data is missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()


# Ensure formatted undervaluation column exists
if "undervalued_pct_formatted" not in df.columns:
    df["undervalued_pct_formatted"] = df["undervalued_pct"].apply(
        lambda x: f"{x * 100:.1f}%"
    )


# =========================
# Sidebar Filters
# =========================
st.sidebar.header("🔍 Filter Candidates")

positions = sorted(df["position_group_raw"].dropna().unique().tolist())
selected_positions = st.sidebar.multiselect(
    "Position Group",
    positions,
    default=positions,
)

min_age = int(df["age_at_valuation"].min())
max_age = int(df["age_at_valuation"].max())
default_max_age = min(25, max_age)

selected_age = st.sidebar.slider(
    "Age Range",
    min_value=min_age,
    max_value=max_age,
    value=(min_age, default_max_age),
)

min_val = int(df["target_market_value"].min())
max_val = int(df["target_market_value"].max())
default_market_value = min(max(5_000_000, min_val), max_val)

selected_val = st.sidebar.slider(
    "Max Actual Market Value (€)",
    min_value=min_val,
    max_value=max_val,
    value=default_market_value,
    step=500_000,
    format="%d",
)

min_mins = int(df["minutes_last_season"].min())
max_mins = int(df["minutes_last_season"].max())
default_min_mins = min(max(900, min_mins), max_mins)

selected_mins = st.sidebar.slider(
    "Minimum Minutes Played",
    min_value=min_mins,
    max_value=max_mins,
    value=default_min_mins,
)


# =========================
# Apply Filters
# =========================
filtered_df = df[
    (df["position_group_raw"].isin(selected_positions))
    & (df["age_at_valuation"] >= selected_age[0])
    & (df["age_at_valuation"] <= selected_age[1])
    & (df["target_market_value"] <= selected_val)
    & (df["minutes_last_season"] >= selected_mins)
].copy()

filtered_df = filtered_df.sort_values(by="undervalued_pct", ascending=False)


# =========================
# KPI Section
# =========================
st.markdown("### 📊 Key Metrics")
st.caption("Summary of filtered candidates based on current sidebar settings.")

avg_age = filtered_df["age_at_valuation"].mean() if not filtered_df.empty else 0
avg_undervalued = (
    filtered_df["undervalued_pct"].mean() if not filtered_df.empty else 0
)
median_val = (
    filtered_df["target_market_value"].median() if not filtered_df.empty else 0
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    kpi_card("Candidates Found", f"{len(filtered_df):,}")

with kpi2:
    kpi_card("Average Age", f"{avg_age:.1f} yrs")

with kpi3:
    kpi_card("Average Undervaluation %", f"{avg_undervalued * 100:.1f}%")

with kpi4:
    kpi_card("Median Market Value", f"€{median_val:,.0f}")

st.markdown("")
methodology_box(
    "This dashboard uses the performance-only model to identify players whose "
    "recent statistical output exceeds their current market value baseline."
)

st.divider()


# =========================
# Main Content
# =========================
if filtered_df.empty:
    st.warning("No candidates match the selected filters. Try adjusting the sliders in the sidebar.")
    st.stop()


# =========================
# Visualization
# =========================
st.markdown("### 📈 Undervalued Candidates Analysis")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.scatter(
        filtered_df,
        x="age_at_valuation",
        y="undervalued_pct",
        color="position_group_raw",
        size="target_market_value",
        hover_name="name",
        hover_data={
            "age_at_valuation": ":.1f",
            "target_market_value": ":,.0f",
            "predicted_value": ":,.0f",
            "undervalued_pct_formatted": True,
            "minutes_last_season": True,
        },
        title="Age vs Undervaluation Ratio",
        labels={
            "age_at_valuation": "Age",
            "undervalued_pct": "Undervaluation Ratio",
            "position_group_raw": "Position",
        },
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=px.colors.qualitative.Safe,
    )

    fig1.update_layout(
        legend_title_text="Position",
        margin=dict(l=10, r=10, t=55, b=10),
    )

    st.plotly_chart(fig1, use_container_width=True)
    st.caption("Bubble size represents the player's actual market value.")

with col2:
    fig2 = px.scatter(
        filtered_df,
        x="target_market_value",
        y="predicted_value",
        color="position_group_raw",
        hover_name="name",
        hover_data={
            "age_at_valuation": ":.1f",
            "target_market_value": ":,.0f",
            "predicted_value": ":,.0f",
            "undervalued_pct_formatted": True,
            "minutes_last_season": True,
        },
        title="Actual Market Value vs Predicted Value",
        labels={
            "target_market_value": "Actual Market Value (€)",
            "predicted_value": "Predicted Market Value (€)",
            "position_group_raw": "Position",
        },
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=px.colors.qualitative.Safe,
    )

    max_limit = max(
        filtered_df["target_market_value"].max(),
        filtered_df["predicted_value"].max(),
    )

    fig2.add_shape(
        type="line",
        line=dict(dash="dash", color=REFERENCE_LINE_COLOR, width=1),
        x0=0,
        y0=0,
        x1=max_limit,
        y1=max_limit,
    )

    fig2.update_layout(
        legend_title_text="Position",
        margin=dict(l=10, r=10, t=55, b=10),
    )

    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Points **above** the dashed line indicate players whose predicted value "
        "is higher than their actual market value."
    )

st.divider()


# =========================
# Candidate Table
# =========================
st.markdown("### 📋 Top Undervalued Candidates (Filtered)")

col_dl_1, col_dl_2 = st.columns([4, 1])

with col_dl_1:
    st.caption("The table below is sorted by Undervaluation % in descending order.")

with col_dl_2:
    st.download_button(
        label="💾 Download CSV",
        data=filtered_df.to_csv(index=False),
        file_name="undervalued_candidates_shortlist.csv",
        mime="text/csv",
        use_container_width=True,
    )


display_cols = [
    "name",
    "age_at_valuation",
    "position_group_raw",
    "minutes_last_season",
    "target_market_value",
    "predicted_value",
    "undervalued_pct_formatted",
]

display_df = filtered_df[display_cols].copy()

display_df["age_at_valuation"] = display_df["age_at_valuation"].round(1)
display_df["minutes_last_season"] = display_df["minutes_last_season"].round(0).astype(int)
display_df["target_market_value"] = display_df["target_market_value"].apply(
    lambda x: f"€{x:,.0f}"
)
display_df["predicted_value"] = display_df["predicted_value"].apply(
    lambda x: f"€{x:,.0f}"
)

display_df.columns = [
    "Player Name",
    "Age",
    "Position",
    "Minutes Played",
    "Actual Value",
    "Predicted Value",
    "Undervaluation %",
]

st.dataframe(display_df, use_container_width=True, hide_index=True)

disclaimer_box(
    "Candidates identified here are statistical leads based on predictive modeling "
    "and should be interpreted as a starting point for professional human scouting."
)