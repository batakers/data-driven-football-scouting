import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Page Configuration
st.set_page_config(
    page_title="Football Talent Scouting Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for adaptive KPI cards
st.markdown("""
<style>
:root {
    --kpi-card-bg: #ffffff;
    --kpi-card-border: #e5e7eb;
    --kpi-label-color: #64748b;
    --kpi-value-color: #0f172a;
    --kpi-shadow: rgba(15, 23, 42, 0.08);
}

@media (prefers-color-scheme: dark) {
    :root {
        --kpi-card-bg: #111827;
        --kpi-card-border: #374151;
        --kpi-label-color: #cbd5e1;
        --kpi-value-color: #f8fafc;
        --kpi-shadow: rgba(0, 0, 0, 0.35);
    }
}

.kpi-card {
    background-color: var(--kpi-card-bg);
    border: 1px solid var(--kpi-card-border);
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 4px 14px var(--kpi-shadow);
    min-height: 105px;
}

.kpi-label {
    color: var(--kpi-label-color);
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 10px;
}

.kpi-value {
    color: var(--kpi-value-color);
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.15;
}

div[data-testid="stExpander"] {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

def kpi_card(label, value):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.title("⚽ Transfermarkt Talent Scouting Dashboard")
st.markdown("""
Welcome to the Data-Driven Talent Scouting Dashboard! 
This tool explores **initial undervalued candidates** identified by our **Performance-Only XGBoost Model**. 
Use the filters on the left to find players whose statistical output significantly exceeds their current market value.
""")

# Load Data
@st.cache_data
def load_data():
    file_path = os.path.join("outputs", "undervalued_candidates_overall.csv")
    if not os.path.exists(file_path):
        return pd.DataFrame()
    return pd.read_csv(file_path)

df = load_data()

if df.empty:
    st.error("No scouting data found. Please run the pipeline first to generate `outputs/undervalued_candidates_overall.csv`.")
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.header("🔍 Filter Candidates")

# 1. Position Filter
positions = sorted(df['position_group_raw'].unique().tolist())
selected_positions = st.sidebar.multiselect("Position Group", positions, default=positions)

# 2. Age Filter
min_age = int(df['age_at_valuation'].min())
max_age = int(df['age_at_valuation'].max())
selected_age = st.sidebar.slider("Age Range", min_value=min_age, max_value=max_age, value=(min_age, 25))

# 3. Market Value Filter
min_val = int(df['target_market_value'].min())
max_val = int(df['target_market_value'].max())
selected_val = st.sidebar.slider(
    "Max Actual Market Value (€)", 
    min_value=min_val, 
    max_value=max_val, 
    value=5000000, 
    step=500000,
    format="%d"
)

# 4. Minutes Played Filter
min_mins = int(df['minutes_last_season'].min())
max_mins = int(df['minutes_last_season'].max())
selected_mins = st.sidebar.slider("Minimum Minutes Played", min_value=min_mins, max_value=max_mins, value=900)

# --- APPLY FILTERS ---
filtered_df = df[
    (df['position_group_raw'].isin(selected_positions)) &
    (df['age_at_valuation'] >= selected_age[0]) &
    (df['age_at_valuation'] <= selected_age[1]) &
    (df['target_market_value'] <= selected_val) &
    (df['minutes_last_season'] >= selected_mins)
].copy()

# Sort by Undervalued Pct Descending by default
filtered_df = filtered_df.sort_values(by="undervalued_pct", ascending=False)

# --- KPIS ---
st.markdown("### 📊 Key Metrics")
st.caption("Summary of filtered candidates based on current sidebar settings.")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

avg_age = filtered_df['age_at_valuation'].mean() if not filtered_df.empty else 0
avg_undervalued = filtered_df['undervalued_pct'].mean() if not filtered_df.empty else 0
median_val = filtered_df['target_market_value'].median() if not filtered_df.empty else 0

with kpi1:
    kpi_card("Candidates Found", f"{len(filtered_df):,}")

with kpi2:
    kpi_card("Average Age", f"{avg_age:.1f} yrs")

with kpi3:
    kpi_card("Average Undervaluation %", f"{avg_undervalued * 100:.1f}%")

with kpi4:
    kpi_card("Median Market Value", f"€{median_val:,.0f}")

st.info("💡 **Methodology:** This dashboard uses the performance-only model to identify players whose recent statistical output exceeds their current market value baseline.")

st.divider()

if not filtered_df.empty:
    # --- VISUALIZATION ---
    st.markdown("### 📈 Undervalued Candidates Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Scatter Plot: Age vs Undervalued Pct
        fig1 = px.scatter(
            filtered_df, 
            x='age_at_valuation', 
            y='undervalued_pct', 
            color='position_group_raw',
            size='target_market_value',
            hover_name='name',
            hover_data={
                'age_at_valuation': ':.1f',
                'target_market_value': ':.0f',
                'predicted_value': ':.0f',
                'undervalued_pct_formatted': True,
                'minutes_last_season': True
            },
            title="Age vs Undervaluation Ratio",
            labels={
                'age_at_valuation': 'Age',
                'undervalued_pct': 'Undervaluation Ratio',
                'position_group_raw': 'Position'
            },
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.caption("Bubble size represents the player's actual market value.")
        
    with col2:
        # Scatter Plot: Actual vs Predicted Value
        fig2 = px.scatter(
            filtered_df, 
            x='target_market_value', 
            y='predicted_value', 
            color='position_group_raw',
            hover_name='name',
            title="Actual Market Value vs Predicted Value",
            labels={
                'target_market_value': 'Actual Market Value (€)',
                'predicted_value': 'Predicted Market Value (€)',
                'position_group_raw': 'Position'
            },
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        
        # Add a reference line (y=x)
        max_limit = max(filtered_df['target_market_value'].max(), filtered_df['predicted_value'].max())
        fig2.add_shape(
            type="line",
            line=dict(dash="dash", color="#64748b", width=1),
            x0=0, y0=0,
            x1=max_limit, y1=max_limit
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Points **above** the dashed line indicate players whose predicted value is higher than their actual market value.")

    st.divider()

    # --- DATA TABLE ---
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
            use_container_width=True
        )
    
    # Format the dataframe for display
    display_cols = [
        'name', 'age_at_valuation', 'position_group_raw', 'minutes_last_season', 
        'target_market_value', 'predicted_value', 'undervalued_pct_formatted'
    ]
    
    display_df = filtered_df[display_cols].copy()
    display_df['age_at_valuation'] = display_df['age_at_valuation'].round(1)
    display_df['target_market_value'] = display_df['target_market_value'].apply(lambda x: f"€{x:,.0f}")
    display_df['predicted_value'] = display_df['predicted_value'].apply(lambda x: f"€{x:,.0f}")
    
    display_df.columns = [
        'Player Name', 'Age', 'Position', 'Minutes Played', 
        'Actual Value', 'Predicted Value', 'Undervaluation %'
    ]
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.warning("⚠️ **Disclaimer:** Candidates identified here are statistical leads based on predictive modeling and should be interpreted as a starting point for professional human scouting.")

else:
    st.warning("No candidates match the selected filters. Try adjusting the sliders in the sidebar.")
