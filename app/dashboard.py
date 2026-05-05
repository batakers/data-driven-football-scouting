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

st.title("⚽ Transfermarkt Talent Scouting Dashboard")
st.markdown("""
Welcome to the Data-Driven Talent Scouting Dashboard! 
This tool explores **initial undervalued candidates** identified by our Performance-Only XGBoost Model. 
Use the filters on the left to find hidden gems whose statistical output exceeds their current market value.
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
positions = df['position_group_raw'].unique().tolist()
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

# --- KPIS ---
st.markdown("### 📊 Key Metrics")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Candidates Found", len(filtered_df))
avg_age = filtered_df['age_at_valuation'].mean() if not filtered_df.empty else 0
kpi2.metric("Average Age", f"{avg_age:.1f} yrs")
avg_undervalued = filtered_df['undervalued_pct'].mean() if not filtered_df.empty else 0
kpi3.metric("Avg Undervalued Pct", f"{avg_undervalued*100:.1f}%")
median_val = filtered_df['target_market_value'].median() if not filtered_df.empty else 0
kpi4.metric("Median Market Value", f"€{median_val:,.0f}")

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
            title="Age vs Undervalued Percentage",
            labels={
                'age_at_valuation': 'Age',
                'undervalued_pct': 'Undervalued Ratio',
                'position_group_raw': 'Position'
            },
            template="plotly_white"
        )
        st.plotly_chart(fig1, use_container_width=True)
        
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
            template="plotly_white"
        )
        
        # Add a reference line (y=x)
        max_limit = max(filtered_df['target_market_value'].max(), filtered_df['predicted_value'].max())
        fig2.add_shape(
            type="line", line=dict(dash='dash', color="gray"),
            x0=0, y0=0, x1=max_limit, y1=max_limit
        )
        
        st.plotly_chart(fig2, use_container_width=True)

    # --- DATA TABLE ---
    st.markdown("### 📋 Candidate Shortlist")
    
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
        'Actual Value', 'Predicted Value', 'Undervalued %'
    ]
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

else:
    st.warning("No candidates match the selected filters. Try adjusting the sliders in the sidebar.")
