# Walkthrough - Extension 2: Player Similarity Search (v1.2)

I have successfully implemented the **Player Similarity Search** engine and integrated it into the Transfermarkt Scouting Dashboard. This update allows scouts to find statistical "twins" for any player, specifically focusing on identifying younger, cheaper, or undervalued alternatives.

## Key Changes

### 1. Similarity Engine (`src/similarity.py`)
- **Dataset**: Uses the **full player pool** (4,852 players with 900+ minutes) from `data/processed/featured_players.csv`, merged with ML model predictions.
- **Reliability Threshold**: The engine was fitted on 4,852 players with at least **900 minutes played** in the recent performance window. This threshold was used to reduce noise from low-sample players, though it may exclude very young prospects with limited senior minutes.
- **Methodology**: Uses K-Nearest Neighbors (KNN) with **Cosine Similarity**.
- **Features**: Focused on performance-only metrics to avoid market bias:
    - `age_at_valuation`
    - `goals_per_90`
    - `assists_per_90`
    - `cards_per_90`
- **Strict Position Matching**: Ensures Goalkeepers are only compared to Goalkeepers, Forwards to Forwards, etc.
- **Similarity Score**: Calculated as `1 - cosine_distance`, where 1.0 is a perfect statistical match.

> [!IMPORTANT]
> **Methodology Note**: Market value, predicted value, and undervaluation flags are NOT used to calculate similarity. They are only used for interpretation after similar players are retrieved.

### 2. Dashboard Integration (`app/dashboard.py`)
- **Tabbed Layout**: Refactored the UI into two main sections:
    - **💎 Scouting Overview**: The original undervalued candidates dashboard.
    - **🔍 Similarity Search**: The new interactive "Statistical Twin" tool.
- **Interactive Search**:
    - Select any player from the dataset.
    - View their "Target Profile" (Age, Value, Minutes).
    - Get a list of the Top 10 most similar players in the same position group.
- **Actionable Scouting Leads**:
    - ✅/❌ flags for **Cheaper** or **Younger** alternatives.
    - 💎 flag for **Undervalued** candidates (as predicted by our ML model).

## Verification Results
- **Engine Performance**: Successfully fitted on 4,852 players.
- **Manual Sanity Checks**: Showed plausible outputs, such as Erling Haaland returning Benjamin Sesko as a lower-cost forward alternative, or Lamine Yamal returning other high-potential young wingers.
- **Full Pool Search**: Verified that the search covers all high-reliability players in the dataset, not just the undervalued shortlist.

## How to Run
1. Ensure dependencies are installed: `pip install -r requirements.txt`
2. Generate the similarity engine artifact:
   ```bash
   python src/similarity.py
   ```
3. Launch the dashboard:
   ```bash
   streamlit run app/dashboard.py
   ```
