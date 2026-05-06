# Walkthrough - Extension 2: Player Similarity Search (v1.2)

I have successfully implemented the **Player Similarity Search** engine and integrated it into the Transfermarkt Scouting Dashboard. This update allows scouts to find statistical "twins" for any player, specifically focusing on identifying younger, cheaper, or undervalued alternatives.

## Key Changes

### 1. Similarity Engine (`src/similarity.py`)
- **Dataset**: Uses the **full player pool** (4,852 players with 900+ minutes) from `data/processed/featured_players.csv`, merged with ML model predictions.
- **⚠️ Important Notes & Limitations**:
    - **Minutes played (>= 900)** is used as a reliability filter to ensure per-90 metrics are representative.
    - **Goalkeeper & Defender Similarity:** Results for defensive roles are based on age, height, and general activity (cards). They may be less granular than forwards due to the lack of specialized defensive metrics (saves, tackles, etc.) in the current dataset.
    - **Market Value:** Not used as a similarity feature to avoid price bias in player profiling.
- **Methodology**: Uses **Manual Weighted Cosine Similarity** (Heuristic Position-Aware).
- **Position-Aware Weighting**:
    - **Forward**: High emphasis on `goals_per_90_ls` (0.35) and `assists_per_90_ls` (0.25).
    - **Midfielder**: High emphasis on `assists_per_90_ls` (0.35).
    - **Defender**: Focused on physical profile (`height`) and discipline (`cards`).
    - **Goalkeeper**: Equal split between `age` and `height` profile.
- **Search Pool**: Over 4,800 players with at least 900 minutes played.
- **Strict Matching**: Restricts comparisons to the same position group by default.
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
    - **Dynamic Tables**: Table columns adapt automatically to the target player's position (e.g., hiding goals/assists for Goalkeepers/Defenders).
- **Recruitment Alternative Mode**: Quick filter for candidates that are both **younger AND cheaper** than the target player.
- **Actionable Scouting Leads**:
    - ✅/❌ flags for **Cheaper** or **Younger** alternatives.
    - 💎 flag for **Undervalued** candidates (as predicted by our ML model).
- **Explanatory Captions**: Honest documentation of data limitations for defensive and GK roles.

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
