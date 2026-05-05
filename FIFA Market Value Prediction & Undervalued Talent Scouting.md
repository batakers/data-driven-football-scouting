# ⚽ Real-World Talent Scouting: Market Value Prediction V2

> **Transition from Video Game Analytics to Real-World Data Engineering**
> 
> A machine learning project that predicts real-world football player market values using the massive Transfermarkt relational dataset. We will aggregate millions of match appearances to build a purely statistical scouting model.

---

## 📋 Problem Statement & V2 Objective

In V1, we relied on EA Sports' "Scout's Eye" (`Overall_Rating`, `Future_Potential`). In V2, we are stripping away subjective video game ratings. Can an AI model predict a player's real-world market value strictly by looking at their real-world performance metrics (Goals, Assists, Minutes Played, Age, Caps)?

### Key Questions
1. Is it possible to accurately predict a player's price tag without a predefined "rating" system?
2. How strongly do basic metrics like `Minutes_Played` and `Goals_per_90` correlate with market value?
3. How much of a player's valuation is driven by their age vs their actual performance?
4. **Who are the true hidden gems in world football right now based on raw statistics?**

---

## 📊 Dataset Structure (Transfermarkt)

We are dealing with a **Relational Database** structure.
- `players.csv`: Demographics (Age, Position, Country, Height, Caps).
- `appearances.csv` (146MB): Match-by-match data (Minutes played, Goals, Assists, Cards).
- `player_valuations.csv`: Time-series of market values.

### Target Variable
- `market_value_in_eur` (Latest available valuation per player).

### Proposed Feature Matrix
| Feature | Source | Description |
|---------|--------|-------------|
| `Age` | `players` | Calculated from `date_of_birth` |
| `Position` | `players` | Aggregated into Position Groups (FW, MF, DF, GK) |
| `Height_cm` | `players` | Physical attribute |
| `Int_Caps` | `players` | International Caps |
| `Total_Minutes` | `appearances` | Total minutes played (indicates regular starter vs bench) |
| `Total_Goals` | `appearances` | Aggregate goals |
| `Total_Assists` | `appearances` | Aggregate assists |
| `Goals_per_90` | Engineered | `(Total_Goals / Total_Minutes) * 90` |
| `Assists_per_90` | Engineered | `(Total_Assists / Total_Minutes) * 90` |

---

## 🔬 Methodology: The 6-Phase Pipeline

### Phase 1: Data Engineering & Aggregation (The Hardest Part)
- Load `players.csv` and filter out inactive players (e.g., `last_season` >= 2023).
- Aggregate `appearances.csv` by `player_id` (SUM goals, assists, minutes, cards).
- Join the aggregated stats with `players.csv`.
- Use the most recent `market_value_in_eur` from `players.csv` as our target.

### Phase 2: Data Cleaning
- Drop players with zero or missing `market_value_in_eur`.
- Handle missing `height_in_cm` (impute by position median).
- Clean dates and calculate `Age`.

### Phase 3: Exploratory Data Analysis (EDA)
- Analyze the severe skewness of `market_value_in_eur`.
- Plot correlation heatmap of real-world stats vs Value.
- Visualize `Minutes_Played` vs `Value_M`.

### Phase 4: Feature Engineering
- Create advanced per-90 metrics.
- One-Hot Encode `Position_Group`.
- Target Encoding for `current_club_id` (because playing for Real Madrid automatically inflates your value compared to playing for a lower-tier club, even if stats are identical).

### Phase 5: Modeling & Evaluation
- Split 80/20.
- Compare Random Forest, Gradient Boosting, XGBoost.
- Target transformation: `log1p(market_value_in_eur)`.
- Metric Goals: Maximize R², Minimize MAE.

### Phase 6: Scouting Analysis & SHAP
- Identify the new "Top 20 Undervalued" players based on pure real-world statistics.
- SHAP Waterfall plots to explain *why* the model thinks a player should be more expensive based on their stats.

---

## 🚧 Risiko Tingkat Kesulitan (Difficulty & Risks)

> [!WARNING]
> **Data Size & Memory Risks**
> `appearances.csv` contains millions of rows. Aggregating this using Pandas `.groupby()` can crash low-RAM machines. We must optimize memory usage (e.g., dropping unused columns before groupby).

> [!CAUTION]
> **Data Leakage Risk**
> We must ensure we are predicting the *current* market value based on *historical* stats, avoiding any future data sneaking into the training set.

> [!NOTE]
> **Missing the "Eye Test"**
> Defenders and Goalkeepers don't score goals. We will have to rely heavily on `Minutes_Played`, `Int_Caps`, and `Club Prestige` to value them accurately.
