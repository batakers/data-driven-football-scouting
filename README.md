# Transfermarkt Market Value Prediction & Undervalued Talent Scouting

This repository contains the full implementation of the pipeline, including data engineering, feature construction, model training, evaluation, and scouting shortlist generation.

The full pipeline was successfully executed end-to-end, from temporal feature engineering to model evaluation and undervalued candidate detection. This project demonstrates a robust, data-driven approach to predicting market values and identifying statistically undervalued football players using real-world Transfermarkt data.

## 📌 Project Versions

- `v1.2-player-similarity`: Player similarity search (statistical twin) extension. [LATEST]
- `v1.1-streamlit-dashboard`: Interactive scouting dashboard extension.
- `v1.0-core-pipeline`: Core machine learning pipeline release.
- `main`: Latest development version.

## 🖼️ Dashboard Preview
![Scouting Dashboard Preview](https://github.com/batakers/data-driven-football-scouting/raw/main/outputs/figures/dashboard_preview.png)
*Interactive dashboard for filtering, visualizing undervalued candidates, and finding statistical twins.*

## 1. 🏗️ Temporal Data Engineering (No Leakage)
To ensure the integrity of the predictive models, strict temporal boundaries were enforced:
- **Valuation Anchoring:** The target variable (`market_value_in_eur`) was anchored to the latest available `valuation_date` for each player.
- **Leakage Prevention:** All match appearances (`appearances.csv` joined with `games.csv`) occurring *after* the `valuation_date` were discarded.
- **Recent Performance Window:** The models were trained exclusively on player statistics (minutes, goals, assists, cards per 90) generated within exactly **365 days prior to their valuation date**.

## 2. 🤖 Dual-Model Evaluation
The dataset was split into training and testing sets after temporal feature construction. For this version, the evaluation uses a standard 80/20 train-test split; future work should include stricter time-based validation.

We developed two distinct XGBoost models to serve different analytical purposes:

| Model | Features | Purpose | $R^2$ Score | Median APE |
|---|---|---|---:|---:|
| **Model A** | Performance-only | Scouting lens | 0.446 | 68.4% |
| **Model B** | Performance + previous value | Accuracy benchmark | 0.969 | 15.65% |

> [!NOTE]
> **Model B (Market-Aware Model)**
> This model suggests that historical market value is an extremely strong predictor of future value. However, because it heavily relies on existing market prestige, it is not ideal for discovering truly undervalued talent based purely on recent athletic performance.

> [!TIP]
> **Model A (Performance-Only Model)**
> While the predictive accuracy is inherently lower without historical market data, this model is much better suited for exploratory scouting because it evaluates players without historical market bias. *However, the results should be treated as initial statistical candidates requiring manual validation.*

## 3. 💎 Scouting: Undervalued Candidates
Using the objective predictions from **Model A**, we calculated the `undervalued_pct` to find players whose statistical output significantly exceeds their actual price tag. 

To ensure realistic results, strict scouting criteria were applied:
- **Age:** $\le$ 25 years old
- **Playing Time:** $\ge$ 900 minutes in the last 365 days (approx. 10 full matches)
- **Market Value Range:** Between €500,000 and €20,000,000 (prevents exploding percentages from near-zero values)

**Result:** The pipeline identified **647 initial undervalued candidates**. 

To make this actionable for a scouting department, we generated targeted shortlists (available in `outputs/shortlists/`):
- `top_20_overall.csv`
- `top_10_forward.csv`, `top_10_midfielder.csv`, `top_10_defender.csv`, `top_10_goalkeeper.csv`
- `top_5_under_5m.csv`
- `top_5_u21.csv`

## 4. 🔍 Extension 2: Player Similarity Search
The **Statistical Twin** engine allows scouts to select any player in the dataset and find the most similar alternatives based purely on performance statistics (Goals, Assists, Cards per 90) and Age.

- **Similarity Engine**: Uses Position-Aware Weighted Cosine Similarity.
- **Dynamic Views**: Result tables adapt columns based on the target position (e.g., Goalkeepers show Height/Minutes instead of Goals/Assists).
- **Dual Mode**: Toggle between **Statistical Twins** (pure similarity) and **Recruitment Alternatives** (younger + cheaper).
- **Actionable Insights**: Highlights if alternatives are **Cheaper**, **Younger**, or **Undervalued** compared to the target player.

---

## ⚠️ Limitations
- **Moderate Predictive Power of Model A:** The performance-only model has moderate predictive power (Median APE ~68.4%). Scouting results should be interpreted as initial candidates, not final recommendations.
- **Defensive & Goalkeeping Metrics:** Performance metrics for defenders and goalkeepers are limited, as the current dataset does not include granular defensive actions (e.g., tackles, interceptions) or save statistics.
- **External Market Factors:** Market value is heavily influenced by external factors absent from this dataset, such as contract length remaining, injury history, club reputation, agent influence, and transfer rumors.
- **Historical Bias vs Discovery:** While `previous_market_value` strongly improves prediction accuracy (Model B), incorporating it reduces the model's usefulness for discovering genuinely undervalued players.

## 🚀 Future Improvements & Extensions
- [DONE] **Extension 1: Streamlit Scouting Dashboard** - Interactive visualization for filtering candidates.
- [DONE] **Extension 2: Player Similarity Search** - Find statistical twins for any player.
- Add league-strength or club-prestige features.
- Build separate models for each position group.
- Add SHAP explainability to understand model feature influence.
- Validate shortlisted candidates against future market value growth.

---

## ⚙️ Reproducibility

### Data Setup
Place the raw Transfermarkt CSV files inside the data directory:
```text
data/raw/
├── players.csv
├── appearances.csv
├── games.csv
└── player_valuations.csv
```

### Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

### How to Run
To reproduce the pipeline, execute the following scripts in order:

```bash
# 1. Prepare and filter temporal data
python src/data_engineering.py

# 2. Clean data and calculate exact age
python src/data_cleaning.py

# 3. Construct features (per-90 metrics, one-hot encoding)
python src/feature_engineering.py

# 4. Train both XGBoost models and evaluate performance
python src/modeling.py

# 5. Apply scouting logic and generate shortlists
python src/scouting.py

# 6. Generate the similarity engine artifact (v1.2+)
python src/similarity.py

# 7. Run the interactive dashboard (v1.1+)
streamlit run app/dashboard.py
```

### Outputs
Executing the pipeline will generate the following structure:
```text
outputs/
├── model_evaluation.csv
├── predictions_per_player.csv
├── shortlists/
│   ├── top_20_overall.csv
│   └── ... (positional shortlists)
├── models/
│   ├── performance_only_model.pkl
│   ├── market_aware_model.pkl
│   └── similarity_engine.pkl
└── figures/
```
