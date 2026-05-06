# Transfermarkt Market Value Prediction & Undervalued Talent Scouting

This repository contains the full implementation of an end-to-end data science pipeline for predicting football player market values and identifying statistically undervalued talent. It integrates Transfermarkt market data with advanced performance metrics from elite European leagues.

## 📌 Project Versions
- **v1.3**: Enriched Similarity Search (Advanced Stats Integration). [LATEST]
- **v1.2**: Position-Aware Player Similarity Search (Basic Mode).
- **v1.1**: Streamlit Scouting Dashboard.
- **v1.0**: Core Machine Learning Pipeline (Market Value Prediction).

## 🖼️ Dashboard Preview
![Scouting Dashboard Preview](https://github.com/batakers/data-driven-football-scouting/raw/main/outputs/figures/dashboard_preview.png)
*Interactive dashboard for filtering, visualizing undervalued candidates, and finding statistical twins.*

---

## 1. 🏗️ Core Methodology: Market Value Prediction

### Temporal Data Engineering (No Leakage)
To ensure the integrity of the predictive models, strict temporal boundaries were enforced:
- **Valuation Anchoring:** The target variable (`market_value_in_eur`) was anchored to the latest available `valuation_date` for each player.
- **Leakage Prevention:** All match appearances occurring *after* the `valuation_date` were discarded.
- **Recent Performance Window:** Models were trained exclusively on player statistics (minutes, goals, assists, cards per 90) generated within exactly **365 days prior to their valuation date**.

### Dual-Model Evaluation
We developed two distinct XGBoost models to serve different analytical purposes:
| Model | Features | Purpose | $R^2$ Score | Median APE |
|---|---|---|---:|---:|
| **Model A** | Performance-only | Scouting & Discovery | ~0.45 | ~68% |
| **Model B** | Performance + Prev. Value | Accuracy Benchmark | ~0.97 | ~15% |

### Scouting Criteria for "Undervalued" Detection
Candidates were identified by comparing their Model A predicted value against their actual Transfermarkt market value.
- **Age:** ≤ 25 years old
- **Playing Time:** ≥ 900 minutes in the last season
- **Market Value Range:** €500,000 to €20,000,000

---

## 2. 🔍 Extension 3: Enriched Similarity Search (v1.3)
The v1.3 update bridges the gap between market data and granular on-pitch performance.

- **Advanced Metrics Layer**: Integrates external Top 5 League stats (xG, Key Passes, Progressive Actions, etc.) matched via a multi-stage pipeline with **8.83% overall coverage** across the full Transfermarkt pool and **42.32% coverage** among eligible Top 5 League players after release-gate validation.
- **Dual Similarity Engine**:
    - **Enriched Mode**: Richer position-specific matching for elite league players using granular performance features.
    - **Basic Mode**: Global fallback matching using core stats (Goals, Assists, Cards).
- **Position-Aware Weighted Similarity**: Custom heuristic weights per position (Forward, Midfielder, Defender, Goalkeeper) ensure tactical relevance.
- **Release-Gated Enrichment**: Accepted enriched rows must pass temporal safety, position compatibility, one-to-one Kaggle row assignment checks, and exclude manual-review fuzzy matches before the similarity engine can be built.

---

## ⚠️ Limitations & Future Improvements
- **Defensive & Goalkeeping Metrics**: While v1.3 adds advanced stats, core data for non-Top 5 league defenders remains limited to age, height, and disciplinary profile.
- **External Market Factors**: Market value is influenced by contract length, injury history, and agent influence—factors currently absent from the dataset.
- **Future Work**: Integration of league-strength coefficients and SHAP explainability for prediction transparency.

---

## 🚀 Reproducibility Guide

### 1. Data Setup
Place the raw Transfermarkt CSV files (players, appearances, games, valuations) and the Kaggle Top 5 League CSV in `data/raw/`.

### 2. Execution Pipeline
Run the following scripts in order to reproduce the full system from scratch:

```bash
# Core Data Pipeline
python src/data_engineering.py     # Temporal filtering
python src/data_cleaning.py        # Cleaning & formatting
python src/feature_engineering.py  # Per-90 & encoding

# Modeling & Scouting
python src/modeling.py             # Train XGBoost models
python src/scouting.py             # Generate shortlists

# Enrichment & Similarity (v1.3)
python src/enrich_similarity.py    # Match with Kaggle stats
python src/validate_enrichment.py  # Release gate validation
python src/similarity.py           # Build similarity engine
```

On PowerShell, use the guarded release pipeline instead of separating commands with `;`:

```powershell
.\scripts\run_v1_3_release_pipeline.ps1
```

`src/similarity.py` refuses to build if `outputs/validation_report.json` is missing, failed, or older than the enrichment outputs.

### 3. Launch Dashboard
```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
```

## 📂 Output Structure
- `outputs/predictions_per_player.csv`: Full database with model predictions.
- `outputs/shortlists/`: Targeted scouting lists (Top U21, Hidden Gems, etc.).
- `outputs/models/`: Serialized XGBoost models and Similarity Engine.
- `outputs/matching_report.csv`: Denominator-aware enrichment report with overall, eligible Top 5, league, method, status, release-gate, and final similarity-engine coverage sections.
- `outputs/matching_audit.csv`: Row-level match audit with method, confidence, season, compatibility, review reason, and `enriched_available`.
- `outputs/validation_report.json`: PASS/FAIL release-gate summary used by the similarity engine.
