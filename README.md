# Transfermarkt Market Value Prediction & Role-Aware Talent Scouting

An end-to-end football analytics project that started as a market value prediction pipeline and evolved into a role-aware scouting dashboard for finding undervalued players and realistic recruitment alternatives.

The final system combines:

- Transfermarkt player, club, market value, and appearance data.
- Leakage-safe recent performance features.
- XGBoost market value prediction.
- Undervalued player discovery.
- Top 5 League advanced-stat enrichment.
- Transfermarkt role metadata from `player_bio.csv`.
- Role-aware similarity search with tactical compatibility and foot/side fit.
- SHAP model explainability for global drivers and player-level valuation rationale.

## 🧠 Project Story: From Market Value Prediction to Role-Aware Scouting

Football scouting is not only about identifying players with strong statistics. A useful scouting system needs to answer four connected questions:

1. **How much should a player be worth?**
2. **Which players appear undervalued by the market?**
3. **Who are realistic alternatives with similar performance and tactical roles?**
4. **Why did the model value a player that way?**

This project started as a market value prediction pipeline and evolved into a role-aware scouting dashboard that combines Transfermarkt market data, recent player performance, advanced Top 5 League statistics, tactical role metadata, and model explainability.

The final system does not only predict value. It supports a scouting workflow:

> Identify undervalued players, compare similar alternatives, evaluate whether those alternatives make tactical sense, and explain which model signals drive the valuation.

## 📈 Project Evolution

| Phase | Focus | Why It Was Added |
|---|---|---|
| **Phase 1** | Market Value Prediction | Build a leakage-safe model to estimate player value. |
| **Phase 2** | Scouting Dashboard | Turn static model outputs into an interactive analysis tool. |
| **Phase 3** | Player Similarity Search | Find statistically similar, younger, or cheaper alternatives. |
| **Phase 4** | Advanced Stats Enrichment | Add advanced Top 5 League metrics for richer comparisons. |
| **Phase 5** | Role-Aware Similarity | Add tactical role compatibility, role tags, and foot/side fit. |
| **Phase 6** | Model Explainability | Add SHAP transparency for global model drivers and player-level explanations. |

### Phase 1 — From Raw Transfermarkt Data to Market Value Prediction

The project began with a core machine learning question:

> Can we estimate a player's market value using only information available before their valuation date?

To avoid data leakage, all player performance features were built from matches played before each valuation date. Two models were trained:

- A **performance-only model** for scouting and discovery.
- A **market-aware model** for accuracy benchmarking.

This created the foundation for detecting players whose recent statistical output appeared stronger than their current market value.

### Phase 2 — From Model Output to Scouting Dashboard

The first model produced useful predictions, but a static CSV shortlist was not enough for practical analysis.

The second phase introduced an interactive Streamlit dashboard, allowing users to filter candidates by age, position, market value, and minutes played. This shifted the project from a modeling experiment into an interactive scouting product.

### Phase 3 — From Undervalued Players to Similar Alternatives

After identifying undervalued candidates, the next scouting question became:

> If we cannot afford Player X, who plays similarly but is younger or cheaper?

The third phase introduced a player similarity engine with two workflows:

- **Statistical Twins:** find the closest statistical profiles.
- **Recruitment Alternatives:** find similar players who are younger and cheaper.

This made the system more useful for real scouting scenarios.

### Phase 4 — From Basic Similarity to Enriched Performance Context

Basic similarity based on goals, assists, and cards worked for some players, but it was too limited for midfielders, defenders, and goalkeepers.

The enrichment phase integrated advanced Top 5 League statistics such as xG, key passes, progressive actions, blocks, aerials, and goalkeeper-related indicators. It also added a validation-gated process to avoid bad external data matches.

This made the similarity engine more position-specific and more transparent.

### Phase 5 — From Position-Aware to Role-Aware Scouting

Even enriched position-level similarity was still too broad.

A Forward could be a striker, left winger, right winger, or second striker.
A Defender could be a centre-back, full-back, or wing-back.

The role-aware phase added Transfermarkt role metadata from `player_bio.csv`, enabling:

- Primary role and role tags.
- Compatible role matching.
- Exact Role / Compatible Roles / Broad Position modes.
- Preferred-foot and side-fit scoring.
- Final match score combining statistical similarity, role compatibility, and foot fit.

This made the system more tactically realistic.

### Phase 6 — From Scouting Scores to Model Explainability

After the scouting and similarity workflows became more complete, the next question was interpretability:

> Why does the model think this player should be valued higher or lower?

The explainability phase adds SHAP analysis for both market value models:

- **Model A:** explains the performance-only scouting model used for undervalued discovery.
- **Model B:** explains the market-aware benchmark model and shows how strongly previous valuation drives accuracy.

This turns the project from a ranking tool into a more transparent decision-support workflow. A scout can now inspect not only who is flagged as undervalued, but also which features pushed the model prediction higher or lower.

## ✅ What the Final System Does

The current system supports three connected scouting workflows.

### Scouting Overview

The overview tab identifies undervalued candidates using the performance-only market value model.

It helps answer:

- Which young players appear underpriced relative to recent statistical output?
- How does predicted value compare with current market value?
- Which candidates fit age, position, value, and minutes filters?
- Which players should be reviewed first by a scout or analyst?

### Similarity Search

The similarity tab compares a selected target player with potential alternatives.

It supports:

- **Statistical Twins:** closest performance-profile matches.
- **Recruitment Alternatives:** similar players who are younger and cheaper.
- **Enriched Mode:** advanced-stat similarity for accepted Top 5 League matches.
- **Basic Mode:** global fallback using core performance features.
- **Exact Role:** candidates with the same primary role.
- **Compatible Roles:** candidates above the tactical role-compatibility threshold.
- **Broad Position Group:** fallback logic for broad position matching.

### Model Explainability

The explainability tab helps interpret the market value models.

It supports:

- **Global feature importance:** shows which features drive predictions across the player pool.
- **Model A vs Model B comparison:** contrasts performance-only scouting signals against market-aware benchmark signals.
- **Local player explanations:** explains top undervalued candidates through their strongest positive and negative SHAP drivers.
- **Interpretability caveats:** SHAP values are shown as directional model contributions in log-value space, not direct euro amounts or causal effects.

## 🖼️ Dashboard Preview

![Scouting Dashboard Preview](https://github.com/batakers/data-driven-football-scouting/raw/main/outputs/figures/dashboard_preview.png)

The dashboard is designed as a scouting workflow rather than a model report. It separates player facts, role context, data provenance, score components, and recruitment flags so the results are easier to interpret for non-technical users.

## 🏗️ Core Methodology

### Temporal Data Engineering

The market value target is time-sensitive. To avoid leaking future information into the model, each player row is anchored to a valuation date.

The pipeline enforces:

- **Valuation anchoring:** the target variable is the latest available `market_value_in_eur` for each player.
- **No future appearances:** matches after the valuation date are excluded.
- **Recent performance window:** minutes, goals, assists, and cards are calculated from the 365 days before valuation.
- **Per-90 normalization:** performance features are adjusted for playing time.

### Dual Model Strategy

Two XGBoost models were trained for different analytical purposes.

| Model | Features | Purpose | Approx. R² | Median APE |
|---|---|---|---:|---:|
| **Model A** | Performance-only | Scouting and undervalued discovery | ~0.45 | ~68% |
| **Model B** | Performance + previous value | Accuracy benchmark | ~0.97 | ~15% |

Model A is intentionally harder and less accurate because it avoids using previous market value. That makes it more useful for discovery: it highlights players whose recent statistical output appears stronger than their current market price.

### Undervalued Candidate Criteria

Candidates are generated by comparing Model A predicted value against actual Transfermarkt market value.

The default shortlist criteria are:

- **Age:** 25 or younger.
- **Minutes:** at least 900 minutes in the recent window.
- **Market value:** between €500,000 and €20,000,000.
- **Undervaluation:** predicted value above current market value.

## 🧠 Model Explainability

The explainability layer uses SHAP to make the market value models easier to inspect.

The focus is Model A, because it is the discovery model used for scouting. Model B is still explained as a benchmark, and its SHAP results are expected to show that previous market value is the dominant driver.

The generated outputs include:

- Global SHAP summary and bar plots for Model A.
- Global SHAP summary and bar plots for Model B.
- A feature-importance CSV comparing both models.
- Local waterfall explanations for top undervalued candidates.
- A hidden-gems explanation table summarizing the strongest upward and downward model drivers.

Important interpretation rule:

```text
SHAP values explain the model output in log1p market-value space.
They should be read as directional pressure on the prediction, not direct euro contributions.
```

## 🔍 Similarity & Role-Aware Scouting

The final similarity engine combines statistical profile matching with role intelligence.

### Advanced Stats Enrichment Layer

The enrichment layer adds Top 5 League advanced statistics for players that can be safely matched to the external performance dataset.

The enrichment layer includes:

- Expected goals and shooting metrics.
- Key passes and shot-creating actions.
- Progressive passes and carries.
- Defensive actions such as blocks, clearances, tackles, and interceptions.
- Aerial duel success.
- Goalkeeper-adjacent distribution and recovery indicators where available.

Coverage is intentionally reported with multiple denominators:

- **Overall enrichment coverage:** 8.83% across the full Transfermarkt pool.
- **Eligible Top 5 League coverage:** 42.32% among eligible Top 5 League players.
- **Accepted enriched matches:** 1,792.
- **Enriched players in final similarity engine:** 1,212 after eligibility filters.

This distinction matters because the external Kaggle dataset covers Top 5 League players, while the Transfermarkt pool contains many players outside that coverage.

### Role Metadata Layer

The role metadata layer adds tactical role understanding on top of statistical similarity.

The similarity engine now considers:

- **Statistical similarity** from recent and enriched performance metrics.
- **Role compatibility** from Transfermarkt role metadata.
- **Foot/side fit** for wide roles such as LB, RB, LW, RW, LM, and RM.

Final score:

```text
Final Match Score =
0.80 × Statistical Similarity
+ 0.15 × Role Compatibility
+ 0.05 × Foot / Side Fit
```

Role metadata is available for **19,448 / 20,305 players (95.78%)** in the role-enriched pool.

## ✅ Trust & Validation Layer

Because this project combines multiple datasets, validation was treated as part of the product, not an afterthought.

The pipeline generates:

- `outputs/matching_report.csv` for denominator-aware external data coverage.
- `outputs/matching_audit.csv` for row-level match review.
- `outputs/validation_report.json` for enrichment validation-gate checks.
- `outputs/role_enrichment_report.csv` for role metadata coverage.
- `outputs/role_validation_report.json` for role-aware similarity checks.
- `outputs/explainability/explainability_report.json` for SHAP output metadata.
- `outputs/explainability/explainability_validation_report.json` for explainability output checks.

Key validation gates include:

- No accepted future-stat leakage.
- No accepted position mismatch.
- No conflicting duplicate Kaggle assignments.
- No fuzzy-review matches accepted automatically.
- Goalkeepers are never matched with outfield players.
- Exact Role mode only returns the same primary role.
- Compatible Roles mode only returns candidates above the role-compatibility threshold.
- SHAP outputs are generated from feature matrices aligned to the trained model columns.
- Local explanations are generated for top undervalued candidates from the scouting shortlist.
- Model B explainability is expected to show `previous_market_value` as its dominant global driver.

Latest validation snapshot:

| Check | Result |
|---|---:|
| Enrichment validation status | PASS |
| Role validation status | PASS |
| Future leakage accepted | 0 |
| Position mismatch accepted | 0 |
| Duplicate conflicting Kaggle assignments | 0 |
| Fuzzy-review accepted | 0 |
| Exact Role checks | 260 |
| Compatible Role checks | 260 |
| Role metadata coverage | 95.78% |
| Explainability report status | PASS |
| Explainability validation status | PASS |
| Global SHAP sample size | 5,000 |
| Top candidate explanations | 20 |

## ⚠️ Limitations

- **Market value is not purely statistical:** contract length, injuries, reputation, agent influence, and transfer rumors can materially affect price.
- **Model A is discovery-oriented, not a pricing oracle:** the performance-only model intentionally excludes previous market value, so it is less accurate but more useful for finding potential mispricing.
- **Top 5 League enrichment is partial by design:** enriched similarity is available only for accepted external matches; other players fall back to Basic Mode.
- **Defensive and goalkeeper metrics remain imperfect:** even with enrichment, non-Top 5 League defenders and goalkeepers can have limited event-level data.
- **Role metadata is heuristic:** role tags and preferred-foot fit come from Transfermarkt metadata and compatibility rules, not from tactical event tracking.
- **SHAP is model explanation, not causality:** feature contributions describe how the trained model behaves; they do not prove real-world causal effects on market value.
- **SHAP values are in log-value space:** because the target uses `log1p(market_value_in_eur)`, contributions should not be interpreted as direct euro increases or decreases.
- **Similarity results are leads, not final recommendations:** candidates should be validated by human scouting, video review, and context-specific recruitment constraints.

## 🚀 Reproducibility

### 1. Data Setup

Place the raw datasets in `data/raw/`:

- Transfermarkt player, appearance, game, and valuation CSV files.
- Kaggle Top 5 League advanced-stat CSV.
- Transfermarkt Football Database `player_bio.csv`.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Full Pipeline

The full role-aware pipeline can be reproduced with the guarded PowerShell runner:

```powershell
.\scripts\run_v1_4_role_pipeline.ps1
```

This script runs enrichment, validation, role metadata integration, similarity engine build, and role validation in a fail-fast sequence.

Then generate and validate explainability outputs:

```bash
python src/explainability.py
python src/validate_explainability.py
```

You can also run the core scripts manually:

```bash
python src/data_engineering.py
python src/data_cleaning.py
python src/feature_engineering.py
python src/modeling.py
python src/scouting.py
python src/enrich_similarity.py
python src/validate_enrichment.py
python src/enrich_roles.py
python src/similarity.py
python src/validate_roles.py
python src/explainability.py
python src/validate_explainability.py
```

Important: avoid chaining PowerShell commands with `;` for final pipeline runs, because later commands will continue even if validation fails. Use the guarded script or explicit exit-code checks.

### 4. Launch the Dashboard

```bash
streamlit run app/dashboard.py
```

## 📂 Key Outputs

| Output | Purpose |
|---|---|
| `outputs/predictions_per_player.csv` | Player-level model predictions and undervaluation estimates. |
| `outputs/undervalued_candidates_overall.csv` | Filtered candidates for the dashboard scouting overview. |
| `outputs/shortlists/` | Prebuilt shortlists such as top overall, U21, and under-€5M candidates. |
| `outputs/models/similarity_engine.pkl` | Serialized role-aware similarity engine. |
| `outputs/matching_report.csv` | Denominator-aware enrichment coverage and validation-gate summary. |
| `outputs/matching_audit.csv` | Row-level audit trail for external advanced-stat matching. |
| `outputs/validation_report.json` | PASS/FAIL enrichment validation report used by the guarded pipeline. |
| `data/processed/role_enriched_players.csv` | Role metadata layer merged onto the similarity pool. |
| `outputs/role_enrichment_report.csv` | Role metadata coverage, role distribution, and missingness report. |
| `outputs/role_validation_report.json` | PASS/FAIL role-aware validation summary. |
| `outputs/explainability/shap_feature_importance.csv` | Global SHAP feature importance for both market value models. |
| `outputs/explainability/top_hidden_gems_explanations.csv` | Local explanation summaries for top undervalued candidates. |
| `outputs/explainability/player_explanations/` | Waterfall plots and feature contribution tables for selected candidates. |
| `outputs/explainability/explainability_report.json` | Explainability output metadata and validation snapshot. |
| `outputs/explainability/explainability_validation_report.json` | PASS/FAIL checks for SHAP files, feature alignment, and expected model behavior. |

## 🧩 Project Structure

```text
app/
  dashboard.py                 # Streamlit scouting dashboard

src/
  data_engineering.py           # Temporal feature construction
  data_cleaning.py              # Cleaning and formatting
  feature_engineering.py        # Per-90 features and encoding
  modeling.py                   # XGBoost market value models
  scouting.py                   # Undervalued candidate generation
  enrich_similarity.py          # External advanced-stat matching
  validate_enrichment.py        # Advanced-stat validation gates
  role_mapping.py               # Role rules and compatibility matrix
  enrich_roles.py               # player_bio role metadata integration
  similarity.py                 # Similarity engine
  validate_roles.py             # Role-aware validation
  explainability.py             # SHAP model explainability
  validate_explainability.py    # Explainability output validation

scripts/
  run_v1_4_role_pipeline.ps1    # Guarded role-aware pipeline runner

outputs/
  explainability/
  figures/
  models/
  shortlists/
```

## Final Interpretation

This project is best understood as a data product, not only a model experiment.

The prediction model estimates market value. The scouting layer turns prediction errors into candidate discovery. The similarity engine converts a shortlist into recruitment alternatives. The role-aware layer makes those alternatives more tactically realistic. The explainability layer shows which model signals drive the valuation.

The result is a reproducible football scouting workflow that can answer:

> Who looks undervalued, who is statistically similar, who actually makes sense as a tactical alternative, and why did the model flag them?
