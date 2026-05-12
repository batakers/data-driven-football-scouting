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
- Scout-facing rationale that explains candidate flags in plain language.
- Temporal validation that checks historical scouting leads against future market-value movement.
- League-strength tiering for contextual performance assessment.
- Contract awareness with transfer feasibility signals.
- Form trend analysis comparing recent vs earlier performance windows.
- Personalized scouting rationale with data confidence scoring.
- Excel and PDF export for scout workflow completion.
- Session watchlist for tracking candidates within a scouting session.

## Project Status

This project is complete as a portfolio-ready football analytics data product.

The core analytical scope ends at Phase 7 (temporal validation). Phases 8-12 extend the system into a more complete scouting workflow tool with league context, contract intelligence, form signals, personalized rationale, and export capabilities.

## 🧠 Project Story: From Market Value Prediction to Role-Aware Scouting

Football scouting is not only about identifying players with strong statistics. A useful scouting system needs to answer four connected questions:

1. **How much should a player be worth?**
2. **Which players appear undervalued by the market?**
3. **Who are realistic alternatives with similar performance and tactical roles?**
4. **Why should a scout review this player next?**
5. **Did historical scouting leads gain value after they were flagged?**

This project started as a market value prediction pipeline and evolved into a role-aware scouting dashboard that combines Transfermarkt market data, recent player performance, advanced Top 5 League statistics, tactical role metadata, and plain-language scouting rationale.

The final system does not only predict value. It supports a scouting workflow:

> Identify undervalued players, compare similar alternatives, evaluate whether those alternatives make tactical sense, explain why a player deserves manual scouting review, and check whether historical leads showed future value growth.

## 📈 Project Evolution

| Phase | Focus | Why It Was Added |
|---|---|---|
| **Phase 1** | Market Value Prediction | Build a leakage-safe model to estimate player value. |
| **Phase 2** | Scouting Dashboard | Turn static model outputs into an interactive analysis tool. |
| **Phase 3** | Player Similarity Search | Find statistically similar, younger, or cheaper alternatives. |
| **Phase 4** | Advanced Stats Enrichment | Add advanced Top 5 League metrics for richer comparisons. |
| **Phase 5** | Role-Aware Similarity | Add tactical role compatibility, role tags, and foot/side fit. |
| **Phase 6** | Scouting Rationale | Translate model outputs into scout-friendly evidence, risks, and next steps. |
| **Phase 7** | Temporal Validation | Test whether historical scouting leads gained market value after 6-12 months. |
| **Phase 8** | League-Strength Context | Add competitive tier classification so scouts can assess performance context. |
| **Phase 9** | Contract Awareness | Add contract expiry data to surface transfer feasibility and urgency. |
| **Phase 10** | Personalized Rationale | Replace generic templates with player-specific scouting narratives. |
| **Phase 11** | Form Trend Analysis | Compare recent vs earlier 6-month windows to detect rising or declining form. |
| **Phase 12** | Export and Workflow | Add Excel export, PDF scouting briefs, and session watchlist for workflow completion. |

### Phase 1 - From Raw Transfermarkt Data to Market Value Prediction

The project began with a core machine learning question:

> Can we estimate a player's market value using only information available before their valuation date?

To avoid data leakage, all player performance features were built from matches played before each valuation date. Two models were trained:

- A **performance-only model** for scouting and discovery.
- A **market-aware model** for accuracy benchmarking.

This created the foundation for detecting players whose recent statistical output appeared stronger than their current market value.

### Phase 2 - From Model Output to Scouting Dashboard

The first model produced useful predictions, but a static CSV shortlist was not enough for practical analysis.

The second phase introduced an interactive Streamlit dashboard, allowing users to filter candidates by age, position, market value, and minutes played. This shifted the project from a modeling experiment into an interactive scouting product.

### Phase 3 - From Undervalued Players to Similar Alternatives

After identifying undervalued candidates, the next scouting question became:

> If we cannot afford Player X, who plays similarly but is younger or cheaper?

The third phase introduced a player similarity engine that ranked close statistical profiles first, then supported recruitment-style filtering for age, value, and affordability.

This made the system more useful for real scouting scenarios.

### Phase 4 - From Basic Similarity to Enriched Performance Context

Basic similarity based on goals, assists, and cards worked for some players, but it was too limited for midfielders, defenders, and goalkeepers.

The enrichment phase integrated advanced Top 5 League statistics and converts them into position-specific scouting profiles such as Goal Threat, Ball Progression, Defensive Activity, Duel Strength, and Build-up Support. It also added a validation-gated process to avoid bad external data matches.

This made the similarity engine more position-specific and more transparent.

### Phase 5 - From Position-Aware to Role-Aware Scouting

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

### Phase 6 - From Scouting Scores to Scouting Rationale

After the scouting and similarity workflows became more complete, the next question was usability:

> Why should a scout actually review this player?

The rationale phase removes technical model-debugging language from the dashboard and translates player evidence into a scouting report style:

- **Why This Player:** plain-language explanation for an individual candidate.
- **Key Scouting Signals:** playing time, age profile, value gap, and role context.
- **Scout Checks Before Action:** league strength, tactical role, injury history, and transfer feasibility.
- **Suggested Action:** prioritize review or monitor based on the value-gap threshold.

This turns the project from a ranking tool into a more practical decision-support workflow. A scout can now inspect not only who is flagged as undervalued, but also why that player deserves review and what must be checked next.

### Phase 7 - From Scouting Leads to Outcome Validation

After the dashboard could identify and explain scouting leads, the next question was whether those leads showed positive market-value movement over time.

The temporal validation phase builds historical valuation snapshots, flags scouting leads using only information available at the snapshot date, and checks future market value after 6 and 12 months.

This is a fixed-model retrospective signal audit, not walk-forward retraining. It does not prove causality, but it helps evaluate whether the scouting signal is useful as a prioritization tool.

### Phase 8 - From Raw League Names to Competitive Context

After the temporal validation confirmed the scouting signal had value, the next question was context:

> Is a player's performance from an elite league or a developing one?

Phase 8 introduced a four-tier league classification system (Elite, Strong, Competitive, Developing) covering 32 domestic leagues. Each player now carries a league tier badge and contextual scout notes. This allows scouts to immediately assess whether performance data comes from a high-competition environment or requires additional step-up validation.

### Phase 9 - From Unknown Contracts to Transfer Feasibility

Scouting leads are only actionable if the player can actually be acquired. Phase 9 added contract awareness by combining two data sources (`players.csv` and `player_bio.csv`) to resolve contract expiry dates for 87.9% of the pipeline.

Players are classified as Expiring (<12 months), Short (12-24 months), Medium (24-36 months), or Long (>36 months). Expiring contracts signal potential free transfers or reduced fees. Long contracts signal significant fee requirements. This context helps scouts prioritize leads by transfer feasibility.

### Phase 10 - From Generic Templates to Personalized Rationale

The original rationale layer produced identical text for every player. Phase 10 rewrote the entire rationale system to generate unique, data-specific narratives.

Each player now receives a personalized scouting summary that references their specific age, minutes, value gap in euros, league tier, contract situation, and position-specific performance highlights. Scout checks are contextual rather than generic: a young player from a Tier 4 league gets different warnings than a senior player from the Premier League.

A data confidence score (High, Medium, Low) was added based on league tier, minutes played, contract availability, and advanced stats coverage.

### Phase 11 - From Static Snapshots to Form Trend Signals

A single 365-day window treats all performance equally, but scouts care about trajectory. Phase 11 splits the performance window into two halves (0-180 days and 181-365 days) and computes per-90 deltas.

Players are classified as Rising, Stable, or Declining based on goals/90 improvement and minutes ratio. This helps scouts distinguish between a player who peaked 10 months ago and one who is currently improving. Form trends require at least 450 minutes in each window to avoid noise from small samples.

### Phase 12 - From Dashboard to Complete Workflow

A scouting tool is only useful if it fits into the scout's existing workflow. Phase 12 added three capabilities:

- **Excel export** with multi-sheet output (data + metadata) for shortlists, similarity comparisons, and watchlists.
- **PDF scouting briefs** per player containing key stats, personalized rationale, scouting signals, and scout checks in a clean printable format.
- **Session watchlist** allowing scouts to bookmark candidates during a session and export them as a batch.

## ✅ What the Final System Does

The current system supports four connected scouting workflows, enhanced with league context, contract intelligence, form signals, and export capabilities.

### Find Undervalued Players

The first dashboard tab identifies undervalued candidates using the performance-only market value model.

It helps answer:

- Which young players appear underpriced relative to recent statistical output?
- How does predicted value compare with current market value?
- Which candidates fit age, position, value, minutes, league tier, and contract filters?
- Which players should be reviewed first by a scout or analyst?
- What is the player's recent form trend (rising, stable, or declining)?

Each candidate row shows a contextual "Key Stat" tailored to their position, plus league tier badge, contract status, and form indicator. Scouts can add players to a session watchlist and export filtered shortlists as Excel files.

### Player Alternatives

The alternatives tab compares a selected target player with potential replacements.

It supports:

- **Target player statistics:** position-specific profile metrics displayed before the comparison table.
- **Ranked playing-profile matches:** candidates are sorted by overall fit before recruitment filters are applied.
- **Recruitment filters:** users can filter by max age, max current value, minimum overall fit, minimum role fit, younger-than-target, cheaper-than-target, and undervalued-only flags.
- **Price Status:** candidates are labelled as lower cost, similar cost, or higher cost relative to the target player.
- **League Tier and Contract:** each alternative shows league tier badge and contract status for transfer context.
- **Enriched Mode:** advanced-stat similarity for accepted Top 5 League matches.
- **Basic Mode:** global fallback using core performance features.
- **Exact Role:** candidates with the same explicit role tag.
- **Compatible Roles:** candidates above the tactical role-compatibility threshold.
- **Broad Position Group:** fallback logic for broad position matching.
- **Excel export:** download comparison results with metadata sheet.

### Scouting Rationale

The rationale tab explains candidate flags in non-technical scouting language with personalized, player-specific narratives.

It supports:

- **Personalized Rationale:** unique narrative per player referencing their specific data points (age, minutes, value gap, league tier, contract, form trend).
- **Data Confidence:** High/Medium/Low scoring based on data completeness.
- **Key Scouting Signals:** highlights positive data signals including league context, contract situation, and form trend.
- **Scout Checks Before Action:** contextual checks specific to the player's situation (not generic templates).
- **Evidence Table:** shows the concrete player evidence behind the rationale.
- **Suggested Action:** context-aware labels such as "Prioritize - Expiring Contract" or "Prioritize - Young and Undervalued".
- **PDF Scouting Brief:** downloadable one-page PDF report per player for sharing with sporting directors.

### Backtest Results

The backtest tab validates historical scouting leads against future market-value outcomes.

It supports:

- **Historical snapshots:** replay scouting leads from fixed dates between 2021 and 2025.
- **Future value outcomes:** checks 6-month and 12-month market-value movement.
- **Hit rate tracking:** labels a hit when future value rises at least 25%.
- **Eligible baseline comparison:** compares scouting leads against similar age, minutes, and value-eligible players.
- **Segment analysis:** shows success by value gap bucket, position, age group, and market value bucket.

Latest Phase 7 snapshot summary:

- **Historical scouting leads:** 4,757.
- **6-month hit rate:** 34.10%.
- **12-month hit rate:** 44.47%.
- **Median 12-month growth:** 14.29%.
- **12-month lift vs eligible baseline:** +3.86 percentage points.

## 🖼️ Dashboard Preview

![Scouting Dashboard Preview](outputs/figures/Screenshot%202026-05-12%20165131.png)

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

### Temporal Validation Backtest

Phase 7 replays the scouting signal on historical snapshots instead of using only the latest available valuation.

For each snapshot date:

- **Current value:** latest market value at or before the snapshot date.
- **Features:** appearances and recent performance from the 365 days before the snapshot date.
- **Prediction:** existing performance-only Model A, used as a fixed retrospective signal.
- **Future outcomes:** nearest market valuation in the 6-month and 12-month outcome windows.
- **Hit definition:** future value at least 25% higher than current value.

The backtest also compares scouting leads against an eligible baseline with the same age, minutes, and market-value filters but without requiring a positive value gap.

## 🧠 Scouting Rationale

The rationale layer is a scout-facing explanation layer. It does not show technical model-debugging artifacts in the dashboard. Instead, it turns model output and player evidence into practical review notes.

It explains:

- **Why the player is interesting:** predicted value is higher than listed market value.
- **Whether the sample is reliable:** minutes played are checked against scouting thresholds.
- **Whether the age profile fits recruitment:** U21 and U25 profiles are highlighted.
- **Whether the value gap is meaningful:** small, moderate, large, or very large gaps are labelled.
- **What to check next:** league strength, tactical role, injuries, contract, and transfer feasibility.

The intent is to support a scout's next action, not to claim automatic transfer recommendations.

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
- **Accepted enriched matches:** 1,791.
- **Enriched players in final similarity engine:** 1,211 after eligibility filters.

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

Role metadata is available for **19,448 / 20,304 players (95.78%)** in the role-enriched pool.

## ✅ Trust & Validation Layer

Because this project combines multiple datasets, validation was treated as part of the product, not an afterthought.

The pipeline generates:

- `outputs/matching_report.csv` for denominator-aware external data coverage.
- `outputs/matching_audit.csv` for row-level match review.
- `outputs/validation_report.json` for enrichment validation-gate checks.
- `outputs/role_enrichment_report.csv` for role metadata coverage.
- `outputs/role_validation_report.json` for role-aware similarity checks.
- `outputs/data_quality_audit_report.json` for dataset contract, entity-key, club/league context, and value-range checks.
- `outputs/player_context_resolution_audit.csv` for player-level Club / League resolution lineage.
- `outputs/temporal_validation/` for Phase 7 historical snapshot outputs, success-rate tables, charts, and validation report.

Key validation gates include:

- No accepted future-stat leakage.
- No accepted position mismatch.
- No conflicting duplicate Kaggle assignments.
- No fuzzy-review matches accepted automatically.
- Goalkeepers are never matched with outfield players.
- Exact Role mode only returns candidates with the same explicit role tag.
- Compatible Roles mode only returns candidates above the role-compatibility threshold.
- Scouting rationale uses actual player rows from the shortlist or full prediction pool, not separate technical artifacts.
- Suggested actions are derived from transparent thresholds for age, minutes, and value gap.
- Dashboard Club / League context is audited across featured players, clubs, valuation history, and domestic appearances.
- Temporal validation checks that snapshot features do not use appearances after the snapshot date.
- Future-value outcomes are validated to occur after the snapshot date and inside the configured 6-month or 12-month windows.

Latest validation snapshot:

| Check | Result |
|---|---:|
| Enrichment validation status | PASS |
| Role validation status | PASS |
| Future leakage accepted | 0 |
| Position mismatch accepted | 0 |
| Duplicate conflicting Kaggle assignments | 0 |
| Fuzzy-review accepted | 0 |
| Exact Role checks | 240 |
| Compatible Role checks | 260 |
| Role metadata coverage | 95.78% |
| Data quality audit status | WARN |
| Shortlist Club / League coverage | 100.00% |
| Full prediction Club / League coverage | 99.72% |
| Temporal validation status | WARN |
| Temporal leakage violations | 0 |
| Temporal historical scouting leads | 4,757 |
| Temporal 12M outcome coverage | 4,378 |

The temporal validation status is marked as WARN because the audit found extreme future-growth outliers above 2,000% in 135 six-month rows and 576 twelve-month rows. There are no temporal leakage violations; the warning is retained so those valuation outliers can be inspected rather than hidden.

## ⚠️ Limitations

- **Market value is not purely statistical:** contract length, injuries, reputation, agent influence, and transfer rumors can materially affect price.
- **Model A is discovery-oriented, not a pricing oracle:** the performance-only model intentionally excludes previous market value, so it is less accurate but more useful for finding potential mispricing.
- **Top 5 League enrichment is partial by design:** enriched similarity is available only for accepted external matches; other players fall back to Basic Mode.
- **Defensive and goalkeeper metrics remain imperfect:** even with enrichment, non-Top 5 League defenders and goalkeepers can have limited event-level data.
- **Role metadata is heuristic:** role tags and preferred-foot fit come from Transfermarkt metadata and compatibility rules, not from tactical event tracking.
- **Scouting rationale is rule-based decision support:** it translates model outputs and player evidence into review notes, but it does not prove causality.
- **Temporal validation is a fixed-model signal audit:** it replays the existing Model A on historical snapshots and is not the same as walk-forward retraining.
- **Similarity results are leads, not final recommendations:** candidates should be validated by human scouting, video review, and context-specific recruitment constraints.

## 🔮 Future Work

The core portfolio project is considered complete at Phase 12. The remaining extension is deployment.

Potential future extensions include:

- **Deployment:** publish the dashboard as a hosted scouting demo (Streamlit Cloud or HuggingFace Spaces).
- **Walk-forward retraining:** retrain the model at each historical snapshot for stricter temporal validation.
- **Injury enrichment:** add injury history data to improve transfer risk assessment.
- **Advanced stats expansion:** integrate additional data sources beyond Top 5 Leagues to improve profile metric coverage.

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

After the guarded pipeline completes, run the data quality audit:

```bash
python src/audit_data_quality.py
```

Then run the Phase 7 temporal validation backtest:

```powershell
.\scripts\run_phase_7_temporal_validation.ps1
```

You can also run the core scripts manually:

```bash
python src/data_engineering.py
python src/enrich_contract.py
python src/data_cleaning.py
python src/feature_engineering.py
python src/modeling.py
python src/scouting.py
python src/enrich_similarity.py
python src/validate_enrichment.py
python src/enrich_roles.py
python src/similarity.py
python src/validate_roles.py
python src/audit_data_quality.py
python src/temporal_backtest.py
python src/validate_temporal_backtest.py
```

Note: `data_engineering.py` now includes league tier enrichment (Phase 8) and form trend computation (Phase 11). `enrich_contract.py` (Phase 9) should be run after `data_engineering.py` and before `data_cleaning.py`.

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
| `outputs/data_quality_audit_report.json` | Dataset-level audit summary for schema, IDs, value ranges, validation reports, and Club / League context. |
| `outputs/data_quality_issues.csv` | Row-level issue list for data quality warnings and failures. |
| `outputs/player_context_resolution_audit.csv` | Player-level lineage for resolving Club / League display context. |
| `outputs/temporal_validation/temporal_backtest_candidates.csv` | Historical scouting leads with 6-month and 12-month future value outcomes. |
| `outputs/temporal_validation/temporal_backtest_summary.csv` | Overall and snapshot-level hit rates, growth rates, and baseline comparison. |
| `outputs/temporal_validation/temporal_validation_report.json` | PASS/FAIL temporal validation report for snapshot leakage, outcome windows, and metric consistency. |
| `outputs/contract_enrichment_report.csv` | Contract status coverage and distribution report. |

## 🧩 Project Structure

```text
app/
  dashboard.py                 # Streamlit scouting dashboard

src/
  data_engineering.py           # Temporal feature construction + league tier + form windows
  enrich_contract.py            # Contract awareness enrichment (Phase 9)
  data_cleaning.py              # Cleaning and formatting
  feature_engineering.py        # Per-90 features and encoding
  modeling.py                   # XGBoost market value models
  scouting.py                   # Undervalued candidate generation
  scouting_rationale.py         # Personalized scouting rationale (Phase 10)
  league_tiers.py               # League tier classification (Phase 8)
  enrich_similarity.py          # External advanced-stat matching
  validate_enrichment.py        # Advanced-stat validation gates
  role_mapping.py               # Role rules and compatibility matrix
  enrich_roles.py               # player_bio role metadata integration
  similarity.py                 # Similarity engine with profile metrics
  validate_roles.py             # Role-aware validation
  excluded_players.py           # Retired/inactive player exclusion registry
  audit_data_quality.py          # Dataset quality and dashboard-context audit
  temporal_backtest.py           # Historical scouting lead outcome backtest
  validate_temporal_backtest.py  # Temporal validation checks

scripts/
  run_v1_4_role_pipeline.ps1    # Guarded role-aware pipeline runner
  run_phase_7_temporal_validation.ps1

outputs/
  figures/
  models/
  shortlists/
  temporal_validation/
```

## Final Interpretation

This project is best understood as a data product, not only a model experiment.

The prediction model estimates market value. The scouting layer turns prediction errors into candidate discovery. The similarity engine converts a shortlist into recruitment alternatives. The role-aware layer makes those alternatives more tactically realistic. The rationale layer explains why a player deserves review in scout-friendly language. The temporal validation layer suggests that the scouting signal has modest but measurable value as a prioritization tool, especially over a 12-month horizon.

The league tier, contract, and form layers add real-world context that scouts need for decision-making. The export capabilities (Excel, PDF, watchlist) complete the workflow so scouts can share findings with their team without leaving the tool.

The result is a reproducible football scouting workflow that can answer:

> Who looks undervalued, who is statistically similar, who actually makes sense as a tactical alternative, why should a scout review them, what is their contract and form situation, and did similar historical leads gain value later?
