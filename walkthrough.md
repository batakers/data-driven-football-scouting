# Walkthrough - Extension 4: Role-Aware Similarity Search (v1.4)

I have implemented **Role-Aware Similarity Search** (v1.4), which layers Transfermarkt role metadata on top of the v1.3 enriched performance similarity engine. The goal is to answer whether candidates are similar statistically and tactically plausible by role, side, and foot fit.

## Key Changes

### 1. External Data Enrichment (`src/enrich_similarity.py`)
- **Kaggle Dataset**: Integrated "Top 5 League Football Player Stats (2017-2025)" containing advanced metrics (xG, Key Passes, Progressive Actions, Defensive Blocks, etc.).
- **Matching Pipeline**:
    - **Normalization**: Standardized names and club names across Transfermarkt and Kaggle datasets.
    - **Season Alignment**: Automatically matched Transfermarkt `valuation_date` to the corresponding "completed previous season" in Kaggle (e.g., 24/25 valuation → 23/24 stats) to prevent future-stat leakage.
    - **Confidence Hierarchy**: Multi-stage matching (Exact Name/Season → team-validated exact match → league-filtered fuzzy match).
    - **Release Gates**: Accepted enrichment rows must be position-compatible, use only completed prior-season stats, map one-to-one to a Kaggle source row, and exclude manual-review fuzzy matches from auto-accepted enrichment.
- **Matching Performance**:
    - **Full Audit Rows**: 20,305
    - **Accepted Enriched Matches**: 1,792 (**8.83% overall coverage**)
    - **Eligible Top 5 League Rows**: 4,232
    - **Accepted Eligible Top 5 Matches**: 1,791 (**42.32% eligible Top 5 coverage**)
    - **Similarity Engine Pool (>900 mins)**: 4,852 players, including 1,212 enriched players after the final engine eligibility filters.
    - **Accepted Match Method Breakdown**: 1,553 exact name/team/season, 239 fuzzy high-confidence, 0 fuzzy review.

### 2. Similarity Engine Upgrade (`src/similarity.py`)
- **Dual-Mode Architecture**:
    - **Enriched Mode**: Uses position-specific advanced weights (e.g., xG/SCA for Forwards, PrgP/KP for Midfielders, Blocks/Aerials for Defenders).
    - **Basic Mode (Fallback)**: Uses core stats (Goals/Assists/Cards) for players outside the Top 5 league coverage.
- **Role-Aware Layer**:
    - Adds `statistical_score`, `role_compatibility_score`, `foot_role_fit_score`, and `final_similarity_score`.
    - Uses `final_similarity_score = 0.80 * statistical + 0.15 * role + 0.05 * foot` when role metadata is available.
    - Falls back to v1.3 broad position logic when role metadata is missing.
- **Role Matching Modes**:
    - **Compatible Roles**: Default mode using a role compatibility threshold of 0.55.
    - **Exact Role**: Requires the same primary role.
    - **Broad Position Group**: Uses the v1.3 position-group behavior.
- **Position-Specific Advanced Weights**:
    - **Forward**: Emphasis on `Expected xG`, `SoT%`, and `Shot Creating Actions (SCA)`.
    - **Midfielder**: Focus on `Key Passes`, `Progressive Passes`, and `Tackles Won`.
    - **Defender**: Granular metrics for `Blocks`, `Clearances`, and `Aerial Duel %`.
    - **Goalkeeper**: Metrics for `Pass Completion %` and `Recoveries`.

### 3. Role Metadata Enrichment (`src/enrich_roles.py`)
- **Transfermarkt Join**: Joins `player_bio.tmid` to `player_id`, avoiding fuzzy name matching.
- **Role Fields**: Adds `primary_role`, `secondary_roles`, `role_tags`, `compatible_roles`, `role_family`, `side_preference`, `foot_role_fit`, and `role_metadata_available`.
- **Coverage**: Role metadata is available for 19,448 of 20,305 rows (**95.78%**).
- **Reports**: Generates `outputs/role_enrichment_report.csv` and `outputs/role_enrichment_audit.csv`.
- **Feature Discipline**: Market value, contract, captain, and jersey fields from `player_bio.csv` are not used in similarity scoring.

### 4. Dashboard Integration (`app/dashboard.py`)
- **Advanced Stats Indicators**: Visual badges identify players with available enriched data.
- **Model Selection Toggle**: Users can switch between "Enriched (Advanced Stats)" and "Basic (Core Stats)" for matched players.
- **Role Matching Toggle**: Users can choose Compatible Roles, Exact Role, or Broad Position Group.
- **Role-Aware Target Profile**: Displays primary role, role tags, foot, selected role mode, and metadata availability.
- **Score Breakdown**: Result tables include statistical score, role score, and final match score.
- **Dynamic Enriched Tables**: When in Enriched mode, the results table reveals advanced metrics tailored to the target position.
- **Methodology Context**: Updated tooltips to explain the provenance and limitations of the enriched data.

## Verification Results
- **Validation Status**: `src/validate_enrichment.py` PASS.
- **Role Validation Status**: `src/validate_roles.py` PASS.
- **Temporal Integrity**: 0 accepted future-season leakage cases.
- **Position Gate**: 0 accepted position mismatch cases.
- **Duplicate Gate**: 0 accepted conflicting Kaggle row assignments.
- **Fuzzy Review Gate**: 0 accepted `fuzzy_review` rows without manual review.
- **Role Metadata Coverage**: 95.78% across the role-enriched pool.
- **Role Mode Checks**: Exact Role and Compatible Role validation each checked 260 returned candidates.
- **Similarity Sanity Check**: Daley Blind returns CB-compatible defender recommendations with statistical, role, foot, and final score components.
- **Stability**: Maintained full backward compatibility for non-Top 5 league players using the v1.2 fallback logic.

## How to Run
1. Ensure dependencies are installed: `pip install -r requirements.txt`
2. Run the enrichment pipeline:
   ```powershell
   .\scripts\run_v1_4_role_pipeline.ps1
   ```
3. Or run each release-gated step explicitly:
   ```powershell
   python src/enrich_similarity.py
   if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

   python src/validate_enrichment.py
   if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

   python src/enrich_roles.py
   if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

   python src/similarity.py
   if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

   python src/validate_roles.py
   if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
   ```
4. Launch the dashboard:
   ```bash
   streamlit run app/dashboard.py
   ```
