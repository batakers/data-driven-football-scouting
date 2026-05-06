import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import joblib
from pathlib import Path
import os
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if __name__ == "__main__":
    sys.modules.setdefault("src.similarity", sys.modules[__name__])

from src.role_mapping import (
    COMPATIBLE_ROLE_THRESHOLD,
    foot_role_fit,
    role_compatibility_score,
)

ENRICHED_DATA_PATH = "data/processed/enriched_similarity_players.csv"
ROLE_DATA_PATH = "data/processed/role_enriched_players.csv"
AUDIT_PATH = "outputs/matching_audit.csv"
MATCHING_REPORT_PATH = "outputs/matching_report.csv"
VALIDATION_REPORT_PATH = "outputs/validation_report.json"
SIMILARITY_ENGINE_PATH = "outputs/models/similarity_engine.pkl"

# Position-aware heuristic weights based on available dataset features
POSITION_FEATURE_WEIGHTS = {
    "Forward": {
        "age_at_valuation": 0.20,
        "height_in_cm": 0.10,
        "goals_per_90_ls": 0.35,
        "assists_per_90_ls": 0.25,
        "cards_per_90_ls": 0.10,
    },
    "Midfielder": {
        "age_at_valuation": 0.25,
        "height_in_cm": 0.10,
        "goals_per_90_ls": 0.20,
        "assists_per_90_ls": 0.35,
        "cards_per_90_ls": 0.10,
    },
    "Defender": {
        "age_at_valuation": 0.35,
        "height_in_cm": 0.35,
        "goals_per_90_ls": 0.00,
        "assists_per_90_ls": 0.00,
        "cards_per_90_ls": 0.30,
    },
    "Goalkeeper": {
        "age_at_valuation": 0.50,
        "height_in_cm": 0.50,
        "goals_per_90_ls": 0.00,
        "assists_per_90_ls": 0.00,
        "cards_per_90_ls": 0.00,
    },
}

# Advanced weights using Kaggle enrichment data
ENRICHED_POSITION_FEATURE_WEIGHTS = {
    "Forward": {
        "age_at_valuation": 0.15,
        "adv_Expected_xG": 0.30,
        "adv_Standard_SoT%": 0.20,
        "adv_SCA_SCA90": 0.20,
        "adv_Take-Ons_Succ%": 0.15,
    },
    "Midfielder": {
        "age_at_valuation": 0.15,
        "adv_KP_": 0.30,
        "adv_Progression_PrgP": 0.25,
        "adv_Tackles_TklW": 0.15,
        "adv_Int_": 0.15,
    },
    "Defender": {
        "age_at_valuation": 0.20,
        "adv_Blocks_Blocks": 0.25,
        "adv_Clr_": 0.25,
        "adv_Aerial Duels_Won%": 0.15,
        "adv_Challenges_Tkl%": 0.15,
    },
    "Goalkeeper": {
        "age_at_valuation": 0.25,
        "adv_Total_Cmp%": 0.30,
        "adv_Performance_Recov": 0.25,
        "adv_Aerial Duels_Won%": 0.20,
    },
}

class PlayerSimilarity:
    def __init__(self, basic_features, enriched_features, metadata_cols):
        self.basic_features = basic_features
        self.enriched_features = enriched_features
        self.metadata_cols = metadata_cols
        self.scaler_basic = MinMaxScaler()
        self.scaler_enriched = MinMaxScaler()
        self.data = None
        self.basic_scaled = None
        self.enriched_scaled = None

    @staticmethod
    def _bool_series(series):
        if series.dtype == bool:
            return series.fillna(False)
        return series.astype("string").str.strip().str.lower().isin(["true", "1", "yes"])

    def _enriched_mask(self):
        if "enriched_available" in self.data.columns:
            return self._bool_series(self.data["enriched_available"])
        return self.data["match_status"] == "matched"

    def _role_metadata_mask(self):
        if "role_metadata_available" not in self.data.columns:
            return pd.Series(False, index=self.data.index)
        return self._bool_series(self.data["role_metadata_available"])

    @staticmethod
    def _clean_role(value):
        if pd.isna(value):
            return "UNKNOWN"
        role = str(value).strip()
        return role if role else "UNKNOWN"

    @staticmethod
    def _scalar_bool(value):
        if pd.isna(value):
            return False
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "1", "yes"}

    def _role_filter_mask(self, target_idx, role_mode, same_position, role_threshold):
        target_row = self.data.iloc[target_idx]
        target_pos = target_row["position_group_raw"]
        target_role = self._clean_role(target_row.get("primary_role", "UNKNOWN"))
        role_available = self._role_metadata_mask()
        target_role_available = bool(role_available.iloc[target_idx]) and target_role != "UNKNOWN"

        if role_mode == "exact" and target_role_available:
            return role_available & (self.data["primary_role"] == target_role)

        if role_mode == "compatible" and target_role_available:
            role_scores = self.data["primary_role"].apply(
                lambda candidate_role: role_compatibility_score(target_role, self._clean_role(candidate_role))
            )
            return role_available & (role_scores >= role_threshold)

        if same_position:
            return self.data["position_group_raw"] == target_pos

        return pd.Series(True, index=self.data.index)

    def _score_role_layer(self, target_idx, candidate_row, role_mode, role_threshold):
        target_row = self.data.iloc[target_idx]
        target_role = self._clean_role(target_row.get("primary_role", "UNKNOWN"))
        candidate_role = self._clean_role(candidate_row.get("primary_role", "UNKNOWN"))

        target_has_role = bool(self._role_metadata_mask().iloc[target_idx]) and target_role != "UNKNOWN"
        candidate_has_role = self._scalar_bool(candidate_row.get("role_metadata_available", False)) and candidate_role != "UNKNOWN"

        if role_mode == "broad":
            return 1.0 if target_row["position_group_raw"] == candidate_row["position_group_raw"] else 0.0

        if role_mode == "exact":
            if not target_has_role or not candidate_has_role:
                return 0.0
            return 1.0 if target_role == candidate_role else 0.0

        if role_mode == "compatible":
            if not target_has_role or not candidate_has_role:
                return 0.0
            score = role_compatibility_score(target_role, candidate_role)
            return score if score >= role_threshold else 0.0

        return 0.0

    def _final_score(self, statistical_score, role_score, foot_score, has_role_layer):
        if not has_role_layer:
            return statistical_score
        return (0.80 * statistical_score) + (0.15 * role_score) + (0.05 * foot_score)

    def fit(self, df):
        """
        Prepare and fit the similarity engine.
        df should contain basic_features, enriched_features, and metadata_cols.
        """
        self.data = df.copy().reset_index(drop=True)

        # Impute missing basic values
        if 'height_in_cm' in self.data.columns:
            self.data['height_in_cm'] = self.data['height_in_cm'].fillna(self.data['height_in_cm'].median())
        self.data[self.basic_features] = self.data[self.basic_features].fillna(0)
        
        # Scale basic features
        self.basic_scaled = self.scaler_basic.fit_transform(self.data[self.basic_features])
        
        # Scale enriched features (only for rows that have them)
        enriched_mask = self._enriched_mask()
        if enriched_mask.any():
            self.data[self.enriched_features] = self.data[self.enriched_features].fillna(0)
            # We scale based on the whole dataset to keep range consistent, or just the enriched?
            # Better to scale based on enriched pool to maximize variance
            self.enriched_scaled = self.scaler_enriched.fit_transform(self.data[self.enriched_features])
        
        print(f"Similarity engine fitted with {len(self.data)} players ({enriched_mask.sum()} enriched).")

    def find_similar(
        self,
        player_id,
        top_n=10,
        same_position=True,
        mode='basic',
        role_mode='compatible',
        role_threshold=COMPATIBLE_ROLE_THRESHOLD,
    ):
        """
        Find top N similar players for a given player ID.
        mode: 'basic' or 'enriched'
        role_mode: 'exact', 'compatible', or 'broad'
        """
        if self.data is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        target_row = self.data[self.data['player_id'] == player_id]
        if target_row.empty:
            return pd.DataFrame()
        
        target_idx = target_row.index[0]
        target_pos = target_row.iloc[0]["position_group_raw"]
        target_value = target_row.iloc[0]["target_market_value"]
        target_age = target_row.iloc[0]["age_at_valuation"]

        # Determine mode and features
        target_has_enrichment = bool(self._enriched_mask().iloc[target_idx])
        use_enriched = (mode == 'enriched') and target_has_enrichment
        
        if use_enriched:
            features = self.enriched_features
            scaled_matrix = self.enriched_scaled
            weights_dict = ENRICHED_POSITION_FEATURE_WEIGHTS
            # Filter pool: Only enriched players
            candidate_mask = self._enriched_mask()
        else:
            features = self.basic_features
            scaled_matrix = self.basic_scaled
            weights_dict = POSITION_FEATURE_WEIGHTS
            candidate_mask = pd.Series(True, index=self.data.index)

        target_vector = scaled_matrix[target_idx].reshape(1, -1)

        # Get weights for this position
        pos_weights = weights_dict.get(target_pos, {f: 1.0/len(features) for f in features})
        w_vector = np.array([np.sqrt(pos_weights.get(f, 0.0)) for f in features])

        # Filter candidate pool with role-aware logic. If role metadata is missing,
        # the helper falls back to the v1.3 broad position behavior.
        role_mode = str(role_mode).lower().strip()
        if role_mode not in {"exact", "compatible", "broad"}:
            role_mode = "compatible"
        role_filter = self._role_filter_mask(target_idx, role_mode, same_position, role_threshold)
        candidate_mask = candidate_mask & role_filter

        # Exclude target player
        candidate_mask = candidate_mask & (self.data["player_id"] != player_id)
        candidate_indices = self.data.index[candidate_mask].to_numpy()

        if len(candidate_indices) == 0:
            return pd.DataFrame()

        # Apply weights
        weighted_target = target_vector.flatten() * w_vector
        candidate_vectors = scaled_matrix[candidate_indices]
        weighted_candidates = candidate_vectors * w_vector

        # Cosine distance
        distances = []
        norm_a = np.linalg.norm(weighted_target)
        for cv in weighted_candidates:
            norm_b = np.linalg.norm(cv)
            if norm_a == 0 or norm_b == 0:
                distances.append(1.0)
            else:
                sim = np.dot(weighted_target, cv) / (norm_a * norm_b)
                distances.append(max(0, 1 - sim))
                
        distances = np.array(distances)
        distances = np.nan_to_num(distances, nan=1.0, posinf=1.0, neginf=1.0)

        results = []
        for order_idx in range(len(candidate_indices)):
            idx = candidate_indices[order_idx]
            dist = distances[order_idx]
            row = self.data.iloc[idx]
            statistical_score = max(0, 1 - dist)
            role_score = self._score_role_layer(target_idx, row, role_mode, role_threshold)
            foot_score = foot_role_fit(
                self._clean_role(row.get("primary_role", "UNKNOWN")),
                row.get("foot", None),
            )
            has_role_layer = role_score > 0 and self._scalar_bool(row.get("role_metadata_available", False))
            final_score = self._final_score(statistical_score, role_score, foot_score, has_role_layer)

            player_data = row.to_dict()
            player_data["statistical_score"] = statistical_score
            player_data["statistical_similarity_score"] = statistical_score
            player_data["role_compatibility_score"] = role_score
            player_data["foot_role_fit_score"] = foot_score
            player_data["final_similarity_score"] = final_score
            player_data["similarity_score"] = final_score
            player_data["age_difference"] = row["age_at_valuation"] - target_age
            player_data["value_difference_vs_target"] = row["target_market_value"] - target_value
            player_data["is_cheaper"] = row["target_market_value"] < target_value
            player_data["is_younger"] = row["age_at_valuation"] < target_age
            player_data["is_undervalued"] = row["undervalued_pct"] > 0.15
            player_data["similarity_mode"] = "enriched" if use_enriched else "basic"
            player_data["role_matching_mode"] = role_mode

            results.append(player_data)

        return pd.DataFrame(results).sort_values("final_similarity_score", ascending=False).head(top_n)


PlayerSimilarity.__module__ = "src.similarity"

def require_validation_pass():
    report_path = Path(VALIDATION_REPORT_PATH)
    data_path = Path(ENRICHED_DATA_PATH)
    audit_path = Path(AUDIT_PATH)

    if not data_path.exists():
        print(f"Error: {ENRICHED_DATA_PATH} not found. Run src/enrich_similarity.py first.")
        return False
    if not audit_path.exists():
        print(f"Error: {AUDIT_PATH} not found. Run src/enrich_similarity.py first.")
        return False
    if not report_path.exists():
        print("Error: validation report not found. Run src/validate_enrichment.py before building the similarity engine.")
        return False

    with report_path.open() as f:
        report = json.load(f)

    if report.get("status") != "PASS":
        print("Error: enrichment validation did not pass. Fix validation errors before building the similarity engine.")
        for error in report.get("errors", []):
            print(f" - {error}")
        return False

    report_mtime = report_path.stat().st_mtime
    if report_mtime < data_path.stat().st_mtime or report_mtime < audit_path.stat().st_mtime:
        print("Error: validation report is older than enrichment outputs. Re-run src/validate_enrichment.py.")
        return False

    return True

def update_matching_report_with_similarity_counts(pool_count, enriched_count):
    report_path = Path(MATCHING_REPORT_PATH)
    if not report_path.exists():
        return

    report = pd.read_csv(report_path)
    row_mask = (report["section"] == "similarity_engine") & (
        report["segment"] == "enriched_in_similarity_engine"
    )
    report = report[~row_mask].copy()

    new_row = pd.DataFrame([
        {
            "section": "similarity_engine",
            "segment": "enriched_in_similarity_engine",
            "numerator": int(enriched_count),
            "denominator": int(pool_count),
            "rate": float(enriched_count / pool_count) if pool_count else 0.0,
            "description": "Accepted enriched players available after final similarity engine eligibility filters",
        }
    ])
    pd.concat([report, new_row], ignore_index=True).to_csv(report_path, index=False)

def main():
    if not require_validation_pass():
        return False

    # Load role-enriched data when v1.4 artifacts are available, otherwise fall back to v1.3.
    print("Loading data for similarity engine...")
    df_path = ROLE_DATA_PATH if os.path.exists(ROLE_DATA_PATH) else ENRICHED_DATA_PATH
    if not os.path.exists(df_path):
        print(f"Error: {df_path} not found. Run src/enrich_similarity.py first.")
        return False
        
    df = pd.read_csv(df_path)
    
    # Filter for reliability (900 minutes = 10 full games)
    # This ensures we compare players with enough sample size for per-90 metrics
    df = df[df['minutes_last_season'] >= 900].copy()
    
    basic_features = [
        'age_at_valuation', 'height_in_cm', 
        'goals_per_90_ls', 'assists_per_90_ls', 'cards_per_90_ls'
    ]
    
    # Extract all unique advanced features from ENRICHED_POSITION_FEATURE_WEIGHTS
    enriched_features = set()
    for pos, weights in ENRICHED_POSITION_FEATURE_WEIGHTS.items():
        for f in weights.keys():
            if f.startswith('adv_'):
                enriched_features.add(f)
    enriched_features = sorted(list(enriched_features))
    
    metadata_cols = [
        'player_id', 'name', 'position_group_raw', 'minutes_last_season', 
        'target_market_value', 'predicted_value', 'undervalued_pct', 'match_status',
        'match_method', 'match_score', 'kaggle_season', 'primary_role', 'secondary_roles',
        'role_tags', 'compatible_roles', 'role_family', 'side_preference', 'foot',
        'foot_role_fit', 'role_metadata_available'
    ]
    
    engine = PlayerSimilarity(basic_features, enriched_features, metadata_cols)
    engine.fit(df)
    enriched_count = int(engine._enriched_mask().sum())
    update_matching_report_with_similarity_counts(len(df), enriched_count)
    
    # Save the engine
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    joblib.dump(engine, SIMILARITY_ENGINE_PATH)
    print(f"Similarity engine saved to {SIMILARITY_ENGINE_PATH}")
    
    # Test with a well-known player who is matched
    if "enriched_available" in df.columns:
        enriched_mask = PlayerSimilarity._bool_series(df["enriched_available"])
    else:
        enriched_mask = df['match_status'] == 'matched'
    matched_players = df[enriched_mask]
    if not matched_players.empty:
        test_player = matched_players.iloc[0]
        test_player_id = test_player['player_id']
        test_player_name = test_player['name']
        
        print(f"\nTesting ENRICHED similarity for: {test_player_name} ({test_player['position_group_raw']})")
        similar = engine.find_similar(test_player_id, top_n=5, mode='enriched', role_mode='compatible')
        if similar is not None and not similar.empty:
            cols_to_show = [
                'name', 'primary_role', 'position_group_raw', 'target_market_value',
                'statistical_score', 'role_compatibility_score', 'final_similarity_score',
                'similarity_mode'
            ]
            print(similar[cols_to_show])
    else:
        print("\nNo matched players found for testing enriched mode.")

    return True

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
