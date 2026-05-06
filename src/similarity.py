import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import joblib
from pathlib import Path

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

class PlayerSimilarity:
    def __init__(self, similarity_features, metadata_cols):
        self.similarity_features = similarity_features
        self.metadata_cols = metadata_cols
        self.scaler = MinMaxScaler()
    def __init__(self, similarity_features, metadata_cols):
        self.similarity_features = similarity_features
        self.metadata_cols = metadata_cols
        self.scaler = MinMaxScaler()
        self.data = None
        self.features_scaled = None

    def fit(self, df):
        """
        Prepare and fit the similarity engine.
        df should contain both similarity_features and metadata_cols.
        """
        # Ensure no NaN in similarity features
        self.data = df.copy().reset_index(drop=True)

        # Impute missing values
        # For height, median is more representative than 0
        if 'height_in_cm' in self.data.columns:
            self.data['height_in_cm'] = self.data['height_in_cm'].fillna(self.data['height_in_cm'].median())
            
        self.data[self.similarity_features] = self.data[self.similarity_features].fillna(0)
        
        # Scale features
        self.features_scaled = self.scaler.fit_transform(self.data[self.similarity_features])
        
        print(f"Similarity engine fitted with {len(self.data)} players.")

    def find_similar(self, player_id, top_n=10, same_position=True):
        """
        Find top N similar players for a given player ID.
        Position filtering is applied before ranking, so results do not disappear
        just because the first global neighbors are from different positions.
        """
        if self.data is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        target_idx = self.data[self.data['player_id'] == player_id].index

        if len(target_idx) == 0:
            print(f"Player ID '{player_id}' not found in dataset.")
            return pd.DataFrame()

        target_idx = target_idx[0]
        target_vector = self.features_scaled[target_idx].reshape(1, -1)

        target_pos = self.data.iloc[target_idx]["position_group_raw"]
        target_value = self.data.iloc[target_idx]["target_market_value"]
        target_age = self.data.iloc[target_idx]["age_at_valuation"]

        # Get weights for this position
        pos_weights = POSITION_FEATURE_WEIGHTS.get(target_pos, {f: 1.0/len(self.similarity_features) for f in self.similarity_features})
        
        # Create a weight vector aligned with similarity_features
        # We use the square root of weights for cosine similarity to maintain proper scaling
        w_vector = np.array([np.sqrt(pos_weights.get(f, 0.0)) for f in self.similarity_features])

        # Filter candidate pool first
        if same_position:
            candidate_mask = self.data["position_group_raw"] == target_pos
        else:
            candidate_mask = pd.Series(True, index=self.data.index)

        # Exclude target player
        candidate_mask = candidate_mask & (self.data["player_id"] != player_id)
        candidate_indices = self.data.index[candidate_mask].to_numpy()

        if len(candidate_indices) == 0:
            return pd.DataFrame()

        # Apply weights to target and candidate vectors
        weighted_target = target_vector.flatten() * w_vector
        candidate_vectors = self.features_scaled[candidate_indices]
        weighted_candidates = candidate_vectors * w_vector

        # Manual cosine distance calculation for the filtered pool
        distances = []
        for cv in weighted_candidates:
            # Cosine similarity = (A . B) / (||A|| * ||B||)
            norm_a = np.linalg.norm(weighted_target)
            norm_b = np.linalg.norm(cv)
            
            if norm_a == 0 or norm_b == 0:
                distances.append(1.0)
            else:
                sim = np.dot(weighted_target, cv) / (norm_a * norm_b)
                distances.append(max(0, 1 - sim))
                
        distances = np.array(distances)
        
        # Handle possible numerical issues
        distances = np.nan_to_num(distances, nan=1.0, posinf=1.0, neginf=1.0)

        # Find top N
        top_order = np.argsort(distances)[:top_n]

        results = []
        for order_idx in top_order:
            idx = candidate_indices[order_idx]
            dist = distances[order_idx]
            row = self.data.iloc[idx]

            player_data = row.to_dict()
            player_data["similarity_score"] = max(0, 1 - dist)

            player_data["age_difference"] = row["age_at_valuation"] - target_age
            player_data["value_difference_vs_target"] = row["target_market_value"] - target_value
            player_data["is_cheaper"] = row["target_market_value"] < target_value
            player_data["is_younger"] = row["age_at_valuation"] < target_age
            player_data["is_undervalued"] = row["undervalued_pct"] > 0.15

            results.append(player_data)

        return pd.DataFrame(results)

def main():
    # Load data
    print("Loading data for similarity engine...")
    df_features = pd.read_csv("data/processed/featured_players.csv")
    df_preds = pd.read_csv("outputs/predictions_per_player.csv")
    
    # Merge predictions into features (to get predicted_value and undervalued_pct)
    # This ensures we use the FULL pool of players who have predictions
    df = pd.merge(
        df_features, 
        df_preds[['player_id', 'predicted_value', 'undervalued_pct']], 
        on='player_id', 
        how='inner'
    )
    
    # Filter for reliability (900 minutes = 10 full games)
    # This ensures we compare players with enough sample size for per-90 metrics
    df = df[df['minutes_last_season'] >= 900].copy()
    
    similarity_features = [
        'age_at_valuation', 'height_in_cm', 
        'goals_per_90_ls', 'assists_per_90_ls', 'cards_per_90_ls'
    ]
    
    metadata_cols = [
        'player_id', 'name', 'position_group_raw', 'minutes_last_season', 
        'target_market_value', 'predicted_value', 'undervalued_pct'
    ]
    
    engine = PlayerSimilarity(similarity_features, metadata_cols)
    engine.fit(df)
    
    # Save the engine
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    joblib.dump(engine, "outputs/models/similarity_engine.pkl")
    print("Similarity engine saved to outputs/models/similarity_engine.pkl")
    
    # Test with a well-known player if possible
    test_player_id = df['player_id'].iloc[0]
    test_player_name = df['name'].iloc[0]
    print(f"\nTesting similarity for: {test_player_name} ({df['position_group_raw'].iloc[0]})")
    similar = engine.find_similar(test_player_id, top_n=5)
    if similar is not None and not similar.empty:
        cols_to_show = ['name', 'position_group_raw', 'age_at_valuation', 'target_market_value', 'similarity_score', 'is_cheaper', 'is_undervalued']
        print(similar[cols_to_show])

if __name__ == "__main__":
    main()
