import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import joblib
from pathlib import Path

class PlayerSimilarity:
    def __init__(self, similarity_features, metadata_cols):
        self.similarity_features = similarity_features
        self.metadata_cols = metadata_cols
        self.scaler = MinMaxScaler()
        self.model = NearestNeighbors(metric="cosine")
        self.data = None
        self.features_scaled = None

    def fit(self, df):
        """
        Prepare and fit the similarity engine.
        df should contain both similarity_features and metadata_cols.
        """
        # Ensure no NaN in similarity features
        self.data = df.copy().reset_index(drop=True)
        self.data[self.similarity_features] = self.data[self.similarity_features].fillna(0)
        
        # Scale features
        self.features_scaled = self.scaler.fit_transform(self.data[self.similarity_features])
        
        # Fit KNN model
        self.model.fit(self.features_scaled)
        
        print(f"Similarity engine fitted with {len(self.data)} players.")

    def find_similar(self, player_name, top_n=10, same_position=True):
        """
        Find top N similar players for a given player name.
        """
        if self.data is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        # Find the target player index
        target_idx = self.data[self.data['name'] == player_name].index
        if len(target_idx) == 0:
            print(f"Player '{player_name}' not found in dataset.")
            return None
        
        target_idx = target_idx[0]
        target_vector = self.features_scaled[target_idx].reshape(1, -1)
        target_pos = self.data.iloc[target_idx]['position_group_raw']
        target_value = self.data.iloc[target_idx]['target_market_value']
        target_age = self.data.iloc[target_idx]['age_at_valuation']
        
        # Search for enough neighbors to allow for position filtering
        # We search for a larger number and then filter
        search_n = min(len(self.data), top_n * 20) 
        distances, indices = self.model.kneighbors(target_vector, n_neighbors=search_n)
        
        distances = distances[0]
        indices = indices[0]
        
        results = []
        for dist, idx in zip(distances, indices):
            # Skip self (Check by player_id to be safe)
            if self.data.iloc[idx]['player_id'] == self.data.iloc[target_idx]['player_id']:
                continue
                
            player_data = self.data.iloc[idx].to_dict()
            
            # Position filter (STRICT)
            if same_position and player_data['position_group_raw'] != target_pos:
                continue
            
            # Calculate additional metrics
            # 1 - dist converts cosine distance to cosine similarity
            player_data['similarity_score'] = 1 - dist
            player_data['is_cheaper'] = player_data['target_market_value'] < target_value
            player_data['is_younger'] = player_data['age_at_valuation'] < target_age
            
            # Highlight if the player is considered undervalued by our ML model
            # (Undervalued if predicted value is higher than target market value)
            player_data['is_undervalued'] = player_data['undervalued_pct'] > 0.15 # 15% threshold
            
            results.append(player_data)
            
            if len(results) >= top_n:
                break
                
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
        'age_at_valuation', 'goals_per_90_ls', 'assists_per_90_ls', 'cards_per_90_ls'
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
    
    # Test with a well-known player if possible, or just the first one
    test_player = df['name'].iloc[0]
    print(f"\nTesting similarity for: {test_player} ({df['position_group_raw'].iloc[0]})")
    similar = engine.find_similar(test_player, top_n=5)
    if similar is not None:
        cols_to_show = ['name', 'position_group_raw', 'age_at_valuation', 'target_market_value', 'similarity_score', 'is_cheaper', 'is_undervalued']
        print(similar[cols_to_show])

if __name__ == "__main__":
    main()
