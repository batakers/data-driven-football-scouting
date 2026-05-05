import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def ensure_dirs():
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)

def plot_market_value_distribution(df):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df['target_market_value'] / 1e6, bins=50, kde=True)
    plt.title('Market Value Distribution')
    plt.xlabel('Market Value (Millions EUR)')
    
    plt.subplot(1, 2, 2)
    sns.histplot(np.log1p(df['target_market_value']), bins=50, kde=True)
    plt.title('Log(Market Value) Distribution')
    plt.xlabel('Log(Market Value)')
    
    plt.tight_layout()
    plt.savefig('outputs/figures/eda_market_value_distribution.png')
    plt.close()

def plot_age_vs_value(df):
    plt.figure(figsize=(10, 6))
    sample_df = df.sample(n=min(5000, len(df)), random_state=42)
    sns.scatterplot(data=sample_df, x='age_at_valuation', y='target_market_value', alpha=0.3)
    plt.yscale('log')
    plt.title('Age vs Market Value (Log Scale)')
    plt.xlabel('Age at Valuation')
    plt.ylabel('Market Value (EUR)')
    plt.savefig('outputs/figures/eda_age_vs_value.png')
    plt.close()

def plot_position_vs_value(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='position', y='target_market_value')
    plt.yscale('log')
    plt.title('Market Value by Position')
    plt.savefig('outputs/figures/eda_position_vs_value.png')
    plt.close()

def main():
    print("Starting EDA...")
    ensure_dirs()
    df = pd.read_csv("data/processed/cleaned_players.csv")
    
    print("Plotting Market Value Distribution...")
    plot_market_value_distribution(df)
    
    print("Plotting Age vs Value...")
    plot_age_vs_value(df)
    
    print("Plotting Position vs Value...")
    plot_position_vs_value(df)
    
    print("EDA Complete! Figures saved to outputs/figures/")

if __name__ == '__main__':
    main()
