from newProblemFormulation2 import CropProblem, load_crop_db
import pip
pip.main(["install","pandas", "matplotlib"])
import math
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
def plot_environmental_comparison(results_df):
    """Visualize environmental impact comparison"""
    plt.figure(figsize=(15, 6))
    
    # 1. Average Environmental Score by Algorithm
    plt.subplot(1, 2, 1)
    avg_scores = results_df.groupby('algorithm')['avg_env_score'].mean()
    avg_scores.sort_values().plot(kind='bar', color='green')
    plt.title('Average Environmental Impact Score\n(Lower is Better)')
    plt.ylabel('Environmental Impact Score')
    plt.xticks(rotation=45)
    
    # 2. Score Distribution by Environmental Factor
    plt.subplot(1, 2, 2)
    
    factors = evaluator.environmental_factors
    weights = evaluator.factor_weights
    
    # Create example data - replace with actual decomposition
    factor_contributions = {
        algo: [np.random.random() * weight for factor, weight in weights.items()]
        for algo in results_df['algorithm'].unique()
    }
    
    bottom = None
    for i, factor in enumerate(factors):
        contributions = [factor_contributions[algo][i] 
                       for algo in avg_scores.index]
        if bottom is None:
            plt.bar(avg_scores.index, contributions, label=factor)
            bottom = contributions
        else:
            plt.bar(avg_scores.index, contributions, bottom=bottom, label=factor)
            bottom = [b + c for b, c in zip(bottom, contributions)]
    
    plt.title('Environmental Impact Factor Contributions')
    plt.ylabel('Weighted Contribution')
    plt.xticks(rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
