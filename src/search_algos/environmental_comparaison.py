from newProblemFormulation import cropProblem, load_crop_db
import pip
pip.main(["install","pandas", "matplotlib"])
import math
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from astar import a_star_search 
from csp import CropCSP


class EnvironmentalImpactEvaluator:

    def __init__(self, problem_class):
        self.problem_class = problem_class
        self.environmental_factors = [
            'water_usage_efficiency',
            'fertilizer_usage',
            'irrigation_frequency', 
            'pest_pressure'
        ]
        self.factor_weights = {
            'water_usage_efficiency': 0.25,
            'fertilizer_usage': 0.25,
            'irrigation_frequency': 0.25, 
            'pest_pressure': 0.25
        }
    
    def calculate_environmental_score(self, state, parameters):
        """Special scoring for environmental parameters"""
        score = 0.0
        n = 0
        global_min = {}  # Track global min for normalization (if needed)
        # First pass: Find global min for each parameter (if normalizing)
        for param, (min_val, _, max_val) in parameters.items():
            global_min[param] = min_val

        for param, (min_val, _, max_val) in parameters.items():
            if param in state:
                val = state[param]
                if val is not None:
                    range_size = max_val - min_val
                    if range_size <= 0:  # Prevent division by zero
                        continue

                    if val < min_val:
                        # Below minimum - linear penalty since min_val - val > 0
                        param_score = (min_val - val) / range_size
                    elif val > max_val:
                        # Above maximum - linear penalty since val - max_val > 0
                        param_score = (val - max_val) / range_size
                    else:
                        param_score = ((val - min_val) / range_size) ** 2  # Quadratic penalty
                else:        
                    if global_min[param] == 0:
                        # If global_min is 0, just use min_val directly
                        param_score = min_val  # Lower min_val = better (no division needed)
                    else:
                        param_score = min_val / global_min[param]  # Normalized penalty

                score += param_score
                n += 1
        return math.sqrt(score / n) if n > 0 else 0.0
    
    def evaluate_algorithm(self, algorithm, conditions, top_n=3):
        """Evaluate an algorithm's recommendations for environmental impact"""
        problem = self.problem_class(conditions)
        recommendations = algorithm(problem)[:top_n]
        
        if not recommendations:
            return None
        
        # Calculate environmental scores for each recommendation
        env_scores = []
        
        for crop, _ in recommendations:
            reqs = self.crop_db[crop]
            score = self.calculate_environmental_score(crop,conditions,reqs['environmental'])
            env_scores.append(score)
        
        return {
            'algorithm': algorithm.__name__,
            'conditions': conditions,
            'recommendations': [crop for crop, _ in recommendations],
            'avg_env_score': np.mean(env_scores),
            'best_env_score': min(env_scores),
            'env_score_std': np.std(env_scores)
        }
    


def compare_environmental_impact(algorithms, scenarios):
    """Compare algorithms based on environmental impact minimization"""
    crop_db = load_crop_db('./data/processed/crop_db.txt')
    problem = cropProblem(scenarios, crop_db)
    evaluator = EnvironmentalImpactEvaluator(problem)
    results = []
    factors = evaluator.environmental_factors
    weights = evaluator.factor_weights
    for scenario in scenarios:
        for algo in algorithms:
            result = evaluator.evaluate_algorithm(algo, scenario)
            if result:
                results.append(result)
    
    return pd.DataFrame(results)

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


def analyze_scenario_specific(results_df, scenario_index=0):
    """Deep dive into a specific environmental scenario"""
    scenario_results = results_df[results_df['conditions'].apply(
        lambda x: x == scenarios[scenario_index])]
    
    print(f"\n=== DETAILED ANALYSIS FOR SCENARIO {scenario_index} ===")
    print(f"Environmental Conditions: {scenarios[scenario_index]}")
    
    plt.figure(figsize=(12, 6))
    
    # Algorithm comparison
    plt.subplot(1, 2, 1)
    scenario_results.set_index('algorithm')['avg_env_score'].plot(
        kind='bar', color='teal')
    plt.title('Environmental Impact by Algorithm')
    plt.ylabel('Impact Score (Lower Better)')
    
    # Recommendation comparison
    plt.subplot(1, 2, 2)
    for _, row in scenario_results.iterrows():
        scores = [evaluator.calculate_environmental_score(crop, row['conditions']) 
                 for crop in row['recommendations']]
        plt.plot(scores, 'o-', label=row['algorithm'])
    
    plt.title('Individual Recommendation Scores')
    plt.xlabel('Recommendation Rank')
    plt.ylabel('Environmental Score')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def analyze_scenario_specific(results_df, scenario_index=0):
    """Deep dive into a specific environmental scenario"""
    scenario_results = results_df[results_df['conditions'].apply(
        lambda x: x == scenarios[scenario_index])]
    
    print(f"\n=== DETAILED ANALYSIS FOR SCENARIO {scenario_index} ===")
    print(f"Environmental Conditions: {scenarios[scenario_index]}")
    
    plt.figure(figsize=(12, 6))
    
    # Algorithm comparison
    plt.subplot(1, 2, 1)
    scenario_results.set_index('algorithm')['avg_env_score'].plot(
        kind='bar', color='teal')
    plt.title('Environmental Impact by Algorithm')
    plt.ylabel('Impact Score (Lower Better)')
    
    # Recommendation comparison
    plt.subplot(1, 2, 2)
    for _, row in scenario_results.iterrows():
        scores = [evaluator.calculate_environmental_score(crop, row['conditions']) 
                 for crop in row['recommendations']]
        plt.plot(scores, 'o-', label=row['algorithm'])
    
    plt.title('Individual Recommendation Scores')
    plt.xlabel('Recommendation Rank')
    plt.ylabel('Environmental Score')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from newProblemFormulation import cropProblem
    
    # 1. Define algorithms to compare
    algorithms = [a_star_search, 'greedy_search', CropCSP.get_all_options, 'genetic_algorithm']
    
    # 2. Generate environmental scenarios
    scenarios = [
        {
        'soil': {
            'n': 100.2, 'p': 82.0, 'k': 50.0,
            'ph': 6.0, 'organic_matter': 5.8,
            'soil_moisture': 20.4
        },
        'climate': {
            'temperature': 27.0, 'humidity': 80.0,
            'rainfall': 110.0, 'sunlight_exposure': 9.0
        },
        'environmental': {
            'irrigation_frequency': 3.0,
            'water_usage_efficiency': 3.0,
            'fertilizer_usage': 120.0,
            'pest_pressure': 50.0
        },
        'current_crop': None,
        'growth_stage': None
    },
        {
        'soil': {
            'n': 100.2, 'p': 82.0, 'k': 50.0,
            'ph': 6.0, 'organic_matter': 5.8,
            'soil_moisture': 20.4
        },
        'climate': {
            'temperature': 27.0, 'humidity': 80.0,
            'rainfall': 110.0, 'sunlight_exposure': 9.0
        },
        'environmental': {
            'irrigation_frequency': 3.0,
            'water_usage_efficiency': 3.0,
            'fertilizer_usage': 120.0,
            'pest_pressure': 50.0
        },
        'current_crop': None,
        'growth_stage': None
    },
        {
        'soil': {
            'n': 100.2, 'p': 82.0, 'k': 50.0,
            'ph': 6.0, 'organic_matter': 5.8,
            'soil_moisture': 20.4
        },
        'climate': {
            'temperature': 27.0, 'humidity': 80.0,
            'rainfall': 110.0, 'sunlight_exposure': 9.0
        },
        'environmental': {
            'irrigation_frequency': 3.0,
            'water_usage_efficiency': 3.0,
            'fertilizer_usage': 120.0,
            'pest_pressure': 50.0
        },
        'current_crop': None,
        'growth_stage': None
    }
    ]
    
    # 3. Initialize evaluator
    evaluator = EnvironmentalImpactEvaluator(cropProblem)
    
    # 4. Run comparison
    print("Running environmental impact comparison...")
    env_results = compare_environmental_impact(algorithms, scenarios)
    
    # 5. Save and display results
    env_results.to_csv('environmental_impact_results.csv', index=False)
    print(env_results[['algorithm', 'conditions', 'avg_env_score']])
    
    # 6. Visualize
    plot_environmental_comparison(env_results)
    
    # 7. Scenario deep dive
    analyze_scenario_specific(env_results, 0)