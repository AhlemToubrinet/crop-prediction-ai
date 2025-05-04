from newProblemFormulation import cropProblem, load_crop_db
import pip
pip.main(["install","pandas", "matplotlib"])
import math
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from astar import a_star_search as AStar
from csp import CropCSP as CSP
import random
from genetic import CropGeneticAlgorithm as Genetic



class EnvironmentalImpactEvaluator:

    def __init__(self,problem):
        self.problem = problem
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
            if param in state['environmental']:
                val = state['environmental'][param]
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
    
    def evaluate_algorithm(self, algorithm, conditions, top_n=10):
        """Evaluate an algorithm's recommendations for environmental impact"""
        crop_db = load_crop_db('./data/processed/crop_db.txt')
        problem = cropProblem(conditions, crop_db)
        recommendations = []
        if algorithm == AStar:
            recommendation = algorithm(problem,top_n)
            recommendations =  [t[0] for t in recommendation ]
   
        elif algorithm == CSP:
            problem = CSP(problem,crop_db,conditions)
            recommendation = CSP.get_all_options(problem,top_n)
            recommendations = [ t[0] for t in recommendation ]
           
        elif algorithm == Genetic:
            ga = Genetic(problem)
            recommendation = ga.get_top_n_crops(top_n)
            recommendations = [ item["crop"] for item in recommendation ]
            

        if not recommendations:
            return None
        
        # Calculate environmental scores for each recommendation
        env_scores = []
        
        for crop in recommendations:
            reqs = crop_db[crop]
            score = self.calculate_environmental_score(conditions,reqs['environmental'])
            env_scores.append(score)
        
        return {
            'algorithm': algorithm.__name__,
            'conditions': conditions,
            'recommendations': [rec for rec in recommendations],
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

def compare_algorithm_environmental_impact(scenarios, crop_db, algorithms):
    """
    Compare environmental parameters of best crops recommended by each algorithm.
    
    Args:
        scenarios (list): List of scenario dictionaries
        crop_db (dict): The crop database dictionary
        algorithms (list): List of algorithm functions to compare
        evaluator: EnvironmentalImpactEvaluator instance
    """
    # Initialize storage for environmental parameters
    parameters = evaluator.environmental_factors
    env_data = {algo.__name__: {param: 0.0 for param in parameters} for algo in algorithms}
    counts = {algo.__name__: 0 for algo in algorithms}  # Track how many scenarios succeeded per algorithm
    
    for scenario in scenarios:
        for algo in algorithms:
            try:
                # Get algorithm recommendations
                scenario_results = evaluator.evaluate_algorithm(algo, scenario, top_n=10)
                
                if scenario_results and scenario_results['recommendations']:
                    best_crop = scenario_results['recommendations'][0]
                    counts[algo.__name__] += 1
                    
                    # Accumulate environmental requirements
                    if best_crop in crop_db:
                        for param in parameters:
                            if param in crop_db[best_crop]['environmental']:
                                min_val = crop_db[best_crop]['environmental'][param][1]
                                env_data[algo.__name__][param] += min_val
            except Exception as e:
                print(f"Error processing {algo.__name__} for scenario: {e}")
                continue
    
    # Calculate averages (skip algorithms with no successful scenarios)
    for algo in algorithms:
        algo_name = algo.__name__
        if counts[algo_name] > 0:
            for param in parameters:
                env_data[algo_name][param] /= counts[algo_name]
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame.from_dict(env_data, orient='index')
        
    # Plot each parameter in a separate subplot with values on bars
    num_params = len(parameters)
    cols = 2
    rows = (num_params + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.suptitle('Comparison of Environmental Requirements for Top Recommended Crops', y=1.02)
    axes = axes.flatten()

    for i, param in enumerate(parameters):
        ax = axes[i]
        bars = comparison_df[param].plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title(param)
        ax.set_ylabel('Average Requirement')
        #ax.set_xlabel('Algorithm')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top of each bar
        for bar in bars.patches:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig('algorithm_environmental_comparison_separate_with_values.png')
    plt.show()
        
    return comparison_df
    

def generate_random_agricultural_data():
    """Generate random agricultural data with realistic ranges"""
    return {
        'soil': {
            'n': round(random.uniform(10.0, 200.0), 1),  # Nitrogen (kg/ha)
            'p': round(random.uniform(5.0, 150.0), 1),   # Phosphorus (kg/ha)
            'k': round(random.uniform(20.0, 300.0), 1),  # Potassium (kg/ha)
            'ph': round(random.uniform(4.5, 8.5), 1),    # pH level
            'organic_matter': round(random.uniform(1.0, 10.0)),  # Percentage
            'soil_moisture': round(random.uniform(5.0, 35.0)),  # Percentage
        },
        'climate': {
            'temperature': round(random.uniform(10.0, 40.0), 1),  # Celsius
            'humidity': round(random.uniform(30.0, 95.0)),  # Percentage
            'rainfall': round(random.uniform(0.0, 50.0)),  # mm/week
            'sunlight_exposure': round(random.uniform(4.0, 14.0)),  # hours/day
        },
        'environmental': {
            'irrigation_frequency': random.randint(1, 14),  # times/week
            'water_usage_efficiency': round(random.uniform(10.0, 80.0)),  # Percentage
            'fertilizer_usage': round(random.uniform(5.0, 50.0)),  # kg/ha
            'pest_pressure': round(random.uniform(0.0, 100.0)),  # Index (0-100)
        },
        'current_crop': None,
        'growth_stage': None
    }
    



if __name__ == "__main__":
    
    random.seed(53)
    # 1. Define algorithms to compare
    algorithms = [AStar, CSP, Genetic]
    scenarios = []
    # 2. Generate environmental scenarios
    for _ in range(100):
        scenarios.append(generate_random_agricultural_data())
    
    
    # 3. Initialize evaluator
    crop_db = load_crop_db('./data/processed/crop_db.txt')
    problem = cropProblem(scenarios,crop_db)
    evaluator = EnvironmentalImpactEvaluator(problem)
    
    # 4. Run comparison
    #print("Running environmental impact comparison...")
    #env_results = compare_environmental_impact(algorithms, scenarios)
    
    # 5. Save and display results
    #env_results.to_csv('environmental_impact_results.csv', index=False)
    #print(env_results[['algorithm', 'conditions', 'avg_env_score']])
    
    # 6. Visualize
    #plot_environmental_comparison(env_results)
    
    # 7. Scenario deep dive
    #analyze_scenario_specific(env_results, 10)

    env_comparison = compare_algorithm_environmental_impact(scenarios, crop_db, algorithms)
    print(env_comparison)