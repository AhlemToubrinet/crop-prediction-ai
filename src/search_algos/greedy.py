import queue 
from copy import deepcopy
import sys
import os
import queue as Q
import ast
import random as R

sys.path.append(os.path.abspath("C:/Users/ASUS/Documents/Crop Project/crop-prediction-ai/src/search_algos/newProblemFormulation.py"))
import newProblemFormulation as Problem

class OrderedNode:
    def __init__(self, node_name , heuristic_value):

        self.heuristic_value = heuristic_value
        self.node_name = node_name

    def __lt__(self, other):  # used by PriorityQueue for comparison
        return self.heuristic_value < other.heuristic_value
    
    def __str__(self):
        return f"( {self.node_name} , {self.heuristic_value} ) "


class GreedyBestFirstSearch:
    
    def __init__ (self , problem , heuristics):

        self.problem = problem
        self.heuristicValues = heuristics
        self.initialState = Problem.cropNode(self.problem.state)

    def set_frontier(self , node_list, heuristics):
        frontier = Q.PriorityQueue()
        for node in node_list:
            h_value = heuristics.get(node.state['current_crop'] ,float('inf'))
            frontier.put(OrderedNode( node.state['current_crop'] , h_value))
        return frontier
    
    def search(self , cropresults = 5):

        frontier = self.set_frontier(self.problem.expand_node(self.initialState), self.heuristicValues)
        list = []
        if not frontier.empty():
            for _ in range(cropresults):
                best_node = frontier.get()
                list.append(best_node)

            return list    
        
        return None, None
    
    def choose_random_crop(self , crops_db):
        """
        Choose a random crop from the crops database.
    
        Parameters:
            crops_db (dict): A dictionary where keys are crop names.
        
        Returns:
            str: The name of a randomly chosen crop.
        """
        crop_names = list(crops_db.keys())
        cropname = R.choice(crop_names)
        reqs = self.problem.state
        reqs['current_crop'] = cropname
        initial_crop = Problem.CropNode(reqs)
        return initial_crop


    def print_top_recommendations_using_heuristics(self, top_n=5):
        """Print top crop recommendations based on heuristic values"""
        recommendations = []

        for crop in self.problem.crop_db:
            h_value = self.heuristicValues.get(crop, float('inf'))
            if h_value != float('inf'):
                recommendations.append((crop, h_value))

        if not recommendations:
            print("No suitable crops found for current conditions.")
            return

        # Normalize scores to 0-1 range (min-max normalization)
        scores = [score for _, score in recommendations]
        min_score, max_score = min(scores), max(scores)

        if max_score == min_score:
            normalized_recommendations = [(crop, 0.0) for crop, _ in recommendations]
        else:
            normalized_recommendations = [
                (crop, (score - min_score) / (max_score - min_score))
                for crop, score in recommendations
            ]

        # Sort by score (ascending - lower is better)
        sorted_recommendations = sorted(normalized_recommendations, key=lambda x: x[1])[:top_n]

        print("\n=== TOP CROP RECOMMENDATIONS ===")
        for rank, (crop, norm_score) in enumerate(sorted_recommendations, 1):
            match_percent = 100 * (1 - norm_score)
            print(
                f"{rank}. {crop.capitalize()} - "
                f"Match: {max(0, match_percent):.2f}% "
                f"(Score: {norm_score:.4f})"
            )



def Load_Files(filename):
    with open(filename, 'r') as f:
        data = f.read()
    heuristics = ast.literal_eval(data)
    return heuristics

def main():
    initial_state ={
    'soil': {
        'n': 115,                 # Moderate nitrogen
        'p': 30,                 # Moderate phosphorus
        'k': 30,                 # Moderate potassium
        'ph': 6.8,               # Slightly acidic
        'organic_matter': 4.0,   # Coffee prefers rich, organic soil
        'soil_moisture': 30      # Moist but well-drained
    },
    'climate': {
        'temperature': 25,       # Optimal range: 18–24°C
        'humidity': 68,          # Coffee thrives in high humidity
        'rainfall': 120,        # Annual, in mm — prefers 1500–2500 mm
        'sunlight_exposure': 5.5 # Moderate sunlight (often grown under shade)
    },
    'environmental': {
        'irrigation_frequency': 2,  
        'water_usage_efficiency': 1, 
        'fertilizer_usage': 35,
        'pest_pressure': 0.8       # Relatively low pressure in good environments
    },
    'current_crop': None,
    'growth_stage': None
}

    heuristics = Load_Files("C:/Users/ASUS/Documents/Crop Project/crop-prediction-ai/src/search_algos/heuristics.txt")
    crop_db = Load_Files('./data/processed/crop_db.txt')
    if not crop_db:
        raise ValueError("Crop database is empty!")
    for crop in crop_db:
        if crop not in heuristics:
            heuristics[crop] = float('inf')  # fallback
            print(f"{crop}crop not in heuristic")

    problem = Problem.cropProblem(initial_state, crop_db)
    Try = GreedyBestFirstSearch(problem, heuristics)
    result = Try.search()
    Try.print_top_recommendations_using_heuristics()
if __name__ == '__main__':
    main()

 