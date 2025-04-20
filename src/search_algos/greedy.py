import queue 
from copy import deepcopy
import sys
import os
import queue as Q
import ast
import random as R

sys.path.append(os.path.abspath("C:/Users/ASUS/Documents/Crop Project/crop-prediction-ai/src"))
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
        self.initialState = Problem.CropNode(self.problem.state)

    def set_frontier(self , node_list, heuristics):
        frontier = Q.PriorityQueue()
        for node in node_list:
            h_value = heuristics.get(node.state['current_crop'] ,float('inf'))
            frontier.put(OrderedNode( node.state['current_crop'] , h_value))
        return frontier
    
    def search(self):

        frontier = self.set_frontier(self.problem.expand_node(self.initialState), self.heuristicValues)
        list = []
        if not frontier.empty():
            for _ in range(5):
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


def Load_Files(filename):
    with open(filename, 'r') as f:
        data = f.read()
    heuristics = ast.literal_eval(data)
    return heuristics

def main():
    initial_state = {
    'soil': {
        'n': 20.8,       
        'p': 134.2,       
        'k': 199.9,      
        'ph': 5.9,     
        'organic_matter': 5.1,  
        'soil_moisture': 21.2    
    },
    'climate': {
        'temperature': 22.6,  
        'humidity': 92.3,     
        'rainfall': 112.7,    
        'sunlight_exposure': 8.8  
    },
    'environmental': {
        'irrigation_frequency': 1,  
        'water_usage_efficiency': 1,
        'fertilizer_usage':53.5,
        'pest_pressure': 1.1
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
    for x in result:
        print (x)

if __name__ == '__main__':
    main()

 