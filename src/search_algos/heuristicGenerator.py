import ast
import json
import sys
import os
sys.path.append(os.path.abspath("C:/Users/ASUS/Documents/Crop Project/crop-prediction-ai/src"))
from newProblemFormulation import cropProblem, load_crop_db , CropNode
# jame3 el 7o9o9 ma7fouda li Alaa plz ask before using it 

class HeuristicCalculator:
    def __init__(self, current_state, crop_db_path):
        self.current_state = current_state
        self.crop_db = self.load_crop_db(crop_db_path)
        self.heuristics = {}
    def get_valid_crops(self, initial_state):

        problem = cropProblem(self.current_state, self.crop_db)
        return problem.get_valid_actions(self.current_state)

    def heuristic(self, state, crop_name):
        
        problem = cropProblem(state, self.crop_db)
        return problem.calculate_cost(state, crop_name)
    def load_crop_db(self, filepath):
        with open(filepath, 'r') as file:
            data = file.read()
        return ast.literal_eval(data)
    
    # example heuristic function wlh makhdatha ya sanaa 
    def example_Heuristic (self, initial_state):
        valid_crops = self.get_valid_crops(self.current_state)
        if not valid_crops:
            print("No valid crops found.")
            return []

        crop_scores = [(crop, self.heuristic(self.current_state, crop)) for crop in valid_crops]
        return crop_scores

    def generate_heuristics(self):
        for crop_name, crop_info in self.example_Heuristic(self.current_state):
            score = crop_info
            self.heuristics[crop_name] = score

    def save_to_file(self, filepath):
        with open(filepath, 'w') as f:
            f.write(str(self.heuristics))  # Save as string

    def run(self, output_path):
        self.generate_heuristics()
        self.save_to_file(output_path)
        print(f"Heuristic values saved to: {output_path}")

#  Usage Example just to see that the file will be changed  
if __name__ == "__main__":
    current_state = {
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
            'fertilizer_usage': 53.5,
            'pest_pressure': 1.1
        }
    }

    calculator = HeuristicCalculator(
        current_state=current_state,
        crop_db_path= 'C:/Users/ASUS/Documents/Crop Project/crop-prediction-ai/data/processed/crop_db.txt'
    )

    calculator.run("C:/Users/ASUS/Documents/Crop Project/crop-prediction-ai/src/search_algos/heuristics.txt")

