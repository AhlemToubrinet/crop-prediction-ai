import ast
import json
import sys
import os
sys.path.append(os.path.abspath("C:/Users/ASUS/Documents/Crop Project/crop-prediction-ai/src/search_algos/newProblemFormulation.py"))
from newProblemFormulation import cropProblem, load_crop_db , cropNode
# jame3 el 7o9o9 ma7fouda li Alaa plz ask before using it 

class HeuristicCalculator:
    def __init__(self, current_state, crop_db_path , priorities = None):
        if priorities is not None:
            total = sum(priorities.values())
            self.priorities = {
                'soil': priorities.get('soil', 1) / total,
                'climate': priorities.get('climate', 1) / total,
                'environmental': priorities.get('environmental', 1) / total
            }
        else:
            # Default equal weights if no priorities provided
            self.priorities = {'soil': 0.4, 'climate': 0.3, 'environmental': 0.3}
            
        self.current_state = current_state
        self.crop_db = self.load_crop_db(crop_db_path)
        self.heuristics = {}
    def get_valid_crops(self, initial_state):

        problem = cropProblem(self.current_state, self.crop_db)
        return problem.get_valid_actions(self.current_state)

    def heuristic(self, crop):
        crop_conditions = self.crop_db.get(crop, {})
        if not crop_conditions:
            return float('inf')  # If crop or growth stage not found, assign maximum cost
        total_cost = 0

        if 'soil' in self.current_state:
            total_cost += self._compute_general_cost(self.current_state['soil'], crop_conditions.get('soil', {}), use_min=False, category="soil")

        if 'climate' in self.current_state:
            total_cost += self._compute_general_cost(self.current_state['climate'], crop_conditions.get('climate', {}), use_min=False, category="climate")
                
        if 'environmental' in self.current_state:
           total_cost += self._compute_general_cost(self.current_state['environmental'], crop_conditions.get('environmental', {}), use_min=True, category="environmental")           
        
        return total_cost
    
    def load_crop_db(self, filepath):
        with open(filepath, 'r') as file:
            data = file.read()
        return ast.literal_eval(data)
    
       
        
 


    def _compute_general_cost(self, actual, ideal, use_min, category):
    
        total_cost = 0
 
        for factor, actual_value in actual.items():
            ideal_range = ideal.get(factor)
            if not ideal_range:
                continue  

            min_val, mean_val, max_val = ideal_range

            priority = self.priorities.get(category)  

            target_value = min_val if use_min else mean_val

            # Estimate standard deviation using the range rule

            range_std = (max_val - min_val) / 4 if (max_val - min_val) > 0 else 1  

            z_score = abs((actual_value - mean_val) / range_std)  

            # Compute Mean Squared Error (MSE) 

            mse =  ((actual_value - target_value) / (max_val - min_val))**2 # Penalizes larger deviations more heavily

            cost =  priority * (mse + z_score)

            # Apply penalty if the value is out of range (higher penalty for extreme deviations)
            if actual_value < min_val or actual_value > max_val:

                cost += priority * (z_score ** 1.5)  # Exponential penalty to emphasize extreme deviations

            # give reward reward for values within the acceptable range (closer values get higher rewards)
            if min_val <= actual_value <= max_val:
                cost -= priority * (1 / (abs(actual_value - mean_val) + 1))  

            total_cost += cost

        return total_cost
    
    def Heuristics (self, initial_state):
        valid_crops = self.get_valid_crops(self.current_state)
        if not valid_crops:
            print("No valid crops found.")
            return []

        crop_scores = [(crop, self.heuristic( crop)) for crop in valid_crops]
        return crop_scores

    def generate_heuristics(self):
        for crop_name, crop_info in self.Heuristics(self.current_state):
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


    calculator = HeuristicCalculator(
        current_state=current_state,
        crop_db_path= 'C:/Users/ASUS/Documents/Crop Project/crop-prediction-ai/data/processed/crop_db.txt'
    )

    calculator.run("C:/Users/ASUS/Documents/Crop Project/crop-prediction-ai/src/search_algos/heuristics.txt")

