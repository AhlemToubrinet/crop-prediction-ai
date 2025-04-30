import copy
import math
from typing import List, Dict, Tuple

class CropNode:
    def __init__(self, state, parent=None, action=None, cost=0):
        self.state = state  # Dictionary of environmental conditions + current crop choice
        self.parent = parent
        self.action = action  # Crop assigned in this step
        self.cost = cost  # Suitability cost (lower = better)
        
        if parent is None:  # Root node
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    def __hash__(self):
        """Hash based on immutable state components"""
        return hash(frozenset(self.state.items()))  # For dictionaries

    def __eq__(self, other):
        if isinstance(other , CropNode):
            return self.state == other.state 
        return False
    
class cropProblem:
    def __init__(self, initial_state, state_transition_model,priorities = None, goal_state = None, path_cost = 0, actions = ""):
        """initial_state is of the form:
            {
                'soil':{'n':20.8, 'p':134.2, 'k':199.9, 'ph':5.9, 'organic_matter':5.1, 'soil_moisture':21.2,}
                'climate:{'temperature':22.6, 'humidity':92.3, 'rainfall':112.7, 'sunlight_exposure':8.8,}
                'environmental':{'irrigation_frequency':3.5, 'water_usage_efficiency':2.8}
                'current_crop':None
                'growth_stage':None   
            }
        """
        """ state_transition_model = crop_db = 
            {
                'apple': {
                    '1': {  # Growth stage 1
                        'soil': {'n': (0.0, 20.8, 40.0), ...},
                        'climate': {...},
                        'environmental': {...}
                    },
                    '2': {  # Growth stage 2
                        'soil': {...},
                        ...
                    },
                    '3': {...}
                },
                'banana': {
                    '1': {...},
                    '2': {...},
                    '3': {...}
                }
            } 
        """
        self.state = initial_state 
        self.goal_state = goal_state
        self.crop_db = state_transition_model
        self.actions=actions
        self.path_cost = path_cost
        if priorities is not None:
            total = sum(priorities.values())
            self.weights = {
                'soil': priorities.get('soil', 1) / total,
                'climate': priorities.get('climate', 1) / total,
                'environmental': priorities.get('environmental', 1) / total
            }
        else:
            # Default equal weights if no priorities provided
            self.weights = {'soil': 0.4, 'climate': 0.3, 'environmental': 0.3}
    def is_goal(self, current_state):
        """this will be modified later"""
        return current_state == self.goal_state
    def get_valid_actions(self, state):
        valid_actions = [] # or we can call them valid_crops
        for crop_name, growth_stages in self.crop_db.items():
            for growth_stage, reqs in growth_stages.items():
                """reqs are of the form:
                {'soil': 
                    {'n': (0.0, 20.8, 40.0),
                    'p': (120.0, 134.2, 145.0),
                    'k': (195.0, 199.9, 205.0),
                    'ph': (5.5, 5.9, 6.5),
                    'organic_matter': (1.1, 5.1, 9.5),
                    'soil_moisture': (10.2, 21.2, 30.0)},
                'climate': 
                    {'temperature': (21.0, 22.6, 24.0),
                    'humidity': (90.0, 92.3, 94.9),
                    'rainfall': (100.1, 112.7, 125.0),
                    'sunlight_exposure': (5.1, 8.8, 12.0)},
                'environmental': 
                    {'irrigation_frequency': (1.0, 3.5, 6.0),
                    'water_usage_efficiency': (1.0, 2.8, 5.0),
                    'fertilizer_usage': (53.5, 131.4, 198.3),
                    'pest_pressure': (1.1, 54.3, 99.6)}
                }"""
                if self.is_valid_action(state, reqs):
                    valid_actions.append((crop_name, growth_stage))
        return valid_actions
    def is_valid_action(self, state, reqs):
        # Check at least one soil parameter is in range
        # if there is no parameter within the range return false
        if not any(
            reqs['soil'][param][0] <= state['soil'][param] <= reqs['soil'][param][2]
            for param in reqs['soil']
        ):
            return False
        # Check at least one climate parameter is in range
        # if there is no parameter within the range return false
        if not any(
            reqs['climate'][param][0] <= state['climate'][param] <= reqs['climate'][param][2]
            for param in reqs['climate']
        ):
            return False
        # Check at least one environmental parameter is in range
        # reaching this stage means that there is at least one parameter within the 
        # range in soil and climate parameters so it still just to check the 
        # environmantal parameters
        return any(
            reqs['environmental'][param][0] <= state['environmental'][param] <= reqs['environmental'][param][2]
            for param in reqs['environmental']
        )
    def apply_action(self, state, action):
        # action is now a tuple (crop_name, growth_stage)
        new_state = copy.deepcopy(state)
        new_state['current_crop'] = action[0]
        new_state['growth_stage'] = action[1]  # Store growth stage in state
        return new_state

    def expand_node(self, node):
        state = node.state
        valid_actions = self.get_valid_actions(state)
        child_nodes = []
        for action in valid_actions:
            child_state = self.apply_action(state, action)
            child_cost = node.cost + self.calculate_cost(node.state, action)
            child_node = CropNode(child_state, parent= node, action = action, cost = child_cost)
            child_nodes.append(child_node)
        return child_nodes
    def calculate_cost(self, state, action):
        """
        Calculate how well the current conditions match a crop's ideal requirements.
        Returns a score where 0 = perfect match, higher values = worse match.
        Uses (min, mean, max) ranges for precise scoring.
        """
            
        crop_name, growth_stage = action
        reqs = self.crop_db[crop_name][growth_stage]
        total_score = 0.0
        
        # Calculate soil suitability (weighted)
        soil_score = self._category_score(state['soil'], reqs['soil'])
        
        # Calculate climate suitability (weighted)
        climate_score = self._category_score(state['climate'], reqs['climate'])
        
        # Calculate environmental suitability (weighted)
        env_score = self._env_score(state['environmental'], reqs['environmental'])
        
        # Combine scores with weights
        total_score = (self.weights['soil'] * soil_score + self.weights['climate'] * climate_score + self.weights['environmental'] * env_score)
        return total_score
    def _category_score(self, state, parameters):
        """parameters is of the form: 
            if parameters = reqs['soil']:
            {
                'n': (0.0, 20.8, 40.0),
                'p': (120.0, 134.2, 145.0),
                'k': (195.0, 199.9, 205.0),
                'ph': (5.5, 5.9, 6.5),
                'organic_matter': (1.1, 5.1, 9.5),
                'soil_moisture': (10.2, 21.2, 30.0)
            }  
        """
        score = 0.0
        n = 0
        for param, (min_val, mean_val, max_val) in parameters.items():
            if param in state:
                val = state[param]  # the condition entered by the user
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
                    # Within range - quadratic penalty based on distance from mean
                    if val < mean_val:
                        norm_dist = (mean_val - val) / (mean_val - min_val)
                    else:
                        norm_dist = (val - mean_val) / (max_val - mean_val)
                    param_score = norm_dist ** 2  # Quadratic penalty
                
                score += param_score
                n += 1
        
        # Return Root Mean Square (RMS) score for the category
        return math.sqrt(score / n) if n > 0 else 0.0
    
    def _env_score(self, state, parameters) :
        """Special scoring for environmental parameters"""
        score = 0.0
        n = 0
        global_min = {}  # Track global min for normalization (if needed)
        # First pass: Find global min for each parameter (if normalizing)
        for param, (min_val, _, max_val) in parameters.items():
            global_min[param] = min_val

        for param, (min_val,_, max_val) in parameters.items():
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
                        param_score = ((val - min_val) / range_size ) ** 2  # Quadratic penalty
                else:        
                    if global_min[param] == 0:
                        # If global_min is 0 , just use min_val directly
                        param_score = min_val  # Lower min_val = better (no division needed)
                    else:
                        param_score = min_val / global_min[param]  # Normalized penalty

                score += param_score
                n += 1
        return math.sqrt(score / n) if n > 0 else 0.0
    
    def print_top_recommendations(self, top_n=5):
        # Get raw recommendations
        recommendations = [
            (action, self.calculate_cost(self.state, action)) 
            for action in self.get_valid_actions(self.state)
        ]
        # Find maximum score for normalization
        max_score = max(score for (_, _), score in recommendations) if recommendations else 1.0
        
        # Normalize scores to 0-1 range
        normalized_recommendations = [
            (action, score / max_score) 
            for action, score in recommendations
        ]
        
        # Sort by normalized score
        sorted_recommendations = sorted(normalized_recommendations, key=lambda x: x[1])[:top_n]

        print("\n=== TOP CROP RECOMMENDATIONS ===")
        for rank, ((crop, stage), norm_score) in enumerate(sorted_recommendations, 1):
            match_percent = 100 * (1 - norm_score)  # Now guaranteed 0-100%
            print(
                f"{rank}. {crop.capitalize()} (Stage {stage}) - "
                f"Match: {max(0, match_percent):.2f}% "  # Prevent negative
                f"(Score: {norm_score:.4f})"
            )

  

def load_crop_db(file_path):
    """Load crop database from crop_db.txt"""
    with open(file_path, 'r') as f:
        # Read the file content and evaluate it as a Python dictionary
        return eval(f.read())

def main():
    # 1. Load crop database
    crop_db = load_crop_db('./data/processed/crop_db.txt')
    
    # 2. Create sample initial state (modify with your actual values)
    initial_state = {
    'soil': {
        'n': 20.7,       
        'p': 134.5,       
        'k': 199.9,      
        'ph': 5.9,     
        'organic_matter': 5.5,  
        'soil_moisture': 21.4    
    },
    'climate': {
        'temperature': 22.6,  
        'humidity': 92.4,     
        'rainfall': 112.6,    
        'sunlight_exposure': 8.7  
    },
    'environmental': {
        'irrigation_frequency': 3.4,  
        'water_usage_efficiency': 2.8,
        'fertilizer_usage':57.5,
        'pest_pressure': 3.0
    },
    'current_crop': None,
    'growth_stage': None
}

    # 3. Initialize problem and print recommendations
    problem = cropProblem(initial_state, crop_db)
    problem.print_top_recommendations(top_n=5)

if __name__ == "__main__":
    main()

