import copy
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
        return hash((tuple(self.state['soil'].values()), 
                   tuple(self.state['climate'].values()),
                   self.state['current_crop']))

    def __eq__(self, other):
        return (isinstance(other, CropNode) and 
                self.state['soil'] == other.state['soil'] and
                self.state['climate'] == other.state['climate'] and
                self.state['current_crop'] == other.state['current_crop'])

class CropProblem:
    def __init__(self, initial_state: Dict, goal_crop: str = None, 
                 crop_db: Dict = None, constraints: Dict = None):
        """
        initial_state: {
            'soil': {'N': 50, 'P': 40, 'K': 60, 'pH': 6.2, 'moisture': 0.5},
            'climate': {'temp': 25, 'humidity': 70, 'rainfall': 800},
            'current_crop': None
        }
        crop_db: {
            'wheat': {
                'soil': {'N': (40,60), 'pH': (6.0,7.0), ...},
                'climate': {'temp': (15,25), ...},
                'water_needs': 500
            },
            ...
        }
        constraints: {
            'max_water': 1000,
            'max_fertilizer': 200
        }
        """
        self.state = initial_state
        self.crop_db = crop_db or self.default_crop_db()
        self.constraints = constraints or {}
        self.goal_crop = goal_crop  # Optional: Specific target crop

    def is_goal(self, node: CropNode) -> bool:
        """Goal is either:
        1. Any valid crop (if no goal_crop specified) """
        return self.validate_crop(node.state)

    def validate_crop(self, state: Dict) -> bool:
        """Check if current crop meets all constraints"""
        crop = state['current_crop']
        if not crop:
            return False
            
        crop_reqs = self.crop_db[crop]
        
        # Check soil constraints
        for param, (min_val, max_val) in crop_reqs['soil'].items():
            if not (min_val <= state['soil'][param] <= max_val):
                return False
                
        # Check climate constraints
        for param, (min_val, max_val) in crop_reqs['climate'].items():
            if not (min_val <= state['climate'][param] <= max_val):
                return False
                
        # Check resource constraints
        if 'water_needs' in crop_reqs:
            if crop_reqs['water_needs'] > self.constraints.get('max_water', float('inf')):
                return False
                
        return True

    def get_valid_actions(self, state: Dict) -> List[str]:
        """Returns list of crop names that satisfy basic constraints"""
        valid_crops = []
        for crop_name, reqs in self.crop_db.items():
            # Quick pre-check before full validation
            if self.quick_check(state, reqs):
                valid_crops.append(crop_name)
        return valid_crops

    def quick_check(self, state: Dict, reqs: Dict) -> bool:
        """Fast preliminary constraint checking"""
        # Check at least one soil parameter is in range
        soil_ok = any(
            reqs['soil'][param][0] <= state['soil'][param] <= reqs['soil'][param][1]
            for param in reqs['soil']
        )
        
        # Check at least one climate parameter is in range
        climate_ok = any(
            reqs['climate'][param][0] <= state['climate'][param] <= reqs['climate'][param][1]
            for param in reqs['climate']
        )
        
        return soil_ok and climate_ok

    def apply_action(self, state: Dict, action: str) -> Dict:
        """Assign a crop to the state"""
        new_state = copy.deepcopy(state)
        new_state['current_crop'] = action
        return new_state

    def calculate_cost(self, state: Dict, crop: str) -> float:
        """Calculate suitability cost (lower = better)"""
        cost = 0
        crop_reqs = self.crop_db[crop]
        
        # Soil parameter deviations
        for param, (ideal_min, ideal_max) in crop_reqs['soil'].items():
            val = state['soil'][param]
            ideal = (ideal_min + ideal_max) / 2
            cost += abs(val - ideal) / (ideal_max - ideal_min)
            
        # Climate parameter deviations
        for param, (ideal_min, ideal_max) in crop_reqs['climate'].items():
            val = state['climate'][param]
            ideal = (ideal_min + ideal_max) / 2
            cost += abs(val - ideal) / (ideal_max - ideal_min)
            
        # Resource costs
        if 'water_needs' in crop_reqs:
            cost += crop_reqs['water_needs'] / self.constraints.get('max_water', 1000)
            
        return cost

    def expand_node(self, node: CropNode) -> List[CropNode]:
        """Generate all valid child nodes"""
        valid_crops = self.get_valid_actions(node.state)
        children = []
        
        for crop in valid_crops:
            child_state = self.apply_action(node.state, crop)
            cost = node.cost + self.calculate_cost(node.state, crop)
            children.append(CropNode(child_state, parent=node, action=crop, cost=cost))
            
        return children

    @staticmethod
    def default_crop_db() -> Dict:
        """Example crop database"""
        return {
            'wheat': {
                'soil': {
                    'N': (40, 60), 'P': (30, 50), 'K': (40, 80),
                    'pH': (6.0, 7.0), 'moisture': (0.4, 0.7)
                },
                'climate': {
                    'temp': (15, 25), 'humidity': (40, 80),
                    'rainfall': (500, 1000)
                },
                'water_needs': 600
            },
            'rice': {
                'soil': {
                    'N': (50, 70), 'P': (20, 40), 'K': (50, 90),
                    'pH': (5.0, 6.5), 'moisture': (0.6, 1.0)
                },
                'climate': {
                    'temp': (20, 35), 'humidity': (60, 100),
                    'rainfall': (1000, 2000)
                },
                'water_needs': 1200
            }
        }
    
def main():
    # Example initial state
    initial_state = {
        'soil': {
            'N': 50,  # Nitrogen level
            'P': 40,  # Phosphorus level
            'K': 60,  # Potassium level
            'pH': 6.2,  # Soil pH
            'moisture': 0.5  # Soil moisture
        },
        'climate': {
            'temp': 25,  # Temperature in °C
            'humidity': 70,  # Relative humidity in %
            'rainfall': 800  # Annual rainfall in mm
        },
        'current_crop': None  # No crop currently planted
    }

    # Example constraints
    constraints = {
        'max_water': 1000,  # Maximum available water (mm)
        'max_fertilizer': 200  # Maximum fertilizer usage (kg/ha)
    }

    # Create the crop problem instance
    problem = CropProblem(initial_state, constraints=constraints)

    # Create the root node
    root_node = CropNode(initial_state)

    # Expand the root node to get possible crops
    possible_crops = problem.expand_node(root_node)

    print("Possible crops with their suitability costs (lower is better):")
    for node in sorted(possible_crops, key=lambda x: x.cost):
        print(f"- {node.action}: {node.cost:.2f}")

    # Find the best crop (lowest cost)
    if possible_crops:
        best_crop = min(possible_crops, key=lambda x: x.cost)
        print(f"\nBest crop choice: {best_crop.action} with suitability cost {best_crop.cost:.2f}")
        
        # Validate the choice
        if problem.validate_crop(best_crop.state):
            print("This crop meets all requirements and constraints.")
        else:
            print("Warning: This crop doesn't meet all requirements!")
    else:
        print("No suitable crops found for the given conditions.")

if __name__ == "__main__":
    main()