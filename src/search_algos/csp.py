import copy
import math
from typing import Dict, List, Tuple, Optional

class CropCSP:
    def __init__(self, crop_db: Dict, current_conditions: Dict, 
                 weights: Optional[Dict] = None, tolerance: float = 0.05):
        """
        Complete CSP implementation that shows all options ranked by suitability
        
        Args:
            crop_db: Dictionary of crop requirements
            current_conditions: Current environmental measurements
            weights: Importance weights for categories {'soil': 0.4, ...}
            tolerance: Percentage tolerance for constraints (default: ±5%)
        """
        self.crop_db = crop_db
        self.state = current_conditions
        self.weights = weights or {'soil': 0.4, 'climate': 0.3, 'environmental': 0.3}
        self.tolerance = tolerance
        self.parameter_tolerances = {}

    def set_tolerance(self, tolerance: float):
        """Set global tolerance level"""
        self.tolerance = tolerance

    def set_parameter_tolerance(self, parameter: str, tolerance: float):
        """Set specific tolerance for individual parameters"""
        self.parameter_tolerances[parameter] = tolerance

    def _get_effective_tolerance(self, parameter: str) -> float:
        """Get tolerance for a specific parameter"""
        return self.parameter_tolerances.get(parameter, self.tolerance)

    def _is_in_range_with_tolerance(self, actual: float, min_val: float, 
                                  max_val: float, parameter: str = None) -> bool:
        """Check if value is within tolerated range"""
        if min_val == max_val == 0:
            return actual == 0

        tolerance = self._get_effective_tolerance(parameter)
        
        if parameter == 'water_usage_efficiency':
            return actual >= min_val * (1 - tolerance)
        if parameter == 'pest_pressure':
            return actual <= max_val * (1 + tolerance)
        
        effective_min = min_val * (1 - tolerance)
        effective_max = max_val * (1 + tolerance)
        return effective_min <= actual <= effective_max

    def _check_constraints(self, crop: str) -> Tuple[bool, Dict[str, bool]]:
        """Check constraints and return detailed results"""
        reqs = self.crop_db[crop]
        passes_all = True
        details = {}
        
        for category in ['soil', 'climate', 'environmental']:
            details[category] = {}
            for param in reqs[category]:
                min_val, _, max_val = reqs[category][param]
                actual = self.state[category][param]
                passes = self._is_in_range_with_tolerance(
                    actual, min_val, max_val, f"{category}.{param}"
                )
                details[category][param] = passes
                if not passes:
                    passes_all = False
        
        return passes_all, details

    def _calculate_match_score(self, crop: str) -> float:
        """Calculate weighted suitability score (0=perfect, higher=worse)"""
        reqs = self.crop_db[crop]
        return (
            self.weights['soil'] * self._category_score('soil', reqs) +
            self.weights['climate'] * self._category_score('climate', reqs) +
            self.weights['environmental'] * self._category_score('environmental', reqs)
        )

    def _category_score(self, category: str, reqs: Dict) -> float:
        """Calculate normalized mismatch score for a category"""
        total = 0.0
        n = 0
        
        for param, (min_val, mean_val, max_val) in reqs[category].items():
            actual = self.state[category][param]
            range_size = max_val - min_val
            
            if actual < min_val:
                param_score = (min_val - actual) / range_size
            elif actual > max_val:
                param_score = (actual - max_val) / range_size
            else:
                if actual < mean_val:
                    norm_dist = (mean_val - actual) / (mean_val - min_val)
                else:
                    norm_dist = (actual - mean_val) / (max_val - mean_val)
                param_score = norm_dist ** 2
            
            total += param_score
            n += 1
        
        return math.sqrt(total / n) if n > 0 else 0.0

    def get_all_options(self, top_n: int = 10) -> List[Tuple[str, float, bool, Dict]]:
        """
        Returns all crops ranked by suitability, with constraint info
        
        Returns: List of (crop_name, match_score, passes_constraints, constraint_details)
        """
        results = []
        
        for crop in self.crop_db:
            score = self._calculate_match_score(crop)
            passes, details = self._check_constraints(crop)
            results.append((crop, score, passes, details))
        
        # Sort by match score (lower is better)
        return sorted(results, key=lambda x: x[1])[:top_n]

def print_recommendations(recommendations: List[Tuple[str, float, bool, Dict]]):
    """Print formatted recommendations with constraint status"""
    print("\n=== TOP RECOMMENDATIONS (ALL OPTIONS) ===")
    print("Rank. Crop       Match%  Constraints  Problem Parameters")
    print("------------------------------------------------------")
    
    for rank, (crop, score, passes, details) in enumerate(recommendations, 1):
        match_percent = 100 * (1 - score)
        status = "PASS" if passes else "FAIL"
        
        # Find which parameters failed
        problems = []
        for category in details:
            for param, passed in details[category].items():
                if not passed:
                    problems.append(f"{param}")
        
        problem_str = ", ".join(problems) if problems else "None"
        print(f"{rank:>2}. {crop.capitalize():<10} {match_percent:5.1f}%  {status:<6}  {problem_str}")

def load_crop_db(file_path: str) -> Dict:
    """Load crop database from file"""
    with open(file_path, 'r') as f:
        return eval(f.read())

def main():
    # Load data
    crop_db = load_crop_db('./data/processed/crop_db.txt')
    
    # Example conditions
    current_conditions = {
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
        }
    }
    
    # Initialize solver
    solver = CropCSP(crop_db, current_conditions)
    
    # Set custom tolerances if needed
    solver.set_parameter_tolerance('soil.ph', 0.02)  # Stricter pH tolerance
    solver.set_tolerance(0.2)
    
    # Get and display all options
    all_options = solver.get_all_options(top_n=40)
    print_recommendations(all_options)
    
    # Optional: Show only passing options
    passing_options = [x for x in all_options if x[2]]
    if passing_options:
        print("\n=== PASSING OPTIONS ONLY ===")
        print_recommendations(passing_options)
    else:
        print("\nNo options pass all constraints at current tolerance levels")

if __name__ == "__main__":
    main()











    