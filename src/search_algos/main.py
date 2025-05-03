from newProblemFormulation import cropProblem, load_crop_db
from astar import a_star_search, print_a_star_recommendations


def main():
    # 1. Load crop database
    crop_db = load_crop_db('./data/processed/crop_db.txt')
    
    # 2. Create sample initial state (modify with your actual values)
    initial_state = {
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

    # 3. Initialize problem
    problem = cropProblem(initial_state, crop_db)

    # 4. Run A* search and print results
    recommendations = a_star_search(problem, 10)
    print_a_star_recommendations(recommendations, top_n=10)

    # Also print the original method's recommendations for comparison
    print("\nFor comparison, here are the original method's recommendations:")
    problem.print_top_recommendations(top_n=10)


if __name__ == "__main__":
    main()