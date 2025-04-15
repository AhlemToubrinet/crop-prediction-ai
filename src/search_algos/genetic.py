import random
from copy import deepcopy
from typing import List, Dict, Tuple
from newProblemFormulation import cropNode, cropProblem

class CropGeneticAlgorithm:
    def __init__(self, problem, population_size=30,  # Increased population size
                 crossover_rate=0.85, mutation_rate=0.25,  # Higher mutation rate
                 elitism_ratio=0.1, max_generations=40):  # More generations
        self.problem = problem
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_ratio = elitism_ratio
        self.max_generations = max_generations
        
    def initialize_population(self) -> List[cropNode]:
        population = []
        valid_crops = self.problem.get_valid_actions(self.problem.state)
        """state is of the form:
            {
                'soil': {'n':20.8, 'p':134.2, 'k':199.9, 'ph':5.9, 'organic_matter':5.1, 'soil_moisture':21.2},
                'climate': {'temperature':22.6, 'humidity':92.3, 'rainfall':112.7, 'sunlight_exposure':8.8},
                'environmental': {'irrigation_frequency':3.5, 'water_usage_efficiency':2.8},
                'current_crop': None
            }
        """
        if not valid_crops:
            raise ValueError("No valid crops available for current conditions")
        for _ in range(self.population_size):
            if random.random() < 0.7:
                crop = random.choice(valid_crops)
            else:
                crop = self.greedy_crop_selection(valid_crops, top_n=3)
                
            state = self.problem.apply_action(self.problem.state, crop)
            cost = self.problem.calculate_cost(state, crop)
            population.append(cropNode(state, action=crop, cost=cost))

        """population is of the form:
            population = [cropNode, cropNode, cropNode, ...]
            and cropNode is of the form:
            cropNode = (state = state, action = apple, cost = 0.15)
            state = {'soil': {'n':20.8, 'p':134.2, 'k':199.9, 'ph':5.9, 'organic_matter':5.1, 'soil_moisture':21.2},
                'climate': {'temperature':22.6, 'humidity':92.3, 'rainfall':112.7, 'sunlight_exposure':8.8},
                'environmental': {'irrigation_frequency':3.5, 'water_usage_efficiency':2.8},
                'current_crop': None }
        """    
        return population
    
    def greedy_crop_selection(self, valid_crops, top_n=3) -> str:
        costs = [(crop, self.problem.calculate_cost(
            self.problem.apply_action(self.problem.state, crop), crop)) 
            for crop in valid_crops]
        costs.sort(key=lambda x: x[1])
        """it will return a randome crop among the best three crops let's assume that it'll 
        return apple"""
        return random.choice(costs[:top_n])[0]
    
    def evaluate_fitness(self, node: cropNode) -> float:
        """Added safeguard against division by zero"""
        return 1 / (1 + node.cost) if node.cost != -1 else 0
    
    def selection(self, population: List[cropNode]) -> Tuple[cropNode, cropNode]:
        tournament = random.sample(population, 3)
        # here we will select random 3 cropNodes from the population returned by
        # the function initialize_population
        tournament.sort(key=lambda x: self.evaluate_fitness(x), reverse=True)
        # here we will sort those 3 cropNodes based on their fitness function from higher to 
        # lower and return the best 2 cropNodes
        return tournament[0], tournament[1]
    
    def crossover(self, parent1: cropNode, parent2: cropNode) -> Tuple[cropNode, cropNode]:
        # parent1 and parent2 are the cropNodes returned by the selection function
        crop1 = parent1.action # this will take the current crop assigned to the conditions entered by the user
        crop2 = parent2.action
        reqs1 = self.problem.crop_db[crop1] # reqs1 are the requirement conditions for that crop which are taken from the database
        reqs2 = self.problem.crop_db[crop2]
        
        blended_reqs = {'soil': {}, 'climate': {}, 'environmental': {}}
        
        for param_type in ['soil', 'climate', 'environmental']:
            for param in reqs1[param_type]:
                min_val = (reqs1[param_type][param][0] + reqs2[param_type][param][0]) / 2
                mean_val = (reqs1[param_type][param][1] + reqs2[param_type][param][1]) / 2
                max_val = (reqs1[param_type][param][2] + reqs2[param_type][param][2]) / 2
                blended_reqs[param_type][param] = (min_val, mean_val, max_val)

        """this will blend the required conditions of the two parents by taking the average
        of the min, mean and the max and assaing them to the blended_reqs which is of the form:
        blended_reqs = {
            'soil' = {'n' = (min, mean, max), ..}
            'climate' = {..}
            'environmental' = {..}
        }
        """
        candidate_crops = []
        for crop, reqs in self.problem.crop_db.items():
            similarity = self.calculate_similarity(reqs, blended_reqs)
            # this will calculate the similarity between the requirements of each crop in our
            # database 'crop_db' and the blended_reqs that we've calculated and create a tuple
            # that has the crop and it's similarity with the blended_reqs and append it to 
            # condidate_crops which is a list of tuples 
            candidate_crops.append((crop, similarity))
            """condidate_crops is of the form: 
                [('apple', 12), ('banana', 10),.. ]
            """
        
        candidate_crops.sort(key=lambda x: x[1], reverse=True)
        # we sort the condidate crops from the crop that has the highest 
        # similarity to the crop that has the lowest one
        top_crops = [crop for crop, _ in candidate_crops[:5]]
        # Select top 5 most similar crops
        
        # if we managed to get more than 2 top crops than we will assign the offsprings
        if len(top_crops) >= 2:
            # we will choose randomly 2 from the top_crops and assign the first one 
            # to offsprint1_crop and the second one to offspring2_crop
            offspring1_crop, offspring2_crop = random.sample(top_crops, 2)
        else:
            # if we have not enough crops than we will select the parents as offsprings
            offspring1_crop, offspring2_crop = crop1, crop2
            
        state1 = self.problem.apply_action(self.problem.state, offspring1_crop)
        state2 = self.problem.apply_action(self.problem.state, offspring2_crop)
        cost1 = self.problem.calculate_cost(state1, offspring1_crop)
        cost2 = self.problem.calculate_cost(state2, offspring2_crop)
        
        return (
            cropNode(state1, action=offspring1_crop, cost=cost1),
            cropNode(state2, action=offspring2_crop, cost=cost2)
        )
    
    def calculate_similarity(self, reqs1, reqs2) -> float:
        similarity = 0.0
        for param_type in ['soil', 'climate', 'environmental']:
            for param in reqs1[param_type]:
                if param in reqs2[param_type]:
                    mean1 = reqs1[param_type][param][1]
                    mean2 = reqs2[param_type][param][1]
                    similarity += 1 / (1 + abs(mean1 - mean2))
        return similarity
    
    def mutate(self, node: cropNode) -> cropNode:
        current_crop = node.action
        valid_crops = self.problem.get_valid_actions(self.problem.state)
        
        if len(valid_crops) <= 1:
            return node
            
        current_reqs = self.problem.crop_db[current_crop]
        similarities = []
        
        for crop in valid_crops:
            if crop != current_crop:
                reqs = self.problem.crop_db[crop]
                similarity = self.calculate_similarity(current_reqs, reqs)
                similarities.append((crop, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        if similarities:
            if random.random() < (1 - self.mutation_rate) and len(similarities) >= 3:
                new_crop = random.choice(similarities[:3])[0]
            else:
                new_crop = random.choice(similarities)[0]
                
            new_state = self.problem.apply_action(self.problem.state, new_crop)
            new_cost = self.problem.calculate_cost(new_state, new_crop)
            return cropNode(new_state, action=new_crop, cost=new_cost)
        
        return node
    
    def evolve_population(self, population: List[cropNode]) -> List[cropNode]:
        """Combined and corrected version"""
        new_population = []
        
        # Classify suitability first
        for individual in population:
            individual.suitability = self.classify_suitability(individual.cost)
        
        population.sort(key=lambda x: self.evaluate_fitness(x), reverse=True)
        elitism_count = int(self.elitism_ratio * self.population_size)
        new_population.extend(population[:elitism_count])
        
        while len(new_population) < self.population_size:
            parent1, parent2 = self.selection(population)
            
            if random.random() < self.crossover_rate:
                offspring1, offspring2 = self.crossover(parent1, parent2)
            else:
                offspring1, offspring2 = deepcopy(parent1), deepcopy(parent2)
            
            if random.random() < self.mutation_rate:
                offspring1 = self.mutate(offspring1)
            if random.random() < self.mutation_rate:
                offspring2 = self.mutate(offspring2)
            
            # Classify offspring
            offspring1.suitability = self.classify_suitability(offspring1.cost)
            offspring2.suitability = self.classify_suitability(offspring2.cost)
            
            new_population.extend([offspring1, offspring2])
            
            if len(new_population) > self.population_size:
                break
                
        return new_population[:self.population_size]
    
    
    def classify_suitability(self, cost: float) -> str:
        if cost < 0.2:
            return "Excellent"
        elif cost < 0.4:
            return "Good"
        elif cost < 0.6:
            return "Fair"
        else:
            return "Poor"

    def run(self):
        """
        Pure genetic search implementation that:
        1. Runs full evolution through all generations
        2. Returns the single best crop found
        3. No early stopping or complex filtering
        """
        population = self.initialize_population()
        
        # Run complete evolution
        for _ in range(self.max_generations):
            population = self.evolve_population(population)
        
        # Find the absolute best crop across all generations
        best_crop = min(population, key=lambda x: x.cost)
        
        return {
            'crop': best_crop.action,
            'cost': best_crop.cost,
            'suitability': self.classify_suitability(best_crop.cost),
            'match_percentage': (1 - best_crop.cost) * 100
        }
    
    def get_top_n_crops(self, n: int = 5) -> List[Dict]:
        """
        Returns top N unique crops from the genetic search
        Preserves your original implementation while adding ranking
        """
        population = self.initialize_population()
        
        # Run full evolution
        for _ in range(self.max_generations):
            population = self.evolve_population(population)
        
        # Get unique crops with their best versions
        unique_crops = {}
        for node in population:
            if node.action not in unique_crops or node.cost < unique_crops[node.action].cost:
                unique_crops[node.action] = node
        
        # Sort by cost and take top N
        sorted_crops = sorted(unique_crops.values(), key=lambda x: x.cost)
        top_n = sorted_crops[:n]
        
        # Format results
        return [{
            'crop': node.action,
            'cost': node.cost,
            'suitability': self.classify_suitability(node.cost),
            'match_percentage': (1 - node.cost) * 100
        } for node in top_n]


def load_crop_db(file_path):
    with open(file_path, 'r') as f:
        return eval(f.read())
    
def main():
    try:
        crop_db = load_crop_db('./data/processed/crop_db.txt')
        
        initial_state = {
            'soil': {
                'n': 110, 'p': 29, 'k': 30,
                'ph': 7, 'organic_matter': 5, 'soil_moisture': 18
            },
            'climate': {
                'temperature': 26, 'humidity': 54,
                'rainfall': 150, 'sunlight_exposure': 7
            },
            'environmental': {
                'irrigation_frequency': 2,
                'water_usage_efficiency': 2,
                'fertilizer_usage': 54,
                'pest_pressure': 1
            },
            'current_crop': None
        }

        problem = cropProblem(initial_state, crop_db)
        
        print("\n=== Genetic Algorithm Optimization ===")
        ga = CropGeneticAlgorithm(problem)
        
        # Get top N recommendations
        top_n = 5  # Can be changed to any number
        results = ga.get_top_n_crops(top_n)
        
        print(f"\n=== Top {top_n} Genetic Recommendations ===")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['crop'].capitalize()}:")
            print(f"   - Suitability: {result['suitability']}")
            print(f"   - Cost Score: {result['cost']:.4f}")
            print(f"   - Match Percentage: {result['match_percentage']:.2f}%")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()