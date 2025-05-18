import copy
import math
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tabulate import tabulate
import tracemalloc
import sys
import gc
from statistics import mean
import queue 
import ast
import json
import os
import random
import time
import timeit
import pandas as pd

class cropNode:
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
        """Create a hash based on the current crop and environmental conditions"""
        return hash((
            self.state['current_crop'],
            frozenset(self.state['soil'].items()),
            frozenset(self.state['climate'].items()),
            frozenset(self.state['environmental'].items())
        ))

    def __eq__(self, other):
        if isinstance(other, cropNode):
            return self.state == other.state 
        return False
    

class CropProblem:
    def __init__(self, initial_state, crop_db, priorities=None):
        """initial_state is of the form:
            {
                'soil': {'n':20.8, 'p':134.2, 'k':199.9, 'ph':5.9, 'organic_matter':5.1, 'soil_moisture':21.2},
                'climate': {'temperature':22.6, 'humidity':92.3, 'rainfall':112.7, 'sunlight_exposure':8.8},
                'environmental': {'irrigation_frequency':3.5, 'water_usage_efficiency':2.8},
                'current_crop': None
            }
        """
        """crop_db is of the form:
            {
                'apple': {
                    'soil': {'n': (0.0, 20.8, 40.0), ...},
                    'climate': {...},
                    'environmental': {...}
                },
                'banana': {
                    'soil': {...},
                    'climate': {...},
                    'environmental': {...}
                },
                ...
            }
        """
        self.state = initial_state 
        self.crop_db = crop_db
        
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

    def get_valid_actions(self, state):
        """Returns list of valid crop names that match current conditions"""
        valid_crops = []
        for crop_name, reqs in self.crop_db.items():
            if self.is_valid_action(state, reqs):
                valid_crops.append(crop_name)
        return valid_crops

    def is_valid_action(self, state, reqs):
        # Check at least one soil parameter is in range
        if not any(
            reqs['soil'][param][0] <= state['soil'][param] <= reqs['soil'][param][2]
            for param in reqs['soil']
        ):
            return False
        # Check at least one climate parameter is in range
        if not any(
            reqs['climate'][param][0] <= state['climate'][param] <= reqs['climate'][param][2]
            for param in reqs['climate']
        ):
            return False
        # Check at least one environmental parameter is in range
        return any(
            reqs['environmental'][param][0] <= state['environmental'][param] <= reqs['environmental'][param][2]
            for param in reqs['environmental']
        )

    def apply_action(self, state, action):
        """Action is just the crop name (string)"""
        new_state = copy.deepcopy(state)
        new_state['current_crop'] = action
        return new_state
    
    def expand_node(self, node):
        state = node.state
        valid_actions = self.get_valid_actions(state)
        child_nodes = []
        for action in valid_actions:
            child_state = self.apply_action(state, action)
            child_cost = node.cost + self.calculate_cost(node.state, action)
            child_node = cropNode(child_state, parent= node, action = action, cost = child_cost)
            child_nodes.append(child_node)
        return child_nodes

    def calculate_cost(self, state, action):
        """
        Calculate how well the current conditions match a crop's ideal requirements.
        Returns a score where 0 = perfect match, higher values = worse match.
        """
        reqs = self.crop_db[action]
        total_score = 0.0
        
        # Calculate soil suitability (weighted)
        soil_score = self._category_score(state['soil'], reqs['soil'])
        
        # Calculate climate suitability (weighted)
        climate_score = self._category_score(state['climate'], reqs['climate'])
        
        # Calculate environmental suitability (weighted)
        env_score = self._env_score(state['environmental'], reqs['environmental'])
        
        # Combine scores with weights
        total_score = (
            self.weights['soil'] * soil_score + 
            self.weights['climate'] * climate_score + 
            self.weights['environmental'] * env_score
        )
        return total_score

    def _category_score(self, state_values, requirements):
        """Calculate score for one category (soil, climate, etc.)"""
        score = 0.0
        n = 0
        
        for param, (min_val, mean_val, max_val) in requirements.items():
            if param in state_values:
                val = state_values[param]
                range_size = max_val - min_val
                if range_size <= 0:  # Prevent division by zero
                    continue
                
                if val < min_val:
                    param_score = (min_val - val) / range_size
                elif val > max_val:
                    param_score = (val - max_val) / range_size
                else:
                    if val < mean_val:
                        norm_dist = (mean_val - val) / (mean_val - min_val)
                    else:
                        norm_dist = (val - mean_val) / (max_val - mean_val)
                    param_score = norm_dist ** 2  # Quadratic penalty
                
                score += param_score
                n += 1
        
        return math.sqrt(score / n) if n > 0 else 0.0
    
    def _env_score(self, state, parameters):
        """Special scoring for environmental parameters"""
        score = 0.0
        n = 0
        global_min = {}  # Track global min for normalization (if needed)
        # First pass: Find global min for each parameter (if normalizing)
        for param, (min_val, _, max_val) in parameters.items():
            global_min[param] = min_val

        for param, (min_val, _, max_val) in parameters.items():
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

    def print_top_recommendations(self, top_n=5):
        """Print top crop recommendations based on current state"""
        recommendations = [
            (crop, self.calculate_cost(self.state, crop)) 
            for crop in self.get_valid_actions(self.state)
        ]
        
        if not recommendations:
            print("No suitable crops found for current conditions.")
            return
            
        # Normalize scores to 0-1 range
        max_score = max(score for _, score in recommendations)
        normalized_recommendations = [
            (crop, score / max_score) 
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

# Genetic Search Algorithm Implementation
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
                offspring1, offspring2 = parent1, parent2
            
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
# General Search Algorithm Implementation

BASE_DIR = os.getcwd()
crop_db_path = os.path.join(BASE_DIR, '..', 'data', 'processed', 'crop_db.txt')


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

        problem = CropProblem(self.current_state, self.crop_db)
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
        
        return total_cost* 0.01 # we multiplied by 0.01 to scale values with costs  
    
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
            priority = self.priorities.get(category, 0.33)  # default if missing

            target_value = min_val if use_min else mean_val
            range_span = max_val - min_val + 1e-6  # Prevent division by zero

            # Mean Squared Error + soft z-distance
            mse = ((actual_value - target_value) / range_span) ** 2
            z_score = abs(actual_value - mean_val) / (range_span / 4)  # rough std dev

            cost = priority * (mse + z_score)

            # Penalty: out-of-range -> apply power penalty based on distance from boundary
            if actual_value < min_val:
                distance = min_val - actual_value
            elif actual_value > max_val:
                distance = actual_value - max_val
            else:
                distance = 0

            if distance > 0:
                margin = 0.1 * range_span  # Tolerance buffer
                if distance > margin:
                    penalty = priority * ((distance - margin) / range_span)
                    cost += penalty

            # Reward: inside range and close to mean
            if min_val <= actual_value <= max_val:
                normalized_dist = abs(actual_value - mean_val) / (range_span / 2)
                reward = priority * (1 - normalized_dist)
                cost -= reward

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
        # print(f"Heuristic values saved to: {output_path}")


  
class OrderedNode:
    def __init__(self, node_name , heuristic_value):

        self.heuristic_value = heuristic_value
        self.node_name = node_name

    def __lt__(self, other):  # used by PriorityQueue for comparison
        return self.heuristic_value < other.heuristic_value
    
    def __str__(self):
        return f"( {self.node_name} , {self.heuristic_value} ) "

class GeneralHeuristicBasedSearch:
    
    def __init__(self, problem, heuristics, mode):
        self.problem = problem
        self.heuristicValues = heuristics
        self.mode = mode
        self.initialState = cropNode(self.problem.state)

    def set_frontier(self, node_list, heuristics):
        frontier = queue.PriorityQueue()
        for node in node_list:
            crop_name = node.state['current_crop']
            h_value = heuristics.get(crop_name, float('inf'))

            if self.mode == "a_star":
                g_value = node.cost if hasattr(node, 'cost') else 0  
                f_value = g_value + h_value
            elif self.mode == "greedy":
                f_value = h_value   

            frontier.put(OrderedNode(crop_name, f_value))
        return frontier

    def search(self, cropresults=5):
        frontier = self.set_frontier(self.problem.expand_node(self.initialState), self.heuristicValues)
        result = []

        if not frontier.empty():
            for _ in range(cropresults):
                best_node = frontier.get()
                result.append(best_node)

            return result    
        
        return None, None
    
    def get_top_recommendations(self, top_n=5):
        """get top crop recommendations based on selected search mode (Greedy or A*)"""
        recommendations = []
        child_nodes = self.problem.expand_node(self.initialState)

        for node in child_nodes:
            
            if self.mode == "a_star":
                g_value = node.cost  
                h_value = self.heuristicValues.get(node.action, float('inf'))  
                score = g_value + h_value
            elif self.mode == "greedy":  
                score = self.heuristicValues.get(node.action, float('inf'))

            recommendations.append((node.action, score))

        if not recommendations:
            print("No suitable crops found for current conditions.")
            return

        scores = [score for _, score in recommendations]
        min_score, max_score = min(scores), max(scores)

        if max_score == min_score:
            normalized = [(crop, 0.0) for crop, _ in recommendations]
        else:
            normalized = [(crop, score)for crop, score in recommendations]

        sorted_recommendations = sorted(normalized, key=lambda x: x[1])[:top_n]

        return sorted_recommendations
    def classify_suitability(self, cost: float) -> str:
        if cost < 0.2:
            return "Excellent"
        elif cost < 0.4:
            return "Good"
        elif cost < 0.6:
            return "Fair"
        else:
            return "Poor"
    
    def print_top_recommendations_using_heuristics(self, top_n=5):
        """Print top crop recommendations based on selected search mode (Greedy or A*)"""
        recommendations = []

        
        child_nodes = self.problem.expand_node(self.initialState)

        for node in child_nodes:
            
            if self.mode == "a_star":
                g_value = node.cost  
                h_value = self.heuristicValues.get(node.action, float('inf'))  
                score = g_value + h_value
            elif self.mode == "greedy":  
                score = self.heuristicValues.get(node.action, float('inf'))

            recommendations.append((node.action, score))

        if not recommendations:
            print("No suitable crops found for current conditions.")
            return

        scores = [score for _, score in recommendations]
        min_score, max_score = min(scores), max(scores)

        if max_score == min_score:
            normalized = [(crop, 0.0) for crop, _ in recommendations]
        else:
            normalized = [(crop, (score) / (max_score)) for crop, score in recommendations]

        sorted_recommendations = sorted(normalized, key=lambda x: x[1])[:top_n]



        print(f"\n=== TOP CROP RECOMMENDATIONS ({self.mode.upper()}) ===")
        for rank, (crop, norm_score) in enumerate(sorted_recommendations, 1):
            match_percent = 100 * (1 - norm_score)
            print(
                f"{rank}. {crop.capitalize()} - "
                f"Match: {max(0, match_percent):.2f}% "
                f"(Score: {norm_score:.4f})"
                f"(suitability: {self.classify_suitability(node.cost)})"
                
            )

 