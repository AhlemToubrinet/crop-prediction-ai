import heapq
from typing import List, Tuple
from newProblemFormulation import cropNode


def a_star_search(problem, top_n=10):
    """A* search algorithm implementation for crop recommendation problem with heuristic."""
    open_list = []
    closed_list = set()
    
    initial_node = cropNode(problem.state)
    # For initial node, we don't have an action yet, so heuristic is 0
    initial_h = 0
    heapq.heappush(open_list, (initial_h, initial_node))
    
    recommendations = []
    
    while open_list:
        current_f, current_node = heapq.heappop(open_list)
        
        if current_node in closed_list:
            continue
            
        closed_list.add(current_node)
        
        if current_node.state['current_crop'] is not None:
            recommendations.append((current_node.state['current_crop'], current_node.cost))
            continue
            
        valid_actions = problem.get_valid_actions(current_node.state)
        if not valid_actions and not recommendations:
            valid_actions = problem.crop_db.keys()
            
        for action in valid_actions:
            new_state = problem.apply_action(current_node.state, action)
            action_cost = problem.calculate_cost(new_state, action)
            new_cost = current_node.cost + action_cost
            
            # Calculate heuristic for the new state
            if hasattr(problem, 'heuristic'):
                h_score = problem.heuristic(new_state, action)
            else:
                h_score = 0
                
            f_score = new_cost + h_score
            new_node = cropNode(new_state, current_node, action, new_cost)
            
            if new_node not in closed_list:
                heapq.heappush(open_list, (f_score, new_node))
    
    return sorted(recommendations, key=lambda x: x[1])[:top_n]




def print_a_star_recommendations(recommendations, top_n=10):
    """Print recommendations from A* search."""
    if not recommendations:
        print("No suitable crops found for current conditions.")
        return
    
    # Normalize scores to 0-1 range
    
    normalized_recommendations = [
        (crop, score ) 
        for crop, score in recommendations[:top_n]
    ]
    
    print("\n=== TOP CROP RECOMMENDATIONS (A* SEARCH) ===")
    for rank, (crop, norm_score) in enumerate(normalized_recommendations, 1):
        match_percent = 100 * (1 - norm_score)
        print(
            f"{rank}. {crop.capitalize()} - "
            f"Match: {max(0, match_percent):.2f}% "
            f"(Score: {norm_score:.4f})"
        )

