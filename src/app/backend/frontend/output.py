
from pathlib import Path
from tkinter import Tk, Canvas, Button, PhotoImage, StringVar, Radiobutton
import subprocess
import sys
import json
import os



# Add project root to Python path (assuming output.py is in src/app/frontend/)
project_root = Path(__file__).parents[2]  # Adjust if needed
sys.path.insert(0, str(project_root))

# Now import using full package path
from app.backend.algorithms import CropProblem,HeuristicCalculator,GeneralHeuristicBasedSearch,HeuristicCalculator, CropGeneticAlgorithm,CropCSP, cropNode 

# Path setup for assets
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"./assets/frame0")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

# Main window setup
window = Tk()
window.geometry("900x590")
window.configure(bg="#FFFFFF")

canvas = Canvas(
    window,
    bg="#FFFFFF",
    height=590,
    width=900,
    bd=0,
    highlightthickness=0,
    relief="ridge",
)
canvas.place(x=0, y=0)

# GUI Elements ----------------------------------------------------------------

# Header text
canvas.create_text(
    16.0, 20.0,
    anchor="nw",
    text="📊 Your Crop Recommendations",
    fill="#5BC893",
    font=("Inter SemiBoldItalic", 26 * -1),
)
canvas.create_text(
    40.0, 73.0,
    anchor="nw",
    text="The best crop for your land is:",
    fill="#848484",
    font=("Inter SemiBoldItalic", 18 * -1),
)

best_crop_text = canvas.create_text(
    300.0, 73.0,  # Positioned right after "The best crop for your land is:"
    anchor="nw",
    text="",  # Will be filled dynamically
    fill="#3AB67D",  # Green color to match your theme
    font=("Inter SemiBoldItalic", 18 * -1),
    tags="best_crop_display"
)

# Table headers
canvas.create_rectangle(39.0, 202.0, 780.0, 203.0, fill="#A8A8A8", outline="")
canvas.create_rectangle(39.0, 158.0, 780.0, 159.0, fill="#A8A8A8", outline="")

headers = [
    (84.0, 172.0, "Crop"),
    (165.0, 172.0, "Suitability"),
    (360.0, 172.0, "Cost Score"),
    (489.0, 172.0, "Match Percentage")
]

for x, y, text in headers:
    canvas.create_text(
        x, y,
        anchor="nw",
        text=text,
        fill="#848484",
        font=("Inter Medium", 15 * -1),
    )

# Create text objects with tags for dynamic updating
result_positions = [239, 283, 326, 369, 413, 456]
for i, y in enumerate(result_positions):
    canvas.create_text(69.0, y, anchor="nw", text="", tags=f"crop_{i}",
                     fill="#3AB67D", font=("Inter SemiBoldItalic", 16 * -1))
    canvas.create_text(171.0, y, anchor="nw", text="", tags=f"suitability_{i}",
                     fill="#848484", font=("Inter SemiBoldItalic", 16 * -1))
    canvas.create_text(376.0, y, anchor="nw", text="", tags=f"cost_{i}",
                     fill="#848484", font=("Inter SemiBoldItalic", 16 * -1))
    canvas.create_text(530.0, y, anchor="nw", text="", tags=f"match_{i}",
                     fill="#848484", font=("Inter SemiBoldItalic", 16 * -1))

# Recommendation method selection
canvas.create_text(
    630.0, 7.0,
    anchor="nw",
    text="Choose Recommendation Method",
    fill="#848484",
    font=("Inter SemiBoldItalic", 16 * -1),
)

methods = {
    "A*": (710, 36, "A* Search"),
    "Greedy": (710, 62, "Greedy Search"),
    "Genetic": (710, 89, "Genetic Algorithm"),
    "CSP": (710, 116, "CSP")
}

selected_method = StringVar(value="Genetic")  # Default to Genetic Algorithm

# Create radio buttons and labels
for method, (x, y, label) in methods.items():
    Radiobutton(
        window,
        text="",
        variable=selected_method,
        value=method,
        bg="#FFFFFF",
        activebackground="#FFFFFF",
        highlightthickness=0,
    ).place(x=x, y=y)
    
    canvas.create_text(
        x + 25, y,
        anchor="nw",
        text=label,
        fill="#848484",
        font=("Inter SemiBoldItalic", 14 * -1),
    )

# Buttons
for i, y in enumerate([234, 276, 321, 364, 408, 451], start=1):
    button_image = PhotoImage(file=relative_to_assets(f"button_{i}.png"))
    button = Button(
        image=button_image,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: subprocess.Popen([sys.executable, "./src/app/frontend/More.py"]),
    )
    button.image = button_image
    button.place(x=661.0, y=y, width=90.0, height=30.0)

# Main Functions --------------------------------------------------------------

def display_genetic_algorithm_results(input_data):
    """Display results from genetic algorithm"""
    try:
        with open(project_root / "app" / "backend" / "crop_db.json") as f:
            crop_db = json.load(f)
        
        initial_state = {
            'soil': input_data['soil'],
            'climate': input_data['climate'],
            'environmental': input_data['environmental'],
            'current_crop': None
        }
        
        problem = CropProblem(initial_state, crop_db)
        ga = CropGeneticAlgorithm(problem)
        results = ga.get_top_n_crops(6)  # Get top 6 crops

        # Display the best crop at the top
        if results:
            best_crop = results[0]['crop'].capitalize()
            best_match = f"{results[0]['match_percentage']:.2f}%"
            canvas.itemconfig(
                "best_crop_display",
                text=f"{best_crop} ({best_match} match)"
            )
        
        for i, (result, y) in enumerate(zip(results, result_positions)):
            canvas.itemconfig(f"crop_{i}", text=result['crop'].capitalize())
            canvas.itemconfig(f"suitability_{i}", text=result['suitability'])
            canvas.itemconfig(f"cost_{i}", text=f"{result['cost']:.4f}")
            canvas.itemconfig(f"match_{i}", text=f"{result['match_percentage']:.2f}%")
            
            # Clear any remaining rows if we have fewer than 6 results
            for j in range(len(results), 6):
                canvas.itemconfig(f"crop_{j}", text="")
                canvas.itemconfig(f"suitability_{j}", text="")
                canvas.itemconfig(f"cost_{j}", text="")
                canvas.itemconfig(f"match_{j}", text="")
                
    except Exception as e:
        print(f"Error displaying results: {e}")
def display_CSP_algorithm_results(input_data):
    """Display results from CSP algorithm"""
    try:
        with open(project_root / "app" / "backend" / "crop_db.json") as f:
            crop_db = json.load(f)
        
        initial_state = {
            'soil': input_data['soil'],
            'climate': input_data['climate'],
            'environmental': input_data['environmental'],
            'current_crop': None
        }
        
        solver = CropCSP(crop_db,initial_state)
        solver.set_parameter_tolerance('soil.ph', 0.02)  # Stricter pH tolerance
        solver.set_tolerance(0.2)
        all_options = solver.get_all_options(6)  # Get top 6 crops

        # Display the best crop at the top
        if all_options:
            best_crop = all_options[0][0].capitalize()
            best_match = f"{100 * ( 1 - all_options[0][1]) :.2f}%"
            
            canvas.itemconfig(
                "best_crop_display",
                text=f"{best_crop} ({best_match} match)"
            )
        
        for i, (crop, score, passes, details) in enumerate(all_options, 0):
            canvas.itemconfig(f"crop_{i}", text=crop.capitalize())
            if score < 0.2:
                status = "Excellent"
            elif score < 0.4:
                status = "Good"
            elif score < 0.6:
                status =  "Fair"
            else:
                status = "Poor"
            canvas.itemconfig(f"suitability_{i}", text=status)
            canvas.itemconfig(f"cost_{i}", text=f"{score:.4f}")
            match_score = 100*(1 - score) if score < 1 else 0
            canvas.itemconfig(f"match_{i}", text=f"{match_score:.2f}%")
            
            # Clear any remaining rows if we have fewer than 6 results
            for j in range(len(all_options), 6):
                canvas.itemconfig(f"crop_{j}", text="")
                canvas.itemconfig(f"suitability_{j}", text="")
                canvas.itemconfig(f"cost_{j}", text="")
                canvas.itemconfig(f"match_{j}", text="")
                
    except Exception as e:
        print(f"Error displaying results: {e}")

def display_greedy_algorithm_results(input_data):
    """Display results from greedy algorithm with new folder structure"""
    try:
        # Define paths - all files now in backend folder
        backend_dir = os.path.join(project_root, "app", "backend")
        crop_db_path = os.path.join(backend_dir, "crop_db.json")
        heuristics_path = os.path.join(backend_dir, "heuristics.txt")

        # Initialize calculator with current state
        initial_state = {
            'soil': input_data['soil'],
            'climate': input_data['climate'],
            'environmental': input_data['environmental'],
            'current_crop': None
        }

        calculator = HeuristicCalculator(
            current_state=initial_state,
            crop_db_path=crop_db_path
        )

        # Generate and save heuristics
        calculator.run(heuristics_path)
        
        # Create problem and search instance
        problem = CropProblem(initial_state, calculator.crop_db)
        ga = GeneralHeuristicBasedSearch(problem, calculator.heuristics, "greedy")
        
        # Get top recommendations
        results = ga.search(6)  # Returns list of OrderedNode objects
        
        if not results:
            raise ValueError("No suitable crops found for current conditions")

        # Display the best crop at the top
        if results:
            best_crop = results[0].node_name.capitalize()
            best_score = results[0].heuristic_value
            best_match = f"{100 * (1 - best_score):.2f}%"
            canvas.itemconfig(
                "best_crop_display",
                text=f"{best_crop} ({best_match} match)"
            )
        def classify_suitability(cost: float) -> str:
               if cost < 0.2:
                  return "Excellent"
               elif cost < 0.4:
                   return "Good"
               elif cost < 0.6:
                    return "Fair"
               else:
                    return "Poor"


   
    
        # Display all results
        for i, (result, y_pos) in enumerate(zip(results, result_positions)):
            canvas.itemconfig(f"crop_{i}", text=result.node_name.capitalize())
            suitability = classify_suitability(result.heuristic_value)
            canvas.itemconfig(f"suitability_{i}", text=suitability)
            canvas.itemconfig(f"cost_{i}", text=f"{result.heuristic_value:.4f}")
            canvas.itemconfig(f"match_{i}", text=f"{100 * (1 - result.heuristic_value):.2f}%")
            
        # Clear any remaining rows
        for j in range(len(results), 6):
            canvas.itemconfig(f"crop_{j}", text="")
            canvas.itemconfig(f"suitability_{j}", text="")
            canvas.itemconfig(f"cost_{j}", text="")
            canvas.itemconfig(f"match_{j}", text="")
            
    except Exception as e:
        print(f"Error displaying results: {e}")
        canvas.itemconfig("best_crop_display", text="Calculation Error")
        for i in range(6):
            canvas.itemconfig(f"crop_{i}", text="")
            canvas.itemconfig(f"suitability_{i}", text="Error")
            canvas.itemconfig(f"cost_{i}", text="")
            canvas.itemconfig(f"match_{i}", text="")
    
def display_a_star_algorithm_results(input_data):
    """Display results from a_star algorithm with new folder structure"""
    try:
        # Define paths - all files now in backend folder
        backend_dir = os.path.join(project_root, "app", "backend")
        crop_db_path = os.path.join(backend_dir, "crop_db.json")
        heuristics_path = os.path.join(backend_dir, "heuristics.txt")

        # Initialize calculator with current state
        initial_state = {
            'soil': input_data['soil'],
            'climate': input_data['climate'],
            'environmental': input_data['environmental'],
            'current_crop': None
        }

        calculator = HeuristicCalculator(
            current_state=initial_state,
            crop_db_path=crop_db_path
        )

        # Generate and save heuristics
        calculator.run(heuristics_path)
        
        # Create problem and search instance
        problem = CropProblem(initial_state, calculator.crop_db)
        a = GeneralHeuristicBasedSearch(problem, calculator.heuristics, "a_star")
        
        # Get top recommendations
        results = a.search(6)  # Returns list of OrderedNode objects
        
        if not results:
            raise ValueError("No suitable crops found for current conditions")

        # Display the best crop at the top
        if results:
            best_crop = results[0].node_name.capitalize()
            best_score = results[0].heuristic_value
            best_match = f"{100 * (1 - best_score):.2f}%"
            canvas.itemconfig(
                "best_crop_display",
                text=f"{best_crop} ({best_match} match)"
            )
        def classify_suitability(cost: float) -> str:
               if cost < 0.2:
                  return "Excellent"
               elif cost < 0.4:
                   return "Good"
               elif cost < 0.6:
                    return "Fair"
               else:
                    return "Poor"


   
    
        # Display all results
        for i, (result, y_pos) in enumerate(zip(results, result_positions)):
            canvas.itemconfig(f"crop_{i}", text=result.node_name.capitalize())
            suitability = classify_suitability(result.heuristic_value)
            canvas.itemconfig(f"suitability_{i}", text=suitability)
            canvas.itemconfig(f"cost_{i}", text=f"{result.heuristic_value:.4f}")
            canvas.itemconfig(f"match_{i}", text=f"{100 * (1 - result.heuristic_value):.2f}%")
            
        # Clear any remaining rows
        for j in range(len(results), 6):
            canvas.itemconfig(f"crop_{j}", text="")
            canvas.itemconfig(f"suitability_{j}", text="")
            canvas.itemconfig(f"cost_{j}", text="")
            canvas.itemconfig(f"match_{j}", text="")
            
    except Exception as e:
        print(f"Error displaying results: {e}")
        canvas.itemconfig("best_crop_display", text="Calculation Error")
        for i in range(6):
            canvas.itemconfig(f"crop_{i}", text="")
            canvas.itemconfig(f"suitability_{i}", text="Error")
            canvas.itemconfig(f"cost_{i}", text="")
            canvas.itemconfig(f"match_{i}", text="")
def on_method_change(*args):
    """Handle method selection change"""
    if len(sys.argv) > 1:
        input_data = json.loads(sys.argv[1])
        method = selected_method.get()
        
        if method == "Genetic":
            display_genetic_algorithm_results(input_data)
        elif method == "Greedy":
            display_greedy_algorithm_results(input_data)
        elif method == "CSP":
            display_CSP_algorithm_results(input_data)
        # Add other method handlers here as needed

# Connect the method change handler
selected_method.trace('w', on_method_change)

# Navigation buttons
def switch_to_input():
    window.destroy()
    subprocess.Popen([sys.executable, "./src/app/frontend/input.py"])

button_image_7 = PhotoImage(file=relative_to_assets("button_7.png"))
button_7 = Button(
    image=button_image_7,
    borderwidth=0,
    highlightthickness=0,
    command=switch_to_input,
)
button_7.image = button_image_7
button_7.place(x=699.0, y=530.0, width=171.0, height=48.0)

# Initialization --------------------------------------------------------------

# Display results if data was passed and Genetic is selected
if len(sys.argv) > 1:
    try:
        input_data = json.loads(sys.argv[1])
        # Call the appropriate display function based on initial selection
        if selected_method.get() == "Genetic":
            display_genetic_algorithm_results(input_data)
        elif selected_method.get() == "Greedy":
            display_greedy_algorithm_results(input_data)
    except json.JSONDecodeError:
        print("Invalid input data format")

window.resizable(False, False)
window.mainloop()
