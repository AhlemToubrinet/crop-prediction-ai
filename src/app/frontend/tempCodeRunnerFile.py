import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from pathlib import Path
import sys
import subprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json

try:
    if len(sys.argv) > 1:
        initial_state = json.loads(sys.argv[1])
        input_data = json.loads(sys.argv[1])
    else:
        initial_state = None
        print(" No initial state received.")
except json.JSONDecodeError as e:
    print(" JSON Decode Error:", e)
    initial_state = None

# Relative path to dashboard assets
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"./assets/dashboard")

project_root = Path(__file__).parents[2]  
sys.path.insert(0, str(project_root))

from app.backend.algorithms import (
    CropProblem, 
    SpaceComplexityBenchmark, 
    TimeComplexityBenchmark, 
    CropAlgorithmComparator, 
    EnvironmentalImpactEvaluator,
    GeneralHeuristicBasedSearch,
    CropCSP,
    CropGeneticAlgorithm
)

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def create_rounded_rect(canvas, x1, y1, x2, y2, radius=25, **kwargs):
    return canvas.create_polygon(
        [x1+radius, y1, x2-radius, y1, x2, y1, x2, y1+radius,
         x2, y2-radius, x2, y2, x2-radius, y2, x1+radius, y2,
         x1, y2, x1, y2-radius, x1, y1+radius, x1, y1],
        smooth=True, splinesteps=36, **kwargs
    )

def go_back_to_output():
    root.destroy()
    subprocess.Popen([sys.executable, "./src/app/frontend/output.py", json.dumps(input_data)])

def display_TimeComplexity_graph(initial_state, parent_frame):
    try:
        with open(project_root / "app" / "backend" / "crop_db.json") as f:
            crop_db = json.load(f)
    except Exception as e:
        print(" Failed to load crop_db.json:", e)
        return

    if initial_state is None:
        print(" Error: No initial state provided.")
        return

    try:
        problem = CropProblem(initial_state, crop_db)
    except Exception as e:
        print(" Failed to create CropProblem:", e)
        return

    benchmark = TimeComplexityBenchmark(problem)
    benchmark.run_benchmarks()

    fig = plt.figure(figsize=(10, 3), dpi=100)
    algorithms = [r['algorithm'] for r in benchmark.results]
    times = [r['time_ms'] for r in benchmark.results]

    bars = plt.bar(algorithms, times, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
    plt.title("Execution Time Comparison")
    plt.xlabel("Algorithm")
    plt.ylabel("Time (ms)")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    plt.grid(True, axis='y')
    plt.tight_layout()

    for widget in parent_frame.winfo_children():
        widget.destroy()

    fig_canvas = FigureCanvasTkAgg(fig, master=parent_frame)
    fig_canvas.draw()
    fig_canvas.get_tk_widget().pack(fill="x", expand=False)
    parent_frame.update_idletasks()

def display_SpaceComplexity_graph(initial_state, parent_frame):
    try:
        with open(project_root / "app" / "backend" / "crop_db.json") as f:
            crop_db = json.load(f)
    except Exception as e:
        print(" Failed to load crop_db.json:", e)
        return

    if initial_state is None:
        print(" Error: No initial state provided.")
        return

    try:
        problem = CropProblem(initial_state, crop_db)
    except Exception as e:
        print(" Failed to create CropProblem:", e)
        return

    benchmark = SpaceComplexityBenchmark(problem)
    benchmark.run_benchmarks()

    fig = plt.figure(figsize=(10, 3), dpi=100)
    algorithms = [r['algorithm'] for r in benchmark.results]
    memory = [r['peak_memory_kb'] for r in benchmark.results]

    bars = plt.bar(algorithms, memory, color=['#9A2802', '#FFDD6C', '#D6A99A', '#52F06F'])
    plt.title("Memory Usage Comparison")
    plt.xlabel("Algorithm")
    plt.ylabel("Peak Memory (KB)")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    plt.grid(True, axis='y')
    plt.tight_layout()

    for widget in parent_frame.winfo_children():
        widget.destroy()

    fig_canvas = FigureCanvasTkAgg(fig, master=parent_frame)
    fig_canvas.draw()
    fig_canvas.get_tk_widget().pack(fill="x", expand=False)
    parent_frame.update_idletasks()

def display_AlgorithmComparator_graph(initial_state, parent_frame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    if initial_state is None:
        print(" Error: No initial state provided.")
        return

    try:
        for widget in parent_frame.winfo_children():
            widget.destroy()

        original_show = plt.show
        plt.show = lambda *args, **kwargs: None

        comparator = CropAlgorithmComparator(initial_state, top_n=5, weight_rank=0.4, weight_score=0.6)
        comparator.compare()

        plt.show = original_show
        fig = plt.gcf()
        fig.set_size_inches(10, 3)
        fig.set_dpi(100)

        fig_canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        fig_canvas.draw()
        fig_canvas.get_tk_widget().pack(fill="x", expand=False)
        parent_frame.update_idletasks()

    except Exception as e:
        print(" Failed to display Algorithm Comparator graph:", e)

def display_EnvironmentalImpact_graph(initial_state, parent_frame):
    try:
        for widget in parent_frame.winfo_children():
            widget.destroy()

        with open(project_root / "app" / "backend" / "crop_db.json") as f:
            crop_db = json.load(f)
        
        if initial_state is None:
            raise ValueError("No initial state provided")
            
        problem = CropProblem(initial_state, crop_db)
        evaluator = EnvironmentalImpactEvaluator(problem)

        import matplotlib.pyplot as plt
        original_show = plt.show
        plt.show = lambda *args, **kwargs: None
        
        scenarios = [evaluator.generate_random_scenario() for _ in range(20)]
        algorithms = [GeneralHeuristicBasedSearch, CropCSP, CropGeneticAlgorithm]
        comparison_df = evaluator.compare_algorithms(scenarios, crop_db, algorithms)
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        fig.suptitle('Algorithm Environmental Impact Comparison', y=1.05)
        
        parameters = {
            'water_usage_efficiency': 'Water Efficiency',
            'fertilizer_usage': 'Fertilizer Usage',
            'irrigation_frequency': 'Irrigation Frequency',
            'pest_pressure': 'Pest Pressure'
        }
        
        for (param, title), ax in zip(parameters.items(), axes.flatten()):
            data = comparison_df[param]
            bars = ax.bar(
                ['Heuristic', 'CSP', 'Genetic'],  # Set labels directly
                data.values,
                color=['#4C72B0', '#55A868', '#C44E52']
            )
            
            ax.set_title(title)
            ax.set_ylim(0, 1)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show = original_show

        fig_canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        fig_canvas.draw()
        fig_canvas.get_tk_widget().pack(fill="x", expand=False)
        parent_frame.update_idletasks()
        
    except Exception as e:
        print(f"Error displaying environmental graph: {e}")
        tk.Label(parent_frame, 
                text=f"Graph Error: {str(e)}",
                fg="red", bg="#f9fff7").pack()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Crop Dashboard")
    root.geometry("1122x630")
    root.minsize(1122, 630)
    root.configure(bg="#f9fff7")

    image_refs = []

    # Header Section
    header = tk.Canvas(root, height=130, bg="#0c4e0b", highlightthickness=0)
    header.pack(fill="x")

    leaf_icon = Image.open(relative_to_assets("leaf.png")).resize((30, 30))
    leaf_icon_tk = ImageTk.PhotoImage(leaf_icon)
    image_refs.append(leaf_icon_tk)

    header.create_image(40, 50, image=leaf_icon_tk)
    header.create_text(80, 42, text="Crop Dashboard", anchor="w", font=("Segoe UI", 18, "bold"), fill="white")
    header.create_text(80, 68, text="Real-time algorithms monitoring", anchor="w", font=("Segoe UI", 12), fill="#bbf7d0")

    create_rounded_rect(header, 920, 35, 1090, 75, radius=15, fill="#3c703c", outline="")
    back_text = header.create_text(1005, 55, text="← Back to Output", font=("Segoe UI", 11, "bold"), fill="white")
    header.tag_bind(back_text, "<Button-1>", lambda e: go_back_to_output())

    # Create main container
    main_frame = tk.Frame(root, bg="#f9fff7")
    main_frame.pack(fill="both", expand=True)

    # Create canvas and scrollbar
    main_canvas = tk.Canvas(main_frame, bg="#f9fff7", highlightthickness=0)
    scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=main_canvas.yview)
    
    main_canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    main_canvas.pack(side="left", fill="both", expand=True)

    # Create scrollable frame
    scrollable_frame = tk.Frame(main_canvas, bg="#f9fff7")
    main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    def on_frame_configure(event):
        main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        main_canvas.itemconfig(1, width=event.width)

    scrollable_frame.bind("<Configure>", on_frame_configure)

    # Create graph frames
    time_graph_frame = tk.Frame(scrollable_frame, bg="#f9fff7", width=1100)
    time_graph_frame.pack(pady=10, fill="x")

    memory_graph_frame = tk.Frame(scrollable_frame, bg="#f9fff7", width=1100)
    memory_graph_frame.pack(pady=10, fill="x")

    environmental_graph_frame = tk.Frame(scrollable_frame, bg="#f9fff7", width=1100)
    environmental_graph_frame.pack(pady=10, fill="x")

    comparator_graph_frame = tk.Frame(scrollable_frame, bg="#f9fff7", width=1100)
    comparator_graph_frame.pack(pady=10, fill="x")

    # Display graphs
    if initial_state:
        display_TimeComplexity_graph(initial_state, time_graph_frame)
        display_SpaceComplexity_graph(initial_state, memory_graph_frame)
        display_AlgorithmComparator_graph(initial_state, comparator_graph_frame)
        display_EnvironmentalImpact_graph(initial_state, environmental_graph_frame)
    else:
        for frame in [time_graph_frame, memory_graph_frame, environmental_graph_frame, comparator_graph_frame]:
            tk.Label(frame, text="No data available", bg="#f9fff7").pack()

    root.mainloop()