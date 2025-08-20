import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from pathlib import Path
import sys
import subprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json

# Enhanced DPI scaling detection and handling
def get_scaling_factor():
    """Detect and return the system's DPI scaling factor"""
    try:
        # Method 1: For Windows
        import ctypes
        try:
            # Try the newer API first
            awareness = ctypes.c_int()
            ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
            scale_factor = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100
            return scale_factor
        except:
            # Fallback to older API
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            # Get screen dimensions to calculate scaling
            screen_width = user32.GetSystemMetrics(0)
            screen_height = user32.GetSystemMetrics(1)
            # Assuming 96 DPI is standard (100% scaling)
            # Calculate approximate scaling factor
            hdc = user32.GetDC(0)
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
            user32.ReleaseDC(0, hdc)
            return dpi / 96
    except:
        # Method 2: For other platforms or as fallback
        try:
            root = tk.Tk()
            root.withdraw()  # Hide the window
            dpi = root.winfo_fpixels('1i')
            root.destroy()
            return dpi / 72  # Standard is 72 DPI for Tkinter
        except:
            return 1.0  # Default to no scaling

# Get scaling factor and adjust base dimensions
SCALING = get_scaling_factor()
BASE_WIDTH = 1122
BASE_HEIGHT = 630
ADJUSTED_WIDTH = int(BASE_WIDTH * SCALING)
ADJUSTED_HEIGHT = int(BASE_HEIGHT * SCALING)

try:
    if len(sys.argv) > 1:
        initial_state = json.loads(sys.argv[1])
        input_data = json.loads(sys.argv[1])
    else:
        initial_state = None
        print(" No initial state received , Dashboard cannot be displayed.")
        sys.exit(0)
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
    points = [
        x1+radius, y1, 
        x2-radius, y1, 
        x2, y1, 
        x2, y1+radius,
        x2, y2-radius, 
        x2, y2, 
        x2-radius, y2, 
        x1+radius, y2,
        x1, y2, 
        x1, y2-radius, 
        x1, y1+radius, 
        x1, y1
    ]
    # Scale points according to DPI
    scaled_points = [p * SCALING for p in points]
    return canvas.create_polygon(
        scaled_points,
        smooth=True, 
        splinesteps=36, 
        **kwargs
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

    fig = plt.figure(figsize=(10 * SCALING, 3 * SCALING), dpi=100)
    algorithms = [r['algorithm'] for r in benchmark.results]
    times = [r['time_ms'] for r in benchmark.results]

    bars = plt.bar(algorithms, times, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
    plt.title("Execution Time Comparison", fontsize=12 * SCALING)
    plt.xlabel("Algorithm", fontsize=10 * SCALING)
    plt.ylabel("Time (ms)", fontsize=10 * SCALING)
    plt.xticks(fontsize=9 * SCALING)
    plt.yticks(fontsize=9 * SCALING)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9 * SCALING)
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

    fig = plt.figure(figsize=(10 * SCALING, 3 * SCALING), dpi=100)
    algorithms = [r['algorithm'] for r in benchmark.results]
    memory = [r['peak_memory_kb'] for r in benchmark.results]

    bars = plt.bar(algorithms, memory, color=['#9A2802', '#FFDD6C', '#D6A99A', '#52F06F'])
    plt.title("Memory Usage Comparison", fontsize=12 * SCALING)
    plt.xlabel("Algorithm", fontsize=10 * SCALING)
    plt.ylabel("Peak Memory (KB)", fontsize=10 * SCALING)
    plt.xticks(fontsize=9 * SCALING)
    plt.yticks(fontsize=9 * SCALING)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9 * SCALING)
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
        fig.set_size_inches(10 * SCALING, 3 * SCALING)
        fig.set_dpi(100)

        # Adjust font sizes in the figure
        for ax in fig.axes:
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + 
                         ax.get_xticklabels() + ax.get_yticklabels()):
                if hasattr(item, 'set_fontsize'):
                    current_size = item.get_fontsize()
                    item.set_fontsize(current_size * SCALING)

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
        
        fig, axes = plt.subplots(2, 2, figsize=(10 * SCALING, 6 * SCALING))
        fig.suptitle('Algorithm Environmental Impact Comparison', y=1.05, fontsize=14 * SCALING)
        
        parameters = {
            'water_usage_efficiency': 'Water Efficiency',
            'fertilizer_usage': 'Fertilizer Usage',
            'irrigation_frequency': 'Irrigation Frequency',
            'pest_pressure': 'Pest Pressure'
        }
        
        for (param, title), ax in zip(parameters.items(), axes.flatten()):
            data = comparison_df[param]
            bars = ax.bar(
                ['Heuristic', 'CSP', 'Genetic'],
                data.values,
                color=['#4C72B0', '#55A868', '#C44E52']
            )
            
            ax.set_title(title, fontsize=12 * SCALING)
            ax.set_ylim(0, 1)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            
            # Adjust font sizes
            ax.xaxis.label.set_fontsize(10 * SCALING)
            ax.yaxis.label.set_fontsize(10 * SCALING)
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(9 * SCALING)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=9 * SCALING)
        
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
    root.geometry(f"{ADJUSTED_WIDTH}x{ADJUSTED_HEIGHT}")
    root.minsize(ADJUSTED_WIDTH, ADJUSTED_HEIGHT)
    root.configure(bg="#f9fff7")

    # Set default font sizes based on scaling
    default_font = ("Segoe UI", int(10 * SCALING))
    title_font = ("Segoe UI", int(18 * SCALING), "bold")
    subtitle_font = ("Segoe UI", int(12 * SCALING))
    button_font = ("Segoe UI", int(11 * SCALING), "bold")
    
    # Apply default font to all widgets
    root.option_add("*Font", default_font)

    image_refs = []

    # Header Section
    header_height = int(130 * SCALING)
    header = tk.Canvas(root, height=header_height, bg="#0c4e0b", highlightthickness=0)
    header.pack(fill="x")

    leaf_size = int(30 * SCALING)
    leaf_icon = Image.open(relative_to_assets("leaf.png")).resize((leaf_size, leaf_size))
    leaf_icon_tk = ImageTk.PhotoImage(leaf_icon)
    image_refs.append(leaf_icon_tk)

    header.create_image(int(40 * SCALING), int(50 * SCALING), image=leaf_icon_tk)
    header.create_text(int(80 * SCALING), int(42 * SCALING), 
                      text="Crop Dashboard", anchor="w", font=title_font, fill="white")
    header.create_text(int(80 * SCALING), int(68 * SCALING), 
                      text="Real-time algorithms monitoring", anchor="w", font=subtitle_font, fill="#bbf7d0")

    # Create back button with scaled coordinates
    btn_x1 = int(920 * SCALING)
    btn_y1 = int(35 * SCALING)
    btn_x2 = int(1090 * SCALING)
    btn_y2 = int(75 * SCALING)
    btn_radius = int(15 * SCALING)
    
    create_rounded_rect(header, btn_x1, btn_y1, btn_x2, btn_y2, radius=btn_radius, fill="#3c703c", outline="")
    back_text = header.create_text(int(1005 * SCALING), int(55 * SCALING), 
                                 text="← Back to Output", font=button_font, fill="white")
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
    frame_width = int(1100 * SCALING)
    time_graph_frame = tk.Frame(scrollable_frame, bg="#f9fff7", width=frame_width)
    time_graph_frame.pack(pady=int(10 * SCALING), fill="x")

    memory_graph_frame = tk.Frame(scrollable_frame, bg="#f9fff7", width=frame_width)
    memory_graph_frame.pack(pady=int(10 * SCALING), fill="x")

    environmental_graph_frame = tk.Frame(scrollable_frame, bg="#f9fff7", width=frame_width)
    environmental_graph_frame.pack(pady=int(10 * SCALING), fill="x")

    comparator_graph_frame = tk.Frame(scrollable_frame, bg="#f9fff7", width=frame_width)
    comparator_graph_frame.pack(pady=int(10 * SCALING), fill="x")

    # Display graphs
    display_TimeComplexity_graph(initial_state, time_graph_frame)
    display_SpaceComplexity_graph(initial_state, memory_graph_frame)
    display_AlgorithmComparator_graph(initial_state, comparator_graph_frame)
    display_EnvironmentalImpact_graph(initial_state, environmental_graph_frame)

    root.mainloop()