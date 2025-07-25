import tkinter as tk
from PIL import Image, ImageTk
from pathlib import Path
import subprocess
import sys
import json
import os

project_root = Path(__file__).parents[2]  
sys.path.insert(0, str(project_root))
from app.backend.algorithms import CropProblem,HeuristicCalculator,GeneralHeuristicBasedSearch,HeuristicCalculator, CropGeneticAlgorithm,CropCSP, cropNode 
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"./assets/output")


def classify_suitability(cost: float) -> str:
               if cost < 0.2:
                  return "Excellent"
               elif cost < 0.4:
                   return "Good"
               elif cost < 0.6:
                    return "Fair"
               else:
                    return "Poor"

def algorithm_label_to_key(label):
    mapping = {
        "A* Search": "A*",
        "Greedy": "Greedy",
        "CSP": "CSP",
        "Genetic": "Genetic"
    }
    return mapping.get(label, label)


def format_top_crops(top_crops, method):
    """Convert any algorithm output into a unified dict format."""
    formatted = []
    if method == "Genetic":
        for i, result in enumerate(top_crops):
            formatted.append({
                "crop": result['crop'].capitalize(),
                "suitability": result['suitability'],
                "cost": f"{result['cost']:.4f}",
                "match": f"{result['match_percentage']:.2f}%"
            })
          
    elif method in ["Greedy", "A*"]:
        for i, result in enumerate(top_crops):
            suitability = classify_suitability(result.heuristic_value)
            formatted.append({
                "crop": result.node_name.capitalize(),
                "suitability": suitability,
                "cost": f"{result.heuristic_value:.4f}",  
                "match": f"{100 * (1 - result.heuristic_value):.2f}%"
            })

    elif method == "CSP":
            
        for i, (crop, score, passes, details) in enumerate(top_crops, 0):
            if score < 0.2:
                status = "Excellent"
            elif score < 0.4:
                status = "Good"
            elif score < 0.6:
                status =  "Fair"
            else:
                status = "Poor"
            match_score = 100*(1 - score) if score < 1 else 0
            formatted.append({
                "crop": crop.capitalize(),
                "suitability": status,
                "cost": f"{score:.4f}",  
                "match": f"{match_score:.2f}%"
            })
            

    return formatted

def genetic_algorithm_results(input_data):
    try:
        with open(project_root / "app" / "backend" / "crop_db.json") as f:
            crop_db = json.load(f)
        
        initial_state = {
            'soil': input_data['soil'],
            'climate': input_data['climate'],
            'environmental': input_data['environmental'],
            'current_crop': None
        }
        global top_crops
        problem = CropProblem(initial_state, crop_db)
        ga = CropGeneticAlgorithm(problem)
        top_crops = ga.get_top_n_crops(6) 
        top_crops = format_top_crops(top_crops, "Genetic")
        return top_crops
    except Exception as e:
        print(f"Error displaying results: {e}")    
        
def CSP_algorithm_results(input_data):
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
        global top_crops
        solver = CropCSP(crop_db,initial_state)
        solver.set_parameter_tolerance('soil.ph', 0.02)  
        solver.set_tolerance(0.2)
        top_crops = solver.get_all_options(6)  
        top_crops = format_top_crops(top_crops, "CSP")
        return top_crops    
    except Exception as e:
        print(f"Error displaying results: {e}")     
            
        
                   
def greedy_algorithm_results(input_data):
    try:
        global top_crops
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

        calculator.run(heuristics_path)
        problem = CropProblem(initial_state, calculator.crop_db)
        ga = GeneralHeuristicBasedSearch(problem, calculator.heuristics, "greedy")
        top_crops = ga.search(6)  
        top_crops = format_top_crops(top_crops, "Greedy")
        
        if not top_crops:
            raise ValueError("No suitable crops found for current conditions")
        return top_crops
    except Exception as e:
        print(f"Error displaying greedy results: {e}")

def a_star_algorithm_results(input_data):
    """Display results from a_star algorithm with new folder structure"""
    try:
        global top_crops
        backend_dir = os.path.join(project_root, "app", "backend")
        crop_db_path = os.path.join(backend_dir, "crop_db.json")
        heuristics_path = os.path.join(backend_dir, "heuristics.txt")
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
        calculator.run(heuristics_path)
        problem = CropProblem(initial_state, calculator.crop_db)
        a = GeneralHeuristicBasedSearch(problem, calculator.heuristics, "a_star")
        top_crops = a.search(6)  
        top_crops = format_top_crops(top_crops, "A*")
        if not top_crops:
            raise ValueError("No suitable crops found for current conditions")
        return top_crops

    except Exception as e:
        print(f"Error displaying greedy results: {e}")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def create_rounded_rect(canvas, x1, y1, x2, y2, radius=25, shadow=False, **kwargs):
    if shadow:
        canvas.create_polygon(
            [x1+radius+2, y1+2, x2-radius+2, y1+2, x2+2, y1+2, x2+2, y1+radius+2,
             x2+2, y2-radius+2, x2+2, y2+2, x2-radius+2, y2+2, x1+radius+2, y2+2,
             x1+2, y2+2, x1+2, y2-radius+2, x1+2, y1+radius+2, x1+2, y1+2],
            smooth=True, splinesteps=36, fill="#cbd5e1", outline=""
        )
    return canvas.create_polygon(
        [x1+radius, y1, x2-radius, y1, x2, y1, x2, y1+radius,
         x2, y2-radius, x2, y2, x2-radius, y2, x1+radius, y2,
         x1, y2, x1, y2-radius, x1, y1+radius, x1, y1],
        smooth=True, splinesteps=36, **kwargs
    )


def set_algorithm(opt):
        algo_var.set(opt)
        selected_method.set(opt)
        refresh_results()
def initialize_ui():
    root = tk.Tk()
    selected_method = tk.StringVar(value="Genetic")
    algo_var = tk.StringVar(value="A* Search")
    image_refs = []
    root.title("Crop Recommendation UI")
    root.geometry("1122x880")
    root.configure(bg="#f9fff7")
    header_canvas = tk.Canvas(root, height=250, bg="#0c4e0b", highlightthickness=0)
    header_canvas.pack(fill="x")
    header_frame = tk.Frame(header_canvas, bg="#0c4e0b")
    header_canvas.create_window((70, 20), window=header_frame, anchor="nw")
    icon_path = relative_to_assets("leaf.png")
    icon_image = Image.open(icon_path).resize((36, 36), Image.Resampling.LANCZOS)
    icon_tk = ImageTk.PhotoImage(icon_image)
    image_refs.append(icon_tk)
    icon_label = tk.Label(header_frame, image=icon_tk, bg="#0c4e0b")
    icon_label.image = icon_tk
    icon_label.pack(side="left", padx=(0, 10))
    title_frame = tk.Frame(header_frame, bg="#0c4e0b")
    title_frame.pack(side="left")
    title_label = tk.Label(title_frame, text="Crop Recommendations",font=("Segoe UI", 17, "bold"), fg="white", bg="#0c4e0b")
    title_label.pack(anchor="w")
    subtitle_label = tk.Label(title_frame, text="AI-powered crop selection analysis",font=("Segoe UI", 11), fg="#bccfbc", bg="#0c4e0b")
    subtitle_label.pack(anchor="w")
    create_rounded_rect(header_canvas, 890, 20, 1040, 65, radius=20, fill="#3c703c")
    dash_frame = tk.Frame(header_canvas, bg="#3c703c", cursor="hand2")
    header_canvas.create_window((965, 42), window=dash_frame, anchor="center")
    dash_path = relative_to_assets("graph.png")
    dash_icon = Image.open(dash_path).resize((18, 18), Image.Resampling.LANCZOS)
    dash_icon_tk = ImageTk.PhotoImage(dash_icon)
    image_refs.append(dash_icon_tk)
    dash_img_label = tk.Label(dash_frame, image=dash_icon_tk, bg="#3c703c")
    dash_img_label.image = dash_icon_tk
    dash_img_label.pack(side="left", padx=(0, 6))
    dash_text_label = tk.Label(dash_frame, text="Dashboard", font=("Helvetica", 13, "bold"), fg="white", bg="#3c703c")
    dash_text_label.pack(side="left")
    dash_frame.bind("<Button-1>", lambda e: go_to_dashboard())
    dash_img_label.bind("<Button-1>", lambda e: go_to_dashboard())
    dash_text_label.bind("<Button-1>", lambda e: go_to_dashboard())
    create_rounded_rect(header_canvas, 730, 20, 880, 65, radius=20, fill="#3c703c")
    algo_frame = tk.Frame(header_canvas, bg="#3c703c", cursor="hand2")
    header_canvas.create_window((805, 42), window=algo_frame, anchor="center")
    algo_path = relative_to_assets("algorithm.png")
    algo_icon = Image.open(algo_path).resize((22, 22), Image.Resampling.LANCZOS)
    algo_icon_tk = ImageTk.PhotoImage(algo_icon)
    image_refs.append(algo_icon_tk)
    algo_img_label = tk.Label(algo_frame, image=algo_icon_tk, bg="#3c703c")
    algo_img_label.image = algo_icon_tk
    algo_img_label.pack(side="left", padx=(0, 6))
    algo_text_label = tk.Label(algo_frame, textvariable=algo_var, font=("Helvetica", 13, "bold"), fg="white", bg="#3c703c")
    algo_text_label.pack(side="left")
    algo_options = ["A* Search", "Greedy", "CSP", "Genetic"]
    popup_menu = tk.Menu(root, tearoff=0, bg="#2e6640", fg="white", font=("Segoe UI", 15), bd=15,activebackground="#659265", activeforeground="white")
    for option in algo_options:
        popup_menu.add_command(label=option, command=lambda opt=option: set_algorithm(opt))
    def show_algo_menu(event):
        try:
            x = algo_frame.winfo_rootx()
            y = algo_frame.winfo_rooty() + algo_frame.winfo_height()
            popup_menu.tk_popup(x, y)
        finally:
            popup_menu.grab_release()
    algo_frame.bind("<Button-1>", show_algo_menu)
    algo_img_label.bind("<Button-1>", show_algo_menu)
    algo_text_label.bind("<Button-1>", show_algo_menu)
    create_rounded_rect(header_canvas, 540, 20, 710, 65, radius=20, fill="#3c703c")
    back_frame = tk.Frame(header_canvas, bg="#3c703c", cursor="hand2")
    header_canvas.create_window((625, 42), window=back_frame, anchor="center")
    back_path = relative_to_assets("backarrow.png")
    back_icon = Image.open(back_path).resize((18, 18), Image.Resampling.LANCZOS)
    back_icon_tk = ImageTk.PhotoImage(back_icon)
    image_refs.append(back_icon_tk)
    back_img_label = tk.Label(back_frame, image=back_icon_tk, bg="#3c703c")
    back_img_label.image = back_icon_tk
    back_img_label.pack(side="left", padx=(0, 6))
    back_text_label = tk.Label(back_frame, text="Back to Inputs", font=("Helvetica", 13, "bold"), fg="white", bg="#3c703c")
    back_text_label.pack(side="left")
    back_frame.bind("<Button-1>", lambda e: go_back_to_inputs())
    back_img_label.bind("<Button-1>", lambda e: go_back_to_inputs())
    back_text_label.bind("<Button-1>", lambda e: go_back_to_inputs())
    body_frame = tk.Frame(root, bg="#f6fff5")
    body_frame.pack(fill="both", expand=True)
    bg_canvas = tk.Canvas(body_frame, bg="#f6fff5", highlightthickness=0)
    bg_canvas.pack(expand=True, fill="both", padx=30, pady=20)
    create_rounded_rect(bg_canvas, 90, 20, 990, 570, radius=35, fill="white", outline="#b4dfc0", width=3)
    rounded_frame = tk.Frame(bg_canvas, bg="white", bd=0)
    bg_canvas.create_window(90, 20, anchor="nw", window=rounded_frame, width=900, height=550)
    header = tk.Frame(rounded_frame, bg="white")
    header.pack(fill="x", padx=30, pady=(20, 10))
    tk.Label(header, text="Recommended Crops", font=("Arial", 14, "bold"), fg="#14532d", bg="white").pack(side="left")
    scroll_canvas = tk.Canvas(rounded_frame, bg="white", highlightthickness=0, height=500)
    scroll_canvas.pack(side="left", fill="both", expand=True, padx=10)
    scrollbar = tk.Scrollbar(rounded_frame, orient="vertical", command=scroll_canvas.yview)
    scrollbar.pack(side="right", fill="y")
    scroll_canvas.configure(yscrollcommand=scrollbar.set)
    scroll_frame = tk.Frame(scroll_canvas, bg="white")
    scroll_canvas.create_window((0, 0), window=scroll_frame, anchor="nw", width=840)

    def update_scroll(event):
        scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

    scroll_frame.bind("<Configure>", update_scroll)
    return root, selected_method, algo_var, image_refs, header_canvas, scroll_frame


def BestRecUi(header_canvas, selected_method, image_refs, top_crops):
    if not top_crops:
        print("No crop recommendations available.")
        return
    reco_name = top_crops[0].get("crop", "Unknown").lower()
    reco_path = relative_to_assets(f"{reco_name}.png")
    reco_path = relative_to_assets(reco_path)
    try:
        reco_icon = Image.open(reco_path).resize((48, 48), Image.Resampling.LANCZOS)
    except FileNotFoundError:
        reco_icon = Image.open(relative_to_assets("default.png")).resize((28, 28))
    reco_icon_tk = ImageTk.PhotoImage(reco_icon)
    image_refs.append(reco_icon_tk)
    banner_top = 120
    banner_bottom = 230
    create_rounded_rect(header_canvas, 70, banner_top, 1040, banner_bottom, radius=30, fill="#246025")
    reco_combo_frame = tk.Frame(header_canvas, bg="#246025")
    header_canvas.create_window((150, banner_top + 15), window=reco_combo_frame, anchor="nw")
    reco_icon_label = tk.Label(reco_combo_frame, image=reco_icon_tk, bg="#246025")
    reco_icon_label.image = reco_icon_tk
    reco_icon_label.pack(side="left", padx=(0, 16), pady=(0, 5))
    reco_texts_frame = tk.Frame(reco_combo_frame, bg="#246025")
    reco_texts_frame.pack(side="left", anchor="n")
    crop_name = top_crops[0].get("crop", "Unknown").capitalize()
    match_score = top_crops[0].get("match", "N/A")
    algo_name = selected_method.get() if selected_method else "Algorithm"
    reco_title = tk.Label(reco_texts_frame, text=f"Best Recommendation: {crop_name}",font=("Segoe UI", 16, "bold"), fg="white", bg="#246025")
    reco_title.pack(anchor="w")
    reco_sub = tk.Label(reco_texts_frame, text=f"{match_score} match • {algo_name} Algorithm",font=("Segoe UI", 12), fg="#d4dfd2", bg="#246025")
    reco_sub.pack(anchor="w", pady=(2, 0))

def go_to_dashboard():
    root.destroy()
    subprocess.Popen(
        [
            sys.executable,
            "src/app/frontend/dashboard.py",
        ]
    )

def go_back_to_inputs():
    
    root.destroy()
    subprocess.Popen(
        [
            sys.executable,
            "src/app/frontend/input.py",
        ]
    )
def go_to_More(index):
    global top_crops
    root.destroy()
    try:
        subprocess.run([
            sys.executable,
            "src/app/frontend/More.py",
            json.dumps({
                "crops": [crop["crop"] for crop in top_crops],
                "index": index,
                "inputs": input_data
            })
        ])
    except Exception as e:
        print(f"Failed to open more.py: {e}")


def CreateUI():
    global scroll_frame, image_refs
    for widget in scroll_frame.winfo_children():
        widget.destroy()

    def create_crop_card(parent, rank, name, suitability, cost, match):
        card_width = 830
        card_height = 140
        canvas = tk.Canvas(parent, width=card_width, height=card_height, bg="white", highlightthickness=0)
        canvas.pack(pady=4)

        x1, y1 = 10, 10
        x2, y2 = card_width - 10, card_height - 10
        create_rounded_rect(canvas, x1, y1, x2, y2, radius=16, fill="white", outline="#d1d5db")
        badge_radius = 14
        badge_cx = x1 + 30
        badge_cy = y1 + 30
        canvas.create_oval(badge_cx - badge_radius, badge_cy - badge_radius,
                           badge_cx + badge_radius, badge_cy + badge_radius,
                           fill="#e9fbe7", outline="")
        canvas.create_text(badge_cx, badge_cy, text=str(rank), font=("Arial", 10, "bold"), fill="#14532d")
        try:
            icon_filename = f"{name.lower().replace(' ', '')}.png"
            icon_path = relative_to_assets(icon_filename)
            if not os.path.exists(icon_path):
                default_icon_path = relative_to_assets("default.png")
                icon_path = default_icon_path
            crop_path = relative_to_assets(icon_path)
            crop_icon = Image.open(crop_path).resize((28, 28))
            crop_icon_photo = ImageTk.PhotoImage(crop_icon)
            image_refs.append(crop_icon_photo)
            canvas.image_crop = crop_icon_photo
            icon_x = badge_cx + badge_radius + 10
            icon_y = y1 + 18
            canvas.create_image(icon_x, icon_y, anchor="nw", image=crop_icon_photo)
        except:
            icon_x = badge_cx + badge_radius + 10
            icon_y = y1 + 18
        name_x = icon_x + 35
        name_y = icon_y
        canvas.create_text(name_x, name_y, text=name, anchor="nw", font=("Arial", 14, "bold"), fill="#111827")
        metrics_y = name_y + 25

        try:
            suit_path = relative_to_assets("increase.png")
            suit_icon = Image.open(suit_path).resize((20, 20))
            suit_icon_photo = ImageTk.PhotoImage(suit_icon)
            image_refs.append(suit_icon_photo)
            canvas.suit_icon = suit_icon_photo
            canvas.create_image(name_x, metrics_y + 30, anchor="nw", image=suit_icon_photo)
        except:
            pass
        canvas.create_text(name_x + 25, metrics_y + 34, text=f"Suitability: {suitability}%",anchor="nw", font=("Arial", 10), fill="#065f46")
        cost_x = name_x + 160
        try:
            cost_path = relative_to_assets("money.png")
            cost_icon = Image.open(cost_path).resize((20, 20))
            cost_icon_photo = ImageTk.PhotoImage(cost_icon)
            image_refs.append(cost_icon_photo)
            canvas.cost_icon = cost_icon_photo
            canvas.create_image(cost_x, metrics_y + 30, anchor="nw", image=cost_icon_photo)
        except:
            pass
        canvas.create_text(cost_x + 25, metrics_y + 34, text=f"Cost: {cost}%",anchor="nw", font=("Arial", 10), fill="#1d4ed8")
        match_x = cost_x + 160
        try:
            match_path = relative_to_assets("good.png")
            match_icon = Image.open(match_path).resize((20, 20))
            match_icon_photo = ImageTk.PhotoImage(match_icon)
            image_refs.append(match_icon_photo)
            canvas.match_icon = match_icon_photo
            canvas.create_image(match_x, metrics_y + 30, anchor="nw", image=match_icon_photo)
        except:
            pass
        canvas.create_text(match_x + 25, metrics_y + 33, text=f"Match: {match}%",anchor="nw", font=("Arial", 10), fill="#374151")
        button_w, button_h = 130, 40
        button_x1 = x2 - button_w - 20
        button_y1 = y1 + 45
        button_x2 = button_x1 + button_w
        button_y2 = button_y1 + button_h
        create_rounded_rect(canvas, button_x1, button_y1, button_x2, button_y2,radius=14, fill="#14532d", outline="")
        try:
            info_path = relative_to_assets("info.png")
            info_icon = Image.open(info_path).resize((18, 18))
            info_icon_photo = ImageTk.PhotoImage(info_icon)
            image_refs.append(info_icon_photo)
            canvas.info_icon = info_icon_photo
            icon_item = canvas.create_image(button_x1 + 18, (button_y1 + button_y2) // 2, image=info_icon_photo)
            icon_offset = 26
        except:
            icon_item = None
            icon_offset = 0

        text_item = canvas.create_text(
            button_x1 + 18 + icon_offset,
            (button_y1 + button_y2) // 2,
            text="More Info",
            anchor="w",
            font=("Arial", 10, "bold"),
            fill="white"
        )

        canvas.tag_bind(text_item, "<Button-1>", lambda e, idx=rank: go_to_More(idx))
        if icon_item:
            canvas.tag_bind(icon_item, "<Button-1>", lambda e, idx=rank: go_to_More(idx))

    for i, result in enumerate(top_crops):
        create_crop_card(scroll_frame, i + 1, result['crop'].capitalize(),
                         result['suitability'], result['cost'], result['match'])

def refresh_results():
    global  input_data,image_refs, top_crops  
    image_refs.clear()

    if len(sys.argv) > 1:
        try:
            input_data = json.loads(sys.argv[1])
            method = selected_method.get()
            if method == "Genetic":
                top_crops = genetic_algorithm_results(input_data)
            elif method == "Greedy":
                top_crops = greedy_algorithm_results(input_data)
            elif method == "CSP":
                top_crops = CSP_algorithm_results(input_data)
            elif method == "A* Search":
                top_crops = a_star_algorithm_results(input_data)
            else:
                print("Invalid method selected.")
                return

            BestRecUi(header_canvas, selected_method, image_refs, top_crops)
            CreateUI()
        except json.JSONDecodeError:
            print("Invalid input data format.")


if __name__ == "__main__":
    root, selected_method, algo_var, image_refs, header_canvas, scroll_frame = initialize_ui()
    window = root
    refresh_results()
    root.mainloop()
