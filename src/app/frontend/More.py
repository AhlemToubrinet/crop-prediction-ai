
from pathlib import Path
from tkinter import Tk, Canvas, Button, PhotoImage
import subprocess
import sys

# --- Paths ---
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"./assets/More")



if len(sys.argv) > 1:
    crop_name = sys.argv[1]
else:
    print("No crop name provided. Exiting.")
    sys.exit(1)
    
def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

# --- Helper Functions ---
def load_crop_db(file_path):
    with open(file_path, 'r') as f:
        return eval(f.read())

def get_crop_info(crop_db, crop_name):
    return crop_db.get(crop_name.lower())

def switch_to_output():
    window.destroy()
    subprocess.Popen([sys.executable, "./src/app/frontend/output.py"])

# --- UI Setup ---
window = Tk()
window.geometry("900x590")
window.configure(bg="#f5fef0")  # Light modern background
window.resizable(False, False)

canvas = Canvas(window, bg="#f5fef0", height=590, width=900, bd=0, highlightthickness=0, relief="ridge")
canvas.place(x=0, y=0)

# --- Header ---
header_color = "#82c959"  # Modern darker green
canvas.create_rectangle(0, 0, 900, 100, fill=header_color, outline="")
canvas.create_text(450, 50, text="Crop Environmental Overview", fill="white",
                   font=("Inter Bold", 26), anchor="center")  # Vertically centered in the header

# --- Table Headers ---
headers = ["Stage", "Irrigation (avg)", "Fertilizer (avg)", "Pest Pressure (avg)"]
x_positions = [80, 280, 480, 700]
for i, header in enumerate(headers):
    canvas.create_text(x_positions[i], 125, anchor="w", text=header, fill="#333333", font=("Inter Bold", 15))

# --- Load and Display Data ---
crop_db = load_crop_db("data/processed/big_crop_db.txt")
crop_data = get_crop_info(crop_db, crop_name)

# --- Display 3 Stages of Data ---
y_start = 170
box_height = 80
box_padding = 20

for i in range(1, 4):
    stage_key = str(i)
    stage = crop_data.get(stage_key)
    env = stage['environmental']

    irrigation_avg = env['irrigation_frequency'][1]
    fertilizer_avg = env['fertilizer_usage'][1]
    pest_avg = env['pest_pressure'][1]

    y_top = y_start + (i - 1) * (box_height + box_padding)
    y_center = y_top + box_height // 2 - 10

    # Removed box border — no rectangle drawn

    # Row text
    canvas.create_text(x_positions[0], y_center, text=f"Stage {stage_key}", fill=header_color,
                       font=("Inter SemiBold", 17), anchor="w")
    canvas.create_text(x_positions[1], y_center, text=f"{irrigation_avg:.1f}", fill="#333333",
                       font=("Inter", 15), anchor="w")
    canvas.create_text(x_positions[2], y_center, text=f"{fertilizer_avg:.1f}", fill="#333333",
                       font=("Inter", 15), anchor="w")
    canvas.create_text(x_positions[3], y_center, text=f"{pest_avg:.1f}", fill="#333333",
                       font=("Inter", 15), anchor="w")


window.mainloop()
