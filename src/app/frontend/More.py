import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
from pathlib import Path
import subprocess
import sys
import json
crop_name = "Apple"
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"./assets/More")
if len(sys.argv) > 1:
    try:
        input_json = sys.argv[1]
        input_data = json.loads(input_json)
        crops = input_data["crops"]
        index = input_data["index"] -1
        inputs = input_data["inputs"]

        if 0 <= index < len(crops):
            crop_name = crops[index]
        else:
            crop_name = "Unknown"

        print(f"Selected crop: {crop_name}")  
    except Exception as e:
        print("Error parsing input data:", e)

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)
def load_crop_db(file_path):
    with open(file_path, 'r') as f:
        return eval(f.read())
def get_crop_info(crop_db, crop_name):
    return crop_db.get(crop_name.lower())
def get_irrigation_desc(value):
    if value >= 4:
        return "Water multiple times a day"
    elif value >= 3:
        return "Water daily"
    elif value >= 2:
        return "Water every 3 days"
    elif value >= 1:
        return "Water weekly"
    else:
        return "No irrigation required"
    
def get_pest_desc(value):
    if value >= 4:
        return "Severe pest infestation – act immediately"
    elif value >= 3:
        return "High pest pressure – frequent monitoring needed"
    elif value >= 2:
        return "Moderate pest presence – consider treatment"
    elif value >= 1:
        return "Low pest pressure – light monitoring"
    else:
        return "No pests detected"

def get_fertilizer_desc(value):
    if value >= 4:
        return "Extremely high fertilizer usage"
    elif value >= 3:
        return "High fertilizer required"
    elif value >= 2:
        return "Moderate fertilizer needed"
    elif value >= 1:
        return "Minimal fertilizer needed"
    else:
        return "No fertilizer yet"

def switch_to_output():
    root.destroy()
    subprocess.Popen([sys.executable, "./src/app/frontend/output.py", json.dumps(inputs)])


crop_db = load_crop_db("data/processed/big_crop_db.txt")
crop_data = get_crop_info(crop_db, crop_name)
if not crop_data:
    print(f"Crop '{crop_name}' not found in database.")
    sys.exit(1)

stage1 = crop_data.get("1")
stage2 = crop_data.get("2")
stage3 = crop_data.get("3")
 

irrigation_avg = stage1['environmental']['irrigation_frequency'][1]
fertilizer_avg = stage1['environmental']['fertilizer_usage'][1]
pest_avg = stage1['environmental']['pest_pressure'][1]

# Constants
CARD_WIDTH = 300
CARD_HEIGHT = 90
CARD_RADIUS = 15
CARD_BG = "#ffffff"
BORDER_COLOR = "#e1e5df"
BG = "#f5fef0"

ICON_FONT = ("Segoe UI Emoji", 13)
TITLE_FONT = ("Segoe UI", 15, "bold")
NUMBER_FONT = ("Segoe UI", 13, "bold")
DESC_FONT = ("Segoe UI", 10)
SECTION_FONT = ("Segoe UI", 15, "bold")
CONTENT_FONT = ("Segoe UI", 9)


# Root window
root = tk.Tk()
root.title("Crop Infos")
root.geometry("1122x650")
root.configure(bg=BG)
root.resizable(False, False)

from PIL import ImageDraw

header = tk.Frame(root, bg="#0c4e0b", height=155)
header.pack(fill="x", side="top")
header.pack_propagate(False)

back_btn = tk.Button(
    header,
    text="← Back to Output",
    font=("Segoe UI", 18),
    bg="#0c4e0b",
    fg="#f5fef0",
    bd=0,
    activebackground="#0c4e0b",
    activeforeground="#cceccc",
    cursor="hand2",
    command=lambda: switch_to_output()
)
back_btn.place(x=40, y=45)

separator = tk.Frame(header, bg="#f5fef0", width=2, height=60)
separator.place(x=260, y=45)  
title_frame = tk.Frame(header, bg="#0c4e0b")  
title_frame.place(x=290, y=20)
tk.Label(title_frame, text=crop_name.title(), font=("Segoe UI", 30, "bold"), bg="#0c4e0b", fg="#f5fef0").pack(anchor="w")
tk.Label(
    title_frame,
    text="Crop Growth Recommendation",
    font=("Segoe UI", 16),
    bg="#0c4e0b",
    fg="#dcebd7"
).pack(anchor="w", pady=(2, 0))

def add_rounded_corners(img, radius):
    rounded = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(rounded)
    draw.rounded_rectangle((0, 0) + img.size, radius=radius, fill=255)
    img = img.convert("RGBA")
    img.putalpha(rounded)
    return img

try:
    for ext in [".png", ".jpg", ".jpeg"]:
        image_path = relative_to_assets(f"{crop_name.lower()}{ext}")
        if image_path.exists():
            image = Image.open(image_path)
            break
    else:
        raise FileNotFoundError(f"No image found for crop '{crop_name}' with .png or .jpg")
    
    image = image.resize((250, 125))
    image = add_rounded_corners(image, radius=20)  # <-- rounded corners applied
    img_tk = ImageTk.PhotoImage(image)
    image_label = tk.Label(header, image=img_tk, bg="#0c4e0b")

except Exception as e:
    print(f"Error loading image for crop '{crop_name}': {e}")
    image_label = tk.Label(header, text="[Image]", width=25, height=7, bg="#0c4e0b", fg="#444")

image_label.place(relx=1.0, x=-85, y=15, anchor="ne")


main_area = tk.Frame(root, bg=BG)
main_area.pack(pady=5)
def create_card(parent, icon_path, title, number, desc):
    canvas = tk.Canvas(parent, width=CARD_WIDTH, height=CARD_HEIGHT, bg=BG, highlightthickness=0)
    r = CARD_RADIUS
    canvas.create_polygon(
        r, 0, CARD_WIDTH - r, 0,
        CARD_WIDTH, 0, CARD_WIDTH, r,
        CARD_WIDTH, CARD_HEIGHT - r, CARD_WIDTH, CARD_HEIGHT,
        CARD_WIDTH - r, CARD_HEIGHT, r, CARD_HEIGHT,
        0, CARD_HEIGHT, 0, CARD_HEIGHT - r,
        0, r, 0, 0,
        fill=CARD_BG, outline=BORDER_COLOR, smooth=True
    )
    

    try:
        img = Image.open(relative_to_assets(icon_path)).resize((24, 24), Image.Resampling.LANCZOS)
        print(relative_to_assets(icon_path))
        img_tk = ImageTk.PhotoImage(img)
        canvas.image = img_tk  
        canvas.create_image(20, 20, anchor="w", image=img_tk)
    except Exception as e:
        print(f"Failed to load image {icon_path}: {e}")
        canvas.create_text(20, 20, text="[X]", font=ICON_FONT, anchor="w", fill="#FF0000")


    canvas.create_text(55, 20, text=title, font=TITLE_FONT, anchor="w", fill="#1e1e1e")
    canvas.create_text(20, 45, text=number, font=NUMBER_FONT, anchor="w", fill="#333")
    canvas.create_text(20, 67, text=desc, font=DESC_FONT, anchor="w", fill="#555")

    return canvas

stage_icons_refs = [] 
def add_stage(stage_title, cards_info, stage_icon):
    stage_frame = tk.Frame(main_area, bg=BG)
    stage_frame.pack(anchor="w", pady=(10, 4), padx=5)
    try:
        icon_path = relative_to_assets(stage_icon)
        icon_img = Image.open(icon_path).resize((24, 24), Image.Resampling.LANCZOS)
        icon_tk = ImageTk.PhotoImage(icon_img)
        stage_icons_refs.append(icon_tk)  

        icon_label = tk.Label(stage_frame, image=icon_tk, bg=BG)
        icon_label.pack(side="left", padx=(0, 5))
    except Exception as e:
        print(f"Error loading stage icon {stage_icon}: {e}")
        icon_label = tk.Label(stage_frame, text="[X]", bg=BG)
        icon_label.pack(side="left", padx=(0, 5))

    tk.Label(stage_frame, text=stage_title, font=SECTION_FONT, bg=BG, fg="#0CA60A").pack(side="left")

    row = tk.Frame(main_area, bg=BG)
    row.pack()
    for info in cards_info:
        card = create_card(row, *info)
        card.pack(side="left", padx=12, pady=5)

add_stage("Germination Stage", [
    ("water.png", "Irrigation", stage1['environmental']['irrigation_frequency'][1],
     get_irrigation_desc(stage1['environmental']['irrigation_frequency'][1])),
    ("leaf2.png", "Fertilizer", stage1['environmental']['fertilizer_usage'][1],
     get_fertilizer_desc(stage1['environmental']['fertilizer_usage'][1])),
    ("pest.png", "Pest Pressure", stage1['environmental']['pest_pressure'][1],
     get_pest_desc(stage1['environmental']['pest_pressure'][1]))
], "leaf.png")

add_stage("Vegetative Stage", [
    ("water.png", "Irrigation", stage2['environmental']['irrigation_frequency'][1],
     get_irrigation_desc(stage2['environmental']['irrigation_frequency'][1])),
    ("leaf2.png", "Fertilizer", stage2['environmental']['fertilizer_usage'][1],
     get_fertilizer_desc(stage2['environmental']['fertilizer_usage'][1])),
    ("pest.png", "Pest Pressure", stage2['environmental']['pest_pressure'][1],
     get_pest_desc(stage2['environmental']['pest_pressure'][1]))
], "wheat.png")

add_stage("Reproductive Stage", [
    ("water.png", "Irrigation", stage3['environmental']['irrigation_frequency'][1],
     get_irrigation_desc(stage3['environmental']['irrigation_frequency'][1])),
    ("leaf2.png", "Fertilizer", stage3['environmental']['fertilizer_usage'][1],
     get_fertilizer_desc(stage3['environmental']['fertilizer_usage'][1])),
    ("pest.png", "Pest Pressure", stage3['environmental']['pest_pressure'][1],
     get_pest_desc(stage3['environmental']['pest_pressure'][1]))
], "plant.png")

root.mainloop()

