from tkinter import Tk, Canvas, PhotoImage
from pathlib import Path
import subprocess
import sys
from functools import partial
from tkinter import ttk

def scale_font(base_size, reference_width=1920):
    current_width = window.winfo_width()
    return max(int(base_size * (current_width / reference_width)), base_size//2)

def scale_value(value, reference_width=1920):
    return int(value * (window.winfo_width() / reference_width))

def debounce(wait):
    """Prevent rapid-fire updates during resize"""
    def decorator(fn):
        def debounced(*args, **kwargs):
            if hasattr(debounced, '_timer'):
                window.after_cancel(debounced._timer)
            debounced._timer = window.after(wait, lambda: fn(*args, **kwargs))
        return debounced
    return decorator

# First, add the rounded rectangle function to Canvas class
def create_round_rect(self, x1, y1, x2, y2, radius=15, **kwargs):
    points = [
        x1+radius, y1, x2-radius, y1, x2, y1,
        x2, y1+radius, x2, y2-radius, x2, y2,
        x2-radius, y2, x1+radius, y2, x1, y2,
        x1, y2-radius, x1, y1+radius, x1, y1,
        x1+radius, y1
    ]
    return self.create_polygon(points, **kwargs, smooth=True)

Canvas.create_round_rect = create_round_rect

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"./assets/home")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

window = Tk()
window.geometry("1122x630")
window.resizable(False, False)

canvas = Canvas(window, bg="#ffffff", bd=0, highlightthickness=0, relief="ridge")
canvas.pack(fill="both", expand=True)

try:
    bg_image = PhotoImage(file=relative_to_assets("bgImage.png"))
    has_images = True
except Exception as e:
    print(f"Image loading error: {e}")
    bg_image = None
    has_images = False

# Create UI elements
if has_images:
    bg_image_id = canvas.create_image(0, 0, anchor="nw", image=bg_image)
else:
    canvas.config(bg="#ffffff")
    bg_image_id = None

text_title = canvas.create_text(0, 0, text="", fill="#FFFFFF", anchor="center")
text_subtitle = canvas.create_text(0, 0, text="", fill="#FFFFFF", anchor="center")
line_top = canvas.create_rectangle(0, 0, 0, 0, fill="#FFFFFF", outline="")
line_bottom = canvas.create_rectangle(0, 0, 0, 0, fill="#FFFFFF", outline="")

# Button configuration - Single color for both buttons
BUTTON_CONFIG = [
    {
        "x1": 350, "y1": 430,
        "x2": 550, "y2": 480,
        "radius": 15,
        "fill": "#094909",  
        "text": "Get Started",
        "text_color": "white",
        "font": ("Inter Bold", 16),
        "command": lambda: switch_to_input(window)
    },
    {
        "x1": 600, "y1": 430,
        "x2": 800, "y2": 480,
        "radius": 15,
        "fill": "white",  
        "text": "Learn More",
        "text_color": "#149D14",
        "font": ("Inter Bold", 16),
        "command": lambda: switch_to_LearnMore(window)
    }
]

buttons = []
for config in BUTTON_CONFIG:
    # Create button background
    button_bg = canvas.create_round_rect(
        config["x1"], config["y1"],
        config["x2"], config["y2"],
        radius=config["radius"],
        fill=config["fill"],
        outline=""
    )
    
    # Create button text
    button_text = canvas.create_text(
        (config["x1"] + config["x2"]) // 2,
        (config["y1"] + config["y2"]) // 2,
        text=config["text"],
        fill=config["text_color"],
        font=config["font"]
    )
    
    # Store button elements
    buttons.append((button_bg, button_text, config))

@debounce(50)
def update_layout(event=None):
    win_width = window.winfo_width()
    win_height = window.winfo_height()
    
    # Calculate responsive dimensions
    title_font_size = scale_font(70)
    subtitle_font_size = scale_font(25)
    title_y = scale_value(410)
    subtitle_y = scale_value(575)
    line_y1 = scale_value(525)
    line_y2 = scale_value(630)
    
    # Update text elements
    canvas.itemconfig(text_title,
                     text="crop recommendation system",
                     font=("Anton Regular", title_font_size))
    canvas.coords(text_title, win_width/2, title_y)
    
    canvas.itemconfig(text_subtitle,
                     text="Optimize your farm's productivity with science-backed crop choices.",
                     font=("Inter ExtraLightItalic", subtitle_font_size))
    canvas.coords(text_subtitle, win_width/2, subtitle_y)
    
    # Update lines
    canvas.coords(line_top, win_width*0.23, line_y1, win_width*0.77, line_y1)
    canvas.coords(line_bottom, win_width*0.2, line_y2, win_width*0.8, line_y2)
    
    # Update background image if exists
    if bg_image_id:
        canvas.coords(bg_image_id, 0, 0)

def switch_to_input(win):
    win.destroy()
    subprocess.Popen([sys.executable, "./src/app/frontend/input.py"])

def switch_to_LearnMore(win):
    win.destroy()
    subprocess.Popen([sys.executable, "./src/app/frontend/LearnMore.py"])

# Simplified button interaction - no color change
def on_enter(event):
    window.config(cursor="hand2")

def on_leave(event):
    window.config(cursor="")

# Bind events to buttons
for button_bg, button_text, config in buttons:
    canvas.tag_bind(button_bg, "<Button-1>", lambda e, c=config: c["command"]())
    canvas.tag_bind(button_text, "<Button-1>", lambda e, c=config: c["command"]())
    
    canvas.tag_bind(button_bg, "<Enter>", on_enter)
    canvas.tag_bind(button_bg, "<Leave>", on_leave)
    
    canvas.tag_bind(button_text, "<Enter>", on_enter)
    canvas.tag_bind(button_text, "<Leave>", on_leave)

# Initial setup
update_layout()
window.bind("<Configure>", update_layout)
window.protocol("WM_DELETE_WINDOW", window.quit)

window.mainloop()