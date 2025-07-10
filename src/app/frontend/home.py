from tkinter import Tk, Canvas, Button, PhotoImage
from pathlib import Path
import subprocess
import sys
from functools import partial
from tkinter import ttk

# Helper functions for responsive design
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

# Path configuration
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"./assets/home")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)



window = Tk()
window.geometry("1122x630")
window.resizable(False, False)

canvas = Canvas(window, bg="#ffffff", bd=0, highlightthickness=0, relief="ridge")
canvas.pack(fill="both", expand=True)

# Load images
try:
    bg_image = PhotoImage(file=relative_to_assets("bgImage.png"))
    button_img1 = PhotoImage(file=relative_to_assets("button_1.png"))
    button_img2 = PhotoImage(file=relative_to_assets("button_2.png"))
    has_images = True
except Exception as e:
    print(f"Image loading error: {e}")
    bg_image = None
    button_img1 = None
    button_img2 = None
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

# Replace buttons with clickable image items
if has_images:
    button1_id = canvas.create_image(0, 0, anchor="nw", image=button_img1)
    button2_id = canvas.create_image(0, 0, anchor="nw", image=button_img2)
    
    # Bind click events and cursor changes
    for btn_id in [button1_id, button2_id]:
        canvas.tag_bind(btn_id, "<Button-1>", 
                      lambda e, b=btn_id: switch_to_input(window) if b == button1_id else switch_to_LearnMore(window))
        canvas.tag_bind(btn_id, "<Enter>", lambda e: window.config(cursor="hand2"))
        canvas.tag_bind(btn_id, "<Leave>", lambda e: window.config(cursor=""))

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
    button_y = scale_value(718)
    
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
    
    # Update image positions instead of buttons
    if has_images:
        # Get original image dimensions (or set your preferred sizes)
        button1_width = button_img1.width()
        button2_width = button_img2.width()
        
        # Set your desired fixed spacing between buttons (in pixels)
        fixed_spacing = 20  # Adjust this value as needed
        
        # Calculate total width of both buttons plus spacing
        total_width = button1_width + button2_width + fixed_spacing
        
        # Calculate starting x position to center the group
        start_x = (win_width - total_width) / 2
        
        # Position the first button
        button1_x = start_x
        button1_y = scale_value(780)
        
        # Position the second button with fixed spacing
        button2_x = start_x + button1_width + fixed_spacing
        button2_y = scale_value(780)
        
        canvas.coords(button1_id, button1_x, button1_y)
        canvas.coords(button2_id, button2_x, button2_y)
    
    # Update background image if exists
    if bg_image_id:
        canvas.coords(bg_image_id, 0, 0)

def switch_to_input(win):
    win.destroy()
    subprocess.Popen([sys.executable, "./src/app/frontend/input.py"])

def switch_to_LearnMore(win):
    win.destroy()
    subprocess.Popen([sys.executable, "./src/app/frontend/LearnMore.py"])

# Initial setup
update_layout()
window.bind("<Configure>", update_layout)
window.protocol("WM_DELETE_WINDOW", window.quit)

window.mainloop()