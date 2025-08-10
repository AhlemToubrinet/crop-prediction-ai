from pathlib import Path
from tkinter import Tk, Canvas, PhotoImage
import subprocess
import sys

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"./assets/LearnMore")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

# First, add the rounded rectangle function to Canvas class
def create_round_rect(self, x1, y1, x2, y2, radius=25, **kwargs):
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
        x1, y1,
        x1+radius, y1
    ]
    return self.create_polygon(points, **kwargs, smooth=True)

Canvas.create_round_rect = create_round_rect

window = Tk()
window.geometry("1122x650")
window.configure(bg="#FFFFFF")

canvas = Canvas(
    window,
    bg="#FFFFFF",
    height=630,
    width=1122,
    bd=0,
    highlightthickness=0,
    relief="ridge",
)
canvas.place(x=0, y=0)

# Background images
image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(780.0, 330.0, image=image_image_1)

image_image_2 = PhotoImage(file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(352.0, 330.0, image=image_image_2)

# Text content (unchanged from your original)
canvas.create_text(
    210.0,
    160.0,
    anchor="nw",
    text="Boost your farm's potential with intelligent crop ",
    fill="#000000",
    font=("Inter SemiBoldItalic", 17 * -1),
)
canvas.create_text(
    210.0,
    200.0,
    anchor="nw",
    text="recommendations.",
    fill="#000000",
    font=("Inter SemiBoldItalic", 17 * -1),
)
canvas.create_text(
    210.0,
    240.0,
    anchor="nw",
    text="Our system analyzes your soil and weather conditions using ",
    fill="#000000",
    font=("Inter SemiBoldItalic", 17 * -1),
)
canvas.create_text(
    210.0,
    280.0,
    anchor="nw",
    text="AI-powered algorithms to suggest the most suitable crops ",
    fill="#000000",
    font=("Inter SemiBoldItalic", 17 * -1),
)
canvas.create_text(
    210.0,
    320.0,
    anchor="nw",
    text="for your land. With better decisions, you save time, conserve",
    fill="#000000",
    font=("Inter SemiBoldItalic", 17 * -1),
)
canvas.create_text(
    210.0,
    360.0,
    anchor="nw",
    text="conserve resources, and increase your yield.",
    fill="#000000",
    font=("Inter SemiBoldItalic", 17 * -1),
)

# Create rectangular button with rounded corners
button_bg = canvas.create_round_rect(
    300, 430,  
    500, 480,  
    radius=15,  
    fill="#094909",  
    outline=""
)

button_text = canvas.create_text(
    400, 455,
    text="Start Now",
    fill="white",
    font=("Inter Bold", 16)
)

def on_button_click(event):
    window.destroy()
    subprocess.Popen([sys.executable, "./src/app/frontend/input.py"])

# Make the button clickable
canvas.tag_bind(button_bg, "<Button-1>", on_button_click)
canvas.tag_bind(button_text, "<Button-1>", on_button_click)

# Hover effects
def on_enter(event):
    canvas.itemconfig(button_bg, fill="#094909")  # Slightly darker blue
    window.config(cursor="hand2")

def on_leave(event):
    canvas.itemconfig(button_bg, fill="#094909")  # Original blue
    window.config(cursor="")

canvas.tag_bind(button_bg, "<Enter>", on_enter)
canvas.tag_bind(button_bg, "<Leave>", on_leave)
canvas.tag_bind(button_text, "<Enter>", on_enter)
canvas.tag_bind(button_text, "<Leave>", on_leave)

window.resizable(False, False)
window.mainloop()