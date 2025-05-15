from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, StringVar, Radiobutton
import subprocess
import sys

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(
    r"C:\Users\OASIS\Desktop\Tkinter-Designer-master\CropRecomApp\build\assets\frame0"
)


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


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

canvas.create_text(
    16.0,
    20.0,
    anchor="nw",
    text="📊 Your Crop Recommendations",
    fill="#5BC893",
    font=("Inter SemiBoldItalic", 26 * -1),
)
canvas.create_text(
    40.0,
    73.0,
    anchor="nw",
    text="the best crop for your land is :  ",
    fill="#848484",
    font=("Inter SemiBoldItalic", 18 * -1),
)

# Dummy crop recommendations
for i, y in enumerate([239, 283, 326, 369, 413, 456]):
    canvas.create_text(
        530.0,
        y,
        anchor="nw",
        text="99.87     ",
        fill="#848484",
        font=("Inter SemiBoldItalic", 16 * -1),
    )
    canvas.create_text(
        376.0,
        y,
        anchor="nw",
        text="0.0013",
        fill="#848484",
        font=("Inter SemiBoldItalic", 16 * -1),
    )
    canvas.create_text(
        171.0,
        y,
        anchor="nw",
        text="Excellent",
        fill="#848484",
        font=("Inter SemiBoldItalic", 16 * -1),
    )
    canvas.create_text(
        69.0,
        y,
        anchor="nw",
        text="Appple ",
        fill="#3AB67D",
        font=("Inter SemiBoldItalic", 16 * -1),
    )

canvas.create_rectangle(39.0, 202.0, 780.0, 203.0, fill="#A8A8A8", outline="")
canvas.create_rectangle(39.0, 158.0, 780.0, 159.0, fill="#A8A8A8", outline="")

canvas.create_text(
    489.0,
    172.0,
    anchor="nw",
    text=" Match Percentage",
    fill="#848484",
    font=("Inter Medium", 15 * -1),
)
canvas.create_text(
    360.0,
    172.0,
    anchor="nw",
    text="Cost Score",
    fill="#848484",
    font=("Inter Medium", 15 * -1),
)
canvas.create_text(
    165.0,
    172.0,
    anchor="nw",
    text="Suitability",
    fill="#848484",
    font=("Inter Medium", 15 * -1),
)
canvas.create_text(
    84.0,
    172.0,
    anchor="nw",
    text="Crop",
    fill="#848484",
    font=("Inter Medium", 15 * -1),
)


def switch_to_output():
    window.destroy()
    subprocess.Popen(
        [
            sys.executable,
            "C:/Users/OASIS/Desktop/Tkinter-Designer-master/CropRecomApp/build/More.py",
        ]
    )


# Button images and placement
for i, y in enumerate([234, 276, 321, 364, 408, 451], start=1):
    button_image = PhotoImage(file=relative_to_assets(f"button_{i}.png"))
    button = Button(
        image=button_image,
        borderwidth=0,
        highlightthickness=0,
        command=switch_to_output,
    )

    button.image = button_image  # Prevent garbage collection
    button.place(x=661.0, y=y, width=90.0, height=30.0)

# Search method labels on canvas
canvas.create_text(
    737.0,
    61.0,
    anchor="nw",
    text="Greedy Search",
    fill="#848484",
    font=("Inter SemiBoldItalic", 14 * -1),
)
canvas.create_text(
    754.0,
    36.0,
    anchor="nw",
    text="A* Search",
    fill="#848484",
    font=("Inter SemiBoldItalic", 14 * -1),
)
canvas.create_text(
    739.0,
    87.0,
    anchor="nw",
    text="Genetic algo\n",
    fill="#848484",
    font=("Inter SemiBoldItalic", 14 * -1),
)
canvas.create_text(
    739.0,
    115.0,
    anchor="nw",
    text="CSP",
    fill="#848484",
    font=("Inter SemiBoldItalic", 14 * -1),
)
canvas.create_text(
    630.0,
    7.0,
    anchor="nw",
    text="Choose Recommendation Method",
    fill="#848484",
    font=("Inter SemiBoldItalic", 16 * -1),
)


# ✅ RADIO BUTTONS FOR METHOD SELECTION
selected_method = StringVar(value="A*")

methods = {
    "A*": (710, 36),
    "Greedy Search": (710, 62),
    "Genetic Algorithm": (710, 89),
    "CSP": (710, 116),
}

for method, (x, y) in methods.items():
    rb = Radiobutton(
        window,
        text="",  # invisible text
        variable=selected_method,
        value=method,
        bg="#FFFFFF",
        activebackground="#FFFFFF",
        highlightthickness=0,
    )
    rb.place(x=x, y=y)


button_image_7 = PhotoImage(file=relative_to_assets("button_7.png"))


def switch_to_input():
    window.destroy()
    subprocess.Popen(
        [
            sys.executable,
            "C:/Users/OASIS/Desktop/Tkinter-Designer-master/CropRecomApp/build/input.py",
        ]
    )


button_7 = Button(
    image=button_image_7,
    borderwidth=0,
    highlightthickness=0,
    command=switch_to_input,
)

button_7.image = button_image_7  # Prevent garbage collection
button_7.place(x=699.0, y=530.0, width=171.0, height=48.0)


window.resizable(False, False)
window.mainloop()
