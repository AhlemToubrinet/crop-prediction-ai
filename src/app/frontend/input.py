from pathlib import Path
import json
from tkinter import messagebox
from tkinter import Tk, Canvas, Entry, PhotoImage
import subprocess
import sys
from PIL import Image, ImageTk

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"./assets/input")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

if __name__ == "__main__":
    window = Tk()
    window.geometry("1122x650")
    window.resizable(False, False)
    window.configure(bg="#f5fef0")

    canvas = Canvas(
        window,
        bg="#f5fef0",
        height=630,
        width=1122,
        bd=0,
        highlightthickness=0,
        relief="ridge",
    )
    canvas.place(x=0, y=0)
    canvas.create_rectangle(0, 0, 1122, 80, fill="#0c4e0b", outline="")
    
    try:
        pil_logo = Image.open(relative_to_assets("leaf.png")).resize((80, 80), Image.Resampling.LANCZOS)
        logo_img = ImageTk.PhotoImage(pil_logo)
        canvas.create_image(80, 40, anchor="center", image=logo_img)
    except Exception:
        pass

    canvas.create_text(130, 26, anchor="w", text="Farm Data Input", fill="#ffffff", font=("Segoe UI", 18,"bold"))
    canvas.create_text(130, 50, anchor="w", text="Fill in your farm’s soil/weather data.", fill="#e7f3e6", font=("Segoe UI", 10))

    def create_rounded_rect(canvas, x1, y1, x2, y2, radius=25, **kwargs):
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1
        ]
        return canvas.create_polygon(points, smooth=True, **kwargs)

    create_rounded_rect(canvas, 50, 80, 1065, 650, radius=30, fill="#ffffff", outline="")
    y_shift = 88
    field_offset = 20  
    sections = [
        ("Soil Properties", [
            ("Nitrogen (ppm)", 150, 70),
            ("Phosphorus (ppm)", 370, 70),
            ("Potassium (ppm)", 580, 70),
            ("pH Level", 800, 70),
            ("Organic Matter (%)", 150, 150),
            ("Soil Moisture (%)", 370, 150),
        ], 0),
        ("Climate Conditions", [
            ("Temperature (C)", 150, 290),
            ("Humidity (%)", 370, 290),
            ("Rainfall (mm/mo)", 580, 290),
            ("Sunlight Exposure (h/day)", 800, 290),
        ], 210),
        ("Environmental Factors", [
            ("Irrigation Frequency (x/week)", 150,420),
            ("Water Usage Efficiency (1–5)", 370, 420),
            ("Fertilizer Usage", 580, 420),
            ("Pest Pressure (1–5)", 800, 420),
        ], 340),
    ]

    entries = []
    entry_size = (190, 46)
    entry_img_path = relative_to_assets("entry.png")

    try:
        pil_entry = Image.open(entry_img_path).resize(entry_size, Image.Resampling.LANCZOS)
        entry_img = ImageTk.PhotoImage(pil_entry)
    except Exception:
        entry_img = None

    for section_title, fields, section_title_y in sections:
        canvas.create_text(135, section_title_y + y_shift, anchor="nw", text=section_title, fill="#0c4e0b", font=("Segoe UI", 18, "bold"))
        for label, x, y in fields:
            canvas.create_text(x-13, y - 33 + y_shift, anchor="nw", text=label, fill="#000000", font=("Segoe UI", 11))
            if entry_img:
                canvas.create_image(x + 78.5, y + 13 + y_shift, image=entry_img)
            entry = Entry(bd=0, bg="#ffffff", fg="#000716", highlightthickness=0)
            entry.place(x=x, y=y + y_shift, width=157.0, height=24)
            entries.append(entry)
    try:
        button_image_1 = PhotoImage(file=relative_to_assets("button.png"))
        button_image_2 = PhotoImage(file=relative_to_assets("reset.png"))
        button1_img = canvas.create_image(735, 580, anchor="nw", image=button_image_1)
        button2_img = canvas.create_image(595, 580, anchor="nw", image=button_image_2)
    except Exception:
        from tkinter import Button
        button1 = Button(window, text="Get Recommendation", bg="#0c4e0b", fg="white", font=("Segoe UI", 12, "bold"))
        button1.place(x=900, y=570, width=150, height=40)
        button2 = Button(window, text="Reset", bg="#e0e0e0", fg="#0c4e0b", font=("Segoe UI", 12, "bold"))
        button2.place(x=770, y=570, width=120, height=40)
        button1_img = button2_img = None

    def collect_inputs():
        try:
            values = [e.get().strip() for e in entries]
            if "" in values:
                messagebox.showerror("Missing Data", "Please fill in all fields!", parent=window)
                return None
            float_values = list(map(float, values))
            return {
                'soil': {
                    'n': float_values[0], 'p': float_values[1], 'k': float_values[2],
                    'ph': float_values[3], 'organic_matter': float_values[4], 'soil_moisture': float_values[5]
                },
                'climate': {
                    'temperature': float_values[6], 'humidity': float_values[7],
                    'rainfall': float_values[8], 'sunlight_exposure': float_values[9]
                },
                'environmental': {
                    'irrigation_frequency': float_values[10], 'water_usage_efficiency': float_values[11],
                    'fertilizer_usage': float_values[12], 'pest_pressure': float_values[13]
                }
            }
        except ValueError:
            messagebox.showerror("Invalid Data", "Please enter valid numbers!", parent=window)
            return None

    def on_button1_click(event=None):
        input_data = collect_inputs()
        if input_data:
            window.destroy()
            subprocess.Popen([sys.executable, "./src/app/frontend/output.py", json.dumps(input_data)])

    def on_button2_click(event=None):
        for entry in entries:
            entry.delete(0, 'end')
        entries[0].focus_set()

    if 'button1_img' in locals() and 'button2_img' in locals() and button1_img and button2_img:
        canvas.tag_bind(button1_img, "<Button-1>", on_button1_click)
        canvas.tag_bind(button2_img, "<Button-1>", on_button2_click)
        for btn in [button1_img, button2_img]:
            canvas.tag_bind(btn, "<Enter>", lambda e: window.config(cursor="hand2"))
            canvas.tag_bind(btn, "<Leave>", lambda e: window.config(cursor=""))
    else:
        button1.config(command=on_button1_click)
        button2.config(command=on_button2_click)

    window.mainloop()

""" 
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path
import json
from tkinter import messagebox

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
import subprocess
import sys

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(
    r"./assets/input"
)


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

if __name__ == "__main__":
    
    window = Tk()

    window.geometry("1122x630")
    window.resizable(False, False)
    window.configure(bg="#f5fef0")
    
    
    canvas = Canvas(
        window,
        bg="#f5fef0",
        height=630,
        width=1122,
        bd=0,
        highlightthickness=0,
        relief="ridge",
    )

    canvas.place(x=0, y=0)
    canvas.create_rectangle(26.5, 220.5, 860.0, 221.0, fill="#f5fef0", outline="")

    canvas.create_rectangle(26.5, 362.5, 868.0, 363.0, fill="#f5fef0", outline="")

    canvas.create_text(
        435.0,
        15.0,
        anchor="nw",
        text="Fill in your farm’s soil/weather data.",
        fill="#000000",
        font=("Inter", 14 * -1),
    )
    # --------------------------------------------------
    canvas.create_text(
        27.0,
        29.0,
        anchor="nw",
        text="Soil Properties",
        fill="#000000",
        font=("Inter MediumItalic", 17 * -1),
    )
    # first line
    entry_image_1 = PhotoImage(file=relative_to_assets("entry.png"))
    entry_bg_1 = canvas.create_image(118.5, 115, image=entry_image_1)
    entry_1 = entry_1 = Entry(bd=0,bg="#ffffff", fg="#000716",highlightthickness=0)
    entry_1.place(x=40.0, y=108, width=160.0, height=15)

    canvas.create_text(
        40.0, 70.0, anchor="nw", text="Nitrogen (pmm)", fill="#000000", font=("Inter", 12 * -1),
    )

    entry_image_2 = PhotoImage(file=relative_to_assets("entry.png"))
    entry_bg_2 = canvas.create_image(387.5, 115, image=entry_image_2)
    entry_2 = Entry(bd=0, bg="#ffffff", fg="#000716", highlightthickness=0)
    entry_2.place(x=309, y=108, width=160.0, height=15)

    canvas.create_text( 
        309, 70.0, anchor="nw", text="Phosphorus (pmm)", fill="#000000", font=("Inter", 12 * -1),
    )

    entry_image_3 = PhotoImage(file=relative_to_assets("entry.png"))
    entry_bg_3 = canvas.create_image(656.5, 115, image=entry_image_3)
    entry_3 = Entry(bd=0,bg="#ffffff", fg="#000716",highlightthickness=0)
    entry_3.place(x=578, y=108, width=160.0, height=15)

    canvas.create_text(
        578.0, 70.0, anchor="nw", text="Potassium (pmm) ", fill="#000000", font=("Inter", 12 * -1),
    )

    entry_image_4 = PhotoImage(file=relative_to_assets("entry.png"))
    entry_bg_4 = canvas.create_image(925.5, 115, image=entry_image_4)
    entry_4 = Entry(bd=0,bg="#ffffff", fg="#000716",highlightthickness=0)
    entry_4.place(x=847, y=108, width=160.0, height=15)

    canvas.create_text(
        847, 70.0, anchor="nw", text="pH Level", fill="#000000", font=("Inter", 12 * -1),
    )

    # second line

    entry_image_5 = PhotoImage(file=relative_to_assets("entry.png"))
    entry_bg_5 = canvas.create_image(118.5, 200, image=entry_image_5)
    entry_5 = Entry(bd=0,bg="#ffffff", fg="#000716",highlightthickness=0)
    entry_5.place(x=40.0, y=193, width=157.0, height=15)

    canvas.create_text(
        41, 151.5, anchor="nw", text="organic matter(%)", fill="#000000", font=("Inter", 12 * -1)
    )

    entry_image_6 = PhotoImage(file=relative_to_assets("entry.png"))
    entry_bg_6 = canvas.create_image(387.5, 200, image=entry_image_6)
    entry_6 = Entry(bd=0,bg="#ffffff", fg="#000716",highlightthickness=0)
    entry_6.place(x=309, y=193, width=157.0, height=15)

    canvas.create_text(
        309, 151.5, anchor="nw", text="Soil Moisture (%)", fill="#000000", font=("Inter", 12 * -1),
    )

    # ---------------------------------
    line_top = canvas.create_rectangle(0, 0, 0, 0, fill="#4d4d4d", outline="")
    canvas.coords(line_top,27,234,1070,234)
    # ------------------------------------

    canvas.create_text(
        27.0,
        247.0,
        anchor="nw",
        text="Climate Conditions",
        fill="#000000",
        font=("Inter MediumItalic", 17 * -1),
    )

    # third line
    entry_image_7 = PhotoImage(file=relative_to_assets("entry.png"))
    entry_bg_7 = canvas.create_image(118.5, 336.5, image=entry_image_7)
    entry_7 = Entry(bd=0,bg="#ffffff", fg="#000716",highlightthickness=0)
    entry_7.place(x=40.0, y=329.5, width=157.0, height=15)

    canvas.create_text(
        40, 288.0, anchor="nw", text="temperature (C)", fill="#000000", font=("Inter", 12 * -1),
    )

    entry_image_8 = PhotoImage(file=relative_to_assets("entry.png"))
    entry_bg_8 = canvas.create_image(387.5, 336.5, image=entry_image_8)
    entry_8 = Entry(bd=0,bg="#ffffff", fg="#000716",highlightthickness=0)
    entry_8.place(x=309, y=329.5, width=157.0, height=15)

    canvas.create_text(
        309, 288.0, anchor="nw", text="humidity (%)", fill="#000000", font=("Inter", 12 * -1),
    )

    entry_image_9 = PhotoImage(file=relative_to_assets("entry.png"))
    entry_bg_9 = canvas.create_image(656.5, 336.5, image=entry_image_9)
    entry_9 = Entry(bd=0,bg="#ffffff", fg="#000716",highlightthickness=0)
    entry_9.place(x=578, y=329.5, width=157.0, height=15)

    canvas.create_text(
        578, 288.0, anchor="nw", text="rainfall (mm/mo)", fill="#000000", font=("Inter", 12 * -1),
    )

    entry_image_10 = PhotoImage(file=relative_to_assets("entry.png"))
    entry_bg_10 = canvas.create_image(925.5, 336.5, image=entry_image_9)
    entry_10 = Entry(bd=0,bg="#ffffff", fg="#000716",highlightthickness=0)
    entry_10.place(x=847, y=329.5, width=157.0, height=15)

    canvas.create_text(
        847, 288.0, anchor="nw", text="sunlight exposure(h/day)", fill="#000000", font=("Inter", 12 * -1),
    )
    # -------------------------------------
    line_bottom = canvas.create_rectangle(0, 0, 0, 0, fill="#4d4d4d", outline="")
    canvas.coords(line_bottom,27,373,1070,373)
    # -------------------------------------

    canvas.create_text(
        27.0,
        386.0,
        anchor="nw",
        text=" Environmental Factors",
        fill="#000000",
        font=("Inter MediumItalic", 17 * -1),
    )

    # fourth line
    entry_image_11 = PhotoImage(file=relative_to_assets("entry.png"))
    entry_bg_11 = canvas.create_image(118.5, 475.5, image=entry_image_6)
    entry_11 = Entry(bd=0,bg="#ffffff", fg="#000716",highlightthickness=0)
    entry_11.place(x=40.0, y=468.5, width=157.0, height=15)


    canvas.create_text(
        40,
        427.0,
        anchor="nw",
        text="irrigation frequency(x/week)",
        fill="#000000",
        font=("Inter", 12 * -1),
    )


    entry_image_12 = PhotoImage(file=relative_to_assets("entry.png"))
    entry_bg_12 = canvas.create_image(387.5, 475.5, image=entry_image_11)
    entry_12 = Entry(bd=0,bg="#ffffff", fg="#000716",highlightthickness=0)
    entry_12.place(x=309, y=468.5, width=157.0, height=15)

    canvas.create_text(
        309,
        427.0,
        anchor="nw",
        text="water usage efficiency(1–5)",
        fill="#000000",
        font=("Inter", 12 * -1),
    )

    entry_image_13 = PhotoImage(file=relative_to_assets("entry.png"))
    entry_bg_13 = canvas.create_image(656.5, 475.5, image=entry_image_12)
    entry_13 = Entry(bd=0,bg="#ffffff", fg="#000716",highlightthickness=0)
    entry_13.place(x=578, y=468.5, width=157.0, height=15)

    canvas.create_text(
        578,
        427.0,
        anchor="nw",
        text="fertilizer usage",
        fill="#000000",
        font=("Inter", 12 * -1),
    )

    entry_image_14 = PhotoImage(file=relative_to_assets("entry.png"))
    entry_bg_14 = canvas.create_image(925.5, 475.5, image=entry_image_13)
    entry_14 = Entry(bd=0,bg="#ffffff", fg="#000716",highlightthickness=0)
    entry_14.place(x=847, y=468.5, width=157.0, height=15)

    canvas.create_text(
        847,
        427.0,
        anchor="nw",
        text="pest pressure(1–5)",
        fill="#000000",
        font=("Inter", 12 * -1),
    )


    # Load button images
    button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
    button_image_2 = PhotoImage(file=relative_to_assets("button_2.png"))

    # Replace buttons with clickable images
    button1_img = canvas.create_image(850.0, 555.0, anchor="nw", image=button_image_1)
    button2_img = canvas.create_image(720.0, 555.0, anchor="nw", image=button_image_2)

    def on_button1_click(event):
        input_data = collect_inputs()
        if input_data:  
            window.destroy()
            subprocess.Popen([sys.executable, "./src/app/frontend/output.py", json.dumps(input_data)])

    def on_button2_click(event):
        #Reset all input fields to empty
        entries = [
            entry_1, entry_2, entry_3, entry_4, entry_5,
            entry_6, entry_7, entry_8, entry_9, entry_10,
            entry_11, entry_12, entry_13, entry_14
        ]
        
        for entry in entries:
            entry.delete(0, 'end')  # Clear the entry field
        
        # Optional: Set focus to first field for better UX
        entry_1.focus_set()

    # Bind click events and cursor changes
    canvas.tag_bind(button1_img, "<Button-1>", on_button1_click)
    canvas.tag_bind(button2_img, "<Button-1>", on_button2_click)

    def on_enter(event):
        window.config(cursor="hand2")

    def on_leave(event):
        window.config(cursor="")

    canvas.tag_bind(button1_img, "<Enter>", on_enter)
    canvas.tag_bind(button1_img, "<Leave>", on_leave)
    canvas.tag_bind(button2_img, "<Enter>", on_enter)
    canvas.tag_bind(button2_img, "<Leave>", on_leave)




    def collect_inputs():
        #Collect all input values from the form with validation
        entries = [
            entry_1, entry_2, entry_3, entry_4, entry_5,
            entry_6, entry_7, entry_8, entry_9, entry_10,
            entry_11, entry_12, entry_13, entry_14
        ]
        
        # Check if any field is empty
        for entry in entries:
            if not entry.get().strip():
                messagebox.showerror(
                    "Missing Data",
                    "Please fill in all fields before getting recommendations!",
                    parent=window
                )
                return None  # Return None if validation fails
        
        try:
            return {
                'soil': {
                    'n': float(entry_1.get()),
                    'p': float(entry_2.get()),
                    'k': float(entry_3.get()),
                    'ph': float(entry_4.get()),
                    'organic_matter': float(entry_5.get()),
                    'soil_moisture': float(entry_6.get())
                },
                'climate': {
                    'temperature': float(entry_7.get()),
                    'humidity': float(entry_8.get()),
                    'rainfall': float(entry_9.get()),
                    'sunlight_exposure': float(entry_10.get())
                },
                'environmental': {
                    'irrigation_frequency': float(entry_11.get()),
                    'water_usage_efficiency': float(entry_12.get()),
                    'fertilizer_usage': float(entry_13.get()),
                    'pest_pressure': float(entry_14.get())
                }
            }
        except ValueError:
            messagebox.showerror(
                "Invalid Data",
                "Please enter valid numbers in all fields!",
                parent=window
            )
            return None


    window.resizable(False, False)

    window.mainloop()
   """