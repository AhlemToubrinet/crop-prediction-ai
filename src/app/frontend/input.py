from pathlib import Path
import json
from tkinter import messagebox
from tkinter import Tk, Canvas, Entry, PhotoImage
import subprocess
import sys
from PIL import Image, ImageTk

# shared_data.py (or inside dashboard.py / main.py)
shared_input = None


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
            ("Pest Pressure", 800, 420),
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

