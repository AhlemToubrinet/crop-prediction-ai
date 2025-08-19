from pathlib import Path
import json
from tkinter import messagebox
from tkinter import Tk, Canvas, Entry
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
    for section_title, fields, section_title_y in sections:
        canvas.create_text(135, section_title_y + y_shift, anchor="nw",
                           text=section_title, fill="#0c4e0b", font=("Segoe UI", 18, "bold"))
        for label, x, y in fields:
            canvas.create_text(x-5, y - 33 + y_shift, anchor="nw",
                               text=label, fill="#000000", font=("Segoe UI", 11))
            
            create_rounded_rect(canvas, x-5, y + y_shift-2, x+165, y + y_shift+28,
                                radius=10, fill="#ffffff", outline="#cccccc")
            entry = Entry(bd=0, bg="#ffffff", fg="#000716", highlightthickness=0)
            entry.place(x=x, y=y + y_shift, width=157.0, height=24)
            entries.append(entry)

    
    btn2_bg = create_rounded_rect(canvas, 595, 580, 715, 625,
                                  radius=25, fill="white", outline="#A6AEA6", width=1)
    btn2_text = canvas.create_text((595+715)//2 + 5, (580+625)//2,
                                   text="Reset", fill="#2A362A",
                                   font=("Segoe UI", 12, "bold"))

    
    btn2_icon = canvas.create_text((595+715)//2 - 35, (580+625)//2,
                                   text="⟲", fill="#2A362A",
                                   font=("Segoe UI", 14, "bold"))

    
    btn1_bg = create_rounded_rect(canvas, 735, 580, 1010, 625,
                                  radius=25, fill="#127512", outline="")
    btn1_text = canvas.create_text((735+1010)//2 - 10, (580+625)//2,
                                   text="Get Recommendation", fill="white",
                                   font=("Segoe UI", 12, "bold"))

    
    btn1_icon = canvas.create_text((735+1010)//2 + 100, (580+625)//2,
                                   text="🡺", fill="white",
                                   font=("Segoe UI", 14, "bold"))

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

    
    for btn in [(btn1_bg, btn1_text, on_button1_click),
                (btn2_bg, btn2_text, on_button2_click)]:
        bg, text, func = btn
        canvas.tag_bind(bg, "<Button-1>", func)
        canvas.tag_bind(text, "<Button-1>", func)
        canvas.tag_bind(bg, "<Enter>", lambda e: window.config(cursor="hand2"))
        canvas.tag_bind(text, "<Enter>", lambda e: window.config(cursor="hand2"))
        canvas.tag_bind(bg, "<Leave>", lambda e: window.config(cursor=""))
        canvas.tag_bind(text, "<Leave>", lambda e: window.config(cursor=""))

    window.mainloop()
