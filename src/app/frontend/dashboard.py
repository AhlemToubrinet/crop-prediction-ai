import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os

class AlgorithmDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Algorithm Comparison Dashboard")
        
        # Configure main window
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Create header
        self.create_header()
        
        # Create comparison frame
        self.create_comparison_frame()
        
        # Load and display images
        self.load_images()
        
        # Create zoom window (initially hidden)
        self.zoom_window = None
        self.zoom_image = None
        
    def create_header(self):
        header_frame = tk.Frame(self.root, bg='#82C959', height=80)
        header_frame.pack(fill='x', padx=10, pady=10)
        
        title_label = tk.Label(
            header_frame, 
            text="Algorithm Performance Comparison",
            font=('Helvetica', 20, 'bold'),
            fg='white',
            bg='#82C959'
        )
        title_label.pack(pady=20)
        
    def create_comparison_frame(self):
        comparison_frame = tk.Frame(self.root, bg='#f0f0f0')
        comparison_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create 4 frames for the images
        self.image_frames = []
        self.image_labels = []  # To store references to image labels
        for i in range(4):
            frame = tk.Frame(
                comparison_frame,
                bg='white',
                bd=2,
                relief='groove',
                padx=10,
                pady=10
            )
            frame.grid(row=i//2, column=i%2, padx=10, pady=10, sticky='nsew')
            self.image_frames.append(frame)
            
            # Configure grid weights
            comparison_frame.grid_rowconfigure(i//2, weight=1)
            comparison_frame.grid_columnconfigure(i%2, weight=1)
            
        # Add titles for each frame
        algo_names = [
            "Actual Memory Usage Comparison",
            "Execution Time Comparison",
            "Comparison of Weighted Scoring",
            "Comparison of Environmental Performance"
        ]

        for i, frame in enumerate(self.image_frames):
            title = tk.Label(
                frame,
                text=algo_names[i],
                font=('Helvetica', 14, 'bold'),
                bg='white'
            )
            title.pack(pady=(0, 10))
            
    def load_images(self):
        # Replace these with your actual image paths
        image_paths = [
            "src/app/frontend/assets/dashboard/graph3.jpg",
            "src/app/frontend/assets/dashboard/graph1.jpg",
            "src/app/frontend/assets/dashboard/graph4.jpg",
            "src/app/frontend/assets/dashboard/graph2.jpg"
        ]
        
        # Store original images for zooming
        self.original_images = []
        
        for i, frame in enumerate(self.image_frames):
            # Make graph3 (index 0) smaller than the others
          

            try:
                # Load original image (keep reference)
                img = Image.open(image_paths[i])
                self.original_images.append(img)
                # Create resized version for display
                display_img = img.resize((500, 265), Image.LANCZOS)
                photo = ImageTk.PhotoImage(display_img)

                # Create label for image
                img_label = tk.Label(frame, image=photo, bg='white')
                img_label.image = photo  # Keep reference
                img_label.pack(fill='both', expand=True)
                
                # Bind hover events
                img_label.bind("<Enter>", lambda e, idx=i: self.show_zoom(idx))
                img_label.bind("<Leave>", lambda e: self.hide_zoom())
                
            except FileNotFoundError:
                # Create placeholder if image not found
                placeholder = tk.Label(
                    frame,
                    text=f"Image {image_paths[i]} not found\n(Placeholder for chart)",
                    font=('Helvetica', 12),
                    bg='white',
                    fg='gray'
                )
                placeholder.pack(fill='both', expand=True)
                self.original_images.append(None)  # Keep index alignment
    
   

if __name__ == "__main__":
    root = tk.Tk()
    app = AlgorithmDashboard(root)
    root.mainloop()