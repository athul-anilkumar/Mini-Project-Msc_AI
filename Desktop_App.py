import customtkinter as ctk
import subprocess
import os
from PIL import Image

# 1. App Configuration (Modern Dark Mode)
ctk.set_appearance_mode("Dark")  
ctk.set_default_color_theme("blue")  

class TrafficOptimizerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("MARL Traffic Optimizer - Control Panel")
        self.geometry("900x650")

        # Configure grid layout (1 row, 2 columns)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # --- Sidebar ---
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(5, weight=1) # Pushes bottom text down

        self.logo_label = ctk.CTkLabel(self.sidebar, text="🚦 SUMO Control", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 30))

        self.mode_label = ctk.CTkLabel(self.sidebar, text="Select Simulation Mode:")
        self.mode_label.grid(row=1, column=0, padx=20, pady=(10, 0), sticky="w")

        # Radio Buttons for Mode Selection
        self.mode_var = ctk.StringVar(value="Trained MAPPO Agent")
        
        self.radio_ai = ctk.CTkRadioButton(self.sidebar, text="Trained MAPPO Agent", variable=self.mode_var, value="Trained MAPPO Agent")
        self.radio_ai.grid(row=2, column=0, padx=20, pady=10, sticky="w")
        
        self.radio_base = ctk.CTkRadioButton(self.sidebar, text="Fixed-Timer Baseline", variable=self.mode_var, value="Fixed-Timer Baseline")
        self.radio_base.grid(row=3, column=0, padx=20, pady=10, sticky="w")

        # Launch Button
        self.launch_btn = ctk.CTkButton(self.sidebar, text="🚀 Launch Simulation", command=self.launch_sim, height=40)
        self.launch_btn.grid(row=4, column=0, padx=20, pady=30)
        
        self.status_label = ctk.CTkLabel(self.sidebar, text="Status: Ready", text_color="gray")
        self.status_label.grid(row=6, column=0, padx=20, pady=20, sticky="s")

        # --- Main Viewport ---
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        self.header = ctk.CTkLabel(self.main_frame, text="Multi-Agent Reinforcement Learning", font=ctk.CTkFont(size=26, weight="bold"))
        self.header.pack(pady=(20, 5))
        
        self.sub_header = ctk.CTkLabel(self.main_frame, text="Dynamic Traffic Synchronization Engine", font=ctk.CTkFont(size=14), text_color="gray")
        self.sub_header.pack(pady=(0, 20))

        self.graph_btn = ctk.CTkButton(self.main_frame, text="📊 Load Performance Analytics", command=self.load_graph)
        self.graph_btn.pack(pady=10)

        # Image Display Area
        self.image_label = ctk.CTkLabel(self.main_frame, text="Click 'Load Analytics' to view final AI vs. Baseline data.")
        self.image_label.pack(pady=20, expand=True)

    # 2. Functional Logic
    def launch_sim(self):
        mode = self.mode_var.get()
        self.status_label.configure(text="Status: SUMO Running...", text_color="#00FF00")
        
        try:
            if mode == "Trained MAPPO Agent":
                # Ensure watch_ai.py is your actual live rendering script!
                subprocess.Popen(["python", "watch_ai.py"])
            else:
                # Ensure Baseline.py has the use_gui=True flag set if you want to watch it!
                subprocess.Popen(["python", "Baseline.py"])
        except Exception as e:
            self.status_label.configure(text="Status: Error Launching", text_color="red")
            print(e)

    def load_graph(self):
        # Update this path if your graph is named differently!
        graph_path = "outputs/Final_Comparison.png"
        
        if os.path.exists(graph_path):
            img = Image.open(graph_path)
            
            # Dynamically scale the image to fit the window while maintaining aspect ratio
            img_width, img_height = img.size
            target_width = 550
            ratio = target_width / img_width
            new_size = (int(img_width * ratio), int(img_height * ratio))
            
            # Load into CustomTkinter
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=new_size)
            self.image_label.configure(image=ctk_img, text="") # Clear placeholder text
        else:
            self.image_label.configure(text="Graph not found! Ensure Final_Comparison.png is in the outputs folder.")

if __name__ == "__main__":
    app = TrafficOptimizerApp()
    app.mainloop()