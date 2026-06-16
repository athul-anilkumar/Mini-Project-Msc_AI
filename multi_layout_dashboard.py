import customtkinter as ctk
import subprocess
import threading

# Modern UI setup
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ScalabilityDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Traffic Scalability Benchmarking Suite")
        self.geometry("800x500")

        self.label = ctk.CTkLabel(self, text="Select Grid Layouts to Compare", font=("Arial", 20, "bold"))
        self.label.pack(pady=20)

        # Dashboard layout
        self.btn_frame = ctk.CTkFrame(self)
        self.btn_frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.add_run_button("2x2 Grid", "grid2x2.net.xml", 8813)
        self.add_run_button("3x3 Grid", "grid3x3.net.xml", 8814)
        self.add_run_button("5x5 Grid", "grid5x5.net.xml", 8815)

    def add_run_button(self, name, net_file, port):
        btn = ctk.CTkButton(self.btn_frame, text=f"Run {name}", 
                            command=lambda: self.run_simulation(net_file, port))
        btn.pack(pady=10, padx=20)

    def run_simulation(self, net, port):
        # We launch the simulation in a background thread to prevent UI freezing
        threading.Thread(target=self.launch_process, args=(net, port)).start()

    def launch_process(self, net, port):
        # We pass the port dynamically to avoid collisions
        cmd = ["python", "run_instance.py", "--net", net, "--port", str(port)]
        subprocess.Popen(cmd)

if __name__ == "__main__":
    app = ScalabilityDashboard()
    app.mainloop()