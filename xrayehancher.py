
import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tkinter import Tk, Frame, Label, filedialog, StringVar, DoubleVar, messagebox
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------------- Enhancement Functions ----------------
def gamma_correction(img, gamma=0.6, c=1.0):
    """Power-law (gamma) transformation"""
    table = np.array([((i / 255.0) ** gamma) * 255 * c for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def contrast_stretch(img, r1=60, s1=10, r2=180, s2=245):
    """Piecewise linear contrast stretching"""
    img_f = img.astype(np.float32)
    out = np.zeros_like(img_f)
    out[img_f <= r1] = (s1 / r1) * img_f[img_f <= r1]
    mid = (img_f > r1) & (img_f <= r2)
    out[mid] = ((s2 - s1) / (r2 - r1)) * (img_f[mid] - r1) + s1
    out[img_f > r2] = ((255 - s2) / (255 - r2)) * (img_f[img_f > r2] - r2) + s2
    return np.clip(out, 0, 255).astype("uint8")

# ---------------- GUI Application ----------------
class XrayEnhancementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🩻 Chest X-ray Image Enhancement")
        self.root.geometry("1350x900")
        self.root.configure(bg="#f7f9fa")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 11, "bold"), padding=8,
                        background="#3498db", foreground="white", relief="flat")
        style.map("TButton", background=[("active", "#2980b9")])
        style.configure("TLabel", background="#f7f9fa", font=("Segoe UI", 11))

        self.gray = None
        self.gamma_var = DoubleVar(value=0.6)

        # Header
        header = Frame(root, bg="#2c3e50")
        header.pack(fill="x")
        Label(header, text="Chest X-ray Enhancement Dashboard",
              font=("Segoe UI", 22, "bold"), bg="#2c3e50", fg="white", pady=15).pack()

        # Controls
        controls = Frame(root, bg="#f7f9fa", pady=10)
        controls.pack(fill="x")

        ttk.Button(controls, text="Upload X-ray", command=self.upload_image).grid(row=0, column=0, padx=10)

        Label(controls, text="Enhancement Method:", font=("Segoe UI", 12, "bold")).grid(row=0, column=1, padx=10)

        self.method_var = StringVar(value="Histogram Equalization")
        self.method_menu = ttk.Combobox(
            controls, textvariable=self.method_var, state="readonly",
            values=["Histogram Equalization", "Gamma Correction", "Gamma + Contrast"],
            font=("Segoe UI", 12), width=28
        )
        self.method_menu.grid(row=0, column=2, padx=10)
        self.method_menu.bind("<<ComboboxSelected>>", self.toggle_gamma_input)

        # Gamma controls
        self.gamma_frame = Frame(controls, bg="#f7f9fa")
        Label(self.gamma_frame, text="Gamma:", font=("Segoe UI", 12, "bold"), bg="#f7f9fa").grid(row=0, column=0, padx=5)
        self.gamma_slider = ttk.Scale(self.gamma_frame, from_=0.2, to=1.5, variable=self.gamma_var,
                                      orient="horizontal", length=250, command=self.update_gamma_label)
        self.gamma_slider.grid(row=0, column=1, padx=5)
        self.gamma_label = Label(self.gamma_frame, text=f"{self.gamma_var.get():.2f}",
                                 font=("Segoe UI", 11), bg="#f7f9fa")
        self.gamma_label.grid(row=0, column=2, padx=5)

        ttk.Button(controls, text="Apply Method", command=self.apply_method).grid(row=0, column=3, padx=10)
        ttk.Button(controls, text="Save Report", command=self.save_report).grid(row=0, column=4, padx=10)

        # Display Area
        self.display = Frame(root, bg="white", relief="ridge", bd=2)
        self.display.pack(fill="both", expand=True, padx=15, pady=10)

        # Status bar
        self.status = Label(root, text="Ready", bd=1, relief="sunken", anchor="w",
                            font=("Segoe UI", 10), bg="#bdc3c7")
        self.status.pack(side="bottom", fill="x")

        self.results = {}

    def set_status(self, msg):
        self.status.config(text=msg)

    def toggle_gamma_input(self, event=None):
        if self.method_var.get() in ["Gamma Correction", "Gamma + Contrast"]:
            self.gamma_frame.grid(row=0, column=5, padx=10)
        else:
            self.gamma_frame.grid_forget()

    def update_gamma_label(self, event=None):
        self.gamma_label.config(text=f"{self.gamma_var.get():.2f}")

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif")])
        if not path: return
        img = cv2.imread(path)
        if img is None: 
            messagebox.showerror("Error", "Could not open image file."); return
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.current_name = Path(path).stem
        self.set_status(f"Loaded {self.current_name}")
        self.show_result("Original", self.gray)

    def show_result(self, title, processed):
        fig, axs = plt.subplots(2, 2, figsize=(9, 7))
        fig.suptitle(title, fontsize=14, weight="bold", color="#2c3e50")

        # Original
        axs[0,0].imshow(self.gray, cmap="gray")
        axs[0,0].set_title("Original")
        axs[0,0].axis("off")

        axs[0,1].hist(self.gray.ravel(), bins=256, range=(0,255), color="#3498db")
        axs[0,1].set_title("Original Histogram")
        axs[0,1].set_xlabel("Pixel Intensity")
        axs[0,1].set_ylabel("Frequency")

        # Enhanced
        axs[1,0].imshow(processed, cmap="gray")
        axs[1,0].set_title("Enhanced")
        axs[1,0].axis("off")

        axs[1,1].hist(processed.ravel(), bins=256, range=(0,255), color="#2ecc71")
        axs[1,1].set_title("Enhanced Histogram")
        axs[1,1].set_xlabel("Pixel Intensity")
        axs[1,1].set_ylabel("Frequency")

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # prevent overlap

        for w in self.display.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.display)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

        self.results[title] = processed

    def apply_method(self):
        if self.gray is None: 
            messagebox.showwarning("Warning", "Upload an image first."); return
        method = self.method_var.get()
        gamma = float(self.gamma_var.get())

        if method == "Histogram Equalization":
            enhanced = cv2.equalizeHist(self.gray)
        elif method == "Gamma Correction":
            enhanced = gamma_correction(self.gray, gamma=gamma)
        elif method == "Gamma + Contrast":
            enhanced = contrast_stretch(gamma_correction(self.gray, gamma=gamma))
        else:
            return

        self.show_result(method, enhanced)
        self.set_status(f"Applied {method} (Gamma={gamma:.2f})" if "Gamma" in method else f"Applied {method}")

    def save_report(self):
        if not self.results:
            messagebox.showwarning("Warning", "No results to save."); return
        outdir = "results"
        os.makedirs(outdir, exist_ok=True)
        pdf_path = os.path.join(outdir, f"{self.current_name}_report.pdf")

        with PdfPages(pdf_path) as pdf:
            for name, proc in self.results.items():
                fig, axs = plt.subplots(2, 2, figsize=(8.5, 11))
                fig.suptitle(name, fontsize=14, weight="bold")

                axs[0,0].imshow(self.gray, cmap="gray")
                axs[0,0].set_title("Original")
                axs[0,0].axis("off")

                axs[0,1].hist(self.gray.ravel(), bins=256, range=(0,255))
                axs[0,1].set_title("Original Histogram")
                axs[0,1].set_xlabel("Pixel Intensity")
                axs[0,1].set_ylabel("Frequency")

                axs[1,0].imshow(proc, cmap="gray")
                axs[1,0].set_title("Enhanced")
                axs[1,0].axis("off")

                axs[1,1].hist(proc.ravel(), bins=256, range=(0,255))
                axs[1,1].set_title("Enhanced Histogram")
                axs[1,1].set_xlabel("Pixel Intensity")
                axs[1,1].set_ylabel("Frequency")

                plt.tight_layout(rect=[0,0,1,0.96])
                pdf.savefig(fig)
                plt.close(fig)

        messagebox.showinfo("Success", f"PDF report saved to {pdf_path}")
        self.set_status(f"Report saved: {pdf_path}")

# ---------------- Main ----------------
if __name__ == "__main__":
    root = Tk()
    app = XrayEnhancementApp(root)
    root.mainloop()
