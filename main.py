import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import csv
import os

# Import our backend
import simulation

ZERNIKE_NAMES = {
    1: "Piston", 2: "X Tilt", 3: "Y Tilt", 4: "Defocus",
    5: "Astigmatism X", 6: "Astigmatism Y", 7: "Coma X", 8: "Coma Y",
    9: "Spherical", 10: "Trefoil X", 11: "Trefoil Y",
    12: "Sec. Astigmatism X", 13: "Sec. Astigmatism Y", 14: "Sec. Coma X", 15: "Sec. Coma Y",
    16: "Sec. Spherical", 17: "Tetrafoil X", 18: "Tetrafoil Y",
    19: "Sec. Trefoil X", 20: "Sec. Trefoil Y", 21: "Ter. Astigmatism X", 22: "Ter. Astigmatism Y",
    23: "Ter. Coma X", 24: "Ter. Coma Y", 25: "Ter. Spherical",
    26: "Pentafoil X", 27: "Pentafoil Y", 28: "Sec. Tetrafoil X", 29: "Sec. Tetrafoil Y",
    30: "Ter. Trefoil X", 31: "Ter. Trefoil Y", 32: "Quat. Astigmatism X", 33: "Quat. Astigmatism Y",
    34: "Quat. Coma X", 35: "Quat. Coma Y", 36: "Quat. Spherical"
}

class PartialCoherenceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Partial Coherence Imaging Simulator")
        self.geometry("1400x700")
        
        # State variables
        self.current_img = None
        self.current_mask = None
        self.current_1d = None
        self.current_extent = None
        
        self.custom_mask_data = None
        self.custom_mask_filepath = None
        
        self.slice_dir = "X"
        self.slice_pos_y = 0
        self.slice_pos_x = 0
        self.slice_line = None
        self.dragging_slice = False
        
        self._build_ui()
        
    def _build_ui(self):
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # --- LEFT PANEL (Inputs) ---
        left_frame = ttk.Frame(main_pane, width=550)
        main_pane.add(left_frame, weight=0)
        
        # 1. Optical Parameters
        opt_frame = ttk.LabelFrame(left_frame, text="Optical Parameters")
        opt_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(opt_frame, text="Wavelength λ (nm):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.var_wav = tk.StringVar(value="365.0")
        ttk.Entry(opt_frame, textvariable=self.var_wav, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(opt_frame, text="Lens NA (0-1):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.var_na = tk.StringVar(value="0.1")
        ttk.Entry(opt_frame, textvariable=self.var_na, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(opt_frame, text="Illumin_σ (0-1):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.var_sig = tk.StringVar(value="0.8")
        ttk.Entry(opt_frame, textvariable=self.var_sig, width=10).grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(opt_frame, text="Focus (um):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.var_foc = tk.StringVar(value="0.0")
        ttk.Entry(opt_frame, textvariable=self.var_foc, width=10).grid(row=3, column=1, padx=5, pady=2)
        
        ttk.Label(opt_frame, text="Source Profile:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.var_src_shape = tk.StringVar(value="Top-hat")
        ttk.OptionMenu(opt_frame, self.var_src_shape, "Top-hat", "Top-hat", "Gaussian").grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(opt_frame, text="Gaussian Sigma (1/σ):").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        self.var_g_sigma = tk.StringVar(value="1.0")
        ttk.Entry(opt_frame, textvariable=self.var_g_sigma, width=10).grid(row=5, column=1, padx=5, pady=2)

        ttk.Label(opt_frame, text="Precision:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=2)
        prec_frame = ttk.Frame(opt_frame)
        prec_frame.grid(row=6, column=1, sticky=tk.W)
        self.var_prec = tk.StringVar(value="Fast")
        ttk.Radiobutton(prec_frame, text="Fast", variable=self.var_prec, value="Fast").pack(side=tk.LEFT)
        ttk.Radiobutton(prec_frame, text="High", variable=self.var_prec, value="High").pack(side=tk.LEFT)
        ttk.Radiobutton(prec_frame, text="Very High", variable=self.var_prec, value="VeryHigh").pack(side=tk.LEFT)
        
        # 2. Pattern Source
        pat_source_frame = ttk.LabelFrame(left_frame, text="Pattern Source")
        pat_source_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.var_pat_type = tk.StringVar(value="L&S")
        f1 = ttk.Frame(pat_source_frame)
        f1.pack(fill=tk.X, padx=5, pady=2)
        ttk.Radiobutton(f1, text="Line & Space", variable=self.var_pat_type, value="L&S", command=self._toggle_pat_type).pack(side=tk.LEFT)
        ttk.Radiobutton(f1, text="Custom File", variable=self.var_pat_type, value="Custom", command=self._toggle_pat_type).pack(side=tk.LEFT, padx=10)
        
        self.frame_ls = ttk.Frame(pat_source_frame)
        ttk.Label(self.frame_ls, text="Width(nm):").pack(side=tk.LEFT)
        self.var_w = tk.StringVar(value="1500.0")
        we = ttk.Entry(self.frame_ls, textvariable=self.var_w, width=6)
        we.pack(side=tk.LEFT, padx=2)
        we.bind("<FocusOut>", lambda e: self._update_preview())
        we.bind("<Return>", lambda e: self._update_preview())
        
        ttk.Label(self.frame_ls, text="Lines:").pack(side=tk.LEFT, padx=(5,0))
        self.var_lines = tk.StringVar(value="5")
        le = ttk.Entry(self.frame_ls, textvariable=self.var_lines, width=4)
        le.pack(side=tk.LEFT, padx=2)
        le.bind("<FocusOut>", lambda e: self._update_preview())
        le.bind("<Return>", lambda e: self._update_preview())
        
        ttk.Label(self.frame_ls, text="Ori:").pack(side=tk.LEFT, padx=(5,0))
        self.var_ori = tk.StringVar(value="V")
        ttk.OptionMenu(self.frame_ls, self.var_ori, "V", "V", "H", command=self._update_preview).pack(side=tk.LEFT, padx=2)
        
        self.frame_custom = ttk.Frame(pat_source_frame)
        btn_browse = ttk.Button(self.frame_custom, text="Browse...", command=self._browse_custom)
        btn_browse.grid(row=0, column=0, padx=5, pady=2)
        self.var_filepath = tk.StringVar(value="No file selected")
        ttk.Label(self.frame_custom, textvariable=self.var_filepath, foreground="gray", wraplength=150).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(self.frame_custom, text="1 Cell Size(nm):").grid(row=1, column=0, sticky=tk.E, padx=5, pady=2)
        self.var_cell_size = tk.StringVar(value="10.0")
        ce = ttk.Entry(self.frame_custom, textvariable=self.var_cell_size, width=8)
        ce.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        ce.bind("<FocusOut>", lambda e: self._update_preview())
        ce.bind("<Return>", lambda e: self._update_preview())
        
        self.var_invert = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.frame_custom, text="Invert (0=Block[Black], 1=Pass[White])", variable=self.var_invert, command=self._update_preview).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        self._toggle_pat_type()
        
        # 3. Zernike
        z_frame_container = ttk.LabelFrame(left_frame, text="36 Fringe Zernike Coefficients (waves)")
        z_frame_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add an explicit width to the inner canvas to force it to spread out
        canvas = tk.Canvas(z_frame_container, width=500)
        scrollbar = ttk.Scrollbar(z_frame_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.zernike_entries = []
        for i in range(1, 37):
            name = ZERNIKE_NAMES.get(i, "")
            row = (i - 1) % 18
            col_base = ((i - 1) // 18) * 2
            # Add pad and sticky settings, tighten the gap
            ttk.Label(scrollable_frame, text=f"Z{i} ({name}):").grid(row=row, column=col_base, sticky=tk.E, padx=(5, 2), pady=1)
            e_var = tk.StringVar(value="0.0")
            e = ttk.Entry(scrollable_frame, textvariable=e_var, width=8)
            e.grid(row=row, column=col_base+1, sticky=tk.W, padx=(2, 5), pady=1)
            self.zernike_entries.append(e_var)
            
        # 4. Slice / App Controls
        slice_frame = ttk.LabelFrame(left_frame, text="1D Slice Controls")
        slice_frame.pack(fill=tk.X, padx=5, pady=5)
        self.var_slice_dir = tk.StringVar(value="X")
        ttk.Radiobutton(slice_frame, text="Parallel to X-Axis (Horizontal Cut)", variable=self.var_slice_dir, value="X", command=self._update_slice_dir).pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(slice_frame, text="Parallel to Y-Axis (Vertical Cut)", variable=self.var_slice_dir, value="Y", command=self._update_slice_dir).pack(anchor=tk.W, padx=5)
        
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(btn_frame, text="Run Simulation", command=self.run_simulation).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Export Profile to CSV", command=self.export_csv).pack(fill=tk.X, pady=2)
        
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(left_frame, textvariable=self.status_var, foreground="blue").pack(anchor=tk.W, padx=5)
        
        # --- RIGHT PANEL (Outputs) ---
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=3)
        
        self.fig = Figure(figsize=(12, 5), dpi=100)
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)
        self.ax_mask = self.fig.add_subplot(gs[0, 0])
        self.ax_2d = self.fig.add_subplot(gs[0, 1])
        self.ax_1d = self.fig.add_subplot(gs[0, 2])
        self.fig.tight_layout(pad=3.0)
        
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.canvas_plot.mpl_connect('button_press_event', self._on_press)
        self.canvas_plot.mpl_connect('button_release_event', self._on_release)
        self.canvas_plot.mpl_connect('motion_notify_event', self._on_motion)
        
        # Draw initial L&S preview
        self.after(200, self._update_preview)

    def _toggle_pat_type(self):
        if self.var_pat_type.get() == "L&S":
            self.frame_custom.pack_forget()
            self.frame_ls.pack(fill=tk.X, padx=5, pady=2)
        else:
            self.frame_ls.pack_forget()
            self.frame_custom.pack(fill=tk.X, padx=5, pady=2)
        self._update_preview()

    def _browse_custom(self):
        fp = filedialog.askopenfilename(filetypes=[("All Supported", "*.csv *.dat *.bmp"), ("CSV files", "*.csv"), ("DAT files", "*.dat"), ("BMP files", "*.bmp")])
        if not fp: return
        self.var_filepath.set(os.path.basename(fp))
        self.custom_mask_filepath = fp
        try:
            self.custom_mask_data = simulation.load_custom_pattern(fp)
            self._update_preview()
        except Exception as e:
            messagebox.showerror("File Load Error", str(e))
            
    def _update_preview(self, *args):
        prec = self.var_prec.get()
        if prec == "Fast": target_size = 512
        elif prec == "High": target_size = 1024
        else: target_size = 2048
        
        if self.var_pat_type.get() == "L&S":
            try:
                w = float(self.var_w.get())
                num_lines = int(self.var_lines.get())
                ori = self.var_ori.get()
                target_field_size = 4.0 * num_lines * w
                Nx, Ny = target_size, target_size
                pixel_size = target_field_size / Nx
                mask = simulation.generate_mask(Nx, Ny, pixel_size, w, num_lines, ori)
                self.current_mask = mask
                self._draw_mask_preview(mask, pixel_size)
            except Exception: pass
        elif self.var_pat_type.get() == "Custom":
            if self.custom_mask_data is None: return
            try:
                cell_size = float(self.var_cell_size.get())
                inv = self.var_invert.get()
                mask, px_size = simulation.prepare_custom_mask(self.custom_mask_data, cell_size, inv, target_size=target_size)
                self.current_mask = mask
                self._draw_mask_preview(mask, px_size)
            except Exception: pass
            
    def _draw_mask_preview(self, mask, px_size):
        self.ax_mask.clear()
        ny, nx = mask.shape
        w_um = nx * px_size / 1000.0
        h_um = ny * px_size / 1000.0
        extent = [-w_um/2, w_um/2, -h_um/2, h_um/2]
        self.ax_mask.imshow(mask, extent=extent, cmap='gray', origin='lower', vmin=0, vmax=1)
        self.ax_mask.set_title("Mask Preview", fontsize=11)
        self.ax_mask.set_xlabel("X (um)", fontsize=10)
        self.ax_mask.set_ylabel("Y (um)", fontsize=10)
        self.canvas_plot.draw_idle()

    def run_simulation(self):
        try:
            wav = float(self.var_wav.get())
            na = float(self.var_na.get())
            sig = float(self.var_sig.get())
            foc_um = float(self.var_foc.get())
            prec = self.var_prec.get()
            src_shape = self.var_src_shape.get()
            g_sigma = float(self.var_g_sigma.get())
            z_coeffs = np.array([float(v.get()) for v in self.zernike_entries])
            
            pat_type = self.var_pat_type.get()
            foc_nm = foc_um * 1000.0
            
            if prec == "Fast":
                num_points_single = 120
                target_sim = 512
            elif prec == "High":
                num_points_single = 250
                target_sim = 1024
            else: # VeryHigh
                num_points_single = 400
                target_sim = 2048
                
            src_single = simulation.get_source_points(na, sig, wav, num_points=num_points_single, shape=src_shape, sigma_gauss=g_sigma)
            
            if pat_type == "L&S":
                w = float(self.var_w.get())
                num_lines = int(self.var_lines.get())
                ori = self.var_ori.get()
                target_field_size = 4.0 * num_lines * w
                Nx, Ny = target_sim, target_sim
                pixel_size = target_field_size / Nx
                mask = simulation.generate_mask(Nx, Ny, pixel_size, w, num_lines, ori)
                self._draw_mask_preview(mask, pixel_size)
            else:
                if self.custom_mask_data is None:
                    messagebox.showerror("Error", "Please load a custom pattern file first.")
                    return
                cell_size = float(self.var_cell_size.get())
                inv = self.var_invert.get()
                mask, pixel_size = simulation.prepare_custom_mask(self.custom_mask_data, cell_size, inv, target_size=target_sim)
                Nx, Ny = mask.shape[1], mask.shape[0]
                self._draw_mask_preview(mask, pixel_size)
                
            self.status_var.set("Running simulation... please wait.")
            self.update()
            
            img = simulation.simulate_image(mask, na, sig, wav, foc_nm, z_coeffs, pixel_size, source_points=src_single)
            
            self.current_img = img
            w_um = Nx * pixel_size / 1000.0
            h_um = Ny * pixel_size / 1000.0
            self.current_extent = [-w_um/2, w_um/2, -h_um/2, h_um/2]
            
            self.ax_2d.clear()
            self.ax_2d.imshow(self.current_img, extent=self.current_extent, origin='lower', cmap='viridis')
            self.ax_2d.set_title(f"2D Aerial Image (Focus = {foc_um:.3f} um)", fontsize=11)
            self.ax_2d.set_xlabel("X (um)", fontsize=10)
            self.ax_2d.set_ylabel("Y (um)", fontsize=10)
            
            self._init_slice_line()
            self._draw_1d_profile()
            self.status_var.set("Simulation completed.")
            
        except Exception as e:
            messagebox.showerror("Simulation Error", str(e))
            self.status_var.set("Error occurred.")

    def _init_slice_line(self):
        if self.current_img is None: return
        if self.slice_line:
            try:
                # Check if the artist is still attached to an axes
                if self.slice_line.axes is not None:
                    self.slice_line.remove()
            except (ValueError, RuntimeError):
                pass
            self.slice_line = None
            
        xmin, xmax, ymin, ymax = self.current_extent
        if self.var_slice_dir.get() == "X":
            y_mid = (ymax + ymin) / 2
            self.slice_pos_y = self.current_img.shape[0] // 2
            self.slice_line, = self.ax_2d.plot([xmin, xmax], [y_mid, y_mid], color='red', marker='^', linestyle='--', linewidth=1.5, markevery=[0,-1])
        else:
            x_mid = (xmax + xmin) / 2
            self.slice_pos_x = self.current_img.shape[1] // 2
            self.slice_line, = self.ax_2d.plot([x_mid, x_mid], [ymin, ymax], color='blue', marker='>', linestyle='--', linewidth=1.5, markevery=[0,-1])

    def _update_slice_dir(self):
        if self.current_img is None: return
        self._init_slice_line()
        self.canvas_plot.draw_idle()
        self._draw_1d_profile()

    def _on_press(self, event):
        if event.inaxes != self.ax_2d or self.current_img is None: return
        self.dragging_slice = True
        self._update_slice_position(event.xdata, event.ydata)
        
    def _on_motion(self, event):
        if not self.dragging_slice or event.inaxes != self.ax_2d: return
        self._update_slice_position(event.xdata, event.ydata)
        
    def _on_release(self, event):
        self.dragging_slice = False
        
    def _update_slice_position(self, x_um, y_um):
        if self.current_extent is None: return
        xmin, xmax, ymin, ymax = self.current_extent
        
        x_um = max(xmin, min(xmax, x_um))
        y_um = max(ymin, min(ymax, y_um))
        
        Nx = self.current_img.shape[1]
        Ny = self.current_img.shape[0]
        
        px = int((x_um - xmin) / (xmax - xmin) * (Nx - 1))
        py = int((y_um - ymin) / (ymax - ymin) * (Ny - 1))
        
        if self.var_slice_dir.get() == "X":
            self.slice_pos_y = py
            actual_y = ymin + py * (ymax - ymin) / max(1, (Ny - 1))
            if self.slice_line:
                self.slice_line.set_ydata([actual_y, actual_y])
        else:
            self.slice_pos_x = px
            actual_x = xmin + px * (xmax - xmin) / max(1, (Nx - 1))
            if self.slice_line:
                self.slice_line.set_xdata([actual_x, actual_x])
                
        self.canvas_plot.draw_idle()
        self._draw_1d_profile()

    def _draw_1d_profile(self):
        if self.current_img is None: return
        self.ax_1d.clear()
        
        xmin, xmax, ymin, ymax = self.current_extent
        Nx = self.current_img.shape[1]
        Ny = self.current_img.shape[0]
        x_axis_um = np.linspace(xmin, xmax, Nx)
        y_axis_um = np.linspace(ymin, ymax, Ny)
        
        if self.var_slice_dir.get() == "X":
            prof = self.current_img[self.slice_pos_y, :]
            axis = x_axis_um
            xlabel = "X Position (um)"
            title = f"Horizontal Slice (Y={y_axis_um[self.slice_pos_y]:.3f} um)"
            color = 'red'
        else:
            prof = self.current_img[:, self.slice_pos_x]
            axis = y_axis_um
            xlabel = "Y Position (um)"
            title = f"Vertical Slice (X={x_axis_um[self.slice_pos_x]:.3f} um)"
            color = 'blue'
            
        self.ax_1d.plot(axis, prof, color=color, linestyle='-')
        self.ax_1d.set_title(title, fontsize=11)
        self.ax_1d.set_xlabel(xlabel, fontsize=10)
        self.ax_1d.set_ylabel("Intensity (a.u.)", fontsize=10)
        self.ax_1d.grid(True)
        self.ax_1d.tick_params(axis='both', which='major', labelsize=9)
        if self.var_slice_dir.get() == "X":
            self.ax_1d.set_xlim([xmin, xmax])
        else:
            self.ax_1d.set_xlim([ymin, ymax])
        self.canvas_plot.draw_idle()

    def export_csv(self):
        if self.current_img is None:
            messagebox.showwarning("No Data", "Please run a simulation first!")
            return
            
        dlg = tk.Toplevel(self)
        dlg.title("Select Data to Export")
        dlg.geometry("300x200")
        
        var_mask = tk.BooleanVar(value=True)
        var_2d = tk.BooleanVar(value=True)
        var_1d = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(dlg, text="1. Original Mask (0=Block, 1=Pass)", variable=var_mask).pack(anchor=tk.W, padx=20, pady=5)
        ttk.Checkbutton(dlg, text="2. Aerial Image (2D)", variable=var_2d).pack(anchor=tk.W, padx=20, pady=5)
        ttk.Checkbutton(dlg, text="3. 1D Slice Profile", variable=var_1d).pack(anchor=tk.W, padx=20, pady=5)
        
        def on_export():
            dlg.destroy()
            if not (var_1d.get() or var_2d.get() or var_mask.get()):
                return
                
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                title="Export Data (Base Name)"
            )
            if not filepath: return
            
            try:
                base, ext = os.path.splitext(filepath)
                
                xmin, xmax, ymin, ymax = self.current_extent
                Nx = self.current_img.shape[1]
                Ny = self.current_img.shape[0]
                x_axis_um = np.linspace(xmin, xmax, Nx)
                y_axis_um = np.linspace(ymin, ymax, Ny)
                
                # 1. Export Mask
                if var_mask.get() and self.current_mask is not None:
                    mask_path = f"{base}_mask{ext}"
                    with open(mask_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["--- Mask File ---"])
                        writer.writerow(["Row(y) \\ Col(x)"] + list(x_axis_um))
                        for i, yy in enumerate(y_axis_um):
                            writer.writerow([yy] + list(self.current_mask[i, :]))
                            
                # 2. Export 2D Aerial Image
                if var_2d.get():
                    img2d_path = f"{base}_2D_image{ext}"
                    with open(img2d_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["--- 2D Aerial Image ---"])
                        writer.writerow(["Row(y) \\ Col(x)"] + list(x_axis_um))
                        for i, yy in enumerate(y_axis_um):
                            writer.writerow([yy] + list(self.current_img[i, :]))
                            
                # 3. Export 1D Slice
                if var_1d.get():
                    prof_path = f"{base}_1D_profile{ext}"
                    with open(prof_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["--- 1D Slice Profile ---"])
                        if self.var_slice_dir.get() == "X":
                            writer.writerow([f"Slice along Y = {y_axis_um[self.slice_pos_y]:.5f} um"])
                            writer.writerow(["X Position (um)", "Intensity (a.u.)"])
                            for xx, pp in zip(x_axis_um, self.current_img[self.slice_pos_y, :]):
                                writer.writerow([xx, pp])
                        else:
                            writer.writerow([f"Slice along X = {x_axis_um[self.slice_pos_x]:.5f} um"])
                            writer.writerow(["Y Position (um)", "Intensity (a.u.)"])
                            for yy, pp in zip(y_axis_um, self.current_img[:, self.slice_pos_x]):
                                writer.writerow([yy, pp])
                                
                self.status_var.set("Successfully exported data to CSV files.")
                messagebox.showinfo("Export Successful", "Data saved successfully with prefix:\n" + base)
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to save CSV:\n{str(e)}")

        ttk.Button(dlg, text="Export Selected", command=on_export).pack(pady=20)

if __name__ == "__main__":
    app = PartialCoherenceApp()
    app.mainloop()
