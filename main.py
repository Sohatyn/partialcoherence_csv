import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import csv

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
        self.geometry("1000x800")
        
        # State variables
        self.current_img = None
        self.current_1d = None
        self.current_foc_list = None
        self.current_c_list = None
        self.current_p_list = None
        self.current_contrast = 0.0
        self.current_extent = None
        
        self._build_ui()
        
    def _build_ui(self):
        # Create Main Paned Window
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # --- LEFT PANEL (Inputs) ---
        left_frame = ttk.Frame(main_pane, width=650)
        main_pane.add(left_frame, weight=1)
        
        # Input standard parameters
        param_frame = ttk.LabelFrame(left_frame, text="Optical Parameters")
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(param_frame, text="Wavelength λ (nm):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.var_wav = tk.StringVar(value="365.0")
        ttk.Entry(param_frame, textvariable=self.var_wav, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(param_frame, text="Lens NA (0-1):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.var_na = tk.StringVar(value="0.1")
        ttk.Entry(param_frame, textvariable=self.var_na, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(param_frame, text="Illumination σ (0-1):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.var_sig = tk.StringVar(value="0.8")
        ttk.Entry(param_frame, textvariable=self.var_sig, width=10).grid(row=2, column=1, padx=5, pady=2)
        ttk.Label(param_frame, text="Focus (um):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.var_foc = tk.StringVar(value="0.0")
        ttk.Entry(param_frame, textvariable=self.var_foc, width=10).grid(row=3, column=1, padx=5, pady=2)
        
        ttk.Label(param_frame, text="Focus Sweep ±(um):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        span_frame = ttk.Frame(param_frame)
        span_frame.grid(row=4, column=1, sticky=tk.W)
        self.var_foc_span = tk.StringVar(value="5.0")
        ttk.Entry(span_frame, textvariable=self.var_foc_span, width=5).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(span_frame, text="Step:").pack(side=tk.LEFT)
        self.var_foc_step = tk.StringVar(value="0.5")
        ttk.Entry(span_frame, textvariable=self.var_foc_step, width=5).pack(side=tk.LEFT, padx=(2, 0))

        ttk.Label(param_frame, text="L&S Width (nm):").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        lw_frame = ttk.Frame(param_frame)
        lw_frame.grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)
        self.var_w = tk.StringVar(value="1500.0")
        ttk.Entry(lw_frame, textvariable=self.var_w, width=7).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(lw_frame, text="Lines:").pack(side=tk.LEFT)
        self.var_lines = tk.StringVar(value="5")
        ttk.Entry(lw_frame, textvariable=self.var_lines, width=4).pack(side=tk.LEFT, padx=(2, 0))
        
        ttk.Label(param_frame, text="Orientation:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=2)
        ori_frame = ttk.Frame(param_frame)
        ori_frame.grid(row=6, column=1, sticky=tk.W)
        self.var_ori = tk.StringVar(value="V")
        ttk.Radiobutton(ori_frame, text="Vertical", variable=self.var_ori, value="V").pack(side=tk.LEFT)
        ttk.Radiobutton(ori_frame, text="Horizontal", variable=self.var_ori, value="H").pack(side=tk.LEFT)
        ttk.Radiobutton(ori_frame, text="Both", variable=self.var_ori, value="Both").pack(side=tk.LEFT)
        
        ttk.Label(param_frame, text="Precision:").grid(row=7, column=0, sticky=tk.W, padx=5, pady=2)
        prec_frame = ttk.Frame(param_frame)
        prec_frame.grid(row=7, column=1, sticky=tk.W)
        self.var_prec = tk.StringVar(value="Fast")
        ttk.Radiobutton(prec_frame, text="Fast (Rough)", variable=self.var_prec, value="Fast").pack(side=tk.LEFT)
        ttk.Radiobutton(prec_frame, text="High (Slow)", variable=self.var_prec, value="High").pack(side=tk.LEFT)
        
        # Zernike Parameters (Scrollable)
        z_frame_container = ttk.LabelFrame(left_frame, text="36 Fringe Zernike Coefficients (waves)")
        z_frame_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        canvas = tk.Canvas(z_frame_container)
        scrollbar = ttk.Scrollbar(z_frame_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.zernike_entries = []
        for i in range(1, 37):
            name = ZERNIKE_NAMES.get(i, "")
            row = (i - 1) % 18
            col_base = ((i - 1) // 18) * 2
            
            ttk.Label(scrollable_frame, text=f"Z{i} ({name}):").grid(row=row, column=col_base, sticky=tk.E, padx=5, pady=1)
            e_var = tk.StringVar(value="0.0")
            e = ttk.Entry(scrollable_frame, textvariable=e_var, width=8)
            e.grid(row=row, column=col_base+1, padx=5, pady=1)
            self.zernike_entries.append(e_var)
            
        # Action Buttons
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)
        ttk.Button(btn_frame, text="Run Full Simulation", command=lambda: self.run_simulation(full=True)).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Run 2D & Profile Only", command=lambda: self.run_simulation(full=False)).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Export to CSV", command=self.export_csv).pack(fill=tk.X, pady=2)
        
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(left_frame, textvariable=self.status_var, foreground="blue").pack(anchor=tk.W, padx=5)
        self.contrast_lbl = tk.StringVar(value="Center Contrast: N/A")
        ttk.Label(left_frame, textvariable=self.contrast_lbl, font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=5, pady=5)
        
        # --- RIGHT PANEL (Outputs) ---
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=3)
        
        # Matplotlib Figures
        self.fig = Figure(figsize=(8.5, 11), dpi=100)
        
        # Give more width/height balance to the top plots to avoid squishing
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], width_ratios=[1, 1], hspace=0.35, wspace=0.25)
        self.ax1 = self.fig.add_subplot(gs[0, 0])  # 1D Profile
        self.ax4 = self.fig.add_subplot(gs[0, 1])  # 2D Profile
        self.ax2 = self.fig.add_subplot(gs[1, :])  # Contrast Curve
        self.ax3 = self.fig.add_subplot(gs[2, :])  # Heatmap
        
        self.fig.tight_layout(pad=3.0)
        
        # Colorbar reference for heatmap
        self.cbar = None
        self.cbar_ax = None
        
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _get_inputs(self):
        try:
            wav = float(self.var_wav.get())
            na = float(self.var_na.get())
            sig = float(self.var_sig.get())
            foc_um = float(self.var_foc.get())
            foc_span = float(self.var_foc_span.get())
            foc_step = float(self.var_foc_step.get())
            w = float(self.var_w.get())
            num_lines = int(self.var_lines.get())
            ori = self.var_ori.get()
            prec = self.var_prec.get()
            z_coeffs = np.array([float(v.get()) for v in self.zernike_entries])
            return wav, na, sig, foc_um, foc_span, foc_step, w, num_lines, ori, prec, z_coeffs
        except ValueError:
            messagebox.showerror("Input Error", "Please ensure all inputs are valid numbers.")
            return None

    def run_simulation(self, full=True):
        params = self._get_inputs()
        if not params: return
        wav, na, sig, foc_um, foc_span, foc_step, w, num_lines, ori, prec, z_coeffs = params
        
        mode_text = "full simulation" if full else "2D & Profile calculation"
        self.status_var.set(f"Running {mode_text}... please wait.")
        self.update()
        
        # Set precision points
        if prec == "Fast":
            num_points_single = 120
            num_points_sweep = 50
        else:
            num_points_single = 250
            num_points_sweep = 250
            
        try:
            # Resolution logic
            # Scale target_field_size proportionally with num_lines so it always looks like 5 lines
            target_field_size = 4.0 * num_lines * w
            Nx, Ny = 512, 512
            pixel_size = target_field_size / Nx
            
            foc_nm = foc_um * 1000.0
            src_single = simulation.get_source_points(na, sig, wav, num_points=num_points_single)
            
            # Helper to run single orientation
            def run_single_orientation(o):
                mask = simulation.generate_mask(Nx, Ny, pixel_size, w, num_lines, o)
                img = simulation.simulate_image(mask, na, sig, wav, foc_nm, z_coeffs, pixel_size, source_points=src_single)
                c = simulation.calculate_contrast(img, w, pixel_size, o)
                
                cx, cy = Nx//2, Ny//2
                if o == 'V':
                    profile = img[cy, :]
                    x_axis = (np.arange(Nx) - cx) * pixel_size
                else:
                    profile = img[:, cx]
                    x_axis = (np.arange(Ny) - cy) * pixel_size
                
                return img, c, x_axis, profile
                
            # --- 1. Single Focus Simulation ---
            if ori == "Both":
                # Define primary view based on V
                img_v, c_v, x_axis_v, profile_v = run_single_orientation("V")
                img_h, c_h, x_axis_h, profile_h = run_single_orientation("H")
                
                self.current_img = img_v # Keep V for the 2D plot
                self.current_contrast = {"V": c_v, "H": c_h}
                self.contrast_lbl.set(f"Center Contrast: V={c_v:.4f}, H={c_h:.4f}")
                self.current_1d = (x_axis_v, profile_v) # Keep V as primary 1D
            else:
                img, c, x_axis, profile = run_single_orientation(ori)
                self.current_img = img
                self.current_contrast = c
                self.contrast_lbl.set(f"Center Contrast: {c:.4f}")
                self.current_1d = (x_axis, profile)
            
            self.current_extent = [-Nx/2 * pixel_size, Nx/2 * pixel_size, -Ny/2 * pixel_size, Ny/2 * pixel_size]
            
            foc_um_list, c_list, p_list = [], [], []
            c_list_h, p_list_h = [], []

            # --- 2. Through-Focus Sweep ---
            if full:
                span_um = foc_span
                step_um = foc_step
                if step_um <= 0:
                    raise ValueError("Focus step must be > 0.")
                num_steps = int(round(2 * span_um / step_um)) + 1
                foc_um_list = np.linspace(foc_um - span_um, foc_um - span_um + (num_steps - 1) * step_um, num_steps)
                foc_nm_list = foc_um_list * 1000.0
                
                if ori == "Both":
                    c_list_v, p_list_v = simulation.run_through_focus(
                        w, na, sig, wav, foc_nm_list, z_coeffs, num_lines, "V", Nx, Ny, pixel_size, num_source=num_points_sweep
                    )
                    c_list_h_res, p_list_h_res = simulation.run_through_focus(
                        w, na, sig, wav, foc_nm_list, z_coeffs, num_lines, "H", Nx, Ny, pixel_size, num_source=num_points_sweep
                    )
                    c_list = c_list_v
                    p_list = p_list_v
                    c_list_h = c_list_h_res
                    p_list_h = p_list_h_res
                else:
                    c_list, p_list = simulation.run_through_focus(
                        w, na, sig, wav, foc_nm_list, z_coeffs, num_lines, ori, Nx, Ny, pixel_size, num_source=num_points_sweep
                    )
                
                self.current_foc_list = foc_um_list
                self.current_c_list = c_list
                self.current_p_list = p_list
            
            # Plot
            self._update_plots(self.current_1d[0], self.current_1d[1], foc_um, self.current_contrast, foc_um_list, c_list, p_list, w, num_lines, ori, full, c_list_h, p_list_h)
            self.status_var.set("Simulation completed.")
            
        except Exception as e:
            messagebox.showerror("Simulation Error", str(e))
            self.status_var.set("Error occurred.")
            
    def _update_plots(self, x, prof, f_user, c_user, f_list, c_list, p_list, w, num_lines, ori, full_update=True, c_list_h=None, p_list_h=None):
        # Convert base units (nm) to (um) for plots 1 and 4
        x_um = x / 1000.0
        w_um = w / 1000.0
        extent_um = [e / 1000.0 for e in self.current_extent]
        limit_um = 1.8 * num_lines * w_um
        
        # 1. 1D Profile (Only shows Primary when Both, which is V)
        self.ax1.clear()
        self.ax1.plot(x_um, prof, 'b-', label='Aerial Image (V)' if ori == 'Both' else 'Aerial Image')
        self.ax1.set_title(f"1D Profile (Focus = {f_user:.3f} um)", fontsize=11)
        self.ax1.set_xlabel("Position (um)", fontsize=10)
        self.ax1.set_ylabel("Intensity (a.u.)", fontsize=10)
        self.ax1.grid(True)
        self.ax1.set_xlim([-limit_um, limit_um])
        self.ax1.legend(fontsize=9)
        self.ax1.tick_params(axis='both', which='major', labelsize=9)
        
        # 0. 2D Profile (Only shows Primary when Both, which is V)
        self.ax4.clear()
        im = self.ax4.imshow(self.current_img, extent=extent_um, origin='lower', cmap='viridis')
        self.ax4.set_title(f"2D Aerial Image (V)" if ori == 'Both' else "2D Aerial Image", fontsize=11)
        self.ax4.set_xlabel("X (um)", fontsize=10)
        self.ax4.set_ylabel("Y (um)", fontsize=10)
        self.ax4.set_xlim([-limit_um, limit_um])
        self.ax4.set_ylim([-limit_um, limit_um])
        self.ax4.set_aspect('equal')
        self.ax4.tick_params(axis='both', which='major', labelsize=9)
        
        if full_update:
            # 2. Contrast Curve
            self.ax2.clear()
            if ori == 'Both':
                self.ax2.plot(f_list, c_list, 'b-o', label='V Contrast')
                self.ax2.plot(f_user, c_user["V"], 'b*', markersize=12, label='V Current Focus')
                self.ax2.plot(f_list, c_list_h, 'r-s', label='H Contrast')
                self.ax2.plot(f_user, c_user["H"], 'r*', markersize=12, label='H Current Focus')
            else:
                self.ax2.plot(f_list, c_list, 'k-o', label='Contrast Curve')
                self.ax2.plot(f_user, c_user, 'r*', markersize=12, label='Current Focus')
            
            self.ax2.set_title("Through-Focus Contrast", fontsize=12)
            self.ax2.set_xlabel("Focus (um)", fontsize=11)
            self.ax2.set_ylabel("Contrast", fontsize=11)
            self.ax2.grid(True)
            self.ax2.legend(fontsize=10)
            self.ax2.tick_params(axis='both', which='major', labelsize=10)
            
            # 3. Heatmap
            # Remove previous colorbar if exists
            if self.cbar_ax is not None:
                self.cbar_ax.remove()
                self.cbar_ax = None
                
            self.ax3.clear()
            # If standard heatmap (V or H only)
            cmap = matplotlib.colormaps['RdYlGn_r']
            X_mesh, Y_mesh = np.meshgrid(x_um, f_list)
            
            if ori == "Both":
                # Splitting ax3 into two using gridspec isn't easy here, so let's plot side-by-side using ax3's axes space
                # We can draw two pcolormeshes on the same axes if we shift the X_mesh.
                # Since Position ranges from -limit to +limit, we can shift them
                # Let's shift V to left side [-limit * 2, 0] and H to right side [0, limit * 2] roughly
                gap = limit_um * 0.2
                shift_v = limit_um + gap
                shift_h = limit_um + gap
                
                # Plot V shifted left
                pcm = self.ax3.pcolormesh(X_mesh - shift_v, Y_mesh, p_list, cmap=cmap, shading='auto')
                # Plot H shifted right
                self.ax3.pcolormesh(X_mesh + shift_h, Y_mesh, p_list_h, cmap=cmap, shading='auto')
                
                self.ax3.set_title("Through-Focus Intensity Heatmap (V <<   ||   >> H)", fontsize=12)
                self.ax3.set_xlabel("Position (shifted)", fontsize=11)
                self.ax3.set_ylabel("Focus (um)", fontsize=11)
                self.ax3.set_xlim([-(limit_um + gap * 2 + limit_um), (limit_um + gap * 2 + limit_um)])
            else:
                pcm = self.ax3.pcolormesh(X_mesh, Y_mesh, p_list, cmap=cmap, shading='auto')
                self.ax3.set_title("Through-Focus Intensity Heatmap", fontsize=12)
                self.ax3.set_xlabel("Position (um)", fontsize=11)
                self.ax3.set_ylabel("Focus (um)", fontsize=11)
                self.ax3.set_xlim([-limit_um, limit_um])
                
            self.ax3.tick_params(axis='both', which='major', labelsize=10)
            
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(self.ax3)
            self.cbar_ax = divider.append_axes("right", size="5%", pad=0.1)
            self.cbar = self.fig.colorbar(pcm, cax=self.cbar_ax, orientation='vertical', label='Intensity')
        
        self.fig.tight_layout(pad=3.0)
        self.canvas_plot.draw()
        
    def export_csv(self):
        if self.current_1d is None or self.current_foc_list is None:
            messagebox.showwarning("No Data", "Please run a simulation first!")
            return
            
        dlg = tk.Toplevel(self)
        dlg.title("Select Data to Export")
        dlg.geometry("300x200")
        
        var_1d = tk.BooleanVar(value=True)
        var_c = tk.BooleanVar(value=True)
        var_h = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(dlg, text="1D Image Profile", variable=var_1d).pack(anchor=tk.W, padx=20, pady=5)
        ttk.Checkbutton(dlg, text="Through-Focus Contrast", variable=var_c).pack(anchor=tk.W, padx=20, pady=5)
        ttk.Checkbutton(dlg, text="Through-Focus Heatmap", variable=var_h).pack(anchor=tk.W, padx=20, pady=5)
        
        def on_export():
            dlg.destroy()
            if not (var_1d.get() or var_c.get() or var_h.get()):
                return
            
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                title="Export Simulation Data"
            )
            if not filepath:
                return
                
            try:
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    x_axis, prof = self.current_1d
                    
                    if var_1d.get():
                        writer.writerow(["--- 1D Image Profile ---"])
                        writer.writerow(["Position (nm)", "Intensity (a.u.)"])
                        for xx, pp in zip(x_axis, prof):
                            writer.writerow([xx, pp])
                        writer.writerow([])
                        
                    if var_c.get():
                        writer.writerow(["--- Through-Focus Contrast ---"])
                        writer.writerow(["Focus (um)", "Contrast"])
                        for ff, cc in zip(self.current_foc_list, self.current_c_list):
                            writer.writerow([ff, cc])
                        writer.writerow([])
                        
                    if var_h.get():
                        writer.writerow(["--- Through-Focus Heatmap ---"])
                        writer.writerow(["Focus (um) \\ Position (nm)"] + list(x_axis))
                        for i, ff in enumerate(self.current_foc_list):
                            writer.writerow([ff] + list(self.current_p_list[i]))
                        writer.writerow([])
                        
                self.status_var.set(f"Successfully exported data to CSV.")
                messagebox.showinfo("Export Successful", f"Data saved to {filepath}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to save CSV:\n{str(e)}")

        ttk.Button(dlg, text="Export", command=on_export).pack(pady=20)

if __name__ == "__main__":
    app = PartialCoherenceApp()
    app.mainloop()
