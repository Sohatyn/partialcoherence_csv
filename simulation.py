import numpy as np
import zernike
import os
import csv
from PIL import Image

def generate_mask(Nx, Ny, pixel_size, line_width_nm, num_lines=5, orientation='V', space_width_nm=None, invert=False):
    """
    Generate a 2D transmission mask of alternating dark lines and bright spaces.
    If invert=False (default): Dark line = 0.0, Bright space = 1.0 (transparent background).
    If invert=True: Bright line = 1.0, Dark space = 0.0 (opaque background).
    """
    if space_width_nm is None:
        space_width_nm = line_width_nm
        
    bg_val = 0.0 if invert else 1.0
    line_val = 1.0 if invert else 0.0
    
    mask = np.full((Ny, Nx), bg_val, dtype=float)
    center_x = Nx // 2
    center_y = Ny // 2
    
    # line width and space width in pixels
    lw_p = int(round(line_width_nm / pixel_size))
    if lw_p == 0:
        lw_p = 1
        
    sw_p = int(round(space_width_nm / pixel_size))
    if sw_p == 0:
        sw_p = 1
        
    pitch_p = lw_p + sw_p
        
    start_idx = -(num_lines // 2)
    end_idx = start_idx + num_lines
    
    # Calculate offset so that the middle line (index 0) is centered at center_x or center_y
    if orientation == 'V':
        for i in range(start_idx, end_idx):
            cx = center_x + i * pitch_p
            start = cx - lw_p // 2
            end = start + lw_p
            # Make sure bounds are within array
            start = max(0, start)
            end = min(Nx, end)
            mask[:, start:end] = line_val
    else:  # Horizontal
        for i in range(start_idx, end_idx):
            cy = center_y + i * pitch_p
            start = cy - lw_p // 2
            end = start + lw_p
            start = max(0, start)
            end = min(Ny, end)
            mask[start:end, :] = line_val
            
    return mask

def load_custom_pattern(filepath):
    """
    Load a 2D pattern from CSV, DAT, or BMP file.
    Returns a numpy array of 0s and 1s.
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.csv':
        # Try reading as comma-separated values
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            data = []
            for row in reader:
                # Filter out empty strings which might happen at trailing commas
                r = [float(x) for x in row if x.strip() != '']
                if r:
                    data.append(r)
        return np.flipud(np.array(data))
        
    elif ext == '.dat':
        # Try numpy loadtxt (handles space or tab separated)
        try:
            arr = np.loadtxt(filepath, delimiter=',')
        except ValueError:
            arr = np.loadtxt(filepath)
        return np.flipud(arr)
            
    elif ext == '.bmp':
        # Load image, convert to grayscale, then threshold
        img = Image.open(filepath).convert('L')
        img_arr = np.array(img)
        # Threshold: assume typical 0-255 range, split at 127
        arr = (img_arr > 127).astype(float)
        return np.flipud(arr)
        
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def prepare_custom_mask(data, cell_size_nm, invert=False, target_size=512):
    """
    Prepares the mask for simulation.
    Default: 0 in data = 1.0 (pass), 1 in data = 0.0 (block).
    If invert is True: 0 -> 0.0 (block), 1 -> 1.0 (pass).
    Upsamples the data to match roughly the target_size for FFT stability.
    Returns (mask_array, pixel_size_nm)
    """
    # 1. Apply rules: standard is 0->1.0 (pass), 1->0.0 (block)
    # Threshold at 0.5 to handle generic float inputs
    bin_data = (data > 0.5)
    
    if invert:
        # invert=True: 0->0.0, 1->1.0
        mask_base = bin_data.astype(float)
    else:
        # invert=False: 0->1.0, 1->0.0
        mask_base = (~bin_data).astype(float)
        
    # 2. Upsample
    # Find scale factor to make the max dimension at least target_size
    ny, nx = mask_base.shape
    max_dim = max(nx, ny)
    scale = max(1, target_size // max_dim)
    
    # Scale both dimensions using repeat (nearest neighbor scaling)
    mask = np.repeat(np.repeat(mask_base, scale, axis=0), scale, axis=1)
    
    # physical size of one pixel in the scaled mask
    pixel_size_nm = cell_size_nm / scale
    
    return mask, pixel_size_nm

def get_source_points(NA, sigma, lambda_nm, num_points=100, shape='Top-hat', sigma_gauss=1.0):
    """
    Generate a grid of source points (fx, fy, weight) within the illumination pupil.
    
    shape: 'Top-hat' or 'Gaussian'
    sigma_gauss: 1/sigma parameter for Gaussian (Default=1.0 means 13.5% at Rs)
    """
    if sigma == 0:
        return np.array([[0.0, 0.0, 1.0]])
        
    Rs = sigma * NA / lambda_nm
    
    # For Gaussian, we might want to sample slightly further than Rs if sigma represents std dev,
    # but in lithography terms, sigma usually defines the integration boundary.
    # We will sample within Rs.
    
    N = int(np.sqrt(num_points * 4 / np.pi))
    if N < 2: N = 2
    
    s_1d = np.linspace(-Rs, Rs, N)
    sx, sy = np.meshgrid(s_1d, s_1d)
    
    # Keep points inside the circle
    r2 = sx**2 + sy**2
    valid = r2 <= Rs**2
    sx = sx[valid]
    sy = sy[valid]
    r2 = r2[valid]
    
    if shape == 'Gaussian':
        # Gaussian weight: exp(-2 * (r / (Rs * sigma_gauss))^2)
        # At r = Rs, if sigma_gauss = 1.0, weight = exp(-2) ~ 0.135
        weights = np.exp(-2 * (r2 / (Rs * sigma_gauss)**2))
    else:
        weights = np.ones_like(sx)
        
    return np.column_stack((sx, sy, weights))

def simulate_image(mask, NA, sigma, lambda_nm, focus_nm, zernike_coeffs, pixel_size_nm, source_points=None):
    """
    Simulate the aerial image using Abbe's method.
    Returns the 2D intensity profile.
    """
    Ny, Nx = mask.shape
    X, Y = np.meshgrid(np.arange(Nx) * pixel_size_nm, np.arange(Ny) * pixel_size_nm)
    # Center coordinates
    X -= X[Ny//2, Nx//2]
    Y -= Y[Ny//2, Nx//2]
    
    # Frequency grids
    fx = np.fft.fftfreq(Nx, d=pixel_size_nm)
    fy = np.fft.fftfreq(Ny, d=pixel_size_nm)
    FX, FY = np.meshgrid(fx, fy)
    
    # Pupil coordinates
    rho = np.sqrt(FX**2 + FY**2) / (NA / lambda_nm)
    theta = np.arctan2(FY, FX)
    
    # Pupil limits
    pupil_support = (rho <= 1.0)
    
    # Zernike Aberrations
    # Coefficient array may omit piston, so pass directly to generator
    W_zernike = zernike.generate_aberration_phase(zernike_coeffs, rho, theta)
    
    # Defocus Phase
    # Exact scalar propagation phase: kz = 2pi/lambda * sqrt(1 - (lambda*fx)^2 - (lambda*fy)^2)
    # For Focus shift z, phase change is kz * z.
    # We take the real part to zero out evanescent waves.
    sq = 1.0 - (lambda_nm * FX)**2 - (lambda_nm * FY)**2 - 0j
    kz = (2 * np.pi / lambda_nm) * np.sqrt(sq).real
    defocus_phase = kz * focus_nm
    
    # Total Pupil
    # P = P0 * exp(i * (2pi * W_zernike)) * exp(i * defocus_phase)
    # Note: W_zernike is expected in units of waves, so we multiply by 2pi.
    phase_total = 2 * np.pi * W_zernike + defocus_phase
    P = pupil_support * np.exp(1j * phase_total)
    
    # Illumination Source
    if source_points is None:
        source_points = get_source_points(NA, sigma, lambda_nm, 100, shape='Top-hat')
    
    intensity = np.zeros((Ny, Nx), dtype=float)
    total_weight = 0.0
    
    for sx, sy, w in source_points:
        # Incident wave tilts the mask field
        mask_tilted = mask * np.exp(1j * 2 * np.pi * (sx * X + sy * Y))
        
        # Mask Spectrum
        E_f = np.fft.fft2(mask_tilted)
        
        # Apply Pupil filter
        E_img_f = E_f * P
        
        # Inverse FFT to get image field
        E_img = np.fft.ifft2(E_img_f)
        
        # Add to total intensity (incoherent sum of weighted coherent images)
        intensity += w * np.abs(E_img)**2
        total_weight += w
        
    return intensity / total_weight

def calculate_contrast(image, line_width_nm, pixel_size_nm, orientation='V', space_width_nm=None):
    """
    Calculate the contrast at the center of the image.
    I_min is the minimum at the center line.
    I_max is the maximum strictly in the space between the center line and the adjacent line.
    """
    if space_width_nm is None:
        space_width_nm = line_width_nm
        
    Ny, Nx = image.shape
    cy, cx = Ny // 2, Nx // 2
    lw_p = int(round(line_width_nm / pixel_size_nm))
    sw_p = int(round(space_width_nm / pixel_size_nm))
    pitch_p = lw_p + sw_p
    
    if orientation == 'V':
        profile = image[cy, :]
        # Center line region: [cx - lw_p/2, cx + lw_p/2]
        start_c = max(0, cx - lw_p//2)
        end_c = min(Nx, cx + lw_p//2 + 1)
        center_region = profile[start_c:end_c]
        I_min = np.min(center_region) if len(center_region) > 0 else profile[cx]
        
        # Space strictly between center line and adjacent right line: [cx + lw_p/2, cx + lw_p/2 + sw_p]
        start_s = max(0, cx + lw_p//2)
        end_s = min(Nx, cx + lw_p//2 + sw_p + 1)
        space_region = profile[start_s:end_s]
        I_max = np.max(space_region) if len(space_region) > 0 else profile[min(Nx-1, cx + lw_p//2 + sw_p//2)]
    else:
        profile = image[:, cx]
        # Center line region
        start_c = max(0, cy - lw_p//2)
        end_c = min(Ny, cy + lw_p//2 + 1)
        center_region = profile[start_c:end_c]
        I_min = np.min(center_region) if len(center_region) > 0 else profile[cy]
        
        # Space strictly between center line and adjacent bottom line
        start_s = max(0, cy + lw_p//2)
        end_s = min(Ny, cy + lw_p//2 + sw_p + 1)
        space_region = profile[start_s:end_s]
        I_max = np.max(space_region) if len(space_region) > 0 else profile[min(Ny-1, cy + lw_p//2 + sw_p//2)]
        
    if I_max + I_min == 0:
        return 0.0
    return (I_max - I_min) / (I_max + I_min)

def sweep_focus(mask, NA, sigma, lambda_nm, focus_list, zernike_coeffs, pixel_size_nm, orientation='V', num_source=100, shape='Top-hat', sigma_gauss=1.0, line_width_nm=0, space_width_nm=None):
    """
    Sweep through a list of focus values and return the contrast at each focus.
    """
    source_points = get_source_points(NA, sigma, lambda_nm, num_points=num_source, shape=shape, sigma_gauss=sigma_gauss)
    contrasts = []
    
    for f in focus_list:
        img = simulate_image(mask, NA, sigma, lambda_nm, f, zernike_coeffs, pixel_size_nm, source_points)
        c = calculate_contrast(img, line_width_nm=line_width_nm, pixel_size_nm=pixel_size_nm, orientation=orientation, space_width_nm=space_width_nm)
        contrasts.append(c)
    return contrasts

def run_through_focus(line_width_nm, NA, sigma, lambda_nm, focus_list, zernike_coeffs, num_lines=5, orientation='V', Nx=512, Ny=512, pixel_size_nm=2.0, num_source=100, shape='Top-hat', sigma_gauss=1.0, space_width_nm=None, invert=False):
    mask = generate_mask(Nx, Ny, pixel_size_nm, line_width_nm, num_lines, orientation, space_width_nm, invert)
    source_points = get_source_points(NA, sigma, lambda_nm, num_points=num_source, shape=shape, sigma_gauss=sigma_gauss)
    
    contrasts = []
    profiles = []
    cx, cy = Nx // 2, Ny // 2
    
    for f in focus_list:
        img = simulate_image(mask, NA, sigma, lambda_nm, f, zernike_coeffs, pixel_size_nm, source_points)
        c = calculate_contrast(img, line_width_nm, pixel_size_nm, orientation, space_width_nm)
        contrasts.append(c)
        if orientation == 'V':
            profiles.append(img[cy, :])
        else:
            profiles.append(img[:, cx])
        
    return contrasts, np.array(profiles)

if __name__ == "__main__":
    # Test simulation
    z_coeffs = np.zeros(36)
    Nx = 512
    Ny = 512
    pixel_size = 2.0
    lw = 40.0
    
    print("Generating Mask")
    m = generate_mask(Nx, Ny, pixel_size, lw, 5, 'V')
    print("Simulating Image")
    img = simulate_image(m, NA=1.35, sigma=0.8, lambda_nm=193.0, focus_nm=0.0, zernike_coeffs=z_coeffs, pixel_size_nm=pixel_size)
    c = calculate_contrast(img, lw, pixel_size, 'V')
    print(f"Contrast at focus 0: {c:.4f}")
    
    foc_list = np.linspace(-100, 100, 5)
    print("Running Through-Focus")
    clist, plist = run_through_focus(lw, 1.35, 0.8, 193.0, foc_list, z_coeffs, 5, 'V', Nx, Ny, pixel_size)
    for f, c in zip(foc_list, clist):
        print(f"Focus {f}nm: Contrast {c:.4f}")
