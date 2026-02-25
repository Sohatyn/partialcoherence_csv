import numpy as np
import zernike

def generate_mask(Nx, Ny, pixel_size, line_width_nm, num_lines=5, orientation='V'):
    """
    Generate a 2D transmission mask of alternating dark lines and bright spaces.
    Dark line = 0.0, Bright space = 1.0 (transparent background).
    """
    mask = np.ones((Ny, Nx), dtype=float)
    center_x = Nx // 2
    center_y = Ny // 2
    
    # line width in pixels
    lw_p = int(round(line_width_nm / pixel_size))
    if lw_p == 0:
        lw_p = 1
        
    start_idx = -(num_lines // 2)
    end_idx = start_idx + num_lines
    
    # Pitch = 2 * lw_p
    if orientation == 'V':
        for i in range(start_idx, end_idx):
            cx = center_x + i * 2 * lw_p
            start = cx - lw_p // 2
            end = start + lw_p
            # Make sure bounds are within array
            start = max(0, start)
            end = min(Nx, end)
            mask[:, start:end] = 0.0
    else:  # Horizontal
        for i in range(start_idx, end_idx):
            cy = center_y + i * 2 * lw_p
            start = cy - lw_p // 2
            end = start + lw_p
            start = max(0, start)
            end = min(Ny, end)
            mask[start:end, :] = 0.0
            
    return mask

def get_source_points(NA, sigma, lambda_nm, num_points=100):
    """
    Generate a uniform grid of source points (fx, fy) within the illumination pupil.
    """
    if sigma == 0:
        return np.array([[0.0, 0.0]])
        
    Rs = sigma * NA / lambda_nm
    # Calculate grid size to roughly achieve num_points inside the circle
    N = int(np.sqrt(num_points * 4 / np.pi))
    if N < 2: N = 2
    
    s_1d = np.linspace(-Rs, Rs, N)
    sx, sy = np.meshgrid(s_1d, s_1d)
    
    # Keep points inside the circle
    valid = (sx**2 + sy**2) <= Rs**2
    sx = sx[valid]
    sy = sy[valid]
    
    return np.column_stack((sx, sy))

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
        source_points = get_source_points(NA, sigma, lambda_nm, 100)
    
    intensity = np.zeros((Ny, Nx), dtype=float)
    
    for sx, sy in source_points:
        # Incident wave tilts the mask field
        mask_tilted = mask * np.exp(1j * 2 * np.pi * (sx * X + sy * Y))
        
        # Mask Spectrum
        E_f = np.fft.fft2(mask_tilted)
        
        # Apply Pupil filter
        E_img_f = E_f * P
        
        # Inverse FFT to get image field
        E_img = np.fft.ifft2(E_img_f)
        
        # Add to total intensity (incoherent sum of coherent images)
        intensity += np.abs(E_img)**2
        
    return intensity / len(source_points)

def calculate_contrast(image, line_width_nm, pixel_size_nm, orientation='V'):
    """
    Calculate the contrast at the center of the image.
    I_min is the minimum at the center line.
    I_max is the maximum strictly in the space between the center line and the adjacent line.
    """
    Ny, Nx = image.shape
    cy, cx = Ny // 2, Nx // 2
    lw_p = int(round(line_width_nm / pixel_size_nm))
    
    if orientation == 'V':
        profile = image[cy, :]
        # Center line region: [cx - lw_p/2, cx + lw_p/2]
        start_c = max(0, cx - lw_p//2)
        end_c = min(Nx, cx + lw_p//2 + 1)
        center_region = profile[start_c:end_c]
        I_min = np.min(center_region) if len(center_region) > 0 else profile[cx]
        
        # Space strictly between center line and adjacent right line: [cx + lw_p/2, cx + 1.5*lw_p]
        start_s = max(0, cx + lw_p//2)
        end_s = min(Nx, cx + int(1.5 * lw_p) + 1)
        space_region = profile[start_s:end_s]
        I_max = np.max(space_region) if len(space_region) > 0 else profile[cx + lw_p]
    else:
        profile = image[:, cx]
        # Center line region
        start_c = max(0, cy - lw_p//2)
        end_c = min(Ny, cy + lw_p//2 + 1)
        center_region = profile[start_c:end_c]
        I_min = np.min(center_region) if len(center_region) > 0 else profile[cy]
        
        # Space strictly between center line and adjacent bottom line
        start_s = max(0, cy + lw_p//2)
        end_s = min(Ny, cy + int(1.5 * lw_p) + 1)
        space_region = profile[start_s:end_s]
        I_max = np.max(space_region) if len(space_region) > 0 else profile[cy + lw_p]
        
    if I_max + I_min == 0:
        return 0.0
    return (I_max - I_min) / (I_max + I_min)

def sweep_focus(mask, NA, sigma, lambda_nm, focus_list, zernike_coeffs, pixel_size_nm, orientation='V', num_source=100):
    """
    Sweep through a list of focus values and return the contrast at each focus.
    """
    source_points = get_source_points(NA, sigma, lambda_nm, num_points=num_source)
    contrasts = []
    
    for f in focus_list:
        img = simulate_image(mask, NA, sigma, lambda_nm, f, zernike_coeffs, pixel_size_nm, source_points)
        c = calculate_contrast(img, line_width_nm=0, pixel_size_nm=pixel_size_nm, orientation=orientation)
        # Note: actually we need to pass line_width_nm. Let's fix loop to accept it as param.
    return contrasts

def run_through_focus(line_width_nm, NA, sigma, lambda_nm, focus_list, zernike_coeffs, num_lines=5, orientation='V', Nx=512, Ny=512, pixel_size_nm=2.0, num_source=100):
    mask = generate_mask(Nx, Ny, pixel_size_nm, line_width_nm, num_lines, orientation)
    source_points = get_source_points(NA, sigma, lambda_nm, num_points=num_source)
    
    contrasts = []
    profiles = []
    cx, cy = Nx // 2, Ny // 2
    
    for f in focus_list:
        img = simulate_image(mask, NA, sigma, lambda_nm, f, zernike_coeffs, pixel_size_nm, source_points)
        c = calculate_contrast(img, line_width_nm, pixel_size_nm, orientation)
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
