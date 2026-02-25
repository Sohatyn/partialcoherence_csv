# Partial Coherence Imaging Simulator

This is a personal hobby project for performing partial coherence analysis on optical systems with a defined Numerical Aperture (NA). It simulates the aerial images typically found in semiconductor exposure tools (lithography scanners) using Abbe's method. I built this while trying out Antigravity for the first time.

If you are curious about exposure devices, optics, or computational lithography, please give it a try!

## Features

- **Interactive UI**: A Tkinter-based GUI for real-time parameter configuration, with Matplotlib visualizations.
- **Customizable Optical Parameters**: Adjust Wavelength ($\lambda$), Lens NA, and Illumination coherence ($\sigma$).
- **Mask Definition**: Simulate Line and Space (L&S) patterns with adjustable width, number of lines, and orientation (Vertical, Horizontal, or Both).
- **Aberrations**: Full support for 36 standard Fringe Zernike Coefficients (in waves) for analyzing aberrations like Coma, Astigmatism, and Spherical.
- **Through-Focus Sweep**: Simulate images through various focus steps to evaluate the depth of focus and contrast curves.
- **Visualizations**: View 1D intensity profiles, 2D aerial images, through-focus contrast curves, and through-focus intensity heatmaps.
- **Data Export**: Easily export the resulting 1D profiles, contrast curves, and heatmaps to CSV format.

## Principle of Partial Coherence Imaging

### Abbe's Formulation
The main simulation engine implements Abbe's theorem by modeling the illumination as an incoherent sum of coherent point sources. The aerial image intensity $I(x,y)$ is computed by integrating the mutually independent coherent images formed by each point source $(s_x, s_y)$ in the continuous illumination pupil limit (e.g., standard disk $\sigma$):

$$ I(x,y) = \int \int_{\text{source}} \left| E_{\text{img}}(x, y ; s_x, s_y) \right|^2 ds_x ds_y $$

For a single given source $(s_x, s_y)$, the incident wave plane tilts across the mask. The mask's diffracted spectrum passes through the objective pupil $P(f_x, f_y)$ to form the coherent image $E_{\text{img}}$:

$$ E_{\text{img}}(x, y; s_x, s_y) = \mathcal{F}^{-1} \left\\{ \mathcal{F} \Big\\{ M(x, y) \cdot e^{i 2\pi (s_x x + s_y y)} \Big\\} \cdot P(f_x, f_y) \right\\} $$

where:
- $M(x, y)$ is the complex transmission function of the photomask.
- $\mathcal{F}$ and $\mathcal{F}^{-1}$ represent the Spatial 2-Dimensional Fourier and Inverse Fourier Transforms respectively.

### Pupil Function & Aberrations

The generalized pupil function is described by the numerical aperture cutoff $\rho \le 1$ and total phase error $\Phi_{\text{total}}$, where $\rho = \frac{\sqrt{f_x^2 + f_y^2}}{\text{NA} / \lambda}$:

$$ P(\rho, \theta) = \Pi(\rho) \cdot \exp( i \Phi_{\text{total}} ) $$

$\Phi_{\text{total}}$ consists of modeled **Wavefront Aberrations** and the exact scalar **Defocus** phase shift. 
Aberrations are injected as a weighted sum of Fringe Zernike polynomials $Z_j(\rho, \theta)$ with user-defined coefficients $c_j$ given in wave variants ($2\pi$ radians):

$$ \Phi_{\text{aberration}}(\rho, \theta) = 2\pi \sum_{j} c_j Z_j(\rho, \theta) $$

Defocus ($z$) propagation natively utilizes the scalar diffraction plane-wave propagation phase. While usually approximated parabolically in low NA, to avoid inaccuracies at high NA, the formula evaluated is the rigorous form:

$$ \Phi_{\text{defocus}}(f_x, f_y; z) = \frac{2\pi}{\lambda} z \sqrt{1 - (\lambda f_x)^2 - (\lambda f_y)^2} $$

## Requirements

The application requires Python 3 and the following dependencies:
- `numpy`
- `matplotlib`

You can install the required packages using pip:
```bash
pip install numpy matplotlib
```

## Usage

Run the main application from the terminal:

```bash
python main.py
```

### Optical Parameters
1. **Wavelength $\lambda$ (nm)**: Source illumination wavelength (e.g., 365.0 for i-line, 193.0 for ArF).
2. **Lens NA**: Numerical Aperture of the objective lens.
3. **Illumination $\sigma$**: Partial coherence factor of the illumination source (0 to 1).
4. **Focus & Sweep**: Set the central focus position and perform a sweep over a defined range and step size.
5. **L&S Width (nm)**: Width of the lines corresponding to your mask pattern.
6. **Precision**: Choose between `Fast (Rough)` for quick explorations or `High (Slow)` for detailed, high-resolution rendering.

### Zernike Coefficients
Expand the bottom left section to input values (in waves) for up to 36 Fringe Zernike aberrations to see their impact on the aerial image and contrast. The simulator utilizes the standard 36 Fringe Zernike polynomials $Z_j(\rho, \theta)$, where $\rho$ is the normalized pupil radius and $\theta$ is the azimuthal angle.

Here are the details for the all 36 primary aberrations:

| Fringe Index ($j$) | Radial $n$, Azimuthal $m$ | Aberration Name | Polynomial $Z_j(\rho, \theta)$ |
|:---:|:---:|:---|:---|
| 1 | 0, 0 | Piston | $1$ |
| 2 | 1, 1 | X Tilt | $\rho \cos(\theta)$ |
| 3 | 1, -1 | Y Tilt | $\rho \sin(\theta)$ |
| 4 | 2, 0 | Defocus | $2\rho^2 - 1$ |
| 5 | 2, 2 | Primary Astigmatism X | $\rho^2 \cos(2\theta)$ |
| 6 | 2, -2 | Primary Astigmatism Y | $\rho^2 \sin(2\theta)$ |
| 7 | 3, 1 | Primary Coma X | $(3\rho^3 - 2\rho) \cos(\theta)$ |
| 8 | 3, -1 | Primary Coma Y | $(3\rho^3 - 2\rho) \sin(\theta)$ |
| 9 | 4, 0 | Primary Spherical | $6\rho^4 - 6\rho^2 + 1$ |
| 10 | 3, 3 | Primary Trefoil X | $\rho^3 \cos(3\theta)$ |
| 11 | 3, -3 | Primary Trefoil Y | $\rho^3 \sin(3\theta)$ |
| 12 | 4, 2 | Secondary Astigmatism X | $(4\rho^4 - 3\rho^2) \cos(2\theta)$ |
| 13 | 4, -2 | Secondary Astigmatism Y | $(4\rho^4 - 3\rho^2) \sin(2\theta)$ |
| 14 | 5, 1 | Secondary Coma X | $(10\rho^5 - 12\rho^3 + 3\rho) \cos(\theta)$ |
| 15 | 5, -1 | Secondary Coma Y | $(10\rho^5 - 12\rho^3 + 3\rho) \sin(\theta)$ |
| 16 | 6, 0 | Secondary Spherical | $20\rho^6 - 30\rho^4 + 12\rho^2 - 1$ |
| 17 | 4, 4 | Primary Tetrafoil X | $\rho^4 \cos(4\theta)$ |
| 18 | 4, -4 | Primary Tetrafoil Y | $\rho^4 \sin(4\theta)$ |
| 19 | 5, 3 | Secondary Trefoil X | $(5\rho^5 - 4\rho^3) \cos(3\theta)$ |
| 20 | 5, -3 | Secondary Trefoil Y | $(5\rho^5 - 4\rho^3) \sin(3\theta)$ |
| 21 | 6, 2 | Tertiary Astigmatism X | $(15\rho^6 - 20\rho^4 + 6\rho^2) \cos(2\theta)$ |
| 22 | 6, -2 | Tertiary Astigmatism Y | $(15\rho^6 - 20\rho^4 + 6\rho^2) \sin(2\theta)$ |
| 23 | 7, 1 | Tertiary Coma X | $(35\rho^7 - 60\rho^5 + 30\rho^3 - 4\rho) \cos(\theta)$ |
| 24 | 7, -1 | Tertiary Coma Y | $(35\rho^7 - 60\rho^5 + 30\rho^3 - 4\rho) \sin(\theta)$ |
| 25 | 8, 0 | Tertiary Spherical | $70\rho^8 - 140\rho^6 + 90\rho^4 - 20\rho^2 + 1$ |
| 26 | 5, 5 | Primary Pentafoil X | $\rho^5 \cos(5\theta)$ |
| 27 | 5, -5 | Primary Pentafoil Y | $\rho^5 \sin(5\theta)$ |
| 28 | 6, 4 | Secondary Tetrafoil X | $(6\rho^6 - 5\rho^4) \cos(4\theta)$ |
| 29 | 6, -4 | Secondary Tetrafoil Y | $(6\rho^6 - 5\rho^4) \sin(4\theta)$ |
| 30 | 7, 3 | Tertiary Trefoil X | $(21\rho^7 - 30\rho^5 + 10\rho^3) \cos(3\theta)$ |
| 31 | 7, -3 | Tertiary Trefoil Y | $(21\rho^7 - 30\rho^5 + 10\rho^3) \sin(3\theta)$ |
| 32 | 8, 2 | Quaternary Astigmatism X | $(56\rho^8 - 105\rho^6 + 60\rho^4 - 10\rho^2) \cos(2\theta)$ |
| 33 | 8, -2 | Quaternary Astigmatism Y | $(56\rho^8 - 105\rho^6 + 60\rho^4 - 10\rho^2) \sin(2\theta)$ |
| 34 | 9, 1 | Quaternary Coma X | $(126\rho^9 - 280\rho^7 + 210\rho^5 - 60\rho^3 + 5\rho) \cos(\theta)$ |
| 35 | 9, -1 | Quaternary Coma Y | $(126\rho^9 - 280\rho^7 + 210\rho^5 - 60\rho^3 + 5\rho) \sin(\theta)$ |
| 36 | 10, 0 | Quaternary Spherical | $252\rho^{10} - 630\rho^8 + 560\rho^6 - 210\rho^4 + 30\rho^2 - 1$ |

### Simulation Modes
- **Run Full Simulation**: Computes the through-focus calculations, providing heatmaps and contrast curves.
- **Run 2D & Profile Only**: Computes only the specific focus position (faster, great for tuning).
- **Export to CSV**: Saves the currently displayed simulation data to a CSV file.

## Acknowledgements
Thanks to the Antigravity system for the development experience.
