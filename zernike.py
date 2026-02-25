import numpy as np
import math

# Mapping of Fringe Zernike index (1-36) to (n, m) radial degree and angular frequency
FRINGE_36 = {
    1: (0, 0),
    2: (1, 1),
    3: (1, -1),
    4: (2, 0),
    5: (2, 2),
    6: (2, -2),
    7: (3, 1),
    8: (3, -1),
    9: (4, 0),
    10: (3, 3),
    11: (3, -3),
    12: (4, 2),
    13: (4, -2),
    14: (5, 1),
    15: (5, -1),
    16: (6, 0),
    17: (4, 4),
    18: (4, -4),
    19: (5, 3),
    20: (5, -3),
    21: (6, 2),
    22: (6, -2),
    23: (7, 1),
    24: (7, -1),
    25: (8, 0),
    26: (5, 5),
    27: (5, -5),
    28: (6, 4),
    29: (6, -4),
    30: (7, 3),
    31: (7, -3),
    32: (8, 2),
    33: (8, -2),
    34: (9, 1),
    35: (9, -1),
    36: (10, 0),
    37: (12, 0)
}

def radial_polynomial(n, m, rho):
    """
    Computes the radial polynomial R_n^|m|(rho) for Zernike polynomials.
    """
    m_abs = abs(m)
    if (n - m_abs) % 2 != 0:
        return np.zeros_like(rho)
    
    R = np.zeros_like(rho, dtype=float)
    for k in range((n - m_abs) // 2 + 1):
        num = ((-1)**k) * math.factorial(n - k)
        den = math.factorial(k) * math.factorial((n + m_abs) // 2 - k) * math.factorial((n - m_abs) // 2 - k)
        R += (num / den) * rho**(n - 2 * k)
    return R

def zernike_polynomial(j, rho, theta):
    """
    Computes the Fringe Zernike polynomial Z_j over polar coordinates (rho, theta).
    rho: numpy array of radial coordinates (normalized to 1 at pupil edge)
    theta: numpy array of angular coordinates
    j: Fringe index (1 to 37)
    Returns a numpy array of the same shape as rho and theta.
    Note: These polynomials are NOT normalized to rms of 1. They are the standard 
    fringe Zernike terms with maximum absolute value of 1 at rho=1 (except some constants).
    """
    if j not in FRINGE_36:
        raise ValueError(f"Fringe Zernike index {j} is not defined in this library (only 1-37).")
        
    n, m = FRINGE_36[j]
    
    R = radial_polynomial(n, m, rho)
    
    if m > 0:
        Z = R * np.cos(m * theta)
    elif m < 0:
        Z = R * np.sin(abs(m) * theta)
    else:
        Z = R
        
    # Mask out values outside the unit circle
    Z[rho > 1.0] = 0.0
    return Z

def generate_aberration_phase(coefficients, rho, theta):
    """
    Generate the total wavefront aberration phase (optical path difference) 
    given a list of Zernike coefficients and polar coordinates.
    coefficients: list or array of 36 or 37 coefficients corresponding to j=1 to len(coefficients)
    Returns: Phase array corresponding to the sum.
    """
    phase = np.zeros_like(rho, dtype=float)
    for i, coef in enumerate(coefficients):
        if coef != 0.0:
            j = i + 1  # Fringe index starts at 1
            phase += coef * zernike_polynomial(j, rho, theta)
    return phase

if __name__ == "__main__":
    # Test
    rho = np.linspace(0, 1, 100)
    theta = np.zeros_like(rho)
    # Z9 = Spherical = 6*rho^4 - 6*rho^2 + 1
    z9 = zernike_polynomial(9, rho, theta)
    print("Z9 at rho=1:", z9[-1])  # Should be 1
    print("Z9 at rho=0:", z9[0])   # Should be 1
    print("Z9 at rho=0.5:", z9[50]) # Should be expected value
