'''
Author: gaofei gg21009100584@gmail.com
Date: 2025-04-16 16:41:50
Description: 
'''
import numpy as np

def closure_temps(cooling, temp, iflag):
    """
    Calculates closure temperatures using Dodson's equation for a 1D array of material.

    Parameters
    ----------
    cooling : ndarray
        Cooling rates (°C/Myr), shape (mz,)
    temp : ndarray
        Temperatures (°C), shape (mz,)
    iflag : int
        Thermochronometer index, determines diffusion and activation energy:
            1 - AFT, 2 - ZFT, 3 - AHe, 4 - ZHe, 5 - Hbl, 6 - Mus, 7 - Bio

    Returns
    -------
    closure : ndarray
        Closure temperatures (°C), shape (mz,)
    """

    # Constants (activation energy in J/mol, D0/a² in s⁻¹)
    if iflag == 1:
        energy = 147e3
        diff = 2.05e6 * 3600 * 24 * 365.25e6
    elif iflag == 2:
        energy = 208e3
        diff = 4.0e8 * 3600 * 24 * 365.25e6
    elif iflag == 3:
        energy = 138e3
        diff = 7.64e7 * 3600 * 24 * 365.25e6
    elif iflag == 4:
        energy = 169e3
        diff = 7.03e5 * 3600 * 24 * 365.25e6
    elif iflag == 5:
        energy = 268e3
        diff = 1320 * 3600 * 24 * 365.25e6
    elif iflag == 6:
        energy = 180e3
        diff = 3.91 * 3600 * 24 * 365.25e6
    elif iflag == 7:
        energy = 197e3
        diff = 733. * 3600 * 24 * 365.25e6
    else:
        raise ValueError("Invalid iflag value. Must be 1-7.")

    R = 8.314  # Gas constant, J/mol/K
    geom = 1.0

    cooling = np.maximum(cooling, 1.0 / 10.0)  # Ensure no zero cooling

    temp_K = temp + 273.0  # Convert to Kelvin
    tau = R * temp_K**2 / (energy * cooling)
    closure = energy / (R * np.log(geom * tau * diff)) - 273.0  # Closure temp in °C

    return closure