'''
Author: gaofei gg21009100584@gmail.com
Date: 2025-04-16 15:10:43
Description: 
'''
import numpy as np

def dodson(temp, time, nstep, iflag):
    """
    Compute closure temperature and age based on Dodson's equation.

    Parameters:
        temp (np.ndarray): Temperature history [°C], shape (nstep,)
        time (np.ndarray): Time history [Ma], shape (nstep,)
        nstep (int): Number of timesteps
        iflag (int): Thermochronometer type selector (1–7)

    Returns:
        age (float): Closure age [Ma]
        closure (float): Closure temperature [°C]
        cooling (float): Local cooling rate at closure [°C/Ma]
    """

    # Activation energy (J/mol) and D0/a² (1/sec)
    if iflag == 1:  # AFT
        energy = 147e3
        diff = 2.05e6 * 3600 * 24 * 365.25e6
    elif iflag == 2:  # ZFT
        energy = 208e3
        diff = 4.0e8 * 3600 * 24 * 365.25e6
    elif iflag == 3:  # AHe
        energy = 138e3
        diff = 7.64e7 * 3600 * 24 * 365.25e6
    elif iflag == 4:  # ZHe
        energy = 169e3
        diff = 7.03e5 * 3600 * 24 * 365.25e6
    elif iflag == 5:  # Ar-Ar hbl
        energy = 268e3
        diff = 1320 * 3600 * 24 * 365.25e6
    elif iflag == 6:  # muscovite
        energy = 180e3
        diff = 3.91 * 3600 * 24 * 365.25e6
    elif iflag == 7:  # biotite
        energy = 197e3
        diff = 733.0 * 3600 * 24 * 365.25e6
    else:
        raise ValueError("Invalid iflag. Must be in 1–7.")

    geom = 1.0
    R = 8.314  # J/mol·K

    age = time[0]  # Default age if closure is never passed

    for i in range(nstep - 1, 0, -1):
        # Cooling rate computation
        if i == 1:
            cooling = (temp[i + 1] - temp[i]) / (time[i + 1] - time[i])
        elif i == nstep - 1:
            cooling = (temp[i] - temp[i - 1]) / (time[i] - time[i - 1])
        else:
            cooling = (temp[i + 1] - temp[i - 1]) / (time[i + 1] - time[i - 1])

        cooling = max(cooling, 1.0 / 10.0)  # At least 1°C/10Ma

        T_K = temp[i] + 273.0
        tau = R * T_K**2 / (energy * cooling)
        closure = energy / (R * np.log(geom * tau * diff)) - 273.0

        if temp[i] > closure:
            ratio = (closure_prev - temp_prev) / ((closure_prev - temp_prev) + (temp[i] - closure))
            age = time[i] + (time[i - 1] - time[i]) * ratio
            return age, closure, cooling

        closure_prev = closure
        temp_prev = temp[i]

    return age, closure, cooling