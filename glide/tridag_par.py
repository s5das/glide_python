'''
Author: gaofei gg21009100584@gmail.com
Date: 2025-04-16 15:09:21
Description: 
'''
import numpy as np

def tridag_par(a, b, c, r):
    """
    Solve multiple tridiagonal systems in parallel (along columns).
    
    Parameters:
        a, b, c: (n, chunk) arrays - lower, main, and upper diagonals
        r:      (n, chunk) array - right-hand side
    Returns:
        u:      (n, chunk) array - solution
    """
    n, chunk = b.shape
    u = np.zeros((n, chunk))
    gam = np.zeros((n, chunk))
    bet = b[0, :].copy()

    # First row solution
    u[0, :] = r[0, :] / bet

    # Forward sweep
    for j in range(1, n):
        gam[j, :] = c[j - 1, :] / bet
        bet = b[j, :] - a[j, :] * gam[j, :]
        u[j, :] = (r[j, :] - a[j, :] * u[j - 1, :]) / bet

    # Back substitution
    for j in range(n - 2, -1, -1):
        u[j, :] -= gam[j + 1, :] * u[j + 1, :]

    return u