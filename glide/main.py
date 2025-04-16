'''
Author: gaofei gg21009100584@gmail.com
Date: 2025-04-16 19:15:10
Description: 
'''
from glide.moudule_define  import (Parm,Matr)
from glide.initialize_parameters import initialize_parameters
from glide.build_matrices import build_matrices
from glide.syn_ages import syn_ages
from glide.find_zc import find_zc
from glide.solve_inverse import solve_inverse
import numpy as np
import os


def glide():

    misfit = np.zeros(2)
    params = Parm()
    matrices = Matr()
    initialize_parameters(matrices,params)
    build_matrices(matrices,params)
    matrices.edot = params.edot_mean
    matrices.edot_pr2 = params.edot_mean
    matrices.edot_dat = matrices.edot_pr
    misfit[0] = syn_ages (matrices,params)

    file_path = os.path.join(params.run, 'residualsLOG.txt')
    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass  # File doesn't exist, which is fine

    residu = 1.0e6
    residup = 0.0
    iter = 0
    while iter < params.iterM:
        residup = residu
        iter += 1
        print(f"solving inverse, iteration = {iter}")
        
        residu = solve_inverse(matrices, params, iter)
        
        print("computing closure depths")
        find_zc(matrices, params)

        print(f"changes in residuals {abs(residup - residu)/residu}")  

    return True

if __name__=="__main__":
    glide()