import numpy as np
from glide.moudule_define import (Parm,Matr)
from glide.prior_zc import prior_zc
from typing import Tuple
from numba import jit

def initialize_parameters(matrices: Matr, params: Parm) -> Tuple[Matr, Parm]:
    """Python implementation of the Fortran initialize_parameters subroutine"""
    
    # Set random seed
    np.random.seed()
    
    # Read parameters from 'glide.in'
    with open('glide.in', 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if not line.startswith(('$', ' ','\n'))]
    
    params.run = lines[0][:5]
    params.topofile = lines[1]
    nx, ny = map(int, lines[2].split())
    datafile = lines[3]
    params.lon1, params.lon2, params.lat1, params.lat2 = map(float, lines[4].split())
    params.edot_mean, params.sigma2 = map(float, lines[5].split())
    params.xL, params.angle, params.aspect,_ = map(float, lines[6].split())
    params.deltat = float(lines[7])
    params.t_total = float(lines[8])
    params.zl, params.Ts, params.Tb, params.kappa, params.hp = map(float, lines[9].split())
    params.iterM, params.xmu = map(float, lines[10].split())
    
    # Adjust parameters
    params.hp = 0.0
    params.sigma2 = params.sigma2 ** 2
    
    # Calculate spacing and setup dummy points
    pi = np.pi
    spacin = 1.5 / params.xL
    
    # Calculate xl and yl (width and length of analysis area)
    colat = np.cos(((params.lat1 + params.lat2) / 2) * pi / 180)
    xl = (params.lon2 - params.lon1) * 111.111 * colat
    yl = (params.lat2 - params.lat1) * 111.111
    
    nx_dum = int(np.floor(xl / (spacin * params.xL))) + 1
    ny_dum = int(np.floor(yl / (spacin * params.xL))) + 1
    
    xl = (params.lon2 - params.lon1) / float(nx_dum)
    yl = (params.lat2 - params.lat1) / float(ny_dum)
    
    # Read data file (fission track data)
    with open(datafile.strip('"'), 'r') as f:
        data_lines = f.readlines()
    
    # Count valid data points
    k = 0
    for line in data_lines:
        parts = line.split()
        if len(parts) < 7:
            continue
        lo, la, tmp1, tmp2, tmp3, tmp4, ib = map(float, parts[:7])
        if lo > 180:
            lo -= 360
        if (tmp2 > params.t_total or 
            lo < params.lon1 or lo > params.lon2 or 
            la < params.lat1 or la > params.lat2):
            continue
        k += 1
    
    params.n = k
    params.contr = 0
    params.dummy = params.contr
    
    print(f"total ages {params.n}, control points={params.dummy}")
    
    # Allocate arrays
    matrices.ta = np.zeros(params.n)
    matrices.a_error = np.zeros(params.n)
    matrices.zc = np.zeros(params.n)
    matrices.zcp = np.zeros(params.n)
    matrices.x = np.zeros(params.n)
    matrices.y = np.zeros(params.n)
    matrices.elev = np.zeros(params.n)
    matrices.elev_true = np.zeros(params.n)
    matrices.x_true = np.zeros(params.n)
    matrices.y_true = np.zeros(params.n)
    matrices.misfits = np.zeros(params.n)
    matrices.syn_age = np.zeros(params.n)
    matrices.isys = np.zeros(params.n)
    matrices.depth1 = np.zeros(params.n)
    matrices.iblock = np.zeros(params.n, dtype=int)
    matrices.depth = np.zeros(params.n)
    matrices.zz = np.zeros(params.n)
    
    # Read data into arrays
    k = 0
    for line in data_lines:
        parts = line.split()
        if len(parts) < 7:
            continue
        lo, la, tmp1, tmp2, tmp3, sys, ib = map(float, parts[:7])
        if lo > 180:
            lo -= 360
        if (tmp2 > params.t_total or 
            lo < params.lon1 or lo > params.lon2 or 
            la < params.lat1 or la > params.lat2):
            continue
        
        matrices.x[k] = lo
        matrices.y[k] = la
        matrices.elev[k] = tmp1
        matrices.ta[k] = tmp2
        matrices.a_error[k] = tmp3
        matrices.isys[k] = sys
        matrices.iblock[k] = int(ib)
        k += 1
        if k == params.n:
            break
    
    matrices.elev_true = matrices.elev.copy()
    
    # Read dummy points
    with open('data_ours/grid_dummy.txt', 'r') as f:
        dummy_lines = f.readlines()
    
    # Count valid dummy points
    k = 0
    for line in dummy_lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        lo, la, ib = map(float, parts[:3])
        if lo > 180:
            lo -= 360
        if (lo < params.lon1 or lo > params.lon2 or 
            la < params.lat1 or la > params.lat2):
            continue
        k += 1
    
    params.dummy = k - 1 + params.contr
    print(f'number of dummy points={params.dummy}')
    
    # Allocate dummy arrays
    matrices.x_dum = np.zeros(params.dummy)
    matrices.y_dum = np.zeros(params.dummy)
    matrices.x_dum_true = np.zeros(params.dummy)
    matrices.y_dum_true = np.zeros(params.dummy)
    matrices.idum_block = np.zeros(params.dummy, dtype=int)
    
    # Read dummy points into arrays
    k = 0
    for line in dummy_lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        lo, la, ib = map(float, parts[:3])
        if lo > 180:
            lo -= 360
        if (lo < params.lon1 or lo > params.lon2 or 
            la < params.lat1 or la > params.lat2):
            continue
        
        matrices.x_dum[k] = lo
        matrices.y_dum[k] = la
        matrices.idum_block[k] = int(ib)
        k += 1
        if k == params.dummy - params.contr:
            break
    
    # Read control points if needed
    if params.contr > 0:
        with open("data/control_id.xy", 'r') as f:
            for i in range(params.contr):
                line = f.readline()
                parts = line.split()
                matrices.x_dum[k] = float(parts[0])
                matrices.y_dum[k] = float(parts[1])
                matrices.idum_block[k] = int(parts[2])
                k += 1
    
    print(np.min(matrices.y_dum), np.max(matrices.y_dum))
    print(np.min(matrices.x_dum), np.max(matrices.x_dum))
    
    # Store original positions
    matrices.x_dum_true = matrices.x_dum.copy()
    matrices.y_dum_true = matrices.y_dum.copy()
    matrices.x_true = matrices.x.copy()
    matrices.y_true = matrices.y.copy()
    
    # Convert to local coordinate system
    colat = np.cos(((params.lat1 + params.lat2) / 2 * pi / 180))
    for i in range(params.dummy):
        matrices.x_dum[i] = (matrices.x_dum[i] - params.lon1) * 111.11 * colat
        matrices.y_dum[i] = (matrices.y_dum[i] - params.lat1) * 111.11
    
    # Set default errors if needed
    matrices.a_error[matrices.a_error == 0] = 0.1
    
    # Read topography file
    with open(params.topofile.strip('"'), 'r') as f:
        topo_lines = f.readlines()
    
    # Process topography data (simplified from Fortran version)
    # Note: The full implementation would need to handle the grid properly
    nx0 = nx
    ny0 = ny
    skip = 1
    nx = (nx0 - 1) // skip + 1
    ny = (ny0 - 1) // skip + 1
    nx = nx0
    ny = ny0
    
    # Simplified topography processing - would need proper implementation
    lon_full = np.zeros((nx0, ny0))
    lat_full = np.zeros((nx0, ny0))
    topob_full = np.zeros((nx0, ny0))
    
    # This would need to be properly implemented based on the file format
    # For now, we'll just create dummy data
    @jit(nopython=True)
    def create_grid(ny0,nx0,lon_full,lat_full,topob_full,lon1,lon2,lat1,lat2):  
        for j in range(ny0):
            for i in range(nx0):
                # In a real implementation, you would parse the actual data
                lon_full[i, j] = lon1 + (lon2 - lon1) * i / (nx0 - 1)
                lat_full[i, j] = lat1 + (lat2 - lat1) * j / (ny0 - 1)
                topob_full[i, j] = 1000.0  # Dummy elevation
    create_grid(ny0,nx0,lon_full,lat_full,topob_full,params.lon1,params.lon2,params.lat1,params.lat2)

    lon = lon_full.copy()
    lat = lat_full.copy()
    topob = topob_full.copy()
    
    xmin = np.min(lon)
    xmax = np.max(lon)
    ymin = np.min(lat)
    ymax = np.max(lat)
    print(xmin, xmax, ymin, ymax)
    
    # Calculate time steps
    params.m_max = int(np.floor(params.t_total / params.deltat)) + 1
    matrices.tsteps = np.full(100, params.deltat)
    matrices.tsteps_sum = np.zeros(100)
    
    for i in range(100):
        if i > 0:
            matrices.tsteps_sum[i] = matrices.tsteps_sum[i-1] + matrices.tsteps[i]
        else:
            matrices.tsteps_sum[i] = matrices.tsteps[i]
    
    # Calculate total number of time steps
    summ = 0.0
    k = 1
    params.m_max = 0
    while summ < params.t_total:
        summ += matrices.tsteps[k-1]
        k += 1
    k -= 1
    k += 1
    params.m_max = k
    
    # Allocate more arrays
    matrices.edot_pr = np.zeros(params.n * params.m_max)
    matrices.edot_pr2 = np.zeros(params.n * params.m_max)
    matrices.edot_pr_dum = np.zeros(params.dummy * params.m_max)
    matrices.nsystems = np.zeros(8, dtype=int)
    
    # Build prior model
    xtime = params.t_total
    for j in range(params.m_max):
        xtime -= params.deltat
        for i in range(params.n):
            matrices.edot_pr[j + (i) * params.m_max] = params.edot_mean
        for i in range(params.dummy):
            matrices.edot_pr_dum[j + (i) * params.m_max] = params.edot_mean
    
    # Count samples with different systems
    for i in range(params.n):
        sys = matrices.isys[i]
        if sys == 1:
            matrices.nsystems[0] += 1
        elif sys == 2:
            matrices.nsystems[1] += 1
        elif sys == 3:
            matrices.nsystems[2] += 1
        elif sys == 4:
            matrices.nsystems[3] += 1
        elif sys == 5:
            matrices.nsystems[4] += 1
        elif sys == 6:
            matrices.nsystems[5] += 1
        elif sys == 7:
            matrices.nsystems[6] += 1
        elif sys < 0:
            matrices.nsystems[7] += 1
    
    matrices.ages = np.zeros((7, 5))
    
    # Calculate mean elevation and adjust temperature
    mean_elev = np.mean(topob)
    params.Ts = params.Ts - mean_elev * 0.006
    params.zl = params.zl + (mean_elev / 1000.0)
    
    print(f"Temperature of thermal model at z=0 {params.Ts} at an elevation of {mean_elev}")
    
    # Here you would call prior_zc and isotherms functions
    prior_zc(matrices, params)
    matrices.zz = matrices.zc.copy()
    # isotherms(matrices, params, nx, ny, topob, lon, lat)

    # # Deallocate arrays (in Python we just delete or let garbage collection handle it)
    # del lon, lat, topob

    # # convert to local coordinate system
    # for i in range(params.n):
    #     matrices.x[i] = (matrices.x_true[i] - params.lon1) * 111.11 * colat
    #     matrices.y[i] = (matrices.y_true[i] - params.lat1) * 111.11


    # for i in range(params.n):
    #     if matrices.isys[i] > 0:
    #         matrices.elev[i] = (matrices.elev[i] - matrices.zc[i]) / 1000.0
    #     else:
    #         matrices.elev[i] = 0.0
    #         matrices.zz[i] = -matrices.isys[i]

    # Print closure depths (dummy values)
    print("AFT= 0.0 km, 0.0 degC")
    print("ZFT= 0.0 km, 0.0 degC")
    print("AHE= 0.0 km, 0.0 degC")
    print("ZHE= 0.0 km, 0.0 degC")
    # matrices.zc=matrices.zz
    return matrices, params


