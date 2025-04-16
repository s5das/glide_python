import numpy as np
from numba import jit
from glide.closure_temps import closure_temps
from numba import jit

@jit(nopython=True)
def forward_process(xtime,dt,tend,tsteps_sum,edot_dat,kk,m_max,kappa,dz,mz,Ts,temp,hp,b_flux,ta,temp_pr,temp_age):
    exit_flag = 0
    j=0 
    query = np.zeros(100)
    diag = np.zeros(mz)
    sup = np.zeros(mz)
    inf = np.zeros(mz)
    f = np.zeros(mz) 
  
    # Main time stepping loop
    while xtime < tend:
        query[:] = xtime
        query = tsteps_sum - query
        query[query < 0.0] = -9999.0
        pos = np.argmin(query**2)
        exhum = edot_dat[pos + (kk) * m_max]
        
        # Constants used by tridag
        lambda_ = kappa * dt / (2.0 * (dz**2))
        alpha = exhum * dt / (4.0 * dz)
        ax = (alpha - lambda_)
        bx = (1.0 + (2.0 * lambda_))
        cx = -1.0 * (lambda_ + alpha)
        
        # Set up tridiagonal system
        for i in range(1, mz-1):
            diag[i] = bx
            sup[i] = cx
            inf[i] = ax
            f[i] = (lambda_ - alpha) * temp[i-1] + \
                    (1.0 - (2.0 * lambda_)) * temp[i] + \
                    (lambda_ + alpha) * temp[i+1] + hp * dt
        
        # Apply boundary conditions
        diag[0] = 1.0
        sup[0] = 0.0
        f[0] = Ts
        diag[mz-1] = 1.0
        inf[mz-1] = 0.0
        f[mz-1] = temp[mz-2] + b_flux
        
        # Solve linear equations
        sup[0] = sup[0] / diag[0]
        f[0] = f[0] / diag[0]
        
        for i in range(1, mz):
            sup[i] = sup[i] / (diag[i] - inf[i] * sup[i-1])
            f[i] = (f[i] - inf[i] * f[i-1]) / (diag[i] - inf[i] * sup[i-1])
        
        temp[mz-1] = f[mz-1]
        for i in range(mz-2, -1, -1):
            temp[i] = f[i] - sup[i] * temp[i+1]

        temp[0] = Ts
        temp[mz-1] = temp[mz-2] + b_flux
        
        # Increase time
        xtime += dt
        j += 1
        
        # When the sample age of the model equals the measured age, exit
        if exit_flag == 1:
            temp_age[:] = temp[:]
            break
        
        # Time step before the age is reached, change dt
        if tend - xtime - dt < ta[kk]:
            dt = tend - xtime - ta[kk]
            lambda_ = kappa * dt / (2.0 * (dz**2))
            
            # Update exhum
            query[:] = xtime
            query = tsteps_sum - query
            query[query < 0.0] = -9999.0
            pos = np.argmin(query**2)
            exhum = edot_dat[pos + (kk) * m_max]
            
            # Define constants for timestepping
            alpha = exhum * dt / (4.0 * dz)
            ax = (alpha - lambda_)
            bx = (1.0 + (2.0 * lambda_))
            cx = -1.0 * (lambda_ + alpha)
            temp_pr[:] = temp[:]
            exit_flag = 1
    return exhum,dt

def find_zc(matrices, params):
    """
    This subroutine calculates closure depths
    
    The routine proceeds as follows:
    For each sample do:
    1. Set up initial conditions, a linear increase of temperature with depth
    2. Use crank-nicholson finite differencing to step through time
    3. At the time equivalent to the measured age record the material derivatives
    4. Use these cooling rates to estimate closure temperatures
    5. Find location in depth where closure depth is equal to temperature
    """
    
    # Parameters defining the geometry
    mz = 131
    dz = params.zl / float(mz - 1)
    
    # Allocate arrays
    temp = np.zeros(mz)
    temp2 = np.zeros(mz)

    temp_pr = np.zeros(mz)
    temp_age = np.zeros(mz)
    tdot = np.zeros(mz)
    closure = np.zeros(mz)
    
    
    # Open file for heat flux output
    heat_flux_file = open(params.run + "/stuff/heat_flux.txt", "w")
    
    for kk in range(params.n):
        np.random.seed()
        
        # Time stepping parameters
        dt = 0.5 * (dz**2) / params.kappa
        tstart = 0.0
        tend = params.t_total
        nt = int(tend / dt) + 1
        temp[:] = 0.0
        temp2[:] = 0.0
        
        xtime = tstart
        # Set up initial conditions
        for i in range(mz):
            temp[i] = params.Ts + float(i) / float(mz - 1) * (params.Tb - params.Ts)
        
        # Basal flux
        b_flux = (temp[mz-1] - temp[mz-2])
        
        exhum,dt = forward_process(xtime,dt,tend,matrices.tsteps_sum,matrices.edot_dat,kk,params.m_max,params.kappa,dz,mz,params.Ts,temp,params.hp,b_flux,matrices.ta,temp_pr,temp_age)
        
        # Calculate material derivative, tdot
        tdot_dist = exhum * dt
        for i in range(mz-1):
            tdot[i] = temp_age[i] - temp_pr[i] + (temp_pr[i+1] - temp_pr[i]) * (tdot_dist / dz)
            tdot[i] = tdot[i] / dt
        
        # Calculate tdot dependent closure depths
        closure[:] = 0.0
        iflag = matrices.isys[kk]
        closure = closure_temps(tdot, temp_age, iflag)
        
        # Solve the simultaneous equations to calculate z_c, T_c
        for i in range(1, mz-1):
            if temp_age[i] > closure[i]:
                M = 1.0 * (dz / (closure[i] - closure[i-1]))
                P = 1.0 * (dz / (temp_age[i] - temp_age[i-1]))
                Tc = ((M * closure[i-1]) - (P * temp_age[i-1])) / (M - P)
                xjunk = M * (Tc - closure[i])
                
                if matrices.isys[kk] > 0:
                    matrices.zc[kk] = dz * float(i-1) + xjunk
                break
        
        # Write heat flux out
        heat_flux_file.write(f"{matrices.x_true[kk]} {matrices.y_true[kk]} {2.3 * (temp[1] - temp[0]) / dz}\n")
    
    heat_flux_file.close()
    return matrices