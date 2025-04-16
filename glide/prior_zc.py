import numpy as np
from glide.closure_temps import closure_temps
from numba import jit



@jit(nopython=True)
def run_simulation(query,kk,b_flux,xtime, tend, dt, dz, mz, temp, f, diag, sup, inf, skip,tsteps_sum,edot_pr,m_max,kappa,run,hp,Ts,ta):
    j = 0
    exit_flag = 0
    while xtime < tend:
        query.fill(xtime)
        query = tsteps_sum - query
        query[query < 0] = -9999.0
        pos = np.argmin(query**2)
        exhum = edot_pr[pos + (kk) * m_max]

        # Constants for tridiagonal solver
        lambda_val = kappa * dt / (2.0 * (dz**2))
        alpha = exhum * dt / (4.0 * dz)
        ax = alpha - lambda_val
        bx = 1.0 + (2.0 * lambda_val)
        cx = -1.0 * (lambda_val + alpha)

        # Set up tridiagonal system
        for i in range(1, mz-1):
            diag[i] = bx
            sup[i] = cx
            inf[i] = ax
            f[i] = (lambda_val - alpha) * temp[i-1] + \
                   (1.0 - (2.0 * lambda_val)) * temp[i] + \
                   (lambda_val + alpha) * temp[i+1] + hp * dt

        # Boundary conditions
        diag[0] = 1.0
        sup[0] = 0.0
        f[0] = Ts
        diag[-1] = 1.0
        inf[-1] = 0.0
        f[-1] = temp[-2] + b_flux

        # Solve tridiagonal system using tridag
        temp = np.zeros(mz)
        # Forward elimination
        for i in range(1, mz):
            factor = inf[i] / diag[i-1]
            diag[i] -= factor * sup[i-1]
            f[i] -= factor * f[i-1]
        
        # Back substitution
        temp[-1] = f[-1] / diag[-1]
        for i in range(mz-2, -1, -1):
            temp[i] = (f[i] - sup[i] * temp[i+1]) / diag[i]
            
        temp[0] = Ts
        temp[-1] = temp[-2] + b_flux

        xtime += dt
        j += 1

        if exit_flag == 1:
            temp_age = temp.copy()
            break

        # Adjust time step when approaching sample age
        if tend - xtime - dt < ta[kk]:
            dt = tend - xtime - ta[kk]
            lambda_val = kappa * dt / (2.0 * (dz**2))

            query.fill(xtime)
            query = tsteps_sum - query
            query[query < 0] = -9999.0
            pos = np.argmin(query**2)
            exhum = edot_pr[pos + (kk) * m_max]

            alpha = exhum * dt / (4.0 * dz)
            ax = alpha - lambda_val
            bx = 1.0 + (2.0 * lambda_val)
            cx = -1.0 * (lambda_val + alpha)
            temp_pr = temp.copy()
            exit_flag = 1


    return exhum,temp_age,temp_pr,dt









def prior_zc(matrices, params):
    """
    This function calculates closure depths
    
    The routine proceeds as follows:
    For each sample:
    1. Sets up initial conditions (linear temperature increase with depth)
    2. Uses Crank-Nicholson finite differencing to step through time
    3. At time equivalent to measured age, records material derivatives
    4. Uses cooling rates to estimate closure temperatures
    5. Finds depth where closure temperature equals actual temperature
    """
    
    # Parameters defining the geometry
    mz = 131
    dz = params.zl / float(mz - 1)
    
    # Allocate arrays
    temp = np.zeros(mz)
    temp2 = np.zeros(mz)
    diag = np.zeros(mz)
    sup = np.zeros(mz)
    inf = np.zeros(mz)
    f = np.zeros(mz)
    temp_pr = np.zeros(mz)
    temp_age = np.zeros(mz)
    tdot = np.zeros(mz)
    closure = np.zeros(mz)
    query = np.zeros(100)
    
    skip = np.argmin(matrices.ta)
    
    # Open output files
    
    cross_over_file = open(f"{params.run}/stuff/cross_over.txt", "w")
    anal_file = open(f"{params.run}/stuff/anal.txt", "w")
    
    # Initialize ages matrix
    matrices.ages = np.zeros((7, 5))  # Assuming 7 systems and 5 parameters
    
    # Loop through data points
    for kk in range(params.n):
        np.random.seed()
        
        # Time stepping parameters
        dt = ((dz**2) / params.kappa) / 2.0
        tstart = 0.0
        tend = params.t_total
        nt = int(tend / dt) + 1
        temp.fill(0.0)
        temp2.fill(0.0)
        
        xtime = tstart
        j = 0
        
        # Initial linear temperature profile
        for i in range(mz):
            temp[i] = params.Ts + float(i) / float(mz - 1) * (params.Tb - params.Ts)
        
        if kk == skip:
            print(f"Geotherm at start of model: {(temp[1] - temp[0]) / dz}")
        
        b_flux = temp[-1] - temp[-2]
        
        # while xtime < tend:
        #     query.fill(xtime)
        #     query = matrices.tsteps_sum - query
        #     query[query < 0] = -9999.0
        #     pos = np.argmin(query**2)
        #     exhum = matrices.edot_pr[pos + (kk) * params.m_max]
            
        #     # Constants for tridiagonal solver
        #     lambda_val = params.kappa * dt / (2.0 * (dz**2))
        #     alpha = exhum * dt / (4.0 * dz)
        #     ax = alpha - lambda_val
        #     bx = 1.0 + (2.0 * lambda_val)
        #     cx = -1.0 * (lambda_val + alpha)
            
        #     # Set up tridiagonal system
        #     for i in range(1, mz-1):
        #         diag[i] = bx
        #         sup[i] = cx
        #         inf[i] = ax
        #         f[i] = (lambda_val - alpha) * temp[i-1] + \
        #                (1.0 - (2.0 * lambda_val)) * temp[i] + \
        #                (lambda_val + alpha) * temp[i+1] + params.hp * dt
            
        #     # Boundary conditions
        #     diag[0] = 1.0
        #     sup[0] = 0.0
        #     f[0] = params.Ts
        #     diag[-1] = 1.0
        #     inf[-1] = 0.0
        #     f[-1] = temp[-2] + b_flux
            
        #     # Solve tridiagonal system using tridag
        #     temp = tridag(inf, diag, sup, f,mz)
            
        #     temp[0] = params.Ts
        #     temp[-1] = temp[-2] + b_flux
            
        #     xtime += dt
        #     j += 1
            
        #     if exit_flag == 1:
        #         temp_age = temp.copy()
        #         break
            
        #     # Adjust time step when approaching sample age
        #     if tend - xtime - dt < matrices.ta[kk]:
        #         dt = tend - xtime - matrices.ta[kk]
        #         lambda_val = params.kappa * dt / (2.0 * (dz**2))
                
        #         query.fill(xtime)
        #         query = matrices.tsteps_sum - query
        #         query[query < 0] = -9999.0
        #         pos = np.argmin(query**2)
        #         exhum = matrices.edot_pr[pos + (kk) * params.m_max]
                
        #         alpha = exhum * dt / (4.0 * dz)
        #         ax = alpha - lambda_val
        #         bx = 1.0 + (2.0 * lambda_val)
        #         cx = -1.0 * (lambda_val + alpha)
        #         temp_pr = temp.copy()
        #         exit_flag = 1
            
        #     # Write transient solution for youngest age
        #     if kk == skip and j % 500 == 0:
        #         Tz_file.write("> -Z {:.3f}\n".format(xtime))
        #         for i in range(mz):
        #             Tz_file.write(f"{temp[i]} {float(i) * dz} {xtime}\n")
        exhum,temp_age,temp_pr,dt = run_simulation(query,kk,b_flux,xtime, tend, dt, dz, mz, temp, f, diag, sup, inf, skip,matrices.tsteps_sum,matrices.edot_pr,params.m_max,params.kappa,params.run,params.hp,params.Ts,matrices.ta)
        if kk == skip:
            print(f"Geotherm at end of model: {(temp[1] - temp[0]) / dz}")
        
        # Calculate material derivative
        tdot_dist = exhum * dt
        for i in range(mz-1):
            tdot[i] = temp_age[i] - temp_pr[i] + (temp_pr[i+1] - temp_pr[i]) * (tdot_dist / dz)
            tdot[i] /= dt
        
        # Calculate closure depths
        closure.fill(0.0)
        iflag = matrices.isys[kk]
        closure= closure_temps(tdot, temp_age, iflag)
        
        for i in range(1, mz):
            if temp_age[i] > closure[i]:
                M = 1.0 * (dz / (closure[i] - closure[i-1]))
                P = 1.0 * (dz / (temp_age[i] - temp_age[i-1]))
                Tc = ((M * closure[i-1]) - (P * temp_age[i-1])) / (M - P)
                xjunk = M * (Tc - closure[i])
                
                matrices.ages[int(iflag), 0] += Tc  # closure temp
                matrices.ages[int(iflag), 1] += dz * float(i-1) + xjunk  # closure depth
                matrices.ages[int(iflag), 2] += matrices.ta[kk]  # age
                matrices.ages[int(iflag), 3] += (temp_age[1] - temp_age[0]) / dz  # dT/dz at z=0
                matrices.ages[int(iflag), 4] += (temp_age[i] - temp_age[i-1]) / dz  # dT/dz at z=zc
                matrices.zc[kk] = dz * float(i-1) + xjunk
                cross_over_file.write(f"{Tc} {dz * float(i-1) + xjunk} {(temp_age[1] - temp_age[0]) / dz}\n")
                break
    
    # Average ages matrix for each system
    for i in range(7):
        matrices.ages[i, :] = matrices.ages[i, :] / float(matrices.nsystems[i])
    
    # Write closures file
    with open(f"{params.run}/stuff/closures.txt", "w") as closures_file:
        for j in range(7):
            closures_file.write(f"{matrices.ages[j, 0]} {matrices.ages[j, 1]} {matrices.ages[j, 2]} {matrices.ages[j, 3]} {matrices.ages[j, 4]}\n")
    
    # Close files
    
    cross_over_file.close()
    anal_file.close()

