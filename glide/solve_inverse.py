import numpy as np
from numba import jit, prange
import scipy.linalg as la

@jit(nopython=True, parallel=True)
def compute_edot_dummy(m_max,dummy,n,sigma2,xL, edot_pr_dum,x_dum,x,y_dum,y, BB):
    """
    Compute exhumation rates at dummy points (parallelized with Numba)
    """
    edot = np.zeros(m_max * dummy)
    edot_pr_dum = edot_pr_dum.copy()
    
    for i in prange(m_max * dummy):
        Cmm = np.zeros(n * m_max)
        ik = (i) // m_max
        
        for j in range(n * m_max):
            jk = (j) // m_max
            if (i % m_max) == (j % m_max):
                dist = np.sqrt((x_dum[ik] - x[jk])**2 + 
                       (y_dum[ik] - y[jk])**2)
                Cmm[j] = sigma2 * np.exp(-(dist / xL))
                if Cmm[j] < 1e-6:
                    Cmm[j] = 0.0
        
        xx = 0.0
        for k in range(n * m_max):
            xx += Cmm[k] * BB[k]
        
        edot[i] = np.exp(xx + np.log(edot_pr_dum[i]))
    
    return edot

def solve_inverse(matrices, params, iter):
    """
    Maximum likelihood estimates of the exhumation rates
    
    edot = edot_pr + CG'(GCG'+Cee) (log(Zc) - log(Aedot_pr))
    """
    sigmaD = 0.2
    
    # Allocate matrices
    n = params.n
    m_max = params.m_max
    n_total = n * m_max
    
    # Compute Y2 = CA'
    matrices.Y2 = np.dot(matrices.cov, matrices.G.T)
    
    # Compute Y1 = GCG' + Cee
    matrices.Y1 = np.dot(matrices.G, matrices.Y2)
    for i in range(n):
        matrices.Y1[i,i] += sigmaD**2
    
    # Invert Y1 using LU decomposition
    matrices.Y1, piv = la.lu_factor(matrices.Y1)
    matrices.Y1 = la.lu_solve((matrices.Y1, piv), np.eye(n))
    
    # Compute H = GA'Y1^(-1)
    matrices.Y2 = np.dot(matrices.G.T, matrices.Y1)
    
    # Compute a priori age (A*eprior)
    zz = np.dot(matrices.A, matrices.edot_pr)
    zz = np.log(matrices.zc + matrices.elev) - np.log(zz)
    
    # BB = Y2 * (log(zc) - log(A*eprior))
    BB = np.dot(matrices.Y2, zz)
    
    # Compute edot at control points (parallelized)
    matrices.edot = compute_edot_dummy(params.m_max,params.dummy,params.n,params.sigma2,params.xL, matrices.edot_pr_dum,matrices.x_dum,matrices.x,matrices.y_dum,matrices.y, BB)
    matrices.edot_pr_dum = matrices.edot.copy()
    
    # Ensure positive exhumation rates
    matrices.edot[matrices.edot < 0] = 0
    print(f"edot min/max values at dummy: {np.min(matrices.edot)}/{np.max(matrices.edot)}")
    
    # Compute exhumation rates for data points
    matrices.H = np.dot(matrices.cov, matrices.Y2)
    
    # Using a priori model to calculate depth of sample
    zz = np.dot(matrices.A, matrices.edot_pr)
    zp = np.log(matrices.zc + matrices.elev) - np.log(zz)
    
    # Compute edot_dat = H*(zc-ta*exp(epsilon)) + log(edot_pr)
    matrices.edot_dat = np.dot(matrices.H, zp) + np.log(matrices.edot_pr)
    
    # Calculate residuals
    resid = np.sum(zp**2 / sigmaD**2)
    
    # Compute model residuals
    xresi = np.abs(matrices.edot_dat - np.log(matrices.edot_pr))
    covinv = la.pinv(matrices.cov)
    vresi = np.dot(covinv, xresi)
    resim = np.sum(np.abs(vresi) * xresi)
    
    # Update mean exhumation rate
    params.edot_mean = np.mean(np.exp(matrices.edot_dat))
    
    # Calculate total residual
    total_resid = (resim + resid) / n
    print(f"residuals {np.sqrt(total_resid)}, resi data {np.sqrt(resid)}, resimod {np.sqrt(resim)}")
    
    # Save residuals to file
    with open(params.run + '/residualsLOG.txt', 'a') as f:
        f.write(f"{iter} {total_resid} {resid} {resim}\n")
    
    # Convert back to linear exhumation rates
    matrices.edot_dat = np.exp(matrices.edot_dat)
    print(f"edot min/max values at data: {np.min(matrices.edot_dat)}/{np.max(matrices.edot_dat)}")
    matrices.edot_pr = matrices.edot_dat.copy()
    
    # Update matrix G
    for i in range(n):
        m = 1
        summ = 0.0
        while summ < matrices.ta[i] and m <= len(matrices.tsteps):
            summ += matrices.tsteps[m-1]
            m += 1
        m -= 1
        
        if m == 0:
            summ = 0.0
            rest = 0.0
        else:
            summ -= matrices.tsteps[m-1]
            rest = matrices.ta[i] - summ
        
        k = 0
        start_idx = (m_max - m) + (i) * (m_max - 1)
        end_idx = start_idx + m
        for j in range(start_idx, end_idx):
            if k == 0:
                matrices.A[i,j] = rest
            else:
                matrices.A[i,j] = matrices.tsteps[m - k]
            
            denom = matrices.zc[i] + matrices.elev[i]
            matrices.G[i,j] = (matrices.edot_pr[j] * matrices.A[i,j]) / denom
            k += 1
    
    return total_resid