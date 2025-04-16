import numpy as np
from numba import jit
from glide.moudule_define import (Parm,Matr)

def build_matrices(matrices: Matr, params: Parm):

    """构建正演算子矩阵A和协方差矩阵cov
    """
    
    n = params.n
    m_max = params.m_max
    dummy = params.dummy
    
    matrices.A = np.zeros((n, n * m_max))
    matrices.G = np.zeros((n, n * m_max))
    matrices.edot = np.zeros(dummy * m_max)
    matrices.edot_dat = np.zeros(n * m_max)
    matrices.eps_dum = np.zeros(dummy * m_max)
    matrices.eps = np.zeros(n * m_max)
    matrices.sf = np.zeros(dummy * m_max)
    matrices.cov = np.zeros((n * m_max, n * m_max))
    matrices.cpost = np.zeros((n * m_max, n * m_max))
    
    matrices.eps_dat = np.zeros(n * m_max)
    matrices.eps_res = np.zeros(n * m_max)
    matrices.H = np.zeros((n * m_max, n))
    matrices.Y2 = np.zeros((n * m_max, n))

    # 2. 构建矩阵A和G（离散化年龄矩阵）
    _build_forward_operator(matrices.ta,matrices.tsteps,matrices.A,matrices.G,matrices.zc,matrices.edot_pr,matrices.elev, params.n,params.m_max)

    # 3. 构建协方差矩阵
    _build_covariance_matrix(matrices.x,matrices.y,matrices.iblock,matrices.cov, params.n,params.m_max,params.sigma2,params.xL)
    
    print(f'covariance max value: {np.max(matrices.cov)}')

@jit(nopython=True)
def _build_forward_operator(ta,tsteps,A,G,zc,edot_pr,elev, n,m_max):
    for i in range(n):
        m = 1
        k = 1
        summ = 0.0
        
        # 计算所需时间步数和剩余项
        while summ < ta[i]:
            summ += tsteps[m-1]
            m += 1
        
        m -= 1
        if m == 0:
            summ = 0.0
            rest = 0.0
        else:
            summ -= tsteps[m-1]
            rest = ta[i] - summ
        
        k = 0
        start_idx = (m_max - m) + (i) * m_max
        end_idx = (i) * (m_max - 1) + m_max + (i)
        
        for j in range(start_idx, end_idx + 1):
            k += 1
            A[i, j] = tsteps[m - k] if k > 1 else rest
            G[i, j] = (edot_pr[j] * A[i, j] / 
                               (zc[i] + elev[i]))

@jit(nopython=True)
def _build_covariance_matrix(x,y,iblock,cov, n,m_max,sigma2,xL):
    for i in range(n * m_max):
        ik = (i + m_max - 1) // m_max - 1  # 转换为0-based索引
        jk = ik - 1
        
        # 只计算矩阵的上三角部分（对称矩阵）
        for j in range(i, n * m_max, m_max):
            jk += 1
            
            # 计算点之间的距离
            dx = x[ik] - x[jk]
            dy = y[ik] - y[jk]
            dist = np.sqrt(dx**2 + dy**2)
            
            # 如果不在同一个地壳块中，则设置很大距离
            if iblock[ik] != iblock[jk]:
                dist = 1e6
                
            cov_value = sigma2 * np.exp(-(dist / xL))
            if cov_value < 1e-8:
                cov_value = 0.0
                
            cov[i, j] = cov_value
            cov[j, i] = cov_value  # 对称赋值