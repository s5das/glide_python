import numpy as np
from glide.tridag_par import tridag_par
from glide.dodson import dodson
from glide.moudule_define import (Parm,Matr)
from numba import jit

@jit(nopython=True)
def forward_process(t_total,tsteps_sum,edot_dat,m_max,n,kappa,dt,dz,mz,Ts,b_flux,age,depth,deep,temperature,temp,hp):
    # 向前时间积分解热传导方程
    xtime = 0.0
    j = 0
    while xtime < t_total:
        query = tsteps_sum - xtime
        query[query < 0] = -9999.0
        pos = np.argmin(query**2)
        exhum = edot_dat[pos::m_max][:n]

        lambda_ = kappa * dt / (2 * dz**2)
        alpha = exhum * dt / (4 * dz)
        ax = alpha - lambda_
        bx = 1 + 2 * lambda_
        cx = -(lambda_ + alpha)

        diag = np.zeros((mz, n))
        sup = np.zeros((mz, n))
        inf = np.zeros((mz, n))
        f = np.zeros((mz, n))

        for p in range(1, mz - 1):
            sup[p, :] = cx
            inf[p, :] = ax
            diag[p, :] = bx
            f[p, :] = (lambda_ - alpha) * temp[p - 1, :] + \
                      (1 - 2 * lambda_) * temp[p, :] + \
                      (lambda_ + alpha) * temp[p + 1, :] + hp * dt

        diag[0, :] = 1
        diag[mz - 1, :] = 1
        f[0, :] = Ts
        f[mz - 1, :] = temp[mz - 2, :] + b_flux

###------------------------------------------------------------------------------------------------
###     tridag_par
        nn, chunk = diag.shape
        temp = np.zeros((nn, chunk))
        gam = np.zeros((nn, chunk))
        bet = diag[0, :].copy()

        # First row solution
        temp[0, :] = f[0, :] / bet

        # Forward sweep
        for jj in range(1, nn):
            gam[jj, :] = sup[jj - 1, :] / bet
            bet = diag[jj, :] - inf[jj, :] * gam[jj, :]
            temp[jj, :] = (f[jj, :] - inf[jj, :] * temp[jj - 1, :]) / bet

        # Back substitution
        for jj in range(nn - 2, -1, -1):
            temp[jj, :] -= gam[jj + 1, :] * temp[jj + 1, :]
###------------------------------------------------------------------------------------------------

        age[j, :] = t_total - xtime

        for i in range(n):
            depth[i] -= exhum[i] * dt
            deep[j, i] = depth[i]
            k = min(mz - 2, max(0, int(depth[i] / dz)))
            xint = (depth[i] - dz * k) / dz
            temperature[j, i] = temp[k + 1, i] * xint + temp[k, i] * (1 - xint)

        xtime += dt
        j += 1


def syn_ages(matrices: Matr, params: Parm):
    # 初始化参数
    n = params.n
    m_max = params.m_max
    zl = params.zl
    kappa = params.kappa
    Ts = params.Ts
    Tb = params.Tb
    t_total = params.t_total
    hp = params.hp
    deltat = params.deltat

    # 网格划分
    mz = 231
    dz = zl / (mz - 1)
    dt = dz**2 / kappa
    nt = int(t_total / dt) + 1
    ntsteps = 100

    # 初始化数组
    temp = np.zeros((mz, n))
    temperature = np.zeros((nt, n))
    age = np.zeros((nt, n))
    deep = np.zeros((nt, n))
    geotherm = np.zeros(n)
    syn_age = np.zeros(n)

    # 初始温度分布
    for i in range(mz):
        temp[i, :] = Ts + i / (mz - 1) * (Tb - Ts)

    b_flux = temp[mz - 1, :] - temp[mz - 2, :]
    matrices.depth = -1.0 * matrices.elev.copy()

    # 反推样品最初深度
    xtime = t_total
    j = 0
    while xtime > 0.0:
        j += 1
        query = matrices.tsteps_sum - xtime
        query[query < 0] = -9999.0
        pos = np.argmin(query**2)
        exhum = matrices.edot_dat[pos::m_max][:n]
        matrices.depth += exhum * dt
        xtime -= dt
    nt = j

    forward_process(t_total,matrices.tsteps_sum,matrices.edot_dat,m_max,n,kappa,dt,dz,mz,Ts,b_flux,age,matrices.depth,deep,temperature,temp,hp)

    # 计算闭合温度对应深度
    matrices.zcp = np.zeros(n)
    for i in range(n):
        isys = matrices.isys[i]
        Tt = temperature[:, i]
        t_age, Tc, cooling = dodson(Tt, age[:, i], nt, isys)
        for j in range(1, nt):
            if Tt[j] < Tc:
                frac = (Tc - Tt[j]) / (Tt[j - 1] - Tt[j])
                matrices.zcp[i] = deep[j, i] + frac * (deep[j - 1, i] - deep[j, i])
                break
        geotherm[i] = (temp[1, i] - temp[0, i]) / dz

    # 合成年代计算
    for i in range(n):
        dist = 0.0
        j = m_max-1
        while dist < (matrices.elev[i] + matrices.zcp[i]):
            syn_age[i] += matrices.tsteps[m_max - j]
            dist += matrices.tsteps[m_max - j] * matrices.edot_dat[j + i * m_max]
            j -= 1
        j += 1
        dist -= matrices.tsteps[m_max - j] * matrices.edot_dat[j + i * m_max]
        syn_age[i] -= matrices.tsteps[m_max - j]
        frac = ((matrices.elev[i] + matrices.zcp[i]) - dist) / matrices.edot_dat[j + i * m_max]
        syn_age[i] += frac

    matrices.syn_age = syn_age

    # 计算误差
    misfits = np.abs(syn_age - matrices.ta)
    matrices.misfits = misfits
    misfit = np.sqrt(np.sum((misfits**2) / (matrices.a_error**2)) / n)

    return misfit