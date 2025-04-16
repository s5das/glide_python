'''
Author: gaofei gg21009100584@gmail.com
Date: 2025-04-16 12:05:11
Description: module parameters define
'''

from dataclasses import dataclass
from typing import List, Optional, Any
import numpy as np

@dataclass
class Parm:
    n: int= None
    m_max: int= None
    napt: int= None
    nzirc: int= None
    narar: int= None
    dummy: int= None
    contr: int= None
    iterM: int= None
    edot_mean: float= None
    deltat: float= None
    xL: float= None
    sigma2: float= None
    sigmaD: float= None
    t_total: float= None
    lat1: float= None
    lat2: float= None
    lon1: float= None
    xmu: float= None
    bot: float= None
    lon2: float= None
    dlon: float= None
    kappa: float= None
    hp: float= None
    Ts: float= None
    Tb: float= None
    zl: float= None
    angle: float= None
    aspect: float= None
    run: str  = None# length 5
    topofile: str = None # length 100

@dataclass
class Matr:
    B: Optional[np.ndarray] = None  # integer array
    nsystems: Optional[np.ndarray] = None  # integer array
    isys: Optional[np.ndarray] = None  # double precision array
    edot: Optional[np.ndarray] = None  # double precision array
    edot_pr: Optional[np.ndarray] = None  # double precision array
    edot_pr2: Optional[np.ndarray] = None  # double precision array
    zc: Optional[np.ndarray] = None  # double precision array
    zcp: Optional[np.ndarray] = None  # double precision array
    eps: Optional[np.ndarray] = None  # double precision array
    elev: Optional[np.ndarray] = None  # double precision array
    eps_dat: Optional[np.ndarray] = None  # double precision array
    eps_res: Optional[np.ndarray] = None  # double precision array
    work: Optional[np.ndarray] = None  # double precision array
    syn_age: Optional[np.ndarray] = None  # double precision array
    iblock: Optional[np.ndarray] = None  # integer array
    idum_block: Optional[np.ndarray] = None  # integer array
    x_true: Optional[np.ndarray] = None  # double precision array
    y_true: Optional[np.ndarray] = None  # double precision array
    x_dum_true: Optional[np.ndarray] = None  # double precision array
    y_dum_true: Optional[np.ndarray] = None  # double precision array
    elev_true: Optional[np.ndarray] = None  # double precision array
    BB: Optional[np.ndarray] = None  # double precision array
    edot_dat: Optional[np.ndarray] = None  # double precision array
    edot_pr_dum: Optional[np.ndarray] = None  # double precision array
    Cmm: Optional[np.ndarray] = None  # double precision array
    eps_dum: Optional[np.ndarray] = None  # double precision array
    x_dum: Optional[np.ndarray] = None  # double precision array
    y_dum: Optional[np.ndarray] = None  # double precision array
    misfits: Optional[np.ndarray] = None  # double precision array
    R: Optional[np.ndarray] = None  # double precision array
    kernel: Optional[np.ndarray] = None  # double precision array
    sf: Optional[np.ndarray] = None  # double precision array
    ta: Optional[np.ndarray] = None  # double precision array
    a_error: Optional[np.ndarray] = None  # double precision array
    x: Optional[np.ndarray] = None  # double precision array
    y: Optional[np.ndarray] = None  # double precision array
    cov_eps: Optional[np.ndarray] = None  # double precision array
    depth: Optional[np.ndarray] = None  # double precision array
    depth1: Optional[np.ndarray] = None  # double precision array
    zz: Optional[np.ndarray] = None  # double precision array
    tsteps: Optional[np.ndarray] = None  # double precision array
    tsteps_sum: Optional[np.ndarray] = None  # double precision array
    A: Optional[np.ndarray] = None  # 2D double precision array
    G: Optional[np.ndarray] = None  # 2D double precision array
    cov: Optional[np.ndarray] = None  # 2D double precision array
    cpost: Optional[np.ndarray] = None  # 2D double precision array
    Y1: Optional[np.ndarray] = None  # 2D double precision array
    Y2: Optional[np.ndarray] = None  # 2D double precision array
    Y3: Optional[np.ndarray] = None  # 2D double precision array
    H: Optional[np.ndarray] = None  # 2D double precision array
    II: Optional[np.ndarray] = None  # 2D double precision array
    distm: Optional[np.ndarray] = None  # 2D double precision array
    ages: Optional[np.ndarray] = None  # 2D double precision array
    Y9: Optional[np.ndarray] = None  # 2D double precision array