from grid import *
import os
from eos import *
from consts import *
from setup import *

class LoadSimulation:

    def __init__(self, t0=None, tf=None, dt=0.1*YEAR, params=Params()):
        self.sim_dir = os.path.join(os.getcwd(), params.SIM_NAME)
        self.ts = np.load(os.path.join(self.sim_dir, "tsave.npy"), mmap_mode="r")
        if t0 is None:
            t0 = self.ts[0]
        if tf is None:
            tf = self.ts[-1]

        comparr = np.append(np.arange(t0, tf, dt), tf)
        ts_proj = np.einsum("i,j->ij", np.ones(comparr.shape), self.ts)
        ca_proj = np.einsum("i,j->ij", comparr, np.ones(self.ts.shape))
        indices = np.unique(np.argmin(np.abs(ts_proj - ca_proj), axis=1))

        self.sigma = np.load(os.path.join(self.sim_dir, "sigma.npy"), mmap_mode="r")[indices]
        self.s = np.load(os.path.join(self.sim_dir, "entropy.npy"), mmap_mode="r")[indices]
        self.ts = self.ts[indices]
        self.grid = Grid(grid_array=np.load(os.path.join(self.sim_dir, "rocell.npy")), params=params)
        self.eos = load_table(params.EOS_TABLE)

        self.chi = self.sigma*self.grid.omgko
        self.T = self.eos(self.chi, self.s)

        self.rho = entropy_difference(self.T, self.chi, self.s, just_density=True)

        self.rad_P = RADA*self.T**4/3
        self.gas_P = self.rho*KB*self.T/mu
        self.P = self.rad_P + self.gas_P
        self.U = RADA * self.T ** 4 + 1.5 * self.rho * KB * self.T / mu

        self.be = -1 + 2 * (self.U + self.P) / self.rho * self.grid.r_cell / CONST_G / MBH
        self.H = self.sigma / 2 / self.rho
        self.nu = params.ALPHA * self.H ** 2 * self.grid.omgko

        self.nuf = np.logspace(13, 18, 100)
        self.S_nu = Snu(self.T, self.nuf, self.grid)
        self.L_nu = 4*np.pi*(10*PC)**2*self.S_nu
        self.nuL_nu = np.einsum("j,ij->ij", self.nuf, self.L_nu)

        self.sigv = sig((self.be - params.BE_CRIT) / params.DBE)
        ## calculates velocities at inter_faces
        lc_sigma = 2 * np.pi * self.sigma * self.grid.r_cell
        g = np.sqrt(self.grid.r_cell) / (self.nu + 1e-20)
        d = 3 * self.nu
        dr = self.grid.r_cell[1] - self.grid.r_cell[0]

        g_tild = (g[:, 1:] + g[:, :-1])/2
        d_tild = (d[:, 1:] + d[:,:-1])/2
        lc_sigma_tild = (lc_sigma[:, 1:] + lc_sigma[:, :-1])/2
        self.vr = -d_tild*g_tild/lc_sigma_tild * (lc_sigma[:,1:]/g[:,1:] - lc_sigma[:,:-1]/g[:,:-1])/dr

        ## calculate source terms for density
        sigma_wl = params.FWIND*self.sigma * self.grid.omgko * self.sigv  ## wind loss

        sigma_fb = 1 / np.pi / gamma(params.FBK / 2 + 1) / params.FBR0 ** 2  ## fall back
        sigma_fb *= np.einsum("i,j->ij", (self.ts + MONTH) ** (-5 / 3), (self.grid.r_cell / params.FBR0) ** params.FBK * np.exp(-(self.grid.r_cell / params.FBR0) ** 2))
        sigma_fb *= MSUN * MONTH ** (2 / 3)

        self.sigma_fb = sigma_fb
        self.sigma_wl = sigma_wl
        self.sigma_dot = sigma_fb - sigma_wl

        ## calculate source terms for entropy
        kappa = kappa_interpolator(self.rho, self.T)
        self.kappa = kappa

        qrad = 4 * RADA * self.T ** 4 * c / (1 + kappa * self.sigma)/self.sigma
        qwind = 0.5 * self.grid.omgko * CONST_G * MBH / self.grid.r_cell * self.sigv
        qvis = 2.25 * self.nu * self.grid.omgko ** 2
        qfb = params.FSH*0.5*CONST_G * MBH * sigma_fb / self.grid.r_cell /self.sigma


        self.qvis = qvis/self.sigma
        self.qfb = qfb/self.sigma
        self.qrad = qrad/self.sigma
        self.qwind = qwind/self.sigma
        self.ts_dot = (qvis + qfb - qrad - qwind) / self.T
