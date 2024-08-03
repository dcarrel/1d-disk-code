from params import *
import numpy as np
import os, sys, glob
from simulation import *
from grid import *
from consts import *
from scipy.special import iv
import matplotlib.pyplot as plt

## analytic solution for a spreading accretion disk with constant viscosity
## TVISC = number of viscous times
def analytic_solution(M0, R0, TVISC):
    def func(r):
        x = r/R0
        tau = 12*TVISC
        retval = M0/np.pi/R0**2/tau*x**-0.25*np.exp(-(1+x**2)/tau)*iv(0.25, 2*x/tau)
        retval = np.where(np.logical_or(np.isnan(retval), np.isinf(retval)), 0, retval)
        return retval
    return func

class InitialCondition:
    def __init__(self, m0=0.01, tv=1, ambf=1e-5, params=Params()):
        self.grid = Grid(params=params)
        self.sigma0 = analytic_solution(m0*params.MSTAR, params.RT, tv)(self.grid.r_cell)
        self.params=params

        ## sets the power law profile in the ambient region
        sigma0_max = np.max(self.sigma0)
        sigma0_crit = ambf*sigma0_max
        disk_body = np.where(self.sigma0 > ambf*sigma0_max, 1, 0)
        bdy1 = np.argmax(disk_body)
        r1 = self.grid.r_cell[bdy1]
        bdy2 = len(self.sigma0) - np.argmax(disk_body[::-1])-1
        r2 = self.grid.r_cell[bdy2]
        if r1 < params.RT: self.sigma0 = np.where(self.grid.r_cell < r1, sigma0_crit*(self.grid.r_cell/r1)**-1, self.sigma0)
        if r2 > params.RT: self.sigma0 = np.where(self.grid.r_cell > r2, sigma0_crit*(self.grid.r_cell/r2)**-1, self.sigma0)

        self.entropy0 = 1e14*self.grid.cell_ones()

        ic_pdict = params._pdict.copy()
        ic_pdict["EVOLVE_SIGMA"] = False
        ic_pdict["FB_ON"] = False
        ic_pdict["SAVE"] = False

        sim_dir = os.path.join(os.getcwd(), ic_pdict["SIM_DIR"])
        if not os.path.isdir(sim_dir):
            os.mkdir(sim_dir)


        self.sim = Simulation(self.sigma0, self.entropy0, params=Params(load=ic_pdict))
        self.sim.evolve()
        self.sigma0 = self.sim.var0.sigma
        self.entropy0 = self.sim.var0.s

    def makefig(self):
        save_dir = os.path.join(os.getcwd(),self.params.SIM_DIR)
        fig, axs = plt.subplots(1,3, figsize=(10, 3))
        axs[0].loglog(self.grid.r_cell, self.sigma0)
        axs[0].set_title(r"$\Sigma$ (g/cm$^2$)", fontsize=15)
        axs[1].loglog(self.grid.r_cell, self.entropy0)
        axs[1].set_title(r"$s$ (erg/g$\cdot$K)", fontsize=15)

        q_wind = self.sim.var0.qwind
        q_visc = self.sim.var0.qvis
        q_rad = self.sim.var0.qrad

        ts_face = np.interp(np.log10(self.grid.r_face), np.log10(self.grid.r_cell), self.sim.var0.ts)
        ts_flux = ts_face*self.sim.var0.vr
        net_flux = self.grid.cell_zeros()
        net_flux[1:-1] = (ts_flux[1:]*self.grid.face_area[1:] - ts_flux[:-1]*self.grid.face_area[:-1])/self.grid.cell_vol[1:-1]


        axs[2].loglog(self.grid.r_cell, q_wind/self.sigma0, label="wind")
        axs[2].loglog(self.grid.r_cell, q_visc/self.sigma0, label="visc")
        axs[2].loglog(self.grid.r_cell, q_rad/self.sigma0, label="rad")
        axs[2].loglog(self.grid.r_cell, np.abs(net_flux)/self.sigma0, label="adv")
        axs[2].legend(frameon=False)
        axs[2].set_title(r"$\dot q$ (erg/s$\cdot$g)", fontsize=15)
        axs[2].set_ylim(bottom=10)


        for ax in axs:
            ax.set_xlabel("Radius (cm)", fontsize=15)
        for ax in axs[:2]:
            ax.axvline(x=self.params.RT, linestyle="--", color="black")

        fig.savefig(f"{save_dir}/initial_conditions.png", dpi=300, bbox_inches="tight")









