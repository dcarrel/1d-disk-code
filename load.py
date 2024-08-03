from grid import *
import os
from eos import *
from consts import *

def vector_interp(r_face, r_cell, v_cell):
    v_cell = v_cell[:,:-1]
    v_cell += (v_cell[:,1:] - v_cell[:,:-1])/()*(r_face - r_cell[:-1])
    return v_cell

class LoadSimulation:
    def __init__(self, t0=None, tf=None, dt=0.1*YEAR, mode="CONST", tend=3*MONTH, dts=10*DAY, tstart=10*DAY, params=Params()):
        self.sim_dir = os.path.join(os.getcwd(), params.SIM_DIR)
        self.ts = np.load(os.path.join(self.sim_dir, "tsave.npy"), mmap_mode="r")
        if t0 is None:
            t0 = self.ts[0]
        if tf is None:
            tf = self.ts[-1]

        comparr = []
        if mode.__eq__("CONST"):
            comparr = np.append(np.arange(t0, tf, dt), tf)
        elif mode.__eq__("LINEAR"):
            # Linear ramp
            slope = (dts-dt)/(tend-tstart)
            first = np.arange(t0, tstart, dt)
            n=0
            trans = [tstart]
            while True:
                dtp = dt + slope * (trans[-1]-tstart)
                trans += [trans[-1] + dtp]
                if trans[-1] > tend:
                    break
            last = np.append(np.arange(tend+dts, tf, dts), tf)
            comparr = np.append(first, trans)
            comparr = np.append(comparr, last)
        elif mode.__eq__("LOGARITHMIC"):
            dtlog = np.log10((self.ts[1]+dt)/self.ts[1])
            logsp = np.append([t0], np.arange(np.log10(self.ts[1]), np.log10(tf)+dtlog, dtlog))
            comparr = 10**logsp
        ts_proj = np.einsum("i,j->ij", np.ones(comparr.shape), self.ts)
        ca_proj = np.einsum("i,j->ij", comparr, np.ones(self.ts.shape))
        indices = np.unique(np.argmin(np.abs(ts_proj - ca_proj), axis=1))

        self.sigma = np.load(os.path.join(self.sim_dir, "sigma.npy"), mmap_mode="r")[indices]
        self.s = np.load(os.path.join(self.sim_dir, "entropy.npy"), mmap_mode="r")[indices]
        self.ts = self.ts[indices]
        self.grid = Grid(grid_array=np.load(os.path.join(self.sim_dir, "r_cell.npy")), params=params)
        self.eos = load_table(params.EOS_TABLE)

        self.chi = self.sigma*self.grid.omgko
        self.T = self.eos(self.chi, self.s)

        self.rho = entropy_difference(self.T, self.chi, self.s, just_density=True)

        self.rad_P = RADA*self.T**4/3
        self.gas_P = self.rho*KB*self.T/mu
        self.P = self.rad_P + self.gas_P
        self.U = RADA * self.T ** 4 + 1.5 * self.rho * KB * self.T / mu

        self.be = -1 + 2 * (self.U + self.P) / self.rho/self.grid.vk2o
        self.H = self.sigma / 2 / self.rho
        self.h = self.H/self.grid.r_cell

        if params.CONST_NU:
            self.nu = params.CONST_NU*self.grid.cell_ones()
        else:
            self.nu = params.ALPHA * self.H ** 2 * self.grid.omgk(self.h)

        self.nuf = np.logspace(13, 18, 100)
        self.S_nu = Snu(self.T, self.nuf, self.grid)
        self.L_nu = 4*np.pi*(10*PC)**2*self.S_nu
        self.nuL_nu = np.einsum("j,ij->ij", self.nuf, self.L_nu)

        self.sigv = sig((self.be - params.BE_CRIT) / params.DBE)
        ## calculates velocities at inter_faces
        lc_sigma = 2 * np.pi * self.sigma * self.grid.r_cell
        g = np.sqrt(self.grid.r_cell) / (self.nu + 1e-20)
        d = 3 * self.nu

        g_tild, d_tild, lc_sigma_tild = [], [], []
        if params.INTERP.__eq__("LINEAR"):
            r_cell = self.grid.r_cell
            r_face = self.grid.r_face
            g_tild = vector_interp(r_face, r_cell, g)
            d_tild = vector_interp(r_face, r_cell, d)
            lc_sigma_tild = vector_interp(r_face, r_cell, lc_sigma)
        elif params.INTERP.__eq__("LOGARITHMIC"):
            logr_cell = np.log10(self.grid.r_cell)
            logr_face = np.log10(self.grid.r_face)
            g_tild = vector_interp(logr_face, logr_cell, g)
            d_tild = vector_interp(logr_face, logr_cell, d)
            lc_sigma_tild = vector_interp(logr_face, logr_cell, lc_sigma)

        self.vr = -d_tild*g_tild/lc_sigma_tild * (lc_sigma[:,1:]/g[:,1:] - lc_sigma[:,:-1]/g[:,:-1])/self.grid.ddr

        ## calculate source terms for density
        sigma_wl = params.FWIND*self.sigma * self.grid.omgk(self.h) * self.sigv  ## wind loss
        if not params.WIND_ON: sigma_wl *= 0

        sigma_fb = 1/np.pi/gamma(params.FBK/2+1)/params.FBR0**2
        sigma_fb *= (self.grid.r_cell/params.FBR0)**params.FBK*np.exp(-(self.grid.r_cell/params.FBR0)**2)
        time_part = np.where(self.ts > params.TFB, (params.TFB/self.ts)**(5/3), params.MSTAR/5/params.TFB)
        sigma_fb = np.einsum("i,j->ij", time_part, sigma_fb)
        if not params.FB_ON: sigma_fb *= 0

        self.sigma_fb = sigma_fb
        self.sigma_wl = sigma_wl
        self.sigma_dot = sigma_fb - sigma_wl

        ## calculate source terms for entropy
        kappa = kappa_interpolator(self.rho, self.T)
        self.kappa = kappa

        qrad = 4 * RADA * self.T ** 4 * c / (1 + kappa * self.sigma)
        qwind = params.FWIND*self.grid.omgk(self.h)*self.sigv*self.sigma*self.grid.vk2(self.h)
        if not params.WIND_ON: qwind *= 0

        qvis = 2.25 * self.nu * self.grid.omgk(self.h) ** 2*self.sigma
        qfb = params.FSH*0.5*sigma_fb*self.grid.vk2(self.h)
        if not params.FB_ON: qfb *= 0

        self.qvis = qvis/self.sigma
        self.qfb = qfb/self.sigma
        self.qrad = qrad/self.sigma
        self.qwind = qwind/self.sigma
        self.ts_dot = (qvis + qfb - qrad - qwind) / self.T