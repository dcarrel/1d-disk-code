import numpy as np
from grid import *
from eos import *
from opacity import *
from scipy.special import gamma

class ShastaVariable:
    def __init__(self, grid, data, vf, D):
        self.grid = grid
        self.data = data
        self.vf = vf
        self.D = D

class FullVariable:
    def __init__(self, params):
        self.eos = load_table(params.EOS_TABLE)
        self.params=params
        self.grid = Grid(params=params)

    def update_variables(self, sigma, ts, t=0):
        self.sigma = sigma
        self.chi = self.sigma*self.grid.omgko
        self.ts = ts
        self.s = self.ts/self.sigma
        self.t = t

        self.T = self.eos(self.chi, self.s)
        self.rho = entropy_difference(self.T, self.chi, self.s, just_density=True)
        self.U = RADA*self.T**4 + 1.5*self.rho*KB*self.T/mu
        self.P = RADA*self.T**4/3 + self.rho*KB*self.T/mu

        self.be = -1 + 2*(self.U + self.P)/self.rho/self.grid.vk2o
        self.H = self.sigma/2/self.rho
        self.h = self.H/self.grid.r_cell

        if self.params.CONST_NU:
            self.nu = self.params.CONST_NU*self.grid.cell_ones()
        else:
            self.nu = self.params.ALPHA*self.H**2*self.grid.omgk(self.h)

        self.sigv = sig((self.be-self.params.BE_CRIT)/self.params.DBE)
        ## calculates velocities at inter_faces
        lc_sigma = 2 * np.pi * self.sigma * self.grid.r_cell
        g = np.sqrt(self.grid.r_cell) / (self.nu+1e-20)
        d = 3*self.nu

        lc_sigma_tild, g_tild, d_tild = [], [], []
        if self.params.INTERP.__eq__("LINEAR"):
            lc_sigma_tild = np.interp(self.grid.r_face, self.grid.r_cell, lc_sigma)
            g_tild = np.interp(self.grid.r_face, self.grid.r_cell, g)
            d_tild = np.interp(self.grid.r_face, self.grid.r_cell, d)
        elif self.params.INTERP.__eq__("LOGARITHMIC"):
            lc_sigma_tild = np.interp(np.log10(self.grid.r_face), np.log10(self.grid.r_cell), lc_sigma)
            g_tild = np.interp(np.log10(self.grid.r_face), np.log10(self.grid.r_cell), g)
            d_tild = np.interp(np.log10(self.grid.r_face), np.log10(self.grid.r_cell), d)

        self.vr = -d_tild * g_tild / lc_sigma_tild * (lc_sigma[1:] / g[1:] - lc_sigma[:-1] / g[:-1])/self.grid.ddr

        ## calculate source terms for density
        sigma_wl = self.sigma*self.grid.omgk(self.h)*self.sigv ## wind loss
        if not self.params.WIND_ON: sigma_wl *= 0

        sigma_fb= 1/np.pi/gamma(self.params.FBK/2+1)/self.params.FBR0**2
        sigma_fb *= (self.grid.r_cell/self.params.FBR0)**self.params.FBK*np.exp(-(self.grid.r_cell/self.params.FBR0)**2)
        sigma_fb *= self.params.MSTAR/5/self.params.TFB
        if t > self.params.TFB: sigma_fb *= (self.params.TFB/t)**(5/3)
        if not self.params.FB_ON: sigma_fb *= 0

        self.sigma_dot = sigma_fb - sigma_wl

        # calculate source terms for entropy
        kappa = kappa_interpolator(self.rho, self.T)
        self.qrad = 4*RADA*self.T**4*c/(1+kappa*self.sigma)  # radiative cooling
        self.qwind = self.params.FWIND*self.grid.omgk(self.h)*self.sigv*self.sigma*self.grid.vk2(self.h)  # wind cooling
        if not self.params.WIND_ON: self.qwind *= 0
        self.qvis = 2.25*self.nu*self.grid.omgk(self.H)**2*self.sigma  # viscous cooling
        self.qfb = self.params.FSH*0.5*sigma_fb*self.grid.vk2(self.h) # fallback heating
        if not self.params.FB_ON: self.qfb *= 0

        self.ts_dot = (self.qvis+self.qfb-self.qrad-self.qwind)/self.T

        self.ts_dt = np.abs(self.ts/self.ts_dot)
        self.sigma_dt = np.abs(self.sigma/self.sigma_dot)

    def sigma_var(self):
        return ShastaVariable(self.grid, self.sigma, self.vr, self.sigma_dot)
    def ts_var(self):
        return ShastaVariable(self.grid, self.ts, self.vr, self.ts_dot)