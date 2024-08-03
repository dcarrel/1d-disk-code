import numpy as np
from eos import *
from opacity import kappa_interpolator
from scipy.special import gamma
import os, sys, glob
from params import *

class Grid:
    def __init__(self, grid_array=None, params=Params(MBH=1e6*MSUN, GEOMETRY="LINEAR")):
        self.params = params
        self.hmax = 0.3 ## Add option to fix later
        ri=params.R0
        rf=params.RF
        nt=params.NR
        if grid_array is not None:
            self.r_cell = np.array(grid_array)
            nt = len(self.r_cell)-2
            ri, rf = self.r_cell[1], self.r_cell[-2]

        # TODO: figure out wtf to do with the different drs and other crap
        if params.GEOMETRY.__eq__("LOGARITHMIC"):
            drlog = np.log10(rf/ri)/(nt-1)
            logi, logf = np.log10(ri), np.log10(rf)
            self.r_cell = np.logspace(logi - drlog, logf + drlog, nt + 2)
            self.r_face = np.logspace(logi - 3*drlog/2, logf + 3*drlog/2, nt + 3)
            self.dr = self.r_face[1:] - self.r_face[:-1]
            self.ddr = self.r_face[1:-1]*np.log(10)*drlog  # for calculating cell-centered derivatives

        elif params.GEOMETRY.__eq__("LINEAR"):
            dr = (rf - ri) / (nt - 1)
            self.r_cell = np.linspace(ri - dr, rf + dr, nt + 2)
            self.r_face = np.linspace(ri - 3*dr/2, rf + 3*dr/2, nt + 3)
            self.dr = dr
            self.ddr = dr # for calculating cell-centered derivatives

        self.omgko = np.sqrt(CONST_G*params.MBH/self.r_cell**3)
        self.vk2o = self.r_cell**2*self.omgko**2


        self.cell_vol = np.pi*(self.r_face[1:]**2 - self.r_face[:-1]**2)
        self.r_face = self.r_face[1:-1]
        self.face_area = 2*np.pi*self.r_face

    def omgk(self, h=None):
        if self.params.PH_EFFECT:
            h = np.where(h>self.hmax, self.hmax, h)
            return self.omgko/(1+h**2)
        else:
            return self.omgko
    def vk2(self, h=None):
        if self.params.PH_EFFECT:
            h = np.where(h > self.hmax, self.hmax, h)
            return self.omgko/(1+h**2)
        else:
            return self.vk2o
    def save(self):
        with open(os.path.join(os.getcwd(), self.params.SIM_DIR+"/grid_data.dat")) as f:
            f.write(arr_to_string(self.r_cell))
            f.write(arr_to_string(self.r_face))
    def cell_zeros(self):
        return np.zeros(self.r_cell.shape)
    def cell_ones(self):
        return np.ones(self.r_cell.shape)
    def face_zeros(self):
        return np.zeros(self.r_face.shape)
    def face_ones(self):
        return np.ones(self.r_face.shape)






