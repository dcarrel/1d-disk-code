import numpy as np
from consts import *
import json, os
class Params:
    def __init__(self,
                 MBH=1e6*MSUN,          ## BH mass
                 R0=5,                  ## Inner grid radius in Rsch
                 RF=1000,               ## Outer grid radius in Rsch
                 NR=500,                ## Number of grid points
                 T0=0,                  ## Initial time
                 TF=5*YEAR,             ## Final time
                 TS=0.1*DAY,            ## Increment between saves
                 SIM_DIR="MY_SIM",     ## File name
                 FILE_INT=0.1*YEAR,     ## Increment between different files (doesn't really matter)
                 RESTART=False,         ## Whether or not to restart, not really that useful
                 CFLDT = 0.7,           ## CFL number
                 SDT = 0.5,             ## Source number, kind of like CFL number for sources
                 BE_CRIT=-0.1,          ## Wind parameter
                 DBE=1/300,             ## Wind parameter
                 FWIND=0.5,             ## Wind parameter
                 FBK=-3/2,              ## Fallback parameter
                 FBR0=2   ,             ## Fallback parameter in terms of tidal disruption radius
                 FSH=0.5,               ## Fallback parameter
                 ALPHA=0.05,            ## Viscosity parameter
                 TOL=1e-3,              ## Numerical tolerance, approximated by RK1/2 difference
                 MAXIT=1000,            ## Max number of iterations for adapative timestepping
                 DIFF=1.,               ## Anti-diffusion stage coefficient, 1 is maximum antidiffusion
                 load=None,             ## Option to load parameter file from pre-existing one
                 EOS_TABLE="EOS_TABLE", ## What EOS table to use
                 GEOMETRY="LINEAR",     ## Grid geometry
                 MSTAR=MSUN,            ## Stellar mass
                 PH_EFFECT=False,       ## Whether to include second order correction to rotation
                 INTERP="LINEAR",       ## How to interpolate from cells onto faces
                 ## Options: LINEAR: interpolates linearly on r_cell, LOGARITHMIC: interpolates linearly on log r_cell
                 WIND_ON = True,        ## If the wind should be on
                 FB_ON = True,          ## If the fallback should be on
                 CONST_NU=None,         ## Set to test against analytic solution. Does not evolve entropy
                 EVOLVE_SIGMA=True,     ## If false, fixes sigma profile
                 EVOLVE_ENTROPY=True,    ## If false, fixes entropy profile
                 SAVE=True
                 ):
        
        if load is None:
            self._pdict = {"MBH": MBH,
                           "R0": R0,
                           "RF": RF,
                           "NR": NR,
                           "T0": T0,
                           "TF": TF,
                           "TS": TS,
                           "SIM_DIR": SIM_DIR,
                           "FILE_INT": FILE_INT,
                           "RESTART": RESTART,
                           "CFLDT": CFLDT,
                           "SDT": SDT,
                           "BE_CRIT": BE_CRIT,
                           "DBE": DBE,
                           "FWIND": FWIND,
                           "FBK": FBK,
                           "FBR0": FBR0,
                           "FSH": FSH,
                           "ALPHA":ALPHA,
                           "TOL": TOL,
                           "MAXIT": MAXIT,
                           "DIFF": DIFF,
                           "EOS_TABLE": EOS_TABLE,
                           "GEOMETRY": GEOMETRY,
                           "MSTAR": MSTAR,
                           "PH_EFFECT": PH_EFFECT,  ## makes change to rotation
                           "INTERP": INTERP,
                           "WIND_ON": WIND_ON,
                           "FB_ON": FB_ON,
                           "CONST_NU": CONST_NU,
                           "EVOLVE_ENTROPY": EVOLVE_ENTROPY,
                           "EVOLVE_SIGMA": EVOLVE_SIGMA,
                           "SAVE": SAVE
            }
        else:
            if isinstance(load, dict):
                self._pdict = load.copy()
            elif isinstance(load, str):
                with open(os.path.join(os.getcwd(), load+"/params.json")) as f:
                    self._pdict = json.load(f)
        ## writes param file

        
        self.MBH = self._pdict["MBH"]

        self.RSCH = 2*CONST_G*self.MBH/c**2
        self.R0 = self._pdict["R0"]*self.RSCH
        self.RF = self._pdict["RF"]*self.RSCH
        self.NR = self._pdict["NR"]

        self.T0 = self._pdict["T0"]
        self.TF = self._pdict["TF"]
        self.TS = self._pdict["TS"]

        self.SIM_DIR = "runs/"+self._pdict["SIM_DIR"]
        if not os.path.exists(os.path.join(os.getcwd(), "runs")):
            os.mkdir(os.path.join(os.getcwd(), "runs"))
        self.FILE_INT = self._pdict["FILE_INT"]
        self.RESTART = self._pdict["RESTART"]

        self.CFLDT = self._pdict["CFLDT"]
        self.SDT = self._pdict["SDT"]

        self.BE_CRIT = self._pdict["BE_CRIT"]
        self.DBE = self._pdict["DBE"]
        self.FWIND = self._pdict["FWIND"]

        self.FBK = self._pdict["FBK"]
        self.FBR0 = self._pdict["FBR0"]
        self.FSH = self._pdict["FSH"]

        self.ALPHA = self._pdict["ALPHA"]
        self.TOL = self._pdict["TOL"]
        self.MAXIT = self._pdict["MAXIT"]
        self.DIFF = self._pdict["DIFF"]
        self.EOS_TABLE = self._pdict["EOS_TABLE"]
        self.GEOMETRY = self._pdict["GEOMETRY"]
        self.MSTAR = self._pdict["MSTAR"]
        self.PH_EFFECT = self._pdict["PH_EFFECT"]
        self.INTERP = self._pdict["INTERP"]

        self.WIND_ON = self._pdict["WIND_ON"]
        self.FB_ON = self._pdict["FB_ON"]

        self.CONST_NU = self._pdict["CONST_NU"]
        if self.CONST_NU: self._pdict["EVOLVE_ENTROPY"] = False
        self.EVOLVE_ENTROPY = self._pdict["EVOLVE_ENTROPY"]

        self.EVOLVE_SIGMA = self._pdict["EVOLVE_SIGMA"]
        self.SAVE = self._pdict["SAVE"]

        ## c+p from https://dergipark.org.tr/en/download/article-file/1612778
        def stellar_radius(m):
            r = 0
            if m < 1.66:
                r = 1.054*m**0.935
            else:
                r = 1.371*m**0.542
            return r

        self.TFB = 0.11*YEAR*stellar_radius(self.MSTAR/MSUN)**(3/2)*(self.MBH/1e6/MSUN)**(1/6)*(self.MSTAR/MSUN)**-1
        self.RT = 100*RSUN*stellar_radius(self.MSTAR/MSUN)*(MSUN/self.MSTAR)**(1/3)*(self.MBH/1e6/MSUN)**(1/3)
        self.MDOT = self.MSTAR/4/self.TFB

        self.FBR0 *= self.RT
    def save(self, file_name=None):
        if file_name is None: file_name = "params.json"
        with open(os.path.join(os.getcwd(), self.SIM_DIR + "/" + file_name), "w") as f:
            json.dump(self._pdict, f)



