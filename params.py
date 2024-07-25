import numpy as np
from consts import *
import json, os


class Params:
    def __init__(self, MBH=1e6*MSUN, XH=0.7, XHe=0.3,
                 R0=5, RF=1000, NR=500,
                 T0=0, TF=5*YEAR, TS=0.1*DAY,
                 SIM_NAME="my_sim", FILE_INT=0.1*YEAR, RESTART=False,
                 CFLDT = 0.7, SDT = 0.5, FT=0.8,
                 BE_CRIT=-0.1, DBE=1/300, FWIND=0.5,
                 FBK=-3/2, FBR0=1e13, FSH=0.5,
                 ALPHA=0.05, TOL=1e-2, MAXIT=1000, DIFF=1., load=None,
                 EOS_TABLE="EOS_TABLE"
                 ):
        
        if load is None:
            self._pdict = {"MBH": MBH,
                           "XH": XH,
                           "XHe": XHe,
                           "R0": R0,
                           "RF": RF,
                           "NR": NR,
                           "T0": T0,
                           "TF": TF,
                           "TS": TS,
                           "SIM_NAME": SIM_NAME,
                           "FILE_INT": FILE_INT,
                           "RESTART": RESTART,
                           "CFLDT": CFLDT,
                           "SDT": SDT,
                           "FT": FT,
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
                           "EOS_TABLE": EOS_TABLE
            }
        else:
            with open(os.path.join(os.getcwd(), load+"/params.json")) as f:
                self._pdict = json.load(f)
        ## writes param file

        
        self.MBH = self._pdict["MBH"]
        self.XH = self._pdict["XH"]
        self.XHe = self._pdict["XHe"]

        self.RSCH = 2*CONST_G*self.MBH/c**2
        self.R0 = self._pdict["R0"]*self.RSCH
        self.RF = self._pdict["RF"]*self.RSCH
        self.NR = self._pdict["NR"]

        self.T0 = self._pdict["T0"]
        self.TF = self._pdict["TF"]
        self.TS = self._pdict["TS"]

        self.SIM_NAME = self._pdict["SIM_NAME"]
        self.FILE_INT = self._pdict["FILE_INT"]
        self.RESTART = self._pdict["RESTART"]

        self.FT = self._pdict["FT"]
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

    def save(self, file_name=None):
        if file_name is None: file_name = "params.json"
        with open(os.path.join(os.getcwd(), self.SIM_NAME + "/" + file_name), "w") as f:
            json.dump(self._pdict, f)



