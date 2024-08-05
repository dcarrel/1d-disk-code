from simulation import *
from initial_conditions import *
import numpy as np
import os, sys, glob
from grid import *
from params import *

import multiprocessing as mp

NR=256

main_p0 = Params(SIM_DIR="runs/aug04_main00hr", MBH=1e6*MSUN, MSTAR=MSUN, ALPHA=0.01, R0=4, RF=4000, NR=NR, FSH=0.5, DIFF=1, FWIND=0.5,
                 GEOMETRY="LOGARITHMIC", INTERP="LOGARITHMIC", TOL=5e-3)

my_params=[main_p0]

def run_simulation(param):
    ic = InitialCondition(m0=0.01, tv=0.05, ambf=1e-3, params=param, tf=2*MONTH)
    sim = Simulation(sigma0=ic.sigma0, entropy0=ic.entropy0, params=param)
    sim.evolve()
    print(f"\nSaved to {param.SIM_DIR}")

if __name__.__eq__("__main__"):
    procs = [mp.Process(target=run_simulation, args=(param,)) for param in my_params]

    for proc in procs:
        proc.start()



