from simulation import *
from initial_conditions import *
import numpy as np
import os, sys, glob
from grid import *
from params import *

NR=256

main_p0 = Params(SIM_DIR="runs/aug04_main00hr", MBH=1e6*MSUN, MSTAR=MSUN, ALPHA=0.01, R0=4, RF=1000, NR=NR, FSH=0.5, DIFF=0.99, FWIND=0.5, FB_ON=False,
                 GEOMETRY="LOGARITHMIC", INTERP="LOGARITHMIC", TOL=5e-3)
main_p1 = Params(SIM_DIR="runs/aug04_main01hr",MBH=1e6*MSUN, MSTAR=MSUN, ALPHA=0.01, R0=4, RF=1000, NR=NR, FSH=0.5, DIFF=0.99, FWIND=0.5,
                 GEOMETRY="LOGARITHMIC", INTERP="LOGARITHMIC", TOL=5e-3)
main_p2 = Params(SIM_DIR="runs/aug04_main02hr", MBH=1e6*MSUN, MSTAR=8*MSUN, ALPHA=0.01, R0=4, RF=1000, NR=NR, FSH=0.5, DIFF=0.99, FWIND=0.5,
                 GEOMETRY="LOGARITHMIC", INTERP="LOGARITHMIC", TOL=5e-3)
main_p3 = Params(SIM_DIR="runs/aug04_main03hr", MBH=1e6*MSUN, MSTAR=8*MSUN, ALPHA=0.05, R0=4, RF=1000, NR=NR, FSH=0.5, DIFF=0.99, FWIND=0.5,
                 GEOMETRY="LOGARITHMIC", INTERP="LOGARITHMIC", TOL=5e-3)
main_p4 = Params(SIM_DIR="runs/aug04_main04hr", MBH=1e6*MSUN, MSTAR=8*MSUN, ALPHA=0.05, R0=4, RF=1000, NR=NR, FSH=0.5, DIFF=0.99, FWIND=0.5, FBR0=5,
                 GEOMETRY="LOGARITHMIC", INTERP="LOGARITHMIC", TOL=5e-3)
main_p5 = Params(SIM_DIR="runs/aug04_main05hr", MBH=1e7*MSUN, MSTAR=8*MSUN, ALPHA=0.05, R0=4, RF=1000, NR=NR, FSH=0.5, DIFF=0.99, FWIND=0.5,
                 GEOMETRY="LOGARITHMIC", INTERP="LOGARITHMIC", TOL=5e-3)

main_ps = [main_p0, main_p1, main_p2, main_p3, main_p4, main_p5]
main_ic = [None, main_p0, None, None, main_p3, None]

## Use same initial conditions as main_p0
fsh_p1 = Params(SIM_DIR="runs/aug04_fsh01hr", MBH=1e6*MSUN, MSTAR=MSUN, ALPHA=0.01, R0=4, RF=1000, NR=NR, FSH=0.05, DIFF=0.99, FWIND=0.5, TF=2*YEAR,
                GEOMETRY="LOGARITHMIC", INTERP="LOGARITHMIC", TOL=5e-3)
fsh_p2 = Params(SIM_DIR="runs/aug04_fsh02hr",MBH=1e6*MSUN, MSTAR=MSUN, ALPHA=0.01, R0=4, RF=1000, NR=NR, FSH=0.005, DIFF=0.99, FWIND=0.5, TF=2*YEAR,
                GEOMETRY="LOGARITHMIC", INTERP="LOGARITHMIC", TOL=5e-3)

fsh_ps = [fsh_p1, fsh_p2]
fsh_ic = [main_p0, main_p0]

## Use same initial conditions as main_p0
diff_p1 = Params(SIM_DIR="runs/aug04_diff01hr", MBH=1e6*MSUN, MSTAR=MSUN, ALPHA=0.01, R0=4, RF=1000, NR=NR, FSH=0.5,
                 DIFF=0.95, FWIND=0.5, TF=2*YEAR, GEOMETRY="LOGARITHMIC", INTERP="LOGARITHMIC", TOL=5e-3)
diff_p2 = Params(SIM_DIR="runs/aug04_diff02hr", MBH=1e6*MSUN, MSTAR=MSUN, ALPHA=0.01, R0=4, RF=1000, NR=NR, FSH=0.5,
                 DIFF=0.98, FWIND=0.5, TF=2*YEAR, GEOMETRY="LOGARITHMIC", INTERP="LOGARITHMIC", TOL=5e-3)
diff_p3 = Params(SIM_DIR="runs/aug04_diff03hr", MBH=1e6*MSUN, MSTAR=MSUN, ALPHA=0.01, R0=4, RF=1000, NR=NR, FSH=0.5,
                 DIFF=0.995, FWIND=0.5, TF=2*YEAR, GEOMETRY="LOGARITHMIC", INTERP="LOGARITHMIC", TOL=5e-3)

diff_ps = [diff_p1, diff_p2, diff_p3]
diff_ic = [main_p0, main_p0, main_p0]

## main runs
for (p, ic) in zip(main_ps, main_ic):
    ics = []
    if ic is None:
        print("Generating initial conditions\n")
        ic = InitialCondition(m0=0.01, tv=0.015, ambf=1e-3, params=p, tf=7*DAY)
    else:
        ic = InitialCondition(load_from=ic.SIM_DIR+"/ics")
    sim = Simulation(sigma0=ic.sigma0, entropy0=ic.entropy0, params=p)
    print("Running simulation...\n")
    sim.evolve()
    print(f"Saved to {p.SIM_DIR}\n")

## shock parameter runs
for (p, ic) in zip(fsh_ps, fsh_ic):
    ics = []
    if ic is None:
        print("Generating initial conditions")
        ic = InitialCondition(m0=0.01, tv=0.015, ambf=1e-3, params=p, tf=14*DAY)
    else:
        ic = InitialCondition(load_from=ic.SIM_DIR+"/ics")
    sim = Simulation(sigma0=ic.sigma0, entropy0=ic.entropy0, params=p)
    print("Running simulation...")
    sim.evolve()

## diffusion runs
for (p, ic) in zip(diff_ps, diff_ic):
    ics = []
    if ic is None:
        print("Generating initial conditions")
        ic = InitialCondition(m0=0.01, tv=0.03, ambf=1e-3, params=p, tf=2*MONTH)
    else:
        ic = InitialCondition(load_from=ic.SIM_DIR+"/ics")
    sim = Simulation(sigma0=ic.sigma0, entropy0=ic.entropy0, params=p)
    print("Running simulation...")
    sim.evolve()


