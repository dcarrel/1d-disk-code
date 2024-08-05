from load import *
from params import *
from celluloid import Camera
import matplotlib.pyplot as plt
import os

simulation_directory = "runs/aug04_main02"
load_sim = LoadSimulation(params=Params(load=simulation_directory), mode="LINEAR", dt=0.2*DAY)

fig, axs = plt.subplots(1, 6, figsize=(15,4))
cam = Camera(fig)

for ax in axs:
    ax.set_xlabel("r (cm)", fontsize=15)
axs[-1].set_xlabel(r"$\nu$ (Hz)", fontsize=15)

axs[0].set_title(r"$\Sigma$ (g/cm$^2$)", fontsize=15)
axs[1].set_title(r"$s$ (erg/g$\cdot$K)", fontsize=15)
axs[2].set_title(r"$T$ (K)", fontsize=15)
axs[3].set_title(r"$\rho$ (g/cm$^3$)", fontsize=15)
axs[4].set_title(r"$h$", fontsize=15)
axs[5].set_title(r"$\nu L_\nu$ (erg/s)", fontsize=15)

for i, t in enumerate(load_sim.ts):
    axs[0].loglog(load_sim.grid.r_cell, load_sim.sigma[i], color="tab:blue")
    axs[0].text(load_sim.grid.r_cell[60], 100, f"t={load_sim.ts[i] / DAY:2.2f} days")
    axs[1].loglog(load_sim.grid.r_cell, load_sim.s[i], color="tab:blue")
    axs[2].loglog(load_sim.grid.r_cell, load_sim.T[i], color="tab:blue")
    axs[3].loglog(load_sim.grid.r_cell, load_sim.rho[i], color="tab:blue")
    axs[4].loglog(load_sim.grid.r_cell, load_sim.h[i], color="tab:blue")
    axs[5].loglog(load_sim.nuf, load_sim.nuL_nu[i], color="tab:blue")
    plt.tight_layout()
    cam.snap()
anim = cam.animate(interval = 100)
anim.save(os.path.join(os.getcwd(), simulation_directory+"/animation.mp4"))

