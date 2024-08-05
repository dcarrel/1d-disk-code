from params import *
import os, sys, glob
import numpy as np
from grid import *
from variable import *

## Initial conditions are a surface density and an entropy per unit mass
class Simulation:
    def __init__(self, sigma0, entropy0, params=Params(), verbose=True):
        self.verbose = verbose
        ## sets up simulation directory
        if not os.path.isdir(os.path.join(os.getcwd(), params.SIM_DIR)):
            os.makedirs(os.path.join(os.getcwd(), params.SIM_DIR))

        self.sim_dir = os.path.join(os.getcwd(), params.SIM_DIR)
        self.params=params
        self.sigmafd = os.path.join(os.getcwd(), params.SIM_DIR + "/sigma.000.dat")
        self.entropyfd = os.path.join(os.getcwd(), params.SIM_DIR + "/entropy.000.dat")
        self.dt = np.inf

        if self.params.RESTART and self.params.SAVE:
            sigma0 = np.load(os.path.join(self.sim_dir, "sigma.npy"), mmap_mode="r")[-1]
            entropy0 = np.load(os.path.join(self.sim_dir, "entropy.npy"), mmap_mode="r")[-1]
            r_cell = np.load(os.path.join(self.sim_dir, "rocell.npy"), mmap_mode="r")
            self.grid = Grid(grid_array=r_cell)
            tarr = np.load(os.path.join(self.sim_dir, "tsave.npy"), mmap_mode="r")
            self.t0, self.t = tarr[0], tarr[-1]
            self.file_start = np.arange(self.t, self.params.TF, self.params.FILE_INT)
            self.file_start[0] = 3*self.params.TF
        else:
            self.grid = Grid(params=params)
            self.t0 = params.T0
            self.t = params.T0
            self.file_start = np.arange(self.t0, self.params.TF, self.params.FILE_INT)
            self.file_start[0] = 3 * self.params.TF

            if self.params.SAVE:
                self.params.save()
                old_dat = glob.glob(os.path.join(self.sim_dir, "*.dat"))
                for f in old_dat:
                    os.remove(f)


                sigmafile = open(self.sigmafd, "w")
                sigmafile.write(arr_to_string(self.grid.r_cell) + arr_to_string(sigma0, t=self.t))
                sigmafile.close()

                sfile = open(self.entropyfd, "w")
                sfile.write(arr_to_string(self.grid.r_cell) + arr_to_string(entropy0, t=self.t))
                sfile.close()

        self.tsave = np.append(np.arange(self.t, self.params.TF, params.TS), [self.params.TF])
        self.tsave[0] = 3*self.params.TF  ## Initial data is already saved, this makes sure that it is not saved again I guess
        self.var0 = FullVariable(params)
        self.var0.update_variables(sigma0, entropy0*sigma0, self.t0)

        self.vhalf = FullVariable(params)

    def take_step(self):
        dr = []
        if self.params.GEOMETRY.__eq__("LOGARITHMIC"):
            dr_left = self.grid.r_face - self.grid.r_cell[:-1]
            dr_right = self.grid.r_cell[1:] - self.grid.r_face
            dr = np.where(self.var0.vr < 0, dr_left, dr_right)
        elif self.params.GEOMETRY.__eq__("LINEAR"):
            dr = self.grid.dr / 2

        ts_arr = (self.params.SDT*np.abs(self.var0.ts/(self.var0.ts_dot+1e-50)))[1:-1]
        ts_loc = np.argmin(ts_arr)
        ts_dt = ts_arr[ts_loc]
        if not self.params.EVOLVE_ENTROPY: ts_dt=np.inf

        sigma_arr = (self.params.SDT*np.abs(self.var0.sigma/(self.var0.sigma_dot+1e-50)))[1:-1]
        sigma_loc = np.argmin(sigma_arr)
        sigma_dt = sigma_arr[sigma_loc]
        if not self.params.EVOLVE_SIGMA: sigma_dt = np.inf

        cfl_arr = self.params.CFLDT*np.abs(dr/(self.var0.vr+1e-50))
        cfl_loc = np.argmin(cfl_arr)
        cfl_dt = cfl_arr[cfl_loc]

        sim_dt = np.minimum(np.minimum(ts_dt, sigma_dt), cfl_dt)
        self.dt = np.minimum(self.dt, sim_dt)

        arr = np.array([ts_dt, sigma_dt, cfl_dt])
        st = ["s", "sigma", "cfl"][np.argmin(arr)]
        loc = [ts_loc, sigma_loc, cfl_loc][np.argmin(arr)]
        loc = self.grid.r_cell[loc]/self.grid.r_cell[0]

        ## I have no fucking clue what any of this dumb shit is
        n, ts_full, sigma_full, max_sigma_err, max_ts_err = 0, [], [], np.inf, np.inf
        vr = np.copy(self.var0.vr)

        while True: ## adaptive timestepping based on RK12
            sigma = self.var0.sigma_var()
            ts = self.var0.ts_var()

            ts_full_o1, sigma_full_o1 = ts.data, sigma.data
            if self.params.EVOLVE_ENTROPY: ts_full_o1 = shasta_step(ts, vr, self.dt, interp=self.params.INTERP, diff=self.params.DIFF)
            if self.params.EVOLVE_SIGMA: sigma_full_o1 = shasta_step(sigma, vr, self.dt, interp=self.params.INTERP, diff=self.params.DIFF)

            ts_half, sigma_half = ts.data, sigma.data
            if self.params.EVOLVE_ENTROPY: ts_half = shasta_step(ts, vr, self.dt/2, interp=self.params.INTERP, diff=self.params.DIFF)
            if self.params.EVOLVE_SIGMA: sigma_half = shasta_step(sigma, vr, self.dt/2, interp=self.params.INTERP, diff=self.params.DIFF)
            self.vhalf.update_variables(sigma_half, ts_half, t=self.t+self.dt/2)

            vf_half = np.copy(self.vhalf.vr)
            sigma.D = self.vhalf.sigma_dot
            ts.D = self.vhalf.ts_dot

            ts_full, sigma_full = ts.data, sigma.data
            if self.params.EVOLVE_ENTROPY: ts_full = shasta_step(ts, vf_half, self.dt, interp=self.params.INTERP, diff=self.params.DIFF)
            if self.params.EVOLVE_SIGMA: sigma_full = shasta_step(sigma, vf_half, self.dt, interp=self.params.INTERP, diff=self.params.DIFF)

            ts_err = np.abs(ts_full_o1 - ts_full)/ts_full
            sigma_err = np.abs(sigma_full_o1 - sigma_full)/sigma_full

            max_ts_err = np.max(ts_err)
            max_sigma_err = np.max(sigma_err)

            err_condition = max_ts_err > self.params.TOL or max_sigma_err > self.params.TOL
            inv_condition = is_invalid(ts_full) or is_invalid(sigma_full)
            if err_condition or inv_condition:
                self.dt *= 0.85
                n += 1
            elif n > self.params.MAXIT:
                print(f"Could not converge to desired tolerances in {self.params.MAXIT} iterations")
                print(f"sigma_max {max_sigma_err:2.2e}; entropy_max {max_ts_err:2.2e}; dt {(self.dt/(self.params.TF-self.t0)):2.2e}")
                break
            else:
                self.dt /= 0.85
                break

        self.t += self.dt
        self.var0.update_variables(sigma_full, ts_full, t=self.t)
        return self.dt, st, loc, max_sigma_err, max_ts_err

    def evolve(self):
        sigmafile, sfile = None, None
        if self.params.SAVE:
            sigmafile = open(self.sigmafd, "a")
            sfile = open(self.entropyfd, "a")
        while True:

            old_sigma, old_entropy = [], []

            dt, st, loc, sig_err, ts_err = self.take_step()

            new_sigma, new_entropy = [], []


            cond = self.tsave - self.t <= 0 ## checks if simulation needs to save
            condf = self.file_start - self.t <= 0 ## checks if at the end of a file

            if np.any(condf) and self.params.SAVE:
                sigmafile.close()
                sfile.close()
                n = np.argmin(self.file_start)
                self.sigmafd = os.path.join(os.getcwd(), self.params.SIM_DIR + f"/sigma.{n:03d}.dat")
                self.entropyfd = os.path.join(os.getcwd(), self.params.SIM_DIR + f"/entropy.{n:03d}.dat")

                sigmafile = open(self.sigmafd, "w")
                sigmafile.write(arr_to_string(self.grid.r_cell)) ## write grid locations
                sigmafile.close()

                sfile = open(self.entropyfd, "w")
                sfile.write(arr_to_string(self.grid.r_cell))
                sfile.close()

                sigmafile = open(self.sigmafd, "a")
                sfile = open(self.entropyfd, "a")
                self.file_start[n] = 3*self.params.TF

            if np.any(cond) and self.params.SAVE:
                sigmafile.write(arr_to_string(self.var0.sigma, t=self.t))
                sfile.write(arr_to_string(self.var0.s, t=self.t))

                self.tsave = np.where(cond, self.params.TF*3, self.tsave)

            ## loop breaking conditions

            if self.verbose:
                print(
                    f"\rpct: {(self.t / self.params.TF * 100):2.2f}%\tdt={dt / (self.params.TF - self.t0):2.2e}\t{st}\tloc={loc:2.2f}"
                    + f"\tsig {sig_err:2.2e}\tts {ts_err:2.2e}\t\t\t\t\t", end="")
            if np.isnan(dt) or np.any(self.var0.ts<0): break
            if self.t > self.params.TF: break

        if self.params.SAVE:
            sigmafile.close()
            sfile.close()

            print("Saving converting into numpy arrays\t\t\t\t\t\t\t\t\t\t")
            sigma_files = sorted(glob.glob(os.path.join(self.sim_dir, "sigma.*.dat")))
            entropy_files = sorted(glob.glob(os.path.join(self.sim_dir, "entropy.*.dat")))

            sigma_array, entropy_array, t_array, nn = [], [], [], 0
            if glob.glob(os.path.join(self.sim_dir, "*.npy")):
                sigma_array = np.load(os.path.join(self.sim_dir, "sigma.npy"))
                entropy_array = np.load(os.path.join(self.sim_dir, "entropy.npy"))
                t_array = np.load(os.path.join(self.sim_dir, "tsave.npy"))
            else:
                sigma_array = np.loadtxt(sigma_files[0], skiprows=1)[:, 1:]
                entropy_array = np.loadtxt(entropy_files[0], skiprows=1)[:, 1:]
                t_array = np.loadtxt(entropy_files[0], skiprows=1, usecols=0)
                np.save(os.path.join(self.sim_dir, "r_cell.npy"), np.loadtxt(entropy_files[0], max_rows=1))

            for s, e in zip(sigma_files, entropy_files):
                sigma_array = np.vstack((sigma_array, np.loadtxt(s, skiprows=1)[:, 1:]))
                entropy_array = np.vstack((entropy_array, np.loadtxt(e, skiprows=1)[:, 1:]))
                t_array = np.append(t_array, np.loadtxt(s, skiprows=1, usecols=0))
            for s, e in zip(sigma_files, entropy_files):
                os.remove(s)
                os.remove(e)

            np.save(os.path.join(self.sim_dir, "sigma.npy"), sigma_array)
            np.save(os.path.join(self.sim_dir, "entropy.npy"), entropy_array)
            np.save(os.path.join(self.sim_dir, "tsave.npy"), t_array)

def shasta_step(var, vf, dt, interp="LINEAR", diff=1):
    ## add different interpolation options at some point, probably will not really matter
    ## no reason to at the moment
    var_face, vol_face = [], []
    if interp.__eq__("LINEAR"):
        var_face = np.interp(var.grid.r_face, var.grid.r_cell, var.data)
        vol_face = np.interp(var.grid.r_face, var.grid.r_cell, var.grid.cell_vol)
    elif interp.__eq__("LOGARITHMIC"):  ## linear interpolation on a evenly spaced logarithmic grid, I guess
        var_face = np.interp(np.log10(var.grid.r_face), np.log10(var.grid.r_cell), var.data)
        vol_face = np.interp(np.log10(var.grid.r_face), np.log10(var.grid.r_cell), var.grid.cell_vol)

    ## convective update
    var_ast = var.grid.cell_zeros()
    var_ast[1:-1] = var.data[1:-1]
    var_ast[1:-1] += (-dt*var.grid.face_area[1:]*vf[1:]*var_face[1:]+dt*var.grid.face_area[:-1]*vf[:-1]*var_face[:-1])/var.grid.cell_vol[1:-1]
    var_ast[0] = var_ast[1]
    var_ast[-1] = var_ast[-2]


    ## transport update
    var_T = var.grid.cell_zeros()
    var_T[1:-1] = var_ast[1:-1] + dt * var.D[1:-1]
    var_T[0] = var_T[1]
    var_T[-1] = var_T[-2]

    ## diffusive variables
    eps_face = 0.5*var.grid.face_area*vf*dt*(1/var.grid.cell_vol[:-1] + 1/var.grid.cell_vol[1:])
    nu_face = 1/6 + 1/3*eps_face**2
    mu_face = 1/6 - 1/6*eps_face**2
    mu_face *= diff


    ## diffusive update
    var_tild = var.grid.cell_zeros()
    var_tild[1:-1] = var_T[1:-1]
    var_tild[1:-1] += (nu_face[1:]*vol_face[1:]*(var.data[2:]-var.data[1:-1]) -
                       nu_face[:-1]*vol_face[:-1]*(var.data[1:-1] - var.data[:-2]))/var.grid.cell_vol[1:-1]
    var_tild[0] = var_tild[1]
    var_tild[-1] = var_tild[-2]
    ## flux correction
    sign_face = np.sign(var_tild[1:] - var_tild[:-1])
    fad_face = mu_face*vol_face*(var_T[1:] - var_T[:-1])

    flux_face = var.grid.face_zeros()
    flux_left = sign_face[:-1]*var.grid.cell_vol[1:-1]*(var_tild[2:] - var_tild[1:-1])
    flux_right = sign_face[1:]*var.grid.cell_vol[1:-1]*(var_tild[1:-1] - var_tild[:-2])
    diff_face = np.minimum(flux_left[1:], flux_right[:-1])
    flux_face[1:-1] = sign_face[1:-1]*np.maximum(0, np.minimum(np.abs(fad_face[1:-1]), diff_face))
    flux_face[0] = sign_face[0]*np.maximum(0, np.minimum(np.abs(fad_face[0]), flux_left[0]))
    flux_face[-1] = sign_face[-1]*np.maximum(0, np.minimum(np.abs(fad_face[-1]), flux_right[-1]))

    var_ret = np.zeros(var.grid.r_cell.shape)
    var_ret[1:-1] = var_tild[1:-1] - var.grid.cell_vol[1:-1]**-1*(flux_face[1:] - flux_face[:-1])
    var_ret[0] = var_ret[1]
    var_ret[-1] = var_ret[-2]

    return var_ret
