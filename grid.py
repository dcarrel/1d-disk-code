import numpy as np
from eos import *
from opacity import kappa_interpolator
from scipy.special import gamma
from scipy.interpolate import interp1d
import os, sys, glob
from params import *

def arr_to_string(array, t=None):
    str = f"{array[0]:5.5e}"
    if t is not None:
        str = f"{t:5.5e} {array[0]:5.5e}"
    for item in array[1:]:
        str += " "
        str += f"{item:5.5e}"
    str += "\n"
    return str

def sig(x):
    return (1+np.exp(-x))**-1

def Snu(T, nu, grid):
    exp_arg = np.einsum("ij,k->ijk", T**-1, HP*nu/KB)
    denom = np.exp(exp_arg) -1
    num = 2*HP/c/c*np.einsum("j,k->jk", 2*np.pi*grid.r_cell*grid.dr, nu**3)
    integrand = np.einsum("jk,ijk->ijk", num, denom**-1)
    integrand = np.where(exp_arg > 1000, 0, integrand)
    integrand = np.where(exp_arg < 1/1000, 2*KB/c/c*np.einsum("k,ij->ijk", nu**2, T*grid.r_cell*2*np.pi*grid.dr), integrand) ## rayleigh jeans
    return np.sum(integrand, axis=1)/(4*np.pi*(10*PC)**2)


class Grid:
    def __init__(self, ri=None, rf=None, nt=None, grid_array=None, params=Params(MBH=1e6*MSUN, GEOMETRY="LINEAR")):
        self.params = params
        if grid_array is not None:
            self.r_cell = np.array(grid_array)
            nt = len(self.r_cell)-2
            ri, rf = self.r_cell[1], self.r_cell[-2]

        ## TODO: figure out wtf to do with the different drs and other crap
        self.n = nt
        ## define cell centers and cell inter_faces and stuff
        if params.GEOMETRY.__eq__("LOGARITHMIC"):
            drlog = np.log10(rf/ri)/(nt-1)
            logi, logf = np.log10(ri), np.log10(rf)
            self.r_cell = np.logspace(logi - drlog, logf + drlog, nt + 2)
            self.r_face = np.logspace(logi - 3*drlog/2, logf + 3*drlog/2, nt + 3)
            self.dr = self.r_face[1:] - self.r_face[:-1]
            self.ddr = self.r_face[1:-1]*np.log(10)*drlog  ## specifically for derivatives

        elif params.GEOMETRY.__eq__("LINEAR"):
            print("pooopy")
            dr = (rf - ri) / (nt - 1)
            self.r_cell = np.linspace(ri - dr, rf + dr, nt + 2)
            self.r_face = np.linspace(ri - 3*dr/2, rf + 3*dr/2, nt + 3)
            self.dr = dr
            self.ddr = dr

        self.omgko = np.sqrt(CONST_G*params.MBH/self.r_cell**3)
        self.vk2 = CONST_G*params.MBH/self.r_cell
        self.cell_vol = np.pi*(self.r_face[1:]**2 - self.r_face[:-1]**-2)
        self.r_face = self.r_face[1:-1]
        self.face_area = np.pi*self.r_face

    def save(self):
        with open(os.path.join(os.getcwd(), self.params.SIM_NAME+"/grid_data.dat")) as f:
            f.write(arr_to_string(self.r_cell))
            f.write(arr_to_string(self.r_face))
    def cell_zeros(self):
        return np.zeros(self.n+2)
    def face_zeros(self):
        return np.zeros(self.n+1)





class ShastaVariable:
    def __init__(self, grid, data, vf, D):
        self.grid = grid  ## should contain only
        self.data = data  ## contains only interior grid points
        self.vf = vf ## at the interior inter_faces
        self.D = D

class FullVariable:
    def __init__(self, params):
        self.eos = load_table(params.EOS_TABLE)
        self.params=params
        self.grid = Grid(params.R0, params.RF, params.NR, params=params)
        self.be_crit = params.BE_CRIT
        self.dbe = params.DBE
        self.fbk = params.FBK
        self.fbr0 = params.FBR0
        self.alpha = params.ALPHA
        self.fsh = params.FSH
        self.fwind = params.FWIND
        self.ft = params.FT

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

        self.be = -1 + 2*(self.U + self.P)/self.rho/self.grid.vk2
        self.H = self.sigma/2/self.rho
        self.nu = self.alpha*self.H**2*self.grid.omgko

        self.sigv = sig((self.be-self.be_crit)/self.dbe)
        ## calculates velocities at inter_faces
        lc_sigma = 2 * np.pi * self.sigma * self.grid.r_cell
        g = np.sqrt(self.grid.r_cell) / (self.nu+1e-20)
        d = 3*self.nu
        dr = self.grid.r_cell[1] - self.grid.r_cell[0]

        lc_sigma_tild = np.interp(self.grid.r_face, self.grid.r_cell, lc_sigma)
        g_tild = np.interp(self.grid.r_face, self.grid.r_cell, g)
        d_tild = np.interp(self.grid.r_face, self.grid.r_cell, d)

        self.vr = 0
        self.vr = -d_tild * g_tild / lc_sigma_tild / (lc_sigma[1:] / g[1:] - lc_sigma[:-1] / g[:-1])/self.grid.ddr

        ## calculate source terms for density
        sigma_wl = self.sigma*self.grid.omgko*self.sigv ## wind loss

        sigma_fb= 1/np.pi/gamma(self.fbk/2+1)/self.fbr0**2   ## fall back
        sigma_fb *= (t + self.params.TFB)**(-5/3)*(self.grid.r_cell/self.fbr0)**self.fbk
        sigma_fb *= np.exp(-(self.grid.r_cell/self.fbr0)**2)
        sigma_fb *= self.params.TFB**(5/3)*self.params.MDOT

        self.sigma_dot = sigma_fb - sigma_wl

        ## calculate source terms for entropy
        kappa = kappa_interpolator(self.rho, self.T)
        qrad = 4*RADA*self.T**4*c/(1+kappa*self.sigma)
        qwind = self.fwind*self.grid.omgko*self.sigv*self.sigma*self.grid.vk2
        qvis = 2.25*self.nu*self.grid.omgko**2*self.sigma
        qfb = self.fsh*0.5*sigma_fb*self.grid.vk2

        self.ts_dot = (qvis+qfb-qrad-qwind)/self.T

        self.ts_dt = np.abs(self.ts/self.ts_dot)
        self.sigma_dt = np.abs(self.sigma_dot/self.sigma_dot)

    def sigma_var(self):
        return ShastaVariable(self.grid, self.sigma, self.vr, self.sigma_dot)

    def ts_var(self):
        return ShastaVariable(self.grid, self.ts, self.vr, self.ts_dot)

def shasta_step(var, vf, dt):

    ## add different interpolation options
    var_face = np.interp(var.grid.r_face, var.grid.r_cell, var.data)
    vol_face = np.interp(var.grid.r_face, var.grid.r_cell, var.grid.cell_vol)  ## volume at face

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


## input as sur_face density, entropy per mass
class Simulation:
    def __init__(self, sigma0, entropy0, params=Params()):


        ## sets up simulation directory
        if not os.path.isdir(os.path.join(os.getcwd(), params.SIM_NAME)):
            os.mkdir(os.path.join(os.getcwd(), params.SIM_NAME))

        self.sim_dir = os.path.join(os.getcwd(), params.SIM_NAME)
        self.params=params
        self.tf = params.TF
        self.file_int = params.FILE_INT
        self.sim_name = params.SIM_NAME
        self.sfdt = params.SDT
        self.cfl = params.CFLDT
        self.restart=params.RESTART
        self.alpha=params.ALPHA
        self.sigmafd = os.path.join(os.getcwd(), params.SIM_NAME + "/sigma.000.dat")
        self.entropyfd = os.path.join(os.getcwd(), params.SIM_NAME + "/entropy.000.dat")
        self.ft = params.FT
        self.dt = np.inf


        if self.restart:
            sigma0 = np.load(os.path.join(self.sim_dir, "sigma.npy"), mmap_mode="r")[-1]
            entropy0 = np.load(os.path.join(self.sim_dir, "entropy.npy"), mmap_mode="r")[-1]
            r_cell = np.load(os.path.join(self.sim_dir, "rocell.npy"), mmap_mode="r")
            self.grid = Grid(grid_array=r_cell)
            tarr = np.load(os.path.join(self.sim_dir, "tsave.npy"), mmap_mode="r")
            self.t0, self.t = tarr[0], tarr[-1]
            self.file_start = np.arange(self.t, self.tf, self.file_int)
            self.file_start[0] = 3*self.tf
        else:
            ## deletes old files
            ## idk why i am deleting this stuff
            #old_npy = glob.glob(os.path.join(self.sim_dir, "*.npy"))
            #for f in old_npy:
            #    os.remove(f)
            old_dat = glob.glob(os.path.join(self.sim_dir, "*.dat"))
            for f in old_dat:
                os.remove(f)

            self.grid = Grid(params.R0, params.RF, params.NR, params=params)
            self.t0 = params.T0
            self.t = params.T0


            sigmafile = open(self.sigmafd, "w")
            sigmafile.write(arr_to_string(self.grid.r_cell) + arr_to_string(sigma0, t=self.t))
            sigmafile.close()

            sfile = open(self.entropyfd, "w")
            sfile.write(arr_to_string(self.grid.r_cell) + arr_to_string(entropy0, t=self.t))
            sfile.close()

            self.file_start = np.arange(self.t0, self.tf, self.file_int)
            self.file_start[0] = 3 * self.tf


        ## class variables
        self.dtsave = params.TS
        self.tsave = np.append(np.arange(self.t, self.tf, params.TS), [self.tf])
        self.tsave[0] = self.tf*3
        self.var0 = FullVariable(params)
        self.var0.update_variables(sigma0, entropy0*sigma0, self.t0)

        self.vhalf = FullVariable(params)

    def take_step(self, mult=1.0):
        dr = 0
        if self.params.GEOMETRY.__eq__("LOGARITHMIC"):
            dr_left = self.grid.r_face - self.grid.r_cell[:-1]
            dr_right = self.grid.r_cell[1:] - self.grid.r_face
            dr = np.where(self.var0.vr < 0, dr_left, dr_right)
        elif self.params.GEOMETRY.__eq__("LINEAR"):
            dr = self.grid.dr / 2

        ts_arr = (self.sfdt*np.abs(self.var0.ts/(self.var0.ts_dot+1e-50)))[1:-1]
        ts_loc = np.argmin(ts_arr)
        ts_dt = ts_arr[ts_loc+1]

        sigma_arr = (self.sfdt*np.abs(self.var0.sigma/(self.var0.sigma_dot+1e-50)))[1:-1]
        sigma_loc = np.argmin(sigma_arr)
        sigma_dt = sigma_arr[sigma_loc+1]

        cfl_arr = self.cfl*np.abs(dr/(self.var0.vr+1e-50))
        cfl_loc = np.argmin(cfl_arr)
        cfl_dt = cfl_arr[cfl_loc]

        sim_dt = mult*np.minimum(np.minimum(ts_dt, sigma_dt), cfl_dt)
        self.dt = np.minimum(self.dt, sim_dt)

        arr = np.array([ts_dt, sigma_dt, cfl_dt])
        st = ["s", "sigma", "cfl"][np.argmin(arr)]
        loc = [ts_loc, sigma_loc, cfl_loc][np.argmin(arr)]
        loc = self.grid.r_cell[loc]/self.grid.r_cell[0]


        ## start adapative time stepping

        ## take half step
        n=0
        ts_full = []
        sigma_full = []
        max_ts_err = np.inf
        max_sigma_err = np.inf

        vr = np.copy(self.var0.vr)

        while True:
            sigma = self.var0.sigma_var()
            ts = self.var0.ts_var()

            ts_full_o1 = shasta_step(ts, vr, self.dt)
            sigma_full_o1 = shasta_step(sigma, vr, self.dt)

            ts_half = shasta_step(ts, vr, self.dt/2)
            sigma_half = shasta_step(sigma, vr, self.dt/2)

            self.vhalf.update_variables(sigma_half, ts_half, t=self.t+self.dt/2)
            vg = np.copy(self.vhalf.vr)
            sigma.D = self.vhalf.sigma_dot
            ts.D = self.vhalf.ts_dot

            ts_full = shasta_step(ts, vg, self.dt)
            sigma_full = shasta_step(sigma, vg, self.dt)

            ts_err = np.abs(ts_full_o1 - ts_full)/ts_full
            sigma_err = np.abs(sigma_full_o1 - sigma_full)/sigma_full

            max_ts_err = np.max(ts_err)
            max_sigma_err = np.max(sigma_err)

            if max_ts_err > self.params.TOL or max_sigma_err > self.params.TOL:
                self.dt *= 0.7
                n += 1
            elif n > self.params.MAXIT:
                print(f"Could not converge to desired tolerances in {self.params.MAXIT} iterations")
                print(f"sigma_max {max_sigma_err:2.2e}; entropy_max {max_ts_err:2.2e}; dt {(self.dt/(self.tf-self.t0)):2.2e}")
                break
            else:
                self.dt /= 0.7
                break

        self.t += self.dt
        self.var0.update_variables(sigma_full, ts_full, t=self.t)
        return self.dt, st, loc, max_ts_err, max_sigma_err

    def evolve(self, m=None):
        if m is None:
            m = self.ft
        sigmafile = open(self.sigmafd, "a")
        sfile = open(self.entropyfd, "a")
        #output_file = open(self.output_dir, "a")
        #compl = 0
        while True:
            dt, st, loc, sig_err, ts_err = self.take_step(m)
            cond = self.tsave - self.t <= 0
            condf = self.file_start - self.t <= 0

            if np.any(condf):
                sigmafile.close()
                sfile.close()
                n = np.argmin(self.file_start)
                self.sigmafd = os.path.join(os.getcwd(), self.sim_name + f"/sigma.{n:03d}.dat")
                self.entropyfd = os.path.join(os.getcwd(), self.sim_name + f"/entropy.{n:03d}.dat")

                sigmafile = open(self.sigmafd, "w")
                sigmafile.write(arr_to_string(self.grid.r_cell)) ## write grid locations
                sigmafile.close()

                sfile = open(self.entropyfd, "w")
                sfile.write(arr_to_string(self.grid.r_cell))
                sfile.close()

                sigmafile = open(self.sigmafd, "a")
                sfile = open(self.entropyfd, "a")
                self.file_start[n] = 3*self.tf


            if np.any(cond):
                #output_file.write("saving\n")
                ## save or die

                sigmafile.write(arr_to_string(self.var0.sigma, t=self.t))
                sfile.write(arr_to_string(self.var0.s, t=self.t))

                self.tsave = np.where(cond, self.tf*3, self.tsave)
            print(f"\rpct: {(self.t/self.tf*100):2.2f}%\tdt={dt/(self.tf-self.t0):2.2e}\t{st}\tloc={loc:2.2f}"
                             +f"\tsig {sig_err:2.2e}\tts {ts_err:2.2e}\t\t\t\t\t", end="")


            #if np.abs(compl-self.t/self.tf) > 1e-5:
            #    output_file.write(f"{self.t/self.tf:1.6f}\n")
            #    compl = self.t/self.tf

            if self.t > self.tf or (np.isnan(dt) or np.any(self.var0.ts<0)): break

        sigmafile.close()
        sfile.close()

        print("Saving converting into numpy arrays\t\t\t\t\t\t\t\t\t\t\t")
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
            np.save(os.path.join(self.sim_dir, "rocell.npy"), np.loadtxt(entropy_files[0], max_rows=1))

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

        #output_file.close()






