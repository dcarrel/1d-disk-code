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
    num = 2*HP/c/c*np.einsum("j,k->jk", 2*np.pi*grid.ro_cell, nu**3)*grid.dr
    integrand = np.einsum("jk,ijk->ijk", num, denom**-1)
    integrand = np.where(exp_arg > 1000, 0, integrand)
    integrand = np.where(exp_arg < 1/1000, 2*KB/c/c*np.einsum("k,ij->ijk", nu**2, T*grid.ro_cell)*2*np.pi*grid.dr, integrand) ## rayleigh jeans
    return np.sum(integrand, axis=1)/(4*np.pi*(10*PC)**2)


class Grid:
    def __init__(self, ri=None, rf=None, nt=None, geometry="CYLINDRICAL", grid_array=None, params=Params(MBH=1e6*MSUN)):
        if grid_array is not None:
            self.ro_cell = np.array(grid_array)
        else:
            dr = (rf - ri) / (nt - 1)
            self.ro_cell = np.linspace(ri - dr, rf + dr, nt + 2)

        nt = len(self.ro_cell)-2

        ri = self.ro_cell[1]
        rf = self.ro_cell[-2]
        dr = self.ro_cell[1] - self.ro_cell[0]
        self.dr = dr
        self.omgko = np.sqrt(CONST_G*params.MBH/self.ro_cell**3)
        self.vk2 = CONST_G*params.MBH/self.ro_cell
        self.ro_face = np.linspace(ri - dr / 2, rf + dr / 2, nt + 1)

        if geometry.__eq__("CARTESIAN"):
            def tmp_vcell(r_int):
                vol = np.zeros(self.ro_cell.shape)
                vol[1:-1] = r_int[1:] - r_int[:-1]
                vol[0] = vol[1]
                vol[-1] = vol[-2]
                return vol
            self.vol_cell = tmp_vcell
            self.area_face = lambda ro_int, rn_int: np.ones(nt + 3)
            self.alpha = 1
        elif geometry.__eq__("CYLINDRICAL"):
            def tmp_vcell(r_int):
                vol = np.zeros(self.ro_cell.shape)
                vol[1:-1] = np.pi * (r_int[1:] ** 2 - r_int[:-1] ** 2)
                vol[0] = vol[1]
                vol[-1] = vol[-2]
                return vol
            self.vol_cell = tmp_vcell
            self.area_face = lambda ro_int, rn_int: np.pi * (ro_int + rn_int)
            self.alpha = 2
        elif geometry.__eq__("SPHERICAL"):
            def tmp_vcell(r_int):
                vol = np.zeros(self.ro_cell.shape)
                vol[1:-1] = (4 / 3) * np.pi * (r_int[1:] ** 3 - r_int[:-1] ** 3)
                vol[0] = vol[1]
                vol[-1] = vol[-2]
                return vol
            self.vol_cell = tmp_vcell
            self.area_face = lambda ro_int, rn_int: (4 / 3) * np.pi * (ro_int ** 2 + ro_int * rn_int + rn_int ** 2)
            self.alpha = 3

class ShastaVariable:
    def __init__(self, grid, data, vf, D):
        self.grid = grid  ## should contain only
        self.data = data  ## contains only interior grid points
        self.vf = vf ## at the interior interfaces
        self.D = D

class FullVariable:
    def __init__(self, params):
        self.eos = load_table(params.EOS_TABLE)
        self.grid = Grid(params.R0, params.RF, params.NR)
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
        ## calculates velocities at interfaces
        lc_sigma = 2 * np.pi * self.sigma * self.grid.ro_cell
        g = np.sqrt(self.grid.ro_cell) / (self.nu+1e-20)
        d = 3*self.nu
        dr = self.grid.ro_cell[1] - self.grid.ro_cell[0]

        lc_sigma_tild = np.interp(self.grid.ro_face, self.grid.ro_cell, lc_sigma)
        g_tild = np.interp(self.grid.ro_face, self.grid.ro_cell, g)
        d_tild = np.interp(self.grid.ro_face, self.grid.ro_cell, d)

        self.vr = -d_tild*g_tild/lc_sigma_tild/dr*(lc_sigma[1:]/g[1:] - lc_sigma[:-1]/g[:-1])

        ## calculate source terms for density
        sigma_wl = self.sigma*self.grid.omgko*self.sigv ## wind loss

        sigma_fb= 1/np.pi/gamma(self.fbk/2+1)/self.fbr0**2   ## fall back
        sigma_fb *= (t + MONTH)**(-5/3)*(self.grid.ro_cell/self.fbr0)**self.fbk
        sigma_fb *= np.exp(-(self.grid.ro_cell/self.fbr0)**2)
        sigma_fb *= MSUN*MONTH**(2/3)

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

def shasta_step(var, vg, dt):

    rn_face = var.grid.ro_face + vg*dt
    rn_cell = np.copy(var.grid.ro_cell)
    rn_cell[1:-1] = 0.5*(rn_face[1:] + rn_face[:-1])
    rn_cell[0] += rn_cell[1] - var.grid.ro_cell[1]
    rn_cell[-1] += rn_cell[-2] - var.grid.ro_cell[-2]
    dv_face = var.vf - vg
    data_face = 0.5*(var.data[1:] + var.data[:-1])

    ## convective update
    vol0 = var.grid.vol_cell(var.grid.ro_face)
    area = var.grid.area_face(var.grid.ro_face, rn_face)
    var_ast = np.zeros(var.grid.ro_cell.shape)
    var_ast[1:-1] = (var.data[1:-1] - dt*data_face[1:]*area[1:]*dv_face[1:]/vol0[1:-1]
               + dt*data_face[:-1]*area[:-1]*dv_face[:-1]/vol0[1:-1])
    var_ast[0] = var_ast[1]
    var_ast[-1] = var_ast[-2]

    ## transport update
    var_T = np.zeros(var.grid.ro_cell.shape)
    var_T[1:-1] = var_ast[1:-1] + dt*var.D[1:-1]
    var_T[0] = var_T[1]
    var_T[-1] = var_T[-2]

    ## diffusive update
    voln = var.grid.vol_cell(rn_face)
    volh = 0.5*(voln[1:] + voln[:-1])
    #volh[0] = voln[1]
    #volh[-1] = voln[-2]

    eps = 0.5*area*dv_face*dt*(1/voln[:-1] + 1/voln[1:])
    nu = 1/6+1/3*eps**2
    mu = 1*(1/6-1/6*eps**2)

    var_tild = np.zeros(var.grid.ro_cell.shape)

    var_tild[1:-1] = (vol0[1:-1]*var_T[1:-1] + nu[1:]*volh[1:]*(var.data[2:] - var.data[1:-1])
                - nu[:-1]*volh[:-1]*(var.data[1:-1] - var.data[:-2]))

    var_tild[1:-1] /= voln[1:-1]
    var_tild[0] = var_tild[1]; var_tild[-1] = var_tild[-2]

    ## diffusive correction
    shalf = np.sign(var_tild[1:] - var_tild[:-1])
    fadd = mu*volh*(var_T[1:]-var_T[:-1])

    fluxc = np.zeros(var.grid.ro_face.shape)
    fluxc[1:-1] = shalf[1:-1]*np.maximum(0,
                             np.minimum(np.abs(fadd[1:-1]),
                                        np.minimum(shalf[1:-1]*voln[2:-1]*(var_tild[3:]-var_tild[2:-1]),
                                                   shalf[1:-1]*voln[1:-2]*(var_tild[1:-2]-var_tild[:-3])))
                             )

    fluxc[0] = shalf[0]*np.maximum(0, np.minimum(np.abs(fadd[0]), shalf[0]*voln[1]*(var_tild[2]-var_tild[1])))
    fluxc[-1] = shalf[-1]*np.maximum(0, np.minimum(np.abs(fadd[-1]), shalf[-1]*voln[-2]*(var_tild[-2]-var_tild[-3])))

    var_ret = np.zeros(var.grid.ro_cell.shape)
    var_ret[1:-1] = var_tild[1:-1] - voln[1:-1]**-1*(fluxc[1:] - fluxc[:-1])
    var_ret[0] = var_ret[1]
    var_ret[-1] = var_ret[-2]

    var_ret = np.interp(var.grid.ro_cell, rn_cell, var_ret)
    var_ret[0] = var_ret[1]
    var_ret[-1] = var_ret[-2]

    return var_ret


## input as surface density, entropy per mass
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
            ro_cell = np.load(os.path.join(self.sim_dir, "rocell.npy"), mmap_mode="r")
            self.grid = Grid(grid_array=ro_cell)
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

            self.grid = Grid(params.R0, params.RF, params.NR)
            self.t0 = params.T0
            self.t = params.T0


            sigmafile = open(self.sigmafd, "w")
            sigmafile.write(arr_to_string(self.grid.ro_cell) + arr_to_string(sigma0, t=self.t))
            sigmafile.close()

            sfile = open(self.entropyfd, "w")
            sfile.write(arr_to_string(self.grid.ro_cell) + arr_to_string(entropy0, t=self.t))
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
        dr = self.grid.dr/2
        ts_arr = self.sfdt*np.abs(self.var0.ts/(self.var0.ts_dot+1e-50))
        ts_loc = np.argmin(ts_arr)
        ts_dt = ts_arr[ts_loc]

        sigma_arr = self.sfdt*np.abs(self.var0.sigma/(self.var0.sigma_dot+1e-50))
        sigma_loc = np.argmin(sigma_arr)
        sigma_dt = sigma_arr[sigma_loc]

        cfl_arr = self.cfl*np.abs(dr/(self.var0.vr+1e-50))
        cfl_loc = np.argmin(cfl_arr)
        cfl_dt = cfl_arr[cfl_loc]

        sim_dt = mult*np.minimum(np.minimum(ts_dt, sigma_dt), cfl_dt)
        self.dt = np.minimum(self.dt, sim_dt)

        arr = np.array([ts_dt, sigma_dt, cfl_dt])
        st = ["s", "sigma", "cfl"][np.argmin(arr)]
        loc = [ts_loc, sigma_loc, cfl_loc][np.argmin(arr)]
        loc = self.grid.ro_cell[loc]/self.grid.ro_cell[0]


        ## start adapative time stepping

        ## take half step
        sigma = self.var0.sigma_var()
        ts = self.var0.ts_var()
        vr = np.copy(self.var0.vr)
        n=0
        ts_full = []
        sigma_full = []
        max_ts_err = np.inf
        max_sigma_err = np.inf

        while True:
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
                sigmafile.write(arr_to_string(self.grid.ro_cell)) ## write grid locations
                sigmafile.close()

                sfile = open(self.entropyfd, "w")
                sfile.write(arr_to_string(self.grid.ro_cell))
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






