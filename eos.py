import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import fsolve
import pandas as pd
from consts import *

XH   = 0.7
XHe  = 0.3
mu   = XH*MP + 4*XHe*MP
Xi = np.array([XH, XH, XHe])
mi = np.array([MP, ME, 4*MP])
Xi_over_mi = Xi/np.array([MP, MP, 4*MP])

## returns temperature, density
def rad_temp(chi, entropy):
    temp = (9 * entropy * chi ** 2 / 16 / RADA ** 2) ** (1 / 7)
    return temp

def rad_dens(chi, entropy, temp):
    dens = 3 * chi ** 2 / 4 / RADA / temp ** 4
    return dens

## returns temperature, density
def gas_temp(chi, entropy):
    mi = np.array([MP, 4 * MP, ME])
    Xi = np.array([XH, XHe, ME / MP * (XH + 8 * XHe)])
    mu = MP * (XH + 4 * XHe)

    Bi = np.log((2 * np.pi * mi * KB / HP / HP) ** 1.5 * mi / Xi)
    kappa = entropy - np.sum(Xi * KB / mi * (5 / 2 + Bi))
    kappa /= np.sum(Xi * KB / mi)

    temp = np.sqrt(0.5 * chi * np.sqrt(mu / KB) * np.exp(kappa))

    return temp

def gas_dens(chi, entropy, temp,):
    mi = np.array([MP, 4 * MP, ME])
    Xi = np.array([XH, XHe, ME / MP * (XH + 8 * XHe)])
    mu = MP * (XH + 4 * XHe)
    dens = chi / 2 * np.sqrt(mu / KB / temp)
    return dens

## Estimates the entropy for a given temperature
def entropy_difference(temp, chi, entropy, just_density=False, temp_log=False):
    if temp_log:
        temp = np.power(10, temp)
    mi = np.array([MP, 4 * MP, ME])
    Xi = np.array([XH, XHe, ME / MP * (XH + 8 * XHe)])
    mu = MP * (XH + 4 * XHe)
    Bi = (2 * np.pi * mi * KB / HP ** 2) ** 1.5 * mi / Xi

    c1 = mu * RADA * temp ** 3 / 6 / KB
    c2 = 9 * KB * chi ** 2 / mu / RADA ** 2 / temp ** 7
    rho = c1 * (np.sqrt(1 + c2) - 1)

    if just_density: return rho

    entropy_estimate = 4 * RADA * temp ** 3 / 3 / rho
    for i, m in enumerate(mi):
        entropy_estimate += Xi[i] * KB / mi[i] * (2.5 + np.log(Bi[i] * temp**1.5/ rho))

    return (entropy - entropy_estimate) / entropy

def better_temperature_guess(temperature_guess, chi, entropy, f=0.005):
    A0 = np.zeros(temperature_guess.shape)
    A0 = np.where(entropy_difference(temperature_guess, chi, entropy) > 0, 1, A0)
    A0 = np.where(entropy_difference(temperature_guess, chi, entropy) < 0, -1, A0)

    temperature_guess *= 1 + f * A0

    An = np.zeros(temperature_guess.shape)
    while True:
        An = np.where(entropy_difference(temperature_guess, chi, entropy) > 0, 1, An)
        An = np.where(entropy_difference(temperature_guess, chi, entropy) < 0, -1, An)
        An = np.where(np.abs(An - A0) > 0, 0, An)
        # print(np.sum(np.abs(An)))
        temperature_guess *= 1 + f * An
        if not np.any(An):
            break
    return temperature_guess

## full_solver solves the entropy equation everywhere, even where
## the disk is gas/radiation dominated
def full_temp(chi, entropy, xtol=1e-6, temp_guess=None):
    if temp_guess is None:
        temp_guess = rad_temp(chi, entropy)
    temp_guess = better_temperature_guess(temp_guess, chi, entropy)
    temp = fsolve(entropy_difference, temp_guess, args=(chi, entropy, False, False), xtol=xtol)
    # dens = entropy_difference(temp, chi, entropy, just_density=True)
    return temp

def make_table(chi, entropy, table_name="EOS_TABLE", xtol=1e-6):
    m, n = len(chi), len(entropy)

    temperature_table = np.zeros((n, m))

    for i in range(n):
        guess = None
        if i > 0:
            guess = temperature_table[i-1]
        temperature_table[i] = full_temp(chi, entropy[i] * np.ones(m), xtol=xtol, temp_guess=guess)
        print(f"{((i+1)/n*100):2.2f}", end="\r")

    eos_df = pd.DataFrame(np.log10(temperature_table), columns=np.log10(chi), index=np.log10(entropy))
    eos_df.to_csv(table_name)
    return temperature_table


def load_table(table_name, retvars=False):
    df = pd.read_csv(table_name, index_col=0)
    log_entropy = np.array(df.index.values, dtype=np.float64)
    log_chi = np.array(df.columns, dtype=np.float64)
    log_temperature = df.to_numpy()
    interpolator = RectBivariateSpline(log_chi, log_entropy, log_temperature.T)

    ## stuff outside the domain
    chi_p_slopes = log_temperature[:, -1] - log_temperature[:, -2]
    chi_p_slopes /= log_chi[-1] - log_chi[-2]

    chi_m_slopes = log_temperature[:, 1] - log_temperature[:, 0]
    chi_m_slopes /= log_chi[1] - log_chi[0]

    entropy_p_slopes = log_temperature[-1] - log_temperature[-2]
    entropy_p_slopes /= log_entropy[-1] - log_entropy[-2]
    entropy_m_slopes = log_temperature[1] - log_temperature[0]
    entropy_m_slopes /= log_entropy[1] - log_entropy[0]

    chi_p = log_chi[-1]
    chi_m = log_chi[0]

    entropy_p = log_entropy[-1]
    entropy_m = log_entropy[0]

    def EOS(chi, entropy):
        ulog_chi = np.log10(chi)
        ulog_entropy = np.log10(entropy)
        values = interpolator(ulog_chi, ulog_entropy, grid=False)

        ## interpolation outside of boundaries
        values += np.maximum(ulog_chi - chi_p, 0) * np.interp(ulog_entropy, log_entropy, chi_p_slopes)
        values += np.minimum(ulog_chi - chi_m, 0) * np.interp(ulog_entropy, log_entropy, chi_m_slopes)
        values += np.maximum(ulog_entropy - entropy_p, 0) * np.interp(ulog_chi, log_chi, entropy_p_slopes)
        values += np.minimum(ulog_entropy - entropy_m, 0) * np.interp(ulog_chi, log_chi, entropy_m_slopes)

        return np.power(10, values)

    if retvars:
        return (np.power(10, log_chi), np.power(10, log_entropy), EOS)

    return EOS

