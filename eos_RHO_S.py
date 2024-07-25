from consts import *
from setup import *
import numpy as np
from scipy.optimize import fsolve
import pandas as pd
from scipy.interpolate import RectBivariateSpline

## uses specific entropy
def entropy_guess(rho, T):
    rad_part = 4*RADA*T**3/3
    gas_part = 0
    for i, frac in enumerate(Xi_over_mi):
        gas_part += rho*KB*frac*(2.5 + np.log(frac**-1 / rho * (2*np.pi*mi[i]*KB*T/HP**2)**1.5))
    return rad_part + gas_part

def entropy_difference(T, rho, entropy):
    return (entropy - entropy_guess(rho, T))/entropy

def temperature_guess(T0, rho, entropy, f=0.005):
    diff0 = entropy_difference(T0, rho, entropy)
    A0 = np.where(diff0 == 0, np.zeros(T0.shape), np.sign(diff0))

    T0 *= 1 + f * A0  ## updates guess
    An = np.zeros(T0.shape)
    while True:  ## iterates
        diff = entropy_difference(T0, rho, entropy)
        An = np.where(diff == 0, An, np.sign(diff))
        An = np.where(np.abs(An - A0) > 0, 0, An)
        T0 *= 1 + f * An
        if not np.any(An):
            break
    return T0

def solve_temp(T0, rho, entropy, xtol=1e-10):
    temp = fsolve(entropy_difference, T0, args=(rho, entropy), xtol=xtol)
    return temp

##makes table using specific entropy!! (this is easier)
def make_table(rho, entropy, table_name="EOS_TABLE.dat", xtol=1e-6):
    m, n = len(rho), len(entropy)

    table = np.zeros((n, m))

    for i in range(n):
        T0 = (3*entropy[i]*rho/4/RADA)**(1/3) ## guess using radiation temperature
        T0 = temperature_guess(T0, rho, entropy[i]*rho*np.ones(m))
        table[i] = solve_temp(T0, rho, entropy[i]*rho, xtol=xtol)

    eos_df = pd.DataFrame(np.log10(table), columns=np.log10(rho), index=np.log10(entropy))
    eos_df.to_csv(table_name)
    return table

def load_table(table_name):
    df = pd.read_csv(table_name, index_col=0)
    log_entropy = np.array(df.index.values, dtype=np.float64)
    log_rho = np.array(df.columns, dtype=np.float64)
    log_temperature = df.to_numpy()
    interpolator = RectBivariateSpline(log_rho, log_entropy, log_temperature.T)

    ## stuff outside the domain
    rho_p_slopes = log_temperature[:, -1] - log_temperature[:, -2]
    rho_p_slopes /= log_rho[-1] - log_rho[-2]

    rho_m_slopes = log_temperature[:, 1] - log_temperature[:, 0]
    rho_m_slopes /= log_rho[1] - log_rho[0]

    entropy_p_slopes = log_temperature[-1] - log_temperature[-2]
    entropy_p_slopes /= log_entropy[-1] - log_entropy[-2]
    entropy_m_slopes = log_temperature[1] - log_temperature[0]
    entropy_m_slopes /= log_entropy[1] - log_entropy[0]

    rho_p = log_rho[-1]
    rho_m = log_rho[0]

    entropy_p = log_entropy[-1]
    entropy_m = log_entropy[0]

    def EOS(rho, entropy):
        entropy = entropy/rho ## converts to specific entropy, what the table reads
        ulog_rho = np.log10(rho)
        ulog_entropy = np.log10(entropy)
        values = interpolator(ulog_rho, ulog_entropy, grid=False)

        ## interpolation outside of boundaries
        values += np.maximum(ulog_rho - rho_p, 0) * np.interp(ulog_entropy, log_entropy, rho_p_slopes)
        values += np.minimum(ulog_rho - rho_m, 0) * np.interp(ulog_entropy, log_entropy, rho_m_slopes)
        values += np.maximum(ulog_entropy - entropy_p, 0) * np.interp(ulog_rho, log_rho, entropy_p_slopes)
        values += np.minimum(ulog_entropy - entropy_m, 0) * np.interp(ulog_rho, log_rho, entropy_m_slopes)

        return np.power(10, values)

    return EOS

class eos:
    def __init__(self, table_name="EOS_TABLE.dat"):
        self.table = load_table(table_name)

    def TPUchi(self, rho, s):
        T = self.table(rho, s)

        rad_P = RADA*T**4/3
        gas_P = rho*KB*T/mu
        P = rad_P + gas_P

        rad_U = RADA*T**4
        gas_U = 1.5*rho*KB*T/mu
        U = rad_U + gas_U

        chi = 2*np.sqrt(rho*P)

        return T,P,U,chi


