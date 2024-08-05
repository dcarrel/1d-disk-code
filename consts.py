import numpy as np

c=3e10
CONST_G=6.67e-8
MSUN=2e33
RSUN=7e10
RADA = 7e-15
MP = 1.6726e-24
KB = 1.38e-16
HP = 6.626e-27
ME = 9.11e-28
DAY = 86400
PC = 3e18
MONTH = 30*DAY
YEAR = 365*DAY
HOUR = DAY/24


def arr_to_string(array, t=None):
    str = f"{array[0]:5.5e}"
    if t is not None:
        str = f"{t:5.5e} {array[0]:5.5e}"
    for item in array[1:]:
        str += " "
        str += f"{item:5.5e}"
    str += "\n"
    return str

def is_invalid(array):
    return np.any(np.isnan(array)) or np.any(np.isinf(array))

def sig(x):
    return (1+np.exp(-x))**-1

def Snu(T, nu, grid):
    exp_arg = np.einsum("ij,k->ijk", T**-1, HP*nu/KB)
    denom = np.exp(exp_arg) -1
    num = 2*HP/c/c*np.einsum("j,k->jk", 2*np.pi*grid.r_cell*grid.dr, nu**3)
    integrand = np.einsum("jk,ijk->ijk", num, denom**-1)
    ## weens's law
    integrand = np.where(exp_arg > 1000, 0, integrand)
    ## Rayleigh-Jeans tail
    integrand = np.where(exp_arg < 1/1000, 2*KB/c/c*np.einsum("k,ij->ijk", nu**2, T*grid.r_cell*2*np.pi*grid.dr), integrand)
    return np.sum(integrand, axis=1)/(4*np.pi*(10*PC)**2)
