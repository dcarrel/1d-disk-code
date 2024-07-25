import numpy as np
from scipy.interpolate import RectBivariateSpline

highT_opacity_file = 'kap_data/gn93_z0.02_x0.7.data'  # Hrich
lowT_opacity_file = 'kap_data/lowT_fa05_gn93_z0.02_x0.7.data'


def load_file(filename):
    data = np.loadtxt(filename, skiprows=7)[:, 1:]
    logT = np.loadtxt(filename, usecols=0, skiprows=7)
    logR = np.loadtxt(filename, skiprows=5, max_rows=1)
    return data, logT, logR

## GLOBAL SHIT BABY

lowT_opacity, lowT_logT, lowT_logR = load_file(lowT_opacity_file)
lowT_max, lowT_min = np.max(lowT_logT), np.min(lowT_logT)

highT_opacity, highT_logT, highT_logR = load_file(highT_opacity_file)
highT_max, highT_min = np.max(highT_logT), np.min(highT_logT)

highT_interpolator = RectBivariateSpline(highT_logT, highT_logR, highT_opacity)
lowT_interpolator = RectBivariateSpline(lowT_logT, lowT_logR, lowT_opacity)

max_logR = np.max(highT_logR)

def kappa_interpolator(rho, T, verbose=False):
    logT = np.log10(T)
    logR = np.log10(rho / T ** 3) + 18

    lowT = np.logical_and(logT < lowT_max, logT > lowT_min)
    highT = np.logical_and(logT < highT_max, logT > highT_min)
    bothT = np.logical_and(lowT, highT)

    # makes them DISJOINT
    old_highT = np.copy(highT)
    highT = np.logical_and(highT, np.logical_not(lowT))
    lowT = np.logical_and(lowT, np.logical_not(old_highT))

    interpolated_values = np.ones(rho.shape)

    if len(highT) > 0:
        interpolated_values[highT] = highT_interpolator(logT[highT], logR[highT], grid=False)
    if len(lowT) > 0:
        interpolated_values[lowT] = lowT_interpolator(logT[lowT], logR[lowT], grid=False)
    if len(bothT) > 0:
        interpolated_values[bothT] = 0.5 * (highT_interpolator(logT[bothT], logR[bothT], grid=False)
                                            + lowT_interpolator(logT[bothT], logR[bothT], grid=False))

    if verbose:
        if np.max(logT) > highT_max: print("temperature above interpolation")
        if np.min(logT) < lowT_min: print("temperature below interpolation")
        if np.max(logR) > np.max(lowT_logR): print("R variable above interpolation")
        if np.min(logR) < np.min(lowT_logR): print("R variable above interpolation")

    highR = logR > max_logR
    interpolated_values[highR] += logR[highR] - max_logR

    return np.power(10, interpolated_values)

