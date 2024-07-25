import numpy as np
from consts import *

class Filter:
    def __init__(self, file_location = "", photon_counter=False):
        self.photon_counter = photon_counter
        self.file_location = file_location

        int_data = np.loadtxt(self.file_location)
        self.filter_wl, self.filter_weight = int_data.T
        self.filter_wl *= 1e-8 ## given in angstroms
        self.filter_nu = (2*np.pi*c)/self.filter_wl

        self.filter_weight = self.filter_weight[::-1]
        self.filter_nu = self.filter_nu[::-1]
        self.filter_wl = self.filter_wl[::-1]

        if self.photon_counter:
            def tempf(nu, S_nu):
                interp_filter = np.interp(nu, self.filter_nu, self.filter_weight, left=0, right=0)
                num = np.trapz(interp_filter/nu**3*S_nu, nu, axis=1)
                den = np.trapz(interp_filter/nu**3, nu)
                return num/den

            self.filterf = tempf
        else:
            def tempf(nu, S_nu):
                interp_filter = np.interp(nu, self.filter_nu, self.filter_weight, left=0, right=0)
                num = np.trapz(interp_filter/nu**2*S_nu, nu, axis=1)
                den = np.trapz(interp_filter/nu**2, nu)
                return num/den

            self.filterf = tempf

    ## Snu
    def __call__(self, nu, S_nu):
        S_nu *= 4*np.pi
        if len(S_nu.shape) < 2:
            S_nu = np.array([S_nu])

        # return -2.5*np.log10(self.filterf(nu, S_nu)/(3631*1e-23))
        return -2.5 * np.log10(S_nu/(3631 * 1e-23))
