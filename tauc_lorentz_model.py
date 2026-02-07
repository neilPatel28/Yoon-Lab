"""Adapted from the pyelli project:
https://pyelli.readthedocs.io/en/v0.21.1/_modules/elli/dispersions/tauc_lorentz.html
(Accessed Dec 2025)"""

import numpy as np
from numpy.lib.scimath import sqrt



class TaucLorentz():
    @staticmethod
    def eps1(E, Eg, Ai, Ei, Ci):
        gamma2 = sqrt(Ei**2 - Ci**2 / 2) ** 2
        alpha = sqrt(4 * Ei**2 - Ci**2)
        aL = (Eg**2 - Ei**2) * E**2 + Eg**2 * Ci**2 - Ei**2 * (Ei**2 + 3 * Eg**2)
        aA = (E**2 - Ei**2) * (Ei**2 + Eg**2) + Eg**2 * Ci**2
        zeta4 = (E**2 - gamma2) ** 2 + alpha**2 * Ci**2 / 4

        return (
            Ai*Ci*aL/2.0/np.pi/zeta4/alpha/Ei*np.log((Ei**2 + Eg**2 + alpha*Eg)/(Ei**2 + Eg**2 - alpha*Eg)) - \
            Ai*aA/np.pi/zeta4/Ei*(np.pi - np.arctan((2.0*Eg + alpha)/Ci) + np.arctan((alpha - 2.0*Eg)/Ci)) + \
            2.0*Ai*Ei*Eg/np.pi/zeta4/alpha*(E**2 - gamma2)*(np.pi + 2.0*np.arctan(2.0/alpha/Ci*(gamma2 - Eg**2))) - \
            Ai*Ei*Ci*(E**2 + Eg**2)/np.pi/zeta4/E*np.log(abs(E - Eg)/(E + Eg)) + \
            2.0*Ai*Ei*Ci*Eg/np.pi/zeta4 * \
            np.log(abs(E - Eg) * (E + Eg) / sqrt((Ei**2 - Eg**2)**2 + Eg**2 * Ci**2))
        )

    @staticmethod
    def eps2(E, Eg, Ai, Ei, Ci):
        return 1j * (Ai * Ei * Ci * (E - Eg) ** 2
              / ((E ** 2 - Ei ** 2) ** 2 + Ci ** 2 * E ** 2)
              / E
              ) * np.heaviside(E - Eg, 0)

    def Lorentz_oscillator_model(self,energy,*params):
        """# add =  e_infinity term plus term for background to account for lost resonances terms
        add = params[1]
        #shift factor to account for lost resonances terms
        scale = params[0]

        osc_params = params[2:]
        sum_term = 0
        for i in range(0, len(osc_params), 4):
            Ei = osc_params[i]
            Ai = osc_params[i+1]
            Ci = osc_params[i+2]
            Eg = osc_params[i+3]
            #tauc_lorentz model
            sum_term += self.eps1(energy, Eg, Ai, Ei, Ci) + self.eps2(energy, Eg, Ai, Ei, Ci)

        return (sum_term * scale) + add"""
        scale, add = params[0], params[1]
        osc_params = np.array(params[2:]).reshape(-1, 4)  # Each row: [Ei, Ai, Ci, Eg]

        Ei = osc_params[:, 0][:, np.newaxis]  # shape (n_osc, 1)
        Ai = osc_params[:, 1][:, np.newaxis]
        Ci = osc_params[:, 2][:, np.newaxis]
        Eg = osc_params[:, 3][:, np.newaxis]

        E = energy[np.newaxis, :]  # shape (1, n_energy)

        eps1_total = self.eps1(E, Eg, Ai, Ei, Ci)
        eps2_total = self.eps2(E, Eg, Ai, Ei, Ci)

        sum_term = np.sum(eps1_total + eps2_total, axis=0)  # sum over oscillators
        return sum_term * scale + add


