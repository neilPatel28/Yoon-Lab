import numpy as np
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt

class InitialGuessCreator:
    def __init__(self, filedata,energy=None,er=None):
        if filedata is not None:
            self.data = filedata
            #load data w n k
            self.wavelengths = self.data[:, 0]
            self.n_vals = self.data[:, 1]
            self.k_vals = self.data[:, 2]

            mask = (self.wavelengths >= 0.4) & (self.wavelengths <= 0.9)
            self.wavelengths = self.wavelengths[mask]
            self.n_vals = self.n_vals[mask]
            self.k_vals = self.k_vals[mask]

            # convert to energy and create er
            self.energy, self.er_vals = self.make_variables(self.wavelengths, self.n_vals, self.k_vals)

        if energy is not None:
            self.energy = energy
            self.er_vals = er



    @staticmethod
    def make_variables(w, n, k):
            # change input um to eV and create er
            h = 6.626e-34
            c = 299792458
            w_m = w * 1e-6
            J = (h * c) / w_m
            eV = J * 6.241509e18

            N = n + 1j * k
            er = N ** 2
            return eV, er


    @staticmethod
    def find_info(x,y,wantGraph,wantPrinted):
        # find ei
        peaks, _ = find_peaks(y)

        # FWHM
        results_half = peak_widths(y, peaks, rel_height=0.5)

        # Convert left_ips and right_ips to x-values
        left_x = x[(results_half[2]).astype(int)]
        right_x = x[(results_half[3]).astype(int)]
        fwhm = np.abs(right_x - left_x)

        # Peak positions
        peak_positions = x[peaks]

        if wantPrinted:
            # Print results
            for i, pos in enumerate(peak_positions):
                print(f"Peak {i + 1}:")
                if fwhm[i] < 0.015:
                    print(f"filtered")
                print(f"  Position (x): {pos}")
                print(f"  FWHM: {fwhm[i]}")

        if wantGraph:
            # plot
            plt.plot(x, y)
            plt.plot(x[peaks], y[peaks], "x")  # mark peaks
            plt.hlines(results_half[1], left_x, right_x, color='C1')  # mark FWHM
            plt.show()

        return peak_positions, fwhm

    def create_guess(self,a=False,b=False):
        peak_positions, fwhm = self.find_info(self.energy, self.er_vals.imag,a,b)
        n = len(peak_positions)
        initialGuess = np.column_stack((peak_positions, np.full(n, 1 / n), fwhm, np.full(n, 2)))
        filtered_guess = initialGuess[initialGuess[:, 2] >= 0.02]
        p0 = [10.339721, 17.299] + filtered_guess.flatten().tolist()

        return p0





