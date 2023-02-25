import numpy as np
import matplotlib.pyplot as plt
import sys
import re

def import_data(filename):
    X, U, eDensity, hDensity, EF = np.loadtxt(filename, skiprows=1, delimiter=',', unpack=True)
    return X, U, eDensity, hDensity, EF

if __name__ == "__main__":
    fig, ax = plt.subplots(3, 1, sharex=True)
    for filename in sys.argv[1:]:
        voltage = re.findall(r"[-+]?\d*\.\d+|\d+", filename)[-1]
        X, U, eDensity, hDensity, EF = import_data(filename)
        ax[0].plot(X, U, label=voltage)
        ax[1].plot(X, EF, label=voltage)
        ax[2].plot(X, eDensity, label=voltage)
        ax[2].plot(X, hDensity, label=voltage)
    ax[0].set_ylabel('Potential (V)')
    ax[1].set_ylabel('Electric Field (V/cm)')
    ax[1].set_ylabel('Carrier Density (cm-3)')
    ax[1].set_xlabel('x (um)')
    ax[0].legend()
    ax[1].legend()
    plt.show()




