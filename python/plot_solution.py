import numpy as np
import matplotlib.pyplot as plt
import sys


def import_data(filename):
    X, U, eDensity, hDensity = np.loadtxt(filename, skiprows=1, delimiter=',', unpack=True)
    return X, U, eDensity, hDensity

if __name__ == "__main__":
    fig, ax = plt.subplots(2, 1, sharex=True)
    for filename in sys.argv[1:]:
        X, U, eDensity, hDensity = import_data(filename)
        ax[0].plot(X, U, label=filename)
        ax[1].plot(X, eDensity, label=filename)
        ax[1].plot(X, hDensity, label=filename)
    ax[0].set_ylabel('Potential (V)')
    ax[1].set_ylabel('Carrier Density (cm-3)')
    ax[1].set_xlabel('x (nm)')
    ax[0].legend()
    ax[1].legend()
    plt.show()




