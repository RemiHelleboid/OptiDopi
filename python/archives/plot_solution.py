import numpy as np
import matplotlib.pyplot as plt
import sys
import re

def import_data(filename):
    X, U, eDensity, hDensity, EF = np.loadtxt(filename, skiprows=1, delimiter=',', unpack=True)
    return X, U, eDensity, hDensity, EF

def extract_depletion_w(X, EF):
    depletion_w = 0
    depleted = EF > 10000
    X_Depleted = X[depleted]
    dx = X_Depleted[1] - X_Depleted[0]
    W = np.sum(depleted) * dx
    return W

if __name__ == "__main__":
    fig, ax = plt.subplots(3, 1, sharex=True)
    list_of_files = sys.argv[1:]
    list_voltage = []
    for filename in list_of_files:
        voltage = re.findall(r"[-+]?\d*\.\d+|\d+", filename)[-1]
        list_voltage.append(float(voltage))
    arg_sorted = np.argsort(list_voltage)
    list_of_files = np.array(list_of_files)
    list_of_files = list_of_files[arg_sorted]
    list_voltage = np.array(list_voltage)
    list_voltage = list_voltage[arg_sorted]
    list_DW = []
    print(list_voltage)
    for idx, filename in enumerate(list_of_files[0::200]):
        voltage = list_voltage[idx*200]
        X, U, eDensity, hDensity, EF = import_data(filename)
        ax[0].plot(X, U, label=voltage)
        ax[1].plot(X, EF, label=voltage)
        ax[2].plot(X, eDensity, label=voltage)
        ax[2].plot(X, hDensity, label=voltage)
    ax[0].set_ylabel('Potential (V)')
    ax[1].set_ylabel('Electric Field (V/cm)')
    ax[2].set_ylabel('Carrier Density (cm-3)')
    ax[1].set_xlabel('x (um)')
    ax[0].legend()
    ax[1].legend()
    plt.show()
    
    for filename in list_of_files:
        X, U, eDensity, hDensity, EF = import_data(filename)
        list_DW.append(extract_depletion_w(X, EF))
    fig, ax = plt.subplots()
    ax.plot(list_voltage, list_DW)
    ax.set_ylabel('Depletion Width (um)')
    ax.set_xlabel('Voltage (V)')
    Nd = Na = 1e18 * 1e6
    epsilon = 11.9 * 8.85e-14
    analytical_DW = np.sqrt(2 * epsilon * ((Nd + Na) / (Nd * Na)) * (list_voltage+0.6) / 1.6e-19)
    ax.plot(list_voltage, analytical_DW, label='Analytical')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    plt.show()




