import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.animation as animation
import glob, re

plt.style.use('seaborn-poster')   

def import_data(filename):
    X, U, eDensity, hDensity, ElectricField = np.loadtxt(filename, skiprows=1, delimiter=',', unpack=True)
    return X, U, eDensity, hDensity, ElectricField

def animation_sol(DIR):
    fig, ax = plt.subplots(1, 1, sharex=True)
    ax_eff = ax.twinx()
    list_files = glob.glob(DIR + '*.csv')
    list_voltage = []
    # Extract the voltage from the file name by using regular expression
    for file in list_files:
        voltage = re.findall(r"[-+]?\d*\.\d+|\d+", file)
        list_voltage.append(float(voltage[0]))
    list_voltage = np.array(list_voltage)
    # Sort the list of files according to the voltage
    list_files = [x for _, x in sorted(zip(list_voltage, list_files))]

    listX, listU, listeDensity, listhDensity, listEF = [], [], [], [], []
    for file in list_files[::10]:
        X, U, eDensity, hDensity, ElectricField = import_data(file)
        listX.append(X)
        listU.append(U)
        listeDensity.append(eDensity)
        listhDensity.append(hDensity)
        listEF.append(ElectricField)
    line1, = ax.plot(listX[0], listU[0], 'r-')
    line2, = ax_eff.plot(listX[0], listEF[0], 'b-')
    ax.set_ylabel('U')
    ax.set_ylabel('Density')
    ax.set_xlabel('X')
    ax.set_title('Solution')
    # ax[0].set_xlim([0, 1])
    ax.set_ylim([-5, 1.1*np.max(list_voltage)])
    ax_eff.set_ylim(0, 1.1*np.max(listEF[-1]))
    # ax[1].set_xlim([0, 1])
    # ax[1].set_ylim([-0.1, 1.1])
    def animate(i):
        line1.set_data(listX[i], listU[i])
        line2.set_data(listX[i], listEF[i])
        return line1, line2 

    ani = animation.FuncAnimation(fig, animate, frames=len(list_files)//10, interval=100, blit=False, repeat=False)
    plt.show()


if __name__ == "__main__":
    DIR = sys.argv[1]
    animation_sol(DIR)