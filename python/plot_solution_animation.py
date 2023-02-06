import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.animation as animation
import glob, re
import scienceplots

plt.style.use(["science", "high-vis", "grid"])

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
    ax.set_xlabel('X')
    ax.set_title('Solution')
    ax.set_ylim([-5, 1.1*np.max(list_voltage)])
    ax_eff.set_ylim(0, 1.1*np.max(listEF[-1]))
    fig.tight_layout()
    def animate(i):
        line1.set_data(listX[i], listU[i])
        line2.set_data(listX[i], listEF[i])
        return line1, line2 

    ani = animation.FuncAnimation(fig, animate, frames=len(list_files)//10, interval=10, blit=False, repeat=False)
    ani.save('solution.mp4', fps=10, extra_args=['-vcodec', 'libx264'], dpi=300)
    plt.show()

def animation_densities(DIR):
    fig, ax = plt.subplots(1, 1, sharex=True)
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
    list_total_density = []
    for file in list_files[::10]:
        X, U, eDensity, hDensity, ElectricField = import_data(file)
        listX.append(X)
        listU.append(U)
        listeDensity.append(np.abs(eDensity))
        listhDensity.append(np.abs(hDensity))
        list_total_density.append(np.abs(eDensity) + np.abs(hDensity))
        listEF.append(ElectricField)
    line1, = ax.plot(listX[0], listeDensity[0], 'r-', label='eDensity')
    line2, = ax.plot(listX[0], listhDensity[0], 'b-', label='hDensity')
    ax.legend()
    ax.set_ylabel('Density')
    ax.set_xlabel('X')
    ax.set_title('Solution')
    ax.set_ylim(1.0e6, 10*np.max(listeDensity[-1]))
    ax.set_yscale('log')
    fig.tight_layout()
    def animate(i):
        line1.set_data(listX[i], listeDensity[i])
        line2.set_data(listX[i], listhDensity[i])
        return line1, line2 

    ani = animation.FuncAnimation(fig, animate, frames=len(list_files)//10, interval=10, blit=False, repeat=False)
    ani.save('densities.mp4', fps=10, extra_args=['-vcodec', 'libx264'], dpi=300)
    plt.show()


if __name__ == "__main__":
    DIR = sys.argv[1]
    animation_sol(DIR)
    animation_densities(DIR)