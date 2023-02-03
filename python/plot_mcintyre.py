import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.animation as animation
import glob, re

plt.style.use('seaborn-poster')   


def import_data(filename):
    X,eBreakdownProbability,hBreakdownProbability,totalBreakdownProbability = np.loadtxt(filename, skiprows=1, delimiter=',', unpack=True)
    return X, eBreakdownProbability, hBreakdownProbability, totalBreakdownProbability

def animation_sol(DIR):
    fig, ax = plt.subplots(1, 1, sharex=True)
    list_files = glob.glob(DIR + 'MCI*.csv')
    list_voltage = []
    # Extract the voltage from the file name by using regular expression on float numbers
    for file in list_files:
        voltage = re.findall(r"[-+]?\d*\.\d+|\d+", file)
        list_voltage.append(float(voltage[0]))
    list_voltage = np.array(list_voltage)
    # Sort the list of files according to the voltage
    list_files = [x for _, x in sorted(zip(list_voltage, list_files))]
    list_voltage = np.sort(list_voltage)
    print(list_voltage)

    listX, listeBreakdownProbability, listhBreakdownProbability, listtotalBreakdownProbability = [], [], [], []
    for file in list_files[::]:
        X, eBreakdownProbability, hBreakdownProbability, totalBreakdownProbability = import_data(file)
        listX.append(X)
        listeBreakdownProbability.append(eBreakdownProbability)
        listhBreakdownProbability.append(hBreakdownProbability)
        listtotalBreakdownProbability.append(totalBreakdownProbability)
    line1, = ax.plot(listX[0], listeBreakdownProbability[0], 'r-')
    line2, = ax.plot(listX[0], listhBreakdownProbability[0], 'b-')
    # line3, = ax.plot(listX[0], listtotalBreakdownProbability[0], 'g-')
    ax.set_ylabel('Breakdown Probability')
    ax.set_xlabel('Voltage')
    ax.set_title('Breakdown Probability')
    ax.set_xlim(0, np.max(listX[0]))
    ax.set_ylim([0, 1.1])
    def animate(i):
        fig.suptitle('Voltage = ' + str(list_voltage[i]) + ' V')
        line1.set_data(listX[i], listeBreakdownProbability[i])
        line2.set_data(listX[i], listhBreakdownProbability[i])
        # line3.set_data(listX[i], listtotalBreakdownProbability[i])
        return line1, line2

    ani = animation.FuncAnimation(fig, animate, frames=len(list_files), interval=10, blit=False, repeat=False)
    plt.show()

if __name__ == '__main__':
    DIR = sys.argv[1]
    animation_sol(DIR)
    