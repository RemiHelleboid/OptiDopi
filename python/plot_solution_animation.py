import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.animation as animation
import glob 


def import_data(filename):
    X, U, eDensity, hDensity = np.loadtxt(filename, skiprows=1, delimiter=',', unpack=True)
    return X, U, eDensity, hDensity

def animation_sol(DIR):
    fig, ax = plt.subplots(1, 1, sharex=True)
    list_files = glob.glob(DIR + '/solution_*.csv')
    list_files.sort()
    listX, listU, listeDensity, listhDensity = [], [], [], []
    for file in list_files[::10]:
        X, U, eDensity, hDensity = import_data(file)
        listX.append(X)
        listU.append(U)
        listeDensity.append(eDensity)
        listhDensity.append(hDensity)
    line1, = ax.plot(listX[0], listU[0], 'r-')
    ax.set_ylabel('U')
    ax.set_ylabel('Density')
    ax.set_xlabel('X')
    ax.set_title('Solution')
    # ax[0].set_xlim([0, 1])
    ax.set_ylim([-5, 25])
    # ax[1].set_xlim([0, 1])
    # ax[1].set_ylim([-0.1, 1.1])
    def animate(i):
        line1.set_data(listX[i], listU[i])
        return line1
    
    ani = animation.FuncAnimation(fig, animate, frames=len(list_files)//10, interval=100, blit=False)
    plt.show()

    
    
    anim = animation.FuncAnimation(fig, animate, frames=100, interval=100, blit=False)
    plt.show()

if __name__ == "__main__":
    DIR = sys.argv[1]
    animation_sol(DIR)