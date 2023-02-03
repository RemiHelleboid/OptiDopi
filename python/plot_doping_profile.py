import numpy as np
import matplotlib.pyplot as plt
import sys

def import_data(filename):
    data = np.loadtxt(filename, skiprows=1, delimiter=',')
    return data

def plot_doping_profile(filename):
    data = import_data(filename)
    # plt.plot(data[:,0], data[:,1], label='Donor')
    # plt.plot(data[:,0], data[:,2], label='Acceptor')
    plt.plot(data[:,0], np.abs(data[:,3]), label='Total')
    plt.xlabel('x (nm)')
    plt.ylabel('Doping (cm-3)')
    plt.yscale('log')
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python plot_doping_profile.py <filename>')
        sys.exit(1)
    arg_filename = sys.argv[1]
    plot_doping_profile(arg_filename)