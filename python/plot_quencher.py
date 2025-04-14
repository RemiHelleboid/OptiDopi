import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import glob, os, sys, time



def smoother(x, window_len=11):
    """Smooth the data using a window of length window_len."""
    return gaussian_filter1d(x, window_len)


def plot_quencher(dirname):
    list_files = glob.glob(f"{dirname}/simulation_*.csv")
    list_files = list_files[:np.min([len(list_files), 100])]

    fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    W = 111
    for file in list_files:
        time,voltage,electron_count,electric_field = np.loadtxt(file, delimiter=',', unpack=True, skiprows=1)
        time = time - time[0]
        
        # Smoothing
        voltage = smoother(voltage, window_len=W)
        electron_count = smoother(electron_count, window_len=W)
        electric_field = smoother(electric_field, window_len=W)

        ax[0].plot(time, voltage, label=file)
        ax[1].plot(time, electron_count, label=file)


    ax[0].set_ylabel('Voltage (V)')
    ax[1].set_ylabel('Electron count')
    ax[1].set_xlabel('Time (s)')

    fig.tight_layout()
    fig.savefig(f"{dirname}/quencher.png")
    plt.show()

def quench_time_distrib(dirname):
    file = f"{dirname}/global_results.txt"
    line_number_quench_times = 0
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            if "Quenching times (s):" in line:
                line_number_quench_times = i
                break
    quench_times = []
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            if i > line_number_quench_times:
                quench_times.append(float(line.split()[0]))
    quench_times = np.array(quench_times)
    return quench_times

def plot_quench_time_distrib(dirname):
    quench_times = quench_time_distrib(dirname)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(quench_times, bins=20, density=True)
    ax.set_xlabel('Quenching time (s)')
    ax.set_ylabel('Density')
    fig.tight_layout()
    fig.savefig(f"{dirname}/quenching_time_distribution.png")
    plt.show()


if __name__ == "__main__":
    dirname = sys.argv[1]
    plot_quencher(dirname)
    plot_quench_time_distrib(dirname)


























