import numpy as np
from pathlib import Path
import scipy.stats as st
from scipy.signal import chirp, find_peaks, peak_widths
import sys
import glob
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt



try:
    import scienceplots
    plt.style.use(['science', 'high-vis'])
except:
    print("Could not load the style SciencePlots")
    plt.style.use(['seaborn-paper'])

pico_second = 1e12


def FWHM(X, Y, ratio=0.5):
    """Return the full width at half maximum of a curve.

    Args:
        X(_type_): times
        Y(_type_): population
        ratio(float, optional): If set to a value !=0.5, the FWx%M is computed. Defaults to 0.5.

    Returns:
        _type_: The FWx%M.

    Warning:
        Not completely reliable for the moment.
    """
    half_max = max(Y) * ratio
    # find when function crosses line half_max (when sign of diff flips)
    # take the 'derivative' of signum(half_max - Y[])
    d = np.sign(half_max - np.array(Y[0:-1])) - \
        np.sign(half_max - np.array(Y[1:]))
    # plot(X[0:len(d)],d) #if you are interested
    # find the left and right most indexes
    left_idx = np.where(d > 0)[0][0]
    right_idx = np.where(d < 0)[-1][0]
    print(left_idx, right_idx)
    F = X[right_idx] - X[left_idx]  # return the difference (full width)
    Xleft = X[left_idx]
    Xright = X[right_idx]
    return F, Xleft, Xright

def plotFWHM(ax, X, Y, ratio):
    F, Xleft, Xright = FWHM(X, Y, ratio)
    # Show the FWHM as an double arrow on the plot with annotation centered on the arrow
    MaxY = np.max(Y)
    ax.annotate('', xy=(Xleft, MaxY*ratio), xytext=(Xright, MaxY*ratio),
                arrowprops=dict(arrowstyle="<->"), va='center')
    text = f"FWHM\n{F:.2f} ps" if ratio == 0.5 else f"FW{ratio*100}\%\n{F:.2f} ps"
    print(text)
    ax.text((Xleft+Xright)/2, MaxY*ratio*0.8, text, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))

def cumulative_time(times, ratio):
    """Return the time needed to reach the ratio of the total population.

    Args:
        times(_type_): times
        ratio(float): ratio of the total population

    Returns:
        _type_: The time needed to reach the ratio of the total population.
    """
    return times[np.where(np.cumsum(times) >= ratio*np.sum(times))[0][0]]


def jitter_plot(file_path: str, output_file: str, max_time=5e-9, kde_factor=1e-3, plot=False, axs=None, showFWHM=False):
    times_to_avalanche = np.loadtxt(file_path, skiprows=1)

    times_to_avalanche *= pico_second
    max_time *= pico_second

    density_jitter = st.gaussian_kde(times_to_avalanche, kde_factor)
    x_times = np.linspace(-50, max_time, 10000)

    plot_density_jitter = density_jitter(x_times)
    std_jitter = np.std(times_to_avalanche)
    var_jitter = np.var(times_to_avalanche)
    np.savetxt(output_file, np.array(
        [x_times, plot_density_jitter]).T, delimiter=',', comments='', header="time, population")

    max_jitter_distribution = np.max(plot_density_jitter)
    simulation_name = Path(file_path).stem
    max_size_small_name = 12
    small_name = simulation_name if len(
        simulation_name) <= max_size_small_name else simulation_name[:max_size_small_name:]

    print(
        f"FWHM  : {FWHM(x_times, plot_density_jitter)[0]} pico_second")
    print(
        f"FW90% : {FWHM(x_times, plot_density_jitter, 0.10)[0]} pico_second")

    print(
        f"Cumulative time 90% : {cumulative_time(times_to_avalanche, 0.90)} pico_second")

    print(
        f"Cumulative time 50% : {cumulative_time(times_to_avalanche, 0.50)} pico_second")

    print(
        f"Median time : {np.median(times_to_avalanche)} pico_second")

    axs.plot(x_times, plot_density_jitter, alpha=0.9,
             lw=0.75, label=small_name, ls='-')
    axs.set_ylim((max_jitter_distribution*1e-3, max_jitter_distribution*1.1))

    if showFWHM:
        plotFWHM(axs, x_times, plot_density_jitter, 0.5)
        plotFWHM(axs, x_times, plot_density_jitter, 0.1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="file_path",
                        help="The jitter file to parse.", required=True, action='append', nargs='+')
    parser.add_argument("-o", "--output", dest="output_prefix",
                        help="The prefix for the output file (e.g.) the device ref.")
    parser.add_argument("--kde", dest="kde_factor",
                        help="The kde factor to be used in the density kernel estimator.", type=float, default=1e-3)
    parser.add_argument("--max_time", dest="max_time",
                        help="The maximum time of the output distribution.", type=float, default=5e-9)
    parser.add_argument("-p", "--plot", dest="plot",
                        help="Plot the Jitter.", action="store_true", default=False)
    parser.add_argument("--FWHM", dest="plotFWHM",
                        help="Show the FWHM on the plot", action="store_true", default=False)
    args = parser.parse_args()
    list_file_path = args.file_path[0]
    print(list_file_path)

    fig, axs = plt.subplots(1)
    axs.set_yscale("log")
    for file_path in list_file_path:
        output_path = Path(file_path).stem + "_parsed_jitter_.csv"
        jitter_plot(file_path, output_path, max_time=args.max_time,
                    kde_factor=args.kde_factor, plot=args.plot, axs=axs, showFWHM=args.plotFWHM)
    axs.grid(True, axis="both", which="both")
    axs.set_xlabel("time (ps)")
    axs.set_ylabel("Jitter distribution (a.u.)")
    y_max = axs.get_ylim()[1]
    axs.set_ylim(y_max * 1e-3, y_max*1.1)

    axs.legend()
    # axs.set_title(f"Jitter")
    fig.tight_layout()
    fig_path = f"{Path(file_path).stem}_plot.svg"
    print(f"Saving figure to {fig_path}")
    fig.savefig(f"{fig_path}", dpi=300)
    if args.plot:
        fig.set_size_inches(10, 8)
        fig.tight_layout()
        plt.show()
