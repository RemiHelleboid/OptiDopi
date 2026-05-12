import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def smoother(values, sigma=11):
    if len(values) == 0:
        return values
    return gaussian_filter1d(values, sigma=sigma)


def load_simulation_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True)

    if data.size == 0:
        return None

    return {
        "time": data["time"],
        "voltage": data["voltage"],
        "electron_count": data["electron_count"],
        "hole_count": data["hole_count"],
        "electric_field": data["electric_field"],
        "avalanche_current": data["avalanche_current"],
    }


def plot_quencher(dirname, max_files=100, smoothing_sigma=111):
    files = sorted(glob.glob(os.path.join(dirname, "simulation_*.csv")))
    files = files[: min(len(files), max_files)]

    if not files:
        raise FileNotFoundError(f"No simulation_*.csv files found in {dirname}")

    fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)

    for path in files:
        data = load_simulation_csv(path)
        if data is None:
            continue

        time = data["time"] - data["time"][0]

        voltage = smoother(data["voltage"], smoothing_sigma)
        electron_count = smoother(data["electron_count"], smoothing_sigma)
        hole_count = smoother(data["hole_count"], smoothing_sigma)
        avalanche_current = smoother(data["avalanche_current"], smoothing_sigma)

        label = os.path.basename(path)

        axes[0].plot(time, voltage, linewidth=0.8, alpha=0.7, label=label)
        axes[1].plot(time, electron_count, linewidth=0.8, alpha=0.7)
        axes[2].plot(time, hole_count, linewidth=0.8, alpha=0.7)
        axes[3].plot(time, avalanche_current, linewidth=0.8, alpha=0.7)

    axes[0].set_ylabel("Voltage (V)")
    axes[1].set_ylabel("Electrons")
    axes[2].set_ylabel("Holes")
    axes[3].set_ylabel("Avalanche current (A)")
    axes[3].set_xlabel("Time (s)")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(dirname, "quencher.png"), dpi=200)
    plt.show(block=False)



def time_distribution(dirname, section_name):
    file = os.path.join(dirname, f"summary.csv")
    if not os.path.isfile(file):
        print(f"Warning: {file} not found. Skipping time distribution for {section_name}.")
        return np.array([])
    dataset = pd.read_csv(file)
    if section_name not in dataset.columns:
        print(f"Warning: {section_name} not found in {file}. Skipping time distribution.")
        return np.array([])
    return dataset[section_name].dropna().values


def plot_time_distribution(dirname, section_name, title, xlabel, output_name):
    times = time_distribution(dirname, section_name)

    if times.size == 0:
        print(f"No data found for {section_name}. Skipping {output_name}.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(times, bins=50, density=True)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(dirname, output_name), dpi=200)
    plt.show(block=False)


def plot_event_distributions(dirname):
    plot_time_distribution(
        dirname=dirname,
        section_name="avalanche_time_s",
        title="Avalanche time distribution",
        xlabel="Avalanche time (s)",
        output_name="avalanche_time_distribution.png",
    )

    plot_time_distribution(
        dirname=dirname,
        section_name="quench_time_s",
        title="Quench time distribution",
        xlabel="Quench time (s)",
        output_name="quench_time_distribution.png",
    )

    plot_time_distribution(
        dirname=dirname,
        section_name="recharge_time_s",
        title="Recharge time distribution",
        xlabel="Recharge time (s)",
        output_name="recharge_time_distribution.png",
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python plot_quencher.py <simulation_output_directory>")

    output_dir = sys.argv[1]

    plot_quencher(f"{output_dir}/traces/")
    plot_event_distributions(f"{output_dir}/")
    plt.show(block=True)