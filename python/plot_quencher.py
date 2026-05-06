import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
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


def read_global_results_sections(dirname):
    path = os.path.join(dirname, "global_results.txt")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    sections = {}
    current_section = None

    with open(path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line:
                continue

            if line in {"avalanche_times_s", "quench_times_s", "recharge_times_s"}:
                current_section = line
                sections[current_section] = []
                continue

            if current_section is None:
                continue

            try:
                sections[current_section].append(float(line))
            except ValueError:
                pass

    return {key: np.array(values, dtype=float) for key, values in sections.items()}


def time_distribution(dirname, section_name):
    sections = read_global_results_sections(dirname)
    return sections.get(section_name, np.array([], dtype=float))


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
        section_name="avalanche_times_s",
        title="Avalanche time distribution",
        xlabel="Avalanche time (s)",
        output_name="avalanche_time_distribution.png",
    )

    plot_time_distribution(
        dirname=dirname,
        section_name="quench_times_s",
        title="Quench time distribution",
        xlabel="Quench time (s)",
        output_name="quench_time_distribution.png",
    )

    plot_time_distribution(
        dirname=dirname,
        section_name="recharge_times_s",
        title="Recharge time distribution",
        xlabel="Recharge time (s)",
        output_name="recharge_time_distribution.png",
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python plot_quencher.py <simulation_output_directory>")

    output_dir = sys.argv[1]

    plot_quencher(output_dir)
    plot_event_distributions(output_dir)
    plt.show(block=True)