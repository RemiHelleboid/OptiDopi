import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob

# Run process
import subprocess

try:
    import scienceplots
    plt.style.use(['science', 'grid'])
except ImportError:
    pass

INOUE_BIN="/home/remi/DEV/OptiDopi/build/apps/quencher.x"

# Command : ./apps/quencher.x --num_simulations 1000 -p --C 30e-15 --R 12e3 --V_ex=0.5

def run_inoue_sweep_C(minC, maxC, Nsteps, R, Vex, Nsimu_per_C):
    """Run Inoue sweep for capacitance.
    Run the process for different values of capacitance and wait for the process to finish.
    """
    C_values = np.linspace(minC, maxC, Nsteps)
    for C in C_values:
        print(f"Running Inoue for C={C}")
        subprocess.run([INOUE_BIN, "--num_simulations", str(Nsimu_per_C), "-p", "--C", str(C), "--R", str(R), "--V_ex", str(Vex)])
        print(f"Done for C={C}")
        time.sleep(1)

def run_inoue_sweep_R(minR, maxR, Nsteps, C, Vex, Nsimu_per_R):
    """Run Inoue sweep for resistance.
    Run the process for different values of resistance and wait for the process to finish.
    """
    R_values = np.linspace(minR, maxR, Nsteps)
    for R in R_values:
        print(f"Running Inoue for R={R}")
        subprocess.run([INOUE_BIN, "--num_simulations", str(Nsimu_per_R), "-p", "--C", str(C), "--R", str(R), "--V_ex", str(Vex)])
        print(f"Done for R={R}")
        time.sleep(1)

def run_inoue_sweep_Vex(minVex, maxVex, Nsteps, C, R, Nsimu_per_Vex):
    """Run Inoue sweep for external voltage.
    Run the process for different values of external voltage and wait for the process to finish.
    """
    Vex_values = np.linspace(minVex, maxVex, Nsteps)
    for Vex in Vex_values:
        print(f"Running Inoue for Vex={Vex}")
        subprocess.run([INOUE_BIN, "--num_simulations", str(Nsimu_per_Vex), "-p", "--C", str(C), "--R", str(R), "--V_ex", str(Vex)])
        print(f"Done for Vex={Vex}")
        time.sleep(1)

def plot_results(main_dir):
    """Plot the results of the Inoue sweep.
    """
    list_dirs = glob.glob(f"{main_dir}/**/global_results.txt")
    print(list_dirs)
    # C (F) = 3.00000e-14
    # R (Ohms) = 4.10000e+04
    # RC (ns) = 1.23
    # V_bias (V) = 30.05
    # W (cm) = 8.00e-05
    # Probability of avalanche = 0.08
    # Probability of succesufull quenching = 1.00
    list_C = []
    list_R = []
    list_RC = []
    list_Vbias = []
    list_W = []
    list_Pav = []
    list_Pquench = []
    for dir in list_dirs:
        with open(dir, 'r') as f:
            lines = f.readlines()
            C = float(lines[0].split('=')[1])
            R = float(lines[1].split('=')[1])
            RC = float(lines[2].split('=')[1])
            Vbias = float(lines[3].split('=')[1])
            W = float(lines[4].split('=')[1])
            Pav = float(lines[5].split('=')[1])
            Pquench = float(lines[6].split('=')[1])
            list_C.append(C)
            list_R.append(R)
            list_RC.append(RC)
            list_Vbias.append(Vbias)
            list_W.append(W)
            list_Pav.append(Pav)
            list_Pquench.append(Pquench)
    print(list_C)
    print(list_R)
    print(list_RC)
    print(list_Vbias)
    # Find the sweep parameters
    C_values = np.unique(list_C)
    R_values = np.unique(list_R)
    Vbias_values = np.unique(list_Vbias)
    
    # Plot the results
    fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    if len(C_values) > 1:
        ax[0].plot(C_values, list_Pav, label='Pav')
        ax[1].plot(C_values, list_Pquench, label='Pquench')
        ax[1].set_xlabel('C (F)')
    elif len(R_values) > 1:
        ax[0].plot(R_values, list_Pav, label='Pav')
        ax[1].plot(R_values, list_Pquench, label='Pquench')
        ax[1].set_xlabel('R (Ohms)')
    elif len(Vbias_values) > 1:
        ax[0].plot(Vbias_values, list_Pav, label='Pav')
        ax[1].plot(Vbias_values, list_Pquench, label='Pquench')
        ax[1].set_xlabel('Vbias (V)')
    ax[0].set_ylabel('Pav')
    ax[1].set_ylabel('Pquench')
    fig.tight_layout()
    fig.savefig(f"{main_dir}/quencher.png")
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Inoue sweep")
    parser.add_argument("--C", type=float, help="Capacitance value")
    parser.add_argument("--R", type=float, help="Resistance value")
    parser.add_argument("--Vex", type=float, help="External voltage value")
    parser.add_argument("--minC", type=float, help="Minimum capacitance value")
    parser.add_argument("--maxC", type=float, help="Maximum capacitance value")
    parser.add_argument("--NstepsC", type=int, help="Number of steps for capacitance")
    parser.add_argument("--minR", type=float, help="Minimum resistance value")
    parser.add_argument("--maxR", type=float, help="Maximum resistance value")
    parser.add_argument("--NstepsR", type=int, help="Number of steps for resistance")
    parser.add_argument("--minVex", type=float, help="Minimum external voltage value")
    parser.add_argument("--maxVex", type=float, help="Maximum external voltage value")
    parser.add_argument("--NstepsVex", type=int, help="Number of steps for external voltage")
    parser.add_argument("--Nsimu_per_C", type=int, help="Number of simulations per capacitance value")
    parser.add_argument("--Nsimu_per_R", type=int, help="Number of simulations per resistance value")
    parser.add_argument("--Nsimu_per_Vex", type=int, help="Number of simulations per external voltage value")
    # Arg P to only plot the results
    parser.add_argument("-P", action="store_true", help="Plot the results")
    args = parser.parse_args()

    if args.P:
        plot_results(".")
        sys.exit(0)
    if args.C is not None:
        run_inoue_sweep_C(args.C, args.C, 1, args.R, args.Vex, 1)
    if args.minC is not None:
        run_inoue_sweep_C(args.minC, args.maxC, args.NstepsC, args.R, args.Vex, args.Nsimu_per_C)
    if args.minR is not None:
        run_inoue_sweep_R(args.minR, args.maxR, args.NstepsR, args.C, args.Vex, args.Nsimu_per_R)
    if args.minVex is not None:
        run_inoue_sweep_Vex(args.minVex, args.maxVex, args.NstepsVex, args.C, args.R, args.Nsimu_per_Vex)

        


