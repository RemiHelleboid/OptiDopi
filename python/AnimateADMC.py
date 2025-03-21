import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# 3D
from mpl_toolkits.mplot3d import Axes3D
# 3D objects
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from matplotlib.pyplot import cm, title
import glob
import os, sys, time
from argparse import ArgumentParser

from itertools import combinations, product

# import scienceplots
# plt.style.use(['science', 'high-vis'])


def animate_trajectory_2D(dirname, fileFIELD):
    list_files = glob.glob(f"{dirname}/*State*.csv.*")
    list_iter = [int(f.split(".")[-1]) for f in list_files]
    list_files = [f for _, f in sorted(zip(list_iter, list_files))]
    nb_files = len(list_files)
    print(f"Nb files found : {len(list_files)}")
    list_files = list_files[:nb_files//1:5]
    x0,y0,Z,Vx,Vy,Vz,Type,CumulativeIonizationCoeff = np.loadtxt(list_files[0], skiprows=1, delimiter=',', unpack=True, dtype=float, ndmin=2)
    min_x, max_x = 0, 5
    min_y, max_y = 0, 1
    
    fig, ax = plt.subplots(figsize=(5,2))
    ax.set_title("$t = 0$ ps")
    
    x, y = [], []
    ln = plt.scatter([], [], animated=True, marker="o", s=4.5, c=[], cmap="rainbow", vmin=0, vmax=1)
    text = ax.text(0.5, 0.95, "", transform=ax.transAxes, ha="center")
    
    X,Y,Z,ElectrostaticPotential,ElectronDensity,HoleDensity,ElectricField,DopingConcentration = np.loadtxt(fileFIELD, skiprows=1, delimiter=',', unpack=True, dtype=float, ndmin=2)
    # Create a color map in 2D
    
    nx = X.shape[0]
    ny = Y.shape[0]
    NX = int(np.sqrt(nx))
    NY = int(np.sqrt(ny))
    X = X.reshape((NX, NY))
    Y = Y.reshape((NX, NY))
    ElectricField = ElectricField.reshape((NX, NY)).T
    
    # Create a color map in 2D
    ax.imshow(ElectricField, extent=[min_x, max_x, min_y, max_y], origin='upper', cmap='jet')
    
    def init():
        ax.set_xlabel("$x$ ($\mu$m)")
        ax.set_ylabel("$y$ ($\mu$m)")
        eps = 0.1
        ax.set_xlim(min_x-eps, max_x+eps)
        ax.set_ylim(min_y-eps, max_y+eps)
        # Draw a box around the plot (no eps)
        ax.plot([min_x, max_x, max_x, min_x, min_x], [min_y, min_y, max_y, max_y, min_y], 'k-', lw=3)
        
        
        fig.tight_layout()
        return ln,
    
    def update(frame):
        print(f"\rFrame : {frame} / {len(list_files)}", end="", flush=True)
        x,y,Z,Vx,Vy,Vz,Type,CumulativeIonizationCoeff = np.loadtxt(list_files[frame], skiprows=1, delimiter=',', unpack=True, dtype=float, ndmin=2)
        x = x % 5
        y = y % 1
        c=Type
        ln.set_offsets(np.c_[x, y])
        ln.set_array(c)
        # text.set_text(f"t = {frame*0.02:.1f} ps")
        ax.set_title(f"$t = {frame*0.05:.1f}$ ps")
        
        return ln,
    
    ani = FuncAnimation(fig, update, frames=len(list_files), init_func=init, blit=True)
    plt.show()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    ani.save(f"anim_trajectory_{timestamp}.mp4", fps=60, dpi=300)
    

def animate_trajectory_3D(dirname):
    list_files = glob.glob(f"{dirname}/trajectory*.csv")
    list_files.sort(key=os.path.getmtime)
    print(f"Nb files found : {len(list_files)}")
    list_files = list_files[:200]
    x0, y0, z0 = np.loadtxt(list_files[0], skiprows=1, delimiter=',', unpack=True, dtype=float, ndmin=2, usecols=(1, 2, 3))
    min_x, max_x = np.min(x0), np.max(x0)
    min_y, max_y = np.min(y0), np.max(y0)
    min_z, max_z = np.min(z0), np.max(z0)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # sc = ax.scatter(x0, y0, z0, c=z0, cmap="viridis", marker="o", s=2)
    # text = ax.text2D(0.5, 0.95, "", transform=ax.transAxes, ha="center")
    
    # # def init():
        
    # #     # Draw a box around the plot (no eps)
    # #     ax.plot([min_x, max_x, max_x, min_x, min_x], [min_y, min_y, max_y, max_y, min_y], [min_z, min_z, min_z, min_z, min_z], 'k-', lw=3)
    # #     fig.tight_layout()
    # #     return sc,
    
    # def update(frame):
    #     ax.clear()
    #     ax.set_xlabel("x ($\mu$m)")
    #     ax.set_ylabel("y ($\mu$m)")
    #     ax.set_zlabel("z ($\mu$m)")
    #     eps = 0.1
    #     ax.set_xlim(min_x-eps, max_x+eps)
    #     ax.set_ylim(min_y-eps, max_y+eps)
    #     ax.set_zlim(min_z-eps, max_z+eps)
    #     print(f"\rFrame : {frame} / {len(list_files)}", end="", flush=True)
    #     x, y, z = np.loadtxt(list_files[frame], skiprows=1, delimiter=',', unpack=True, dtype=float, ndmin=2, usecols=(1, 2, 3))
    #     sc2 = ax.scatter(x, y, z, c=z, cmap="viridis", marker="o", s=2)
    #     text.set_text(f"t = {frame*0.02:.1f} ps")
    #     # return sc2, 
    
    # ani = FuncAnimation(fig, update, frames=len(list_files), blit=True)
        
    X = []
    Y = []
    Z = []


    def update(t):
        ax.cla()
        ax.plot([min_x, max_x, max_x, min_x, min_x], [min_y, min_y, max_y, max_y, min_y], [min_z, min_z, min_z, min_z, min_z], 'k-', lw=3)
        ax.plot([min_x, max_x, max_x, min_x, min_x], [min_y, min_y, max_y, max_y, min_y], [max_z, max_z, max_z, max_z, max_z], 'k-', lw=3)
        ax.plot([min_x, max_x, max_x, min_x, min_x], [min_y, min_y, min_y, min_y, min_y], [min_z, min_z, max_z, max_z, min_z], 'k-', lw=3)
        ax.plot([min_x, max_x, max_x, min_x, min_x], [max_y, max_y, max_y, max_y, max_y], [min_z, min_z, max_z, max_z, min_z], 'k-', lw=3)

        print(f"\rFrame : {t} / {len(list_files)}", end="", flush=True)
        x, y, z, type = np.loadtxt(list_files[t], skiprows=1, delimiter=',', unpack=True, dtype=float, ndmin=2, usecols=(1, 2, 3, 5))
        ax.scatter(x, y, z, s = 1, marker = '.', c = type, cmap = 'jet')

        eps = 0.1
        ax.set_xlim(min_x-eps, max_x+eps)
        ax.set_ylim(min_y-eps, max_y+eps)
        ax.set_zlim(min_z-eps, max_z+eps)
        
        ax.set_xlabel("x ($\mu$m)")
        ax.set_ylabel("y ($\mu$m)")
        ax.set_zlabel("z ($\mu$m)")
        fig.tight_layout()

    fig = plt.figure(dpi=200)
    # fig.set_size_inches(4, 3)
    ax = fig.add_subplot(projection='3d')
    # X view
    ax.view_init(elev=20, azim=-100)
    fig.tight_layout()

    ani = FuncAnimation(fig = fig, func = update, frames =len(list_files), interval = 1, repeat = False)

    ani.save("anim_trajectory_3D.mp4", fps=10, dpi=300)
    # plt.show()
    plt.show()


def main():
    dirname = sys.argv[1]
    fileFIELD = sys.argv[2]
    dir_history = f"{dirname}/history"
    dir_trajectories = f"{dirname}/"
    
    animate_trajectory_2D(dir_trajectories, fileFIELD)
    # animate_trajectory_3D(dir_trajectories)
    
    

if __name__ == "__main__":
    main()